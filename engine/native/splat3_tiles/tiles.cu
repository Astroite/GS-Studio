// Independently written tile culling, CUB stable pair sort, and offset encoding.
// Public gsplat 1.5.3 supplies the input/output representation and bbox contract.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <climits>
#include <cmath>
#include <vector>
using Tensor = at::Tensor;

__device__ __forceinline__ float clamped(float v,float lo,float hi) { return fminf(fmaxf(v,lo),hi); }
__device__ __forceinline__ float q_lower(float a,float b,float c,float x,float y) {
    float ax=a*x*x, by=2.f*b*x*y, cy=c*y*y;
    // Rounding safety includes cancellation and optimizer-coordinate rounding.
    return fmaf(a*x,x,fmaf(2.f*b*x,y,cy))-2e-5f*(fabsf(ax)+fabsf(by)+fabsf(cy)+1.f);
}
__device__ __forceinline__ bool conservative(float mx,float my,float a,float b,float c,float tau,
                                           float x0,float y0,float x1,float y1) {
    if(mx>=x0 && mx<=x1 && my>=y0 && my<=y1) return true;
    x0-=mx; x1-=mx; y0-=my; y1-=my;
    // Four edge minima of a convex quadratic; each evaluated lower bound is
    // conservative. Full tile rectangle contains all its valid pixel centers.
    float y=clamped(-b*x0/c,y0,y1);
    float q=q_lower(a,b,c,x0,y);
    y=clamped(-b*x1/c,y0,y1);q=fminf(q,q_lower(a,b,c,x1,y));
    float x=clamped(-b*y0/a,x0,x1);q=fminf(q,q_lower(a,b,c,x,y0));
    x=clamped(-b*y1/a,x0,x1);q=fminf(q,q_lower(a,b,c,x,y1));
    return !isfinite(q) || q<=tau+2e-4f*(1.f+tau);
}
__device__ __forceinline__ bool native_edges(float mx,float my,float a,float b,float c,float tau,
                                           float x0,float y0,float x1,float y1) {
    if(mx>=x0 && mx<=x1 && my>=y0 && my<=y1) return true;
    float y=((y0+y1>2.f*my)?y0:y1)-my, B=2.f*b*y;
    float D=fmaf(B,B,4.f*a*fmaf(-y*y,c,tau));
    float edge=fmaxf(fmaf(x0-mx,2.f*a,B),-fmaf(x1-mx,2.f*a,B));
    if(D>=0.f && (edge<=0.f || D>=edge*edge)) return true;
    float x=((x0+x1>2.f*mx)?x0:x1)-mx; B=2.f*b*x;
    D=fmaf(B,B,4.f*c*fmaf(-x*x,a,tau));
    edge=fmaxf(fmaf(y0-my,2.f*c,B),-fmaf(y1-my,2.f*c,B));
    return D>=0.f && (edge<=0.f || D>=edge*edge);
}

template<bool Emit>
__global__ void intersections(const float* m,const int* radii,const float* depth,const float* conic,
                              const float* opacity,int n,int tile,int tw,int th,int mode,
                              int* counts,const int64_t* prefix,int64_t* keys,int* ids) {
    int g=blockIdx.x*blockDim.x+threadIdx.x;
    if(g>=n) return;
    if(!Emit) counts[g]=0;
    int rx=radii[2*g], ry=radii[2*g+1];
    float mx=m[2*g],my=m[2*g+1];
    if(rx<=0 || ry<=0 || !isfinite(mx) || !isfinite(my)) return;
    float a=conic[3*g],b=conic[3*g+1],c=conic[3*g+2],op=opacity[g];
    float ac=a*c, bb=b*b,det=fmaf(a,c,-bb);
    // Ill-conditioned/invalid data retains public bbox rather than risking a
    // false rejection; never introduce NaNs to the existing renderer.
    bool valid=isfinite(a)&&isfinite(b)&&isfinite(c)&&isfinite(op)&&a>0.f&&c>0.f&&op>0.f&&det>0.f;
    bool safe=valid && det>2e-5f*fmaxf(ac,bb);
    bool native=(mode==1 && valid);
    float tau=0.f;
    int x0,y0,x1,y1;
    if(native) {
        float xx=c/det,xy=-b/det,yy=a/det,mid=.5f*(xx+yy);
        float covdet=fmaf(xx,yy,-xy*xy);
        float eigen=mid+sqrtf(fmaxf(.01f,fmaf(mid,mid,-covdet)));
        tau=2.f*logf(fmaxf(1.f,__uint_as_float(0x437effff)*op));
        float radius=sqrtf(eigen*tau);
        if(radius<=0.f) return;
        if(!isfinite(radius)) native=false;
        else {
            // Clamp in floating point BEFORE conversion. CUDA float-to-int
            // behavior outside int32 range is not a portable support contract.
            // Lower can equal grid extent (empty); upper can equal -1 (empty).
            x0=__float2int_rd(clamped((mx-radius)/tile,0.f,float(tw)));
            y0=__float2int_rd(clamped((my-radius)/tile,0.f,float(th)));
            x1=__float2int_rd(clamped((mx+radius)/tile,-1.f,float(tw-1)));
            y1=__float2int_rd(clamped((my+radius)/tile,-1.f,float(th-1)));
        }
    }
    if(!native) {
        float tx=mx/tile,ty=my/tile,trx=float(rx)/tile,try_=float(ry)/tile;
        x0=min(max(0,__float2int_rd(tx-trx)),tw); y0=min(max(0,__float2int_rd(ty-try_)),th);
        x1=min(max(0,__float2int_ru(tx+trx)),tw)-1; y1=min(max(0,__float2int_ru(ty+try_)),th)-1;
        if(safe) tau=2.f*logf(fmaxf(1.f,op/__uint_as_float(0x3b808081)));
    }
    bool one=(x0==x1 && y0==y1);
    int count=0;
    int64_t start=Emit?(g?prefix[g-1]:0):0;
    for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
        // Native emit skips the edge test for one tile only after its earlier
        // projection/count pass proved nonempty support. We combine both
        // passes here, so native mode still needs this test when bbox size=1.
        bool keep=(!native && one) || !(native?valid:safe);
        if(!keep) {
            float px=float(x*tile),py=float(y*tile);
            keep=native?native_edges(mx,my,a,b,c,tau,px,py,px+tile,py+tile):conservative(mx,my,a,b,c,tau,px,py,px+tile,py+tile);
        }
        if(keep) {
            if(Emit) {
                keys[start+count]=(int64_t(y*tw+x)<<32)|uint32_t(__float_as_uint(depth[g]));
                ids[start+count]=g;
            }
            ++count;
        }
    }
    if(!Emit) counts[g]=count;
}

__global__ void offsets_kernel(const int64_t* keys,int count,int* offsets,int tiles) {
    int tile=blockIdx.x*blockDim.x+threadIdx.x;
    if(tile>=tiles) return;
    int lo=0,hi=count;
    while(lo<hi) { int mid=lo+(hi-lo)/2; if((keys[mid]>>32)<tile) lo=mid+1;else hi=mid; }
    offsets[tile]=lo;
}

std::vector<Tensor> intersect_cuda(const Tensor& m,const Tensor& r,const Tensor& d,const Tensor& c,const Tensor& o,
                                 int64_t width,int64_t height,int64_t tile,int64_t mode) {
    c10::cuda::CUDAGuard guard(m.device());
    auto stream=at::cuda::getCurrentCUDAStream();
    int n=int(m.size(1)),tw=int((width+tile-1)/tile),th=int((height+tile-1)/tile);
    auto counts=at::zeros({1,n},m.options().dtype(at::kInt));
    auto offsets=at::zeros({1,th,tw},m.options().dtype(at::kInt));
    auto keys=at::empty({0},m.options().dtype(at::kLong));
    auto ids=at::empty({0},m.options().dtype(at::kInt));
    if(!n) return {counts,keys,ids,offsets};
    intersections<false><<<(n+255)/256,256,0,stream>>>(m.data_ptr<float>(),r.data_ptr<int>(),d.data_ptr<float>(),c.data_ptr<float>(),o.data_ptr<float>(),n,tile,tw,th,mode,counts.data_ptr<int>(),nullptr,nullptr,nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto prefix=at::cumsum(counts.reshape({n}),0,at::kLong);
    int64_t count=prefix.select(0,n-1).item<int64_t>();
    TORCH_CHECK(count<=INT_MAX,"Tile intersection count exceeds int32 indexing");
    if(!count) return {counts,keys,ids,offsets};
    keys=at::empty({count},m.options().dtype(at::kLong));
    ids=at::empty({count},m.options().dtype(at::kInt));
    intersections<true><<<(n+255)/256,256,0,stream>>>(m.data_ptr<float>(),r.data_ptr<int>(),d.data_ptr<float>(),c.data_ptr<float>(),o.data_ptr<float>(),n,tile,tw,th,mode,nullptr,prefix.data_ptr<int64_t>(),keys.data_ptr<int64_t>(),ids.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto sorted_keys=at::empty_like(keys),sorted_ids=at::empty_like(ids);
    size_t bytes=0;
    int bits=32;for(int t=tw*th;t;t>>=1)++bits;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,bytes,keys.data_ptr<int64_t>(),sorted_keys.data_ptr<int64_t>(),ids.data_ptr<int>(),sorted_ids.data_ptr<int>(),int(count),0,bits,stream));
    auto workspace=at::empty({int64_t(bytes)},m.options().dtype(at::kByte));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(workspace.data_ptr(),bytes,keys.data_ptr<int64_t>(),sorted_keys.data_ptr<int64_t>(),ids.data_ptr<int>(),sorted_ids.data_ptr<int>(),int(count),0,bits,stream));
    offsets_kernel<<<(tw*th+255)/256,256,0,stream>>>(sorted_keys.data_ptr<int64_t>(),int(count),offsets.data_ptr<int>(),tw*th);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {counts,sorted_keys,sorted_ids,offsets};
}

// Test-only exhaustive audit. Same float32 expression, __expf, cutoff and cap as
// the frozen renderer forward; independent of the culling decisions above.
__global__ void audit_kernel(const float* means,const float* conics,const float* opacity,const bool* present,
                            int n,int w,int h,int tile,int tw,int tiles,int* missing) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,total=int64_t(n)*w*h;
    if(i>=total) return;
    int pixel=int(i%(w*h)),g=int(i/(w*h)),y=pixel/w,x=pixel-y*w;
    float dx=means[2*g]-(float(x)+.5f),dy=means[2*g+1]-(float(y)+.5f);
    float sigma=.5f*(conics[3*g]*dx*dx+conics[3*g+2]*dy*dy)+conics[3*g+1]*dx*dy;
    float alpha=fminf(.998046875f,opacity[g]*__expf(-sigma));
    if(sigma<0.f || alpha<__uint_as_float(0x3b808081) || !isfinite(sigma) || !isfinite(opacity[g]))return;
    if(!present[int64_t(g)*tiles+(y/tile)*tw+x/tile])atomicAdd(missing+g,1);
}
Tensor audit_contributors_cuda(const Tensor& means,const Tensor& conics,const Tensor& opacity,const Tensor& present,
                               int64_t w,int64_t h,int64_t tile) {
    c10::cuda::CUDAGuard guard(means.device());
    int n=int(means.size(1)),tw=int((w+tile-1)/tile),tiles=tw*int((h+tile-1)/tile);
    auto missing=at::zeros({n},means.options().dtype(at::kInt));
    int64_t total=int64_t(n)*w*h;
    if(total) audit_kernel<<<unsigned((total+255)/256),256,0,at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(),conics.data_ptr<float>(),opacity.data_ptr<float>(),present.data_ptr<bool>(),n,w,h,tile,tw,tiles,missing.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return missing;
}
