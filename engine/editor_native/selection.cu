// Independent editor helper: original full-frame means/conics/sorted tile list.
// Same float32 arithmetic/caps/termination as the paired frozen statistics.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void roi_kernel(const float* means,const float* conics,const float* opacity,
    const int32_t* offsets,const int32_t* ids,float* maximum,
    int n,int n_ids,int width,int height,int tile,int left,int top,int right,int bottom,bool native) {
    const int x=left+blockIdx.x*blockDim.x+threadIdx.x;
    const int y=top+blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=right || y>=bottom) return;
    const int tw=(width+tile-1)/tile,th=(height+tile-1)/tile;
    const int t=(y/tile)*tw+x/tile;
    const int start=offsets[t],end=t+1==tw*th ? n_ids : offsets[t+1];
    if(start<0 || end<start || end>n_ids) return;
    float transmittance=1.0f;
    for(int k=start;k<end;++k) {
        const int g=ids[k];
        if(g<0 || g>=n) continue;
        const float dx=means[2*g]-(static_cast<float>(x)+.5f);
        const float dy=means[2*g+1]-(static_cast<float>(y)+.5f);
        const float sigma=.5f*(conics[3*g]*dx*dx+conics[3*g+2]*dy*dy)+conics[3*g+1]*dx*dy;
        const float alpha=fminf(native ? .998046875f : .999f,opacity[g]*__expf(-sigma));
        if(sigma<0.0f || alpha<1.0f/255.0f) continue;
        const float next=transmittance*(1.0f-alpha);
        if(!native && next<=1e-4f) break;
        const float weight=transmittance*alpha;
        atomicMax(reinterpret_cast<int*>(maximum+g),__float_as_int(weight));
        transmittance=next;
        if(native && transmittance<1e-4f) break;
    }
}

torch::Tensor roi_max_cuda(torch::Tensor means,torch::Tensor conics,torch::Tensor opacity,
    torch::Tensor offsets,torch::Tensor ids,int64_t width,int64_t height,int64_t tile,
    int64_t left,int64_t top,int64_t right,int64_t bottom,bool native) {
    TORCH_CHECK(means.is_cuda() && means.dim()==3 && means.size(0)==1 && means.size(2)==2,"CUDA [1,N,2] means required");
    const int64_t n=means.size(1);
    TORCH_CHECK(width>0 && height>0 && tile>0 && tile<=32,"Invalid raster dimensions");
    TORCH_CHECK(left>=0 && top>=0 && right<=width && bottom<=height && right>=left && bottom>=top,"Invalid ROI");
    TORCH_CHECK(conics.sizes()==torch::IntArrayRef({1,n,3}) && opacity.numel()==n,"Invalid projected tensors");
    TORCH_CHECK(offsets.dim()==3 && offsets.size(0)==1 && offsets.size(1)==(height+tile-1)/tile && offsets.size(2)==(width+tile-1)/tile && ids.dim()==1,"Invalid tile lists");
    for(const auto& t:{means,conics,opacity,offsets,ids}) {
        TORCH_CHECK(t.device()==means.device() && t.is_contiguous(),"Contiguous same-device tensors required");
    }
    TORCH_CHECK(means.scalar_type()==torch::kFloat32 && conics.scalar_type()==torch::kFloat32 && opacity.scalar_type()==torch::kFloat32 && offsets.scalar_type()==torch::kInt32 && ids.scalar_type()==torch::kInt32,"Float32 projection and int32 tile lists required");
    const c10::cuda::CUDAGuard guard(means.device());
    auto result=torch::zeros({n},means.options().requires_grad(false));
    if(n && ids.numel() && right>left && bottom>top) {
        dim3 threads(16,16),blocks((right-left+15)/16,(bottom-top+15)/16);
        auto stream=c10::cuda::getCurrentCUDAStream(means.get_device());
        roi_kernel<<<blocks,threads,0,stream.stream()>>>(means.data_ptr<float>(),conics.data_ptr<float>(),opacity.data_ptr<float>(),offsets.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),result.data_ptr<float>(),n,ids.numel(),width,height,tile,left,top,right,bottom,native);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return result;
}
