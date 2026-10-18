#include <torch/extension.h>
#include <climits>
#include <vector>
using Tensor = at::Tensor;
std::vector<Tensor> intersect_cuda(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, int64_t, int64_t);
Tensor audit_contributors_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,int64_t,int64_t,int64_t);
std::vector<Tensor> intersect(const Tensor& m, const Tensor& r, const Tensor& d, const Tensor& c, const Tensor& o,
                             int64_t w, int64_t h, int64_t tile, int64_t mode) {
    TORCH_CHECK(m.is_cuda() && m.scalar_type()==at::kFloat && m.is_contiguous(), "CUDA contiguous float32 means required");
    TORCH_CHECK(m.dim()==3 && m.size(0)==1 && m.size(2)==2, "means must be [1,N,2]");
    const auto n=m.size(1);
    TORCH_CHECK(n<=INT_MAX/3 && w>0 && h>0 && w<=INT_MAX && h<=INT_MAX && w*h<=INT_MAX, "invalid dimensions");
    TORCH_CHECK(tile>0 && tile<=32 && (mode==0 || mode==1), "invalid tile size or support mode");
    TORCH_CHECK(r.sizes()==at::IntArrayRef({1,n,2}) && c.sizes()==at::IntArrayRef({1,n,3}), "bad radii/conic shape");
    TORCH_CHECK(d.sizes()==at::IntArrayRef({1,n}), "bad depth shape");
    TORCH_CHECK(o.sizes()==at::IntArrayRef({1,n}) || o.sizes()==at::IntArrayRef({n}), "bad opacity shape");
    for (const auto& v:{d,c,o}) TORCH_CHECK(v.device()==m.device() && v.scalar_type()==at::kFloat && v.is_contiguous(), "float input device/dtype/contiguity mismatch");
    TORCH_CHECK(r.device()==m.device() && r.scalar_type()==at::kInt && r.is_contiguous(), "bad radii device/dtype/contiguity");
    return intersect_cuda(m,r,d,c,o,w,h,tile,mode);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.attr("implementation_version")="tile_support_v1";
    m.def("intersect", &intersect);
    m.def("audit_contributors", [](const Tensor& means,const Tensor& conics,const Tensor& opacity,const Tensor& present,int64_t w,int64_t h,int64_t tile) {
        TORCH_CHECK(means.is_cuda() && means.dim()==3 && means.size(0)==1 && means.size(2)==2,"bad audit means");
        TORCH_CHECK(w>0 && h>0 && w*h<=INT_MAX && tile>0 && tile<=32,"bad audit dimensions");
        auto n=means.size(1);
        TORCH_CHECK(conics.sizes()==at::IntArrayRef({1,n,3}) && opacity.sizes()==at::IntArrayRef({1,n}),"bad audit conic/opacity shape");
        for(const auto& v:{means,conics,opacity})TORCH_CHECK(v.device()==means.device() && v.scalar_type()==at::kFloat && v.is_contiguous(),"bad audit float input");
        TORCH_CHECK(present.device()==means.device() && present.scalar_type()==at::kBool && present.is_contiguous() && present.sizes()==at::IntArrayRef({n,((w+tile-1)/tile)*((h+tile-1)/tile)}),"bad audit tile presence");
        return audit_contributors_cuda(means,conics,opacity,present,w,h,tile);
    });
}
