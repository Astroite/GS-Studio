#include <torch/extension.h>
#include <limits>
#include <climits>
#include <vector>
using Tensor=torch::Tensor;
std::vector<Tensor> render_fwd_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,int64_t,int64_t,int64_t);
std::vector<Tensor> render_bwd_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,int64_t,int64_t,int64_t);
std::vector<Tensor> renderer_stats_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,int64_t,int64_t,int64_t);

void check(const Tensor& v,const Tensor& anchor,at::ScalarType dtype) {
    TORCH_CHECK(v.is_cuda() && v.device()==anchor.device(),"all inputs must share one CUDA device");
    TORCH_CHECK(v.scalar_type()==dtype && v.is_contiguous(),"wrong dtype or noncontiguous input");
}
void common(const Tensor& means,const Tensor& conics,const Tensor& opacity,const Tensor& offsets,
            const Tensor& ids,int64_t width,int64_t height,int64_t tile) {
    TORCH_CHECK(width>0 && height>0 && width<=INT_MAX && height<=INT_MAX && width*height<=INT_MAX,"invalid image dimensions");
    TORCH_CHECK(tile>=1 && tile<=32,"tile_size must be1..32");
    TORCH_CHECK(means.dim()==3 && means.size(0)==1 && means.size(2)==2,"means2d must be[1,N,2]");
    auto n=means.size(1);
    TORCH_CHECK(n<=INT_MAX/3 && ids.numel()<=INT_MAX,"int32 index limit exceeded");
    TORCH_CHECK(conics.sizes()==at::IntArrayRef({1,n,3}),"conics must be[1,N,3]");
    TORCH_CHECK(opacity.sizes()==at::IntArrayRef({n}) || opacity.sizes()==at::IntArrayRef({1,n}),"opacity must be[N] or[1,N]");
    TORCH_CHECK(offsets.sizes()==at::IntArrayRef({1,(height+tile-1)/tile,(width+tile-1)/tile}),"invalid offsets shape");
    TORCH_CHECK(ids.dim()==1,"flatten_ids must be a vector");
    for(const auto& v:{means,conics,opacity})check(v,means,at::kFloat);
    for(const auto& v:{offsets,ids})check(v,means,at::kInt);
}
void rgb_check(const Tensor& means,const Tensor& colors,const Tensor& bg,int64_t tile) {
    TORCH_CHECK(tile==16,"native RGB compositor currently requires tile_size16");
    TORCH_CHECK(colors.sizes()==at::IntArrayRef({1,means.size(1),3}),"colors must be[1,N,3]");
    TORCH_CHECK(bg.sizes()==at::IntArrayRef({1,3}),"background must be[1,3]");
    check(colors,means,at::kFloat);check(bg,means,at::kFloat);
}
std::vector<Tensor> forward(const Tensor& m,const Tensor& c,const Tensor& rgb,const Tensor& o,
    const Tensor& bg,const Tensor& off,const Tensor& ids,int64_t w,int64_t h,int64_t tile) {
    common(m,c,o,off,ids,w,h,tile);rgb_check(m,rgb,bg,tile);
    return render_fwd_cuda(m,c,rgb,o,bg,off,ids,w,h,tile);
}
std::vector<Tensor> backward(const Tensor& m,const Tensor& c,const Tensor& rgb,const Tensor& o,
    const Tensor& bg,const Tensor& off,const Tensor& ids,const Tensor& trans,const Tensor& last,
    const Tensor& grgb,const Tensor& ga,int64_t w,int64_t h,int64_t tile) {
    common(m,c,o,off,ids,w,h,tile);rgb_check(m,rgb,bg,tile);
    for(const auto& v:{trans,ga}) {
        check(v,m,at::kFloat);TORCH_CHECK(v.sizes()==at::IntArrayRef({1,h,w,1}),"bad alpha/trans gradient shape");
    }
    check(last,m,at::kInt);TORCH_CHECK(last.sizes()==at::IntArrayRef({1,h,w}),"bad last IDs shape");
    check(grgb,m,at::kFloat);TORCH_CHECK(grgb.sizes()==at::IntArrayRef({1,h,w,3}),"bad RGB gradient shape");
    return render_bwd_cuda(m,c,rgb,o,bg,off,ids,trans,last,grgb,ga,w,h,tile);
}
std::vector<Tensor> statistics(const Tensor& m,const Tensor& c,const Tensor& o,const Tensor& off,
    const Tensor& ids,const Tensor& error,int64_t w,int64_t h,int64_t tile) {
    common(m,c,o,off,ids,w,h,tile);check(error,m,at::kFloat);
    TORCH_CHECK(error.sizes()==at::IntArrayRef({h,w}),"error_map must be[H,W]");
    return renderer_stats_cuda(m,c,o,off,ids,error,w,h,tile);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward",&forward);m.def("backward",&backward);m.def("contribution_stats",&statistics);
    m.attr("implementation_version")="native_renderer_v1";
    m.attr("alpha_cap")=511.0/512.0;
    m.attr("transmittance_policy")="old_T_ge_float32_1e-4_inclusive_current";
    m.attr("cap_backward")="straight_through";
}
