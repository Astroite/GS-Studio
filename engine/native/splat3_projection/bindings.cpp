#include <torch/extension.h>
#include <limits>
#include <cmath>
#include <vector>
using Tensor=at::Tensor;
std::vector<Tensor> projection_forward_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,int64_t,int64_t,double,double,double,double);
std::vector<Tensor> projection_backward_cuda(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,int64_t,int64_t,double);
std::vector<Tensor> aa_covariance_cuda(const Tensor&,const Tensor&,double);

void input(const Tensor& value,const Tensor& anchor,at::ScalarType type=at::kFloat) {
    TORCH_CHECK(value.is_cuda() && value.device()==anchor.device(),"Inputs must share one CUDA device");
    TORCH_CHECK(value.scalar_type()==type && value.is_contiguous(),"Expected contiguous float32 (radii int32)");
}
void common(const Tensor& m,const Tensor& q,const Tensor& s,const Tensor& v,const Tensor& k,int64_t w,int64_t h,double e) {
    TORCH_CHECK(m.dim()==2 && m.size(1)==3,"means must be[N,3]");
    auto n=m.size(0);
    TORCH_CHECK(n<=std::numeric_limits<int32_t>::max()/4,"Gaussian index limit exceeded");
    TORCH_CHECK(q.sizes()==at::IntArrayRef({n,4}) && s.sizes()==at::IntArrayRef({n,3}),"Expected quats[N,4],scales[N,3]");
    TORCH_CHECK(v.sizes()==at::IntArrayRef({1,4,4}) && k.sizes()==at::IntArrayRef({1,3,3}),"Exactly one pinhole camera required");
    TORCH_CHECK(w>0 && h>0 && w<=INT32_MAX && h<=INT32_MAX,"Invalid image dimensions");
    TORCH_CHECK(std::isfinite(e) && e>0,"eps2d must be finite and positive");
    for(const auto& t:{m,q,s,v,k})input(t,m);
}
std::vector<Tensor> forward(const Tensor& m,const Tensor& q,const Tensor& s,const Tensor& o,
    const Tensor& v,const Tensor& k,int64_t w,int64_t h,double e,double near,double far,double clip) {
    common(m,q,s,v,k,w,h,e);
    TORCH_CHECK(o.sizes()==at::IntArrayRef({m.size(0)}),"opacities must be[N]");input(o,m);
    TORCH_CHECK(std::isfinite(near) && near>0 && far>near && !std::isnan(far),"Invalid near/far");
    TORCH_CHECK(std::isfinite(clip) && clip>=0,"radius_clip must be finite and nonnegative");
    return projection_forward_cuda(m,q,s,o,v,k,w,h,e,near,far,clip);
}
std::vector<Tensor> backward(const Tensor& m,const Tensor& q,const Tensor& s,const Tensor& v,const Tensor& k,
    const Tensor& radii,const Tensor& conic,const Tensor& comp,const Tensor& gm,const Tensor& gz,const Tensor& gc,const Tensor& ga,
    int64_t w,int64_t h,double e) {
    common(m,q,s,v,k,w,h,e);auto n=m.size(0);
    TORCH_CHECK(radii.sizes()==at::IntArrayRef({1,n,2}) && gm.sizes()==radii.sizes(),"Invalid radii/means gradient shape");
    TORCH_CHECK(conic.sizes()==at::IntArrayRef({1,n,3}) && gc.sizes()==conic.sizes(),"Invalid conic shape");
    for(const auto& t:{comp,gz,ga})TORCH_CHECK(t.sizes()==at::IntArrayRef({1,n}),"Invalid scalar metadata shape");
    input(radii,m,at::kInt);
    for(const auto& t:{conic,comp,gm,gz,gc,ga})input(t,m);
    return projection_backward_cuda(m,q,s,v,k,radii,conic,comp,gm,gz,gc,ga,w,h,e);
}
std::vector<Tensor> aa(const Tensor& raw,const Tensor& upstream,double eps) {
    TORCH_CHECK(raw.dim()==2 && raw.size(1)==3 && upstream.sizes()==at::IntArrayRef({raw.size(0)}),"AA diagnostic expects raw[N,3],upstream[N]");
    TORCH_CHECK(raw.size(0)<=INT32_MAX && std::isfinite(eps) && eps>0,"Invalid AA diagnostic parameters");
    input(raw,raw);input(upstream,raw);return aa_covariance_cuda(raw,upstream,eps);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward",&forward);m.def("backward",&backward);m.def("aa_covariance",&aa);
    m.attr("implementation_version")="native_floor_v1";
    m.attr("floor_backward")="straight_through_all_three";
}
