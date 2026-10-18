// Private RGB3 specialization; see LICENSE/NOTICE/provenance.json.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "RasterizeToPixels3DGSFwd.cu"
#include "RasterizeToPixels3DGSBwd.cu"
#include "Statistics.cu"

using Tensor=torch::Tensor;
using Optional=at::optional<Tensor>;

std::vector<Tensor> render_fwd_cuda(const Tensor& means,const Tensor& conics,
    const Tensor& colors,const Tensor& opa,const Tensor& background,const Tensor& offsets,
    const Tensor& ids,int64_t width,int64_t height,int64_t tile) {
    c10::cuda::CUDAGuard guard(means.device());
    auto opts=means.options().requires_grad(false);
    auto rgb=torch::zeros({1,height,width,3},opts);
    auto alpha=torch::zeros({1,height,width,1},opts);
    auto trans=torch::ones({1,height,width,1},opts);
    auto last=torch::full({1,height,width},-1,means.options().dtype(torch::kInt32).requires_grad(false));
    if(ids.numel()==0 || means.size(1)==0) {
        rgb.copy_(background.view({1,1,1,3}).expand_as(rgb));
    } else {
        local_render::launch_rasterize_to_pixels_3dgs_fwd_kernel<3>(means,conics,colors,opa,
            Optional(background),at::nullopt,width,height,tile,offsets,ids,rgb,alpha,last,trans);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {rgb,alpha,trans,last};
}

std::vector<Tensor> render_bwd_cuda(const Tensor& means,const Tensor& conics,
    const Tensor& colors,const Tensor& opa,const Tensor& background,const Tensor& offsets,
    const Tensor& ids,const Tensor& trans,const Tensor& last,const Tensor& grad_rgb,
    const Tensor& grad_alpha,int64_t width,int64_t height,int64_t tile) {
    c10::cuda::CUDAGuard guard(means.device());
    auto gm=torch::zeros_like(means),gc=torch::zeros_like(conics),
         grgb=torch::zeros_like(colors),go=torch::zeros_like(opa);
    if(ids.numel()!=0 && means.size(1)!=0) {
        local_render::launch_rasterize_to_pixels_3dgs_bwd_kernel<3>(means,conics,colors,opa,
            Optional(background),at::nullopt,width,height,tile,offsets,ids,trans,last,
            grad_rgb,grad_alpha,at::nullopt,gm,gc,grgb,go);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    auto gb=(grad_rgb*trans).sum({1,2});
    return {gm,gc,grgb,go,gb};
}

std::vector<Tensor> renderer_stats_cuda(const Tensor& means,const Tensor& conics,
    const Tensor& opa,const Tensor& offsets,const Tensor& ids,const Tensor& error,
    int64_t width,int64_t height,int64_t tile) {
    c10::cuda::CUDAGuard guard(means.device());
    auto opts=means.options().requires_grad(false);
    auto se=torch::zeros({means.size(1)},opts),mw=torch::zeros_like(se),sw=torch::zeros_like(se);
    if(ids.numel()==0 || means.size(1)==0)return {se,mw,sw};
    dim3 threads(16,16),grid((width+15)/16,(height+15)/16);
    auto stream=c10::cuda::getCurrentCUDAStream(means.get_device());
    if(tile==16) {
        renderer_stats_tiled16_kernel<<<grid,threads,0,stream.stream()>>>(
            means.data_ptr<float>(),conics.data_ptr<float>(),opa.data_ptr<float>(),offsets.data_ptr<int32_t>(),
            ids.data_ptr<int32_t>(),error.data_ptr<float>(),se.data_ptr<float>(),mw.data_ptr<float>(),sw.data_ptr<float>(),
            means.size(1),ids.numel(),width,height,(width+15)/16,(height+15)/16);
    } else {
        renderer_stats_kernel<<<grid,threads,0,stream.stream()>>>(
            means.data_ptr<float>(),conics.data_ptr<float>(),opa.data_ptr<float>(),offsets.data_ptr<int32_t>(),
            ids.data_ptr<int32_t>(),error.data_ptr<float>(),se.data_ptr<float>(),mw.data_ptr<float>(),sw.data_ptr<float>(),
            means.size(1),ids.numel(),width,height,tile,(width+tile-1)/tile,(height+tile-1)/tile);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {se,mw,sw};
}
