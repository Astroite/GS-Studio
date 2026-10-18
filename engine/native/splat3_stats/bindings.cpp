// Independent contribution statistics over public gsplat tile metadata.
// No original Postshot source, PTX, binary, or private gsplat function is used.
#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using Tensor = torch::Tensor;

std::vector<Tensor> contribution_stats_cuda(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const Tensor&, int64_t, int64_t, int64_t);

static void validate(const Tensor& means, const Tensor& conics,
                     const Tensor& opacity, const Tensor& offsets,
                     const Tensor& ids, const Tensor& error,
                     int64_t width, int64_t height, int64_t tile_size) {
    TORCH_CHECK(width > 0 && height > 0, "width and height must be positive");
    TORCH_CHECK(tile_size > 0 && tile_size <= 32, "tile_size must be in1..32");
    TORCH_CHECK(width <= std::numeric_limits<int32_t>::max() &&
                height <= std::numeric_limits<int32_t>::max() &&
                width * height <= std::numeric_limits<int32_t>::max(),
                "image dimensions exceed int32 indexing");
    TORCH_CHECK(means.dim() == 3 && means.size(0) == 1 && means.size(2) == 2,
                "means2d must have shape[1,N,2]");
    const int64_t n = means.size(1);
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "N exceeds int32 indexing");
    TORCH_CHECK(conics.dim() == 3 && conics.size(0) == 1 &&
                conics.size(1) == n && conics.size(2) == 3,
                "conics must have shape[1,N,3]");
    TORCH_CHECK((opacity.dim() == 1 && opacity.size(0) == n) ||
                (opacity.dim() == 2 && opacity.size(0) == 1 && opacity.size(1) == n),
                "opacities must have shape[N] or[1,N]");
    const int64_t tile_h = (height + tile_size - 1) / tile_size;
    const int64_t tile_w = (width + tile_size - 1) / tile_size;
    TORCH_CHECK(offsets.dim() == 3 && offsets.size(0) == 1 &&
                offsets.size(1) == tile_h && offsets.size(2) == tile_w,
                "isect_offsets must have shape[1,ceil(H/tile_size),ceil(W/tile_size)]");
    TORCH_CHECK(ids.dim() == 1 && ids.numel() <= std::numeric_limits<int32_t>::max(),
                "flatten_ids must be an int32 vector");
    TORCH_CHECK(error.dim() == 2 && error.size(0) == height && error.size(1) == width,
                "error_map must have shape[H,W]");
    for (const Tensor& t : {means, conics, opacity, error}) {
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, "float inputs must be float32");
        TORCH_CHECK(t.device() == means.device(), "all inputs must share one device");
        TORCH_CHECK(t.is_contiguous(), "all inputs must be contiguous");
    }
    for (const Tensor& t : {offsets, ids}) {
        TORCH_CHECK(t.scalar_type() == torch::kInt32, "tile offsets and flatten_ids must be int32");
        TORCH_CHECK(t.device() == means.device(), "all inputs must share one device");
        TORCH_CHECK(t.is_contiguous(), "all inputs must be contiguous");
    }
    TORCH_CHECK(means.device().is_cpu() || means.device().is_cuda(), "only CPU/CUDA inputs supported");
}

static std::vector<Tensor> contribution_stats_cpu(
    const Tensor& means, const Tensor& conics, const Tensor& opacity,
    const Tensor& offsets, const Tensor& ids, const Tensor& error,
    int64_t width, int64_t height, int64_t tile_size) {
    const int64_t n = means.size(1), count = ids.numel();
    auto options = means.options().requires_grad(false);
    Tensor sumerr = torch::zeros({n}, options);
    Tensor maxw = torch::zeros({n}, options);
    Tensor sumw = torch::zeros({n}, options);
    if (n == 0 || count == 0) return {sumerr, maxw, sumw};
    const auto *xy = means.data_ptr<float>(), *co = conics.data_ptr<float>();
    const auto *op = opacity.data_ptr<float>(), *err = error.data_ptr<float>();
    const auto *off = offsets.data_ptr<int32_t>(), *flat = ids.data_ptr<int32_t>();
    auto *out = sumerr.data_ptr<float>(), *mx = maxw.data_ptr<float>(), *sw = sumw.data_ptr<float>();
    const int64_t tile_w = (width + tile_size - 1) / tile_size;
    const int64_t tile_h = (height + tile_size - 1) / tile_size;
    for (int64_t y = 0; y < height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            const int64_t tile = (y / tile_size) * tile_w + x / tile_size;
            const int64_t start = off[tile];
            const int64_t end = tile + 1 == tile_h * tile_w ? count : off[tile + 1];
            // Public rasterizer metadata is required to be well-formed. Guard
            // corrupted indices without launching unsafe memory accesses.
            if (start < 0 || end < start || end > count) continue;
            float transmittance = 1.0f;
            for (int64_t k = start; k < end; ++k) {
                const int32_t g = flat[k];
                if (g < 0 || g >= n) continue;
                const float dx = xy[2*g] - (static_cast<float>(x) + .5f);
                const float dy = xy[2*g+1] - (static_cast<float>(y) + .5f);
                const float sigma = .5f * (co[3*g]*dx*dx + co[3*g+2]*dy*dy) + co[3*g+1]*dx*dy;
                const float alpha = std::min(.999f, op[g] * std::exp(-sigma));
                if (sigma < 0.0f || alpha < 1.0f/255.0f) continue;
                const float next = transmittance * (1.0f-alpha);
                if (next <= 1e-4f) break; // Exclusive, as public gsplat1.5.3.
                const float weight = transmittance * alpha;
                out[g] += weight * err[y*width+x];
                mx[g] = std::max(mx[g], weight);
                sw[g] += weight;
                transmittance = next;
            }
        }
    }
    return {sumerr, maxw, sumw};
}

std::vector<Tensor> contribution_stats(
    const Tensor& means, const Tensor& conics, const Tensor& opacity,
    const Tensor& offsets, const Tensor& ids, const Tensor& error,
    int64_t width, int64_t height, int64_t tile_size) {
    validate(means, conics, opacity, offsets, ids, error, width, height, tile_size);
    if (means.is_cuda())
        return contribution_stats_cuda(means, conics, opacity, offsets, ids, error, width, height, tile_size);
    return contribution_stats_cpu(means, conics, opacity, offsets, ids, error, width, height, tile_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Independent C=1 public-rasterizer contribution statistics v1; no autograd";
    module.def("contribution_stats", &contribution_stats,
               pybind11::arg("means2d"), pybind11::arg("conics"), pybind11::arg("opacities"),
               pybind11::arg("isect_offsets"), pybind11::arg("flatten_ids"), pybind11::arg("error_map"),
               pybind11::arg("width"), pybind11::arg("height"), pybind11::arg("tile_size"));
    module.attr("implementation_version") = "native_cuda_v1";
}
