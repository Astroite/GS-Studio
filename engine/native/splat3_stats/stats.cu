// Independent per-pixel accumulation over the existing depth-sorted tile list.
// Public gsplat1.5.3 alpha=.999/threshold/exclusive termination semantics.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void contribution_stats_kernel(
    const float* __restrict__ means, const float* __restrict__ conics,
    const float* __restrict__ opacity, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ ids, const float* __restrict__ error,
    float* __restrict__ sumerr, float* __restrict__ maxw, float* __restrict__ sumw,
    int n, int n_ids, int width, int height, int tile_size, int tile_w, int tile_h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int tile = (y / tile_size) * tile_w + x / tile_size;
    const int start = offsets[tile];
    const int end = tile+1 == tile_w*tile_h ? n_ids : offsets[tile+1];
    if (start < 0 || end < start || end > n_ids) return;
    float transmittance = 1.0f;
    const float pixel_error = error[y*width+x];
    for (int k = start; k < end; ++k) {
        const int g = ids[k];
        if (g < 0 || g >= n) continue;
        const float dx = means[2*g] - (static_cast<float>(x)+.5f);
        const float dy = means[2*g+1] - (static_cast<float>(y)+.5f);
        const float sigma = .5f*(conics[3*g]*dx*dx + conics[3*g+2]*dy*dy) + conics[3*g+1]*dx*dy;
        const float alpha = fminf(.999f, opacity[g]*__expf(-sigma));
        if (sigma < 0.0f || alpha < 1.0f/255.0f) continue;
        const float next = transmittance*(1.0f-alpha);
        if (next <= 1e-4f) break; // Do not include the terminating Gaussian.
        const float weight = transmittance*alpha;
        atomicAdd(sumerr+g, weight*pixel_error);
        // All retained weights are nonnegative, so IEEE float bits are ordered.
        atomicMax(reinterpret_cast<int*>(maxw+g), __float_as_int(weight));
        atomicAdd(sumw+g, weight);
        transmittance = next;
    }
}

// The common public rasterizer tile is16x16. Load each Gaussian once per block,
// then aggregate the32 pixel contributions in each warp before global atomics.
// All256 threads remain alive, including off-image/done pixels, until the next
// block-wide batch barrier. No proprietary code or kernel is used here.
__global__ void contribution_stats_tiled16_kernel(
    const float* __restrict__ means, const float* __restrict__ conics,
    const float* __restrict__ opacity, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ ids, const float* __restrict__ error,
    float* __restrict__ sumerr, float* __restrict__ maxw, float* __restrict__ sumw,
    int n, int n_ids, int width, int height, int tile_w, int tile_h) {
    constexpr unsigned FULL = 0xffffffffu;
    const int rank = threadIdx.y*16 + threadIdx.x;
    const int lane = rank & 31;
    const int x = blockIdx.x*16 + threadIdx.x;
    const int y = blockIdx.y*16 + threadIdx.y;
    const bool inside = x < width && y < height;
    bool done = !inside;
    const float pixel_error = inside ? error[y*width+x] : 0.0f;
    const int tile = blockIdx.y*tile_w + blockIdx.x;
    const int start = offsets[tile];
    const int end = tile+1 == tile_w*tile_h ? n_ids : offsets[tile+1];
    // Range validity is block-uniform, so this early return cannot break sync.
    if (start < 0 || end < start || end > n_ids) return;
    __shared__ int batch_ids[256];
    __shared__ float batch_x[256], batch_y[256], batch_opacity[256];
    __shared__ float batch_a[256], batch_b[256], batch_c[256];
    float transmittance = 1.0f;
    for (int begin = start; begin < end; begin += 256) {
        if (__syncthreads_count(!done) == 0) break;
        const int index = begin + rank;
        if (index < end) {
            const int g = ids[index];
            batch_ids[rank] = g >= 0 && g < n ? g : -1;
            if (g >= 0 && g < n) {
                batch_x[rank] = means[2*g];
                batch_y[rank] = means[2*g+1];
                batch_opacity[rank] = opacity[g];
                batch_a[rank] = conics[3*g];
                batch_b[rank] = conics[3*g+1];
                batch_c[rank] = conics[3*g+2];
            }
        }
        __syncthreads();
        const int size = min(256, end-begin);
        for (int k = 0; k < size; ++k) {
            const int g = batch_ids[k];
            float weight = 0.0f;
            if (!done && g >= 0) {
                const float dx = batch_x[k]-(static_cast<float>(x)+.5f);
                const float dy = batch_y[k]-(static_cast<float>(y)+.5f);
                const float sigma = .5f*(batch_a[k]*dx*dx + batch_c[k]*dy*dy) + batch_b[k]*dx*dy;
                const float alpha = fminf(.999f,batch_opacity[k]*__expf(-sigma));
                if (sigma >= 0.0f && alpha >= 1.0f/255.0f) {
                    const float next = transmittance*(1.0f-alpha);
                    if (next <= 1e-4f) {
                        done = true; // Exclusive: this contribution stays zero.
                    } else {
                        weight = transmittance*alpha;
                        transmittance = next;
                    }
                }
            }
            float total_error = weight*pixel_error;
            float total_weight = weight;
            float maximum = weight;
            #pragma unroll
            for (int shift = 16; shift > 0; shift /= 2) {
                total_error += __shfl_down_sync(FULL,total_error,shift);
                total_weight += __shfl_down_sync(FULL,total_weight,shift);
                maximum = fmaxf(maximum,__shfl_down_sync(FULL,maximum,shift));
            }
            if (lane == 0 && total_weight > 0.0f && g >= 0) {
                atomicAdd(sumerr+g,total_error);
                atomicAdd(sumw+g,total_weight);
                atomicMax(reinterpret_cast<int*>(maxw+g),__float_as_int(maximum));
            }
        }
    }
}

std::vector<torch::Tensor> contribution_stats_cuda(
    const torch::Tensor& means, const torch::Tensor& conics, const torch::Tensor& opacity,
    const torch::Tensor& offsets, const torch::Tensor& ids, const torch::Tensor& error,
    int64_t width, int64_t height, int64_t tile_size) {
    const c10::cuda::CUDAGuard device_guard(means.device());
    const auto options = means.options().requires_grad(false);
    const int64_t n = means.size(1);
    auto sumerr = torch::zeros({n}, options);
    auto maxw = torch::zeros({n}, options);
    auto sumw = torch::zeros({n}, options);
    if (n == 0 || ids.numel() == 0) return {sumerr,maxw,sumw};
    dim3 threads(16,16);
    dim3 blocks((width+15)/16,(height+15)/16);
    auto stream = c10::cuda::getCurrentCUDAStream(means.get_device());
    if (tile_size == 16) {
        contribution_stats_tiled16_kernel<<<blocks,threads,0,stream.stream()>>>(
            means.data_ptr<float>(),conics.data_ptr<float>(),opacity.data_ptr<float>(),
            offsets.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),error.data_ptr<float>(),
            sumerr.data_ptr<float>(),maxw.data_ptr<float>(),sumw.data_ptr<float>(),
            static_cast<int>(n),static_cast<int>(ids.numel()),static_cast<int>(width),
            static_cast<int>(height),static_cast<int>((width+15)/16),static_cast<int>((height+15)/16));
    } else {
        contribution_stats_kernel<<<blocks,threads,0,stream.stream()>>>(
            means.data_ptr<float>(), conics.data_ptr<float>(), opacity.data_ptr<float>(),
            offsets.data_ptr<int32_t>(), ids.data_ptr<int32_t>(), error.data_ptr<float>(),
            sumerr.data_ptr<float>(), maxw.data_ptr<float>(), sumw.data_ptr<float>(),
            static_cast<int>(n), static_cast<int>(ids.numel()), static_cast<int>(width),
            static_cast<int>(height), static_cast<int>(tile_size),
            static_cast<int>((width+tile_size-1)/tile_size),static_cast<int>((height+tile_size-1)/tile_size));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {sumerr,maxw,sumw};
}
