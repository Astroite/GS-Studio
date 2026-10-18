// Editor-only ROI visibility. Never linked into training kernels.
#include <torch/extension.h>
torch::Tensor roi_max_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
                          torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,
                          int64_t,int64_t,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("roi_max",&roi_max_cuda); }
