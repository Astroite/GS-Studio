"""Private optional extension; built using existing pinned Torch/CUDA only."""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT=Path(__file__).resolve().parent
setup(
    name="local_splat3_stats", version="0.1.1",
    description="Independent public gsplat contribution statistics for LocalPostshot",
    ext_modules=[CUDAExtension("local_splat3_stats",sources=[str(ROOT/"bindings.cpp"),str(ROOT/"stats.cu")],
        extra_compile_args={"cxx":["/O2"],"nvcc":["-O3"]})],
    cmdclass={"build_ext":BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
)
