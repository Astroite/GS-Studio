from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension,BuildExtension
ROOT=Path(__file__).resolve().parent
setup(name="local_splat3_render",version="0.1.0",description="Private RGB3 Gaussian compositor with explicit alpha/T policy",
    license="Apache-2.0",license_files=["LICENSE","NOTICE"],
    ext_modules=[CUDAExtension("local_splat3_render",[str(ROOT/"bindings.cpp"),str(ROOT/"renderer.cu")],
        include_dirs=[str(ROOT)],extra_compile_args={"cxx":["/O2"],"nvcc":["-O3"]})],
    cmdclass={"build_ext":BuildExtension.with_options(use_ninja=True)},python_requires=">=3.10")
