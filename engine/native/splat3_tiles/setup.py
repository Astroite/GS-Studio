from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
ROOT = Path(__file__).resolve().parent
setup(name="local_splat3_tiles", version="0.1.0",
      description="Private conservative and recovered-algebra Gaussian tile support",
      license="Apache-2.0", license_files=["LICENSE", "NOTICE"],
      ext_modules=[CUDAExtension("local_splat3_tiles", [str(ROOT/"bindings.cpp"), str(ROOT/"tiles.cu")],
                    extra_compile_args={"cxx": ["/O2"], "nvcc": ["-O3"]})],
      cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)}, python_requires=">=3.10")
