from pathlib import Path
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
ROOT = Path(__file__).resolve().parent
GLM = Path(os.environ.get('LP_GSPLAT_SOURCE', r'E:\AI\LocalPostshot\sources\gsplat'))/'gsplat/cuda/csrc/third_party/glm'
if not (GLM/'glm/glm.hpp').is_file():
    raise RuntimeError('Set LP_GSPLAT_SOURCE to the pinned gsplat 1.5.3 source checkout including GLM')
setup(name='local_splat3_projection', version='0.1.0',
    description='Private fused pinhole projection with explicit native AA floor/VJP',
    license='Apache-2.0', license_files=['LICENSE', 'NOTICE', 'GLM-LICENSE.txt'],
    ext_modules=[CUDAExtension('local_splat3_projection', [str(ROOT/'bindings.cpp'), str(ROOT/'projection.cu')],
        include_dirs=[str(ROOT),str(GLM)], extra_compile_args={'cxx':['/O2'],'nvcc':['-O3']})],
    cmdclass={'build_ext':BuildExtension.with_options(use_ninja=True)}, python_requires='>=3.10')
