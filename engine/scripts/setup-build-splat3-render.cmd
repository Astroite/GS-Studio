@echo off
setlocal
rem Build only the private Apache-2.0 RGB3 compositor. No install or gsplat edits.
if not defined LP_ROOT set "LP_ROOT=E:\AI\LocalPostshot"
set "LP_JOB=%LP_ROOT%\build\local-splat3-render"
set "LP_PYTHON=%LP_ROOT%\runtime\python-install\cpython-3.10.22-windows-x86_64-none\python.exe"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=8.9"
set "MAX_JOBS=2"
set "PYTORCH_NVML_BASED_CUDA_CHECK=1"
set "TORCH_EXTENSIONS_DIR=%LP_ROOT%\cache\torch_extensions"
set "UV_CACHE_DIR=%LP_ROOT%\cache\uv"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "PATH=%CUDA_HOME%\bin;%LP_ROOT%\runtime\python-install\cpython-3.10.22-windows-x86_64-none\Scripts;%PATH%"
if not defined LP_VCVARS set "LP_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
call "%LP_VCVARS%" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0..\engine\native\splat3_render"
"%LP_PYTHON%" setup.py build --build-base "%LP_JOB%\build" bdist_wheel --dist-dir "%LP_ROOT%\wheels"
exit /b %errorlevel%
