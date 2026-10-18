@echo off
setlocal
if not defined LP_ROOT set "LP_ROOT=E:\AI\LocalPostshot"
set "LP_PYTHON=%LP_ROOT%\runtime\python-install\cpython-3.10.22-windows-x86_64-none\python.exe"
set "UV_CACHE_DIR=%LP_ROOT%\cache\uv"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%CUDA_HOME%"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=8.9"
set "MAX_JOBS=6"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "PATH=%CUDA_HOME%\bin;%LP_ROOT%\runtime\python-install\cpython-3.10.22-windows-x86_64-none\Scripts;%PATH%"
if not defined LP_VCVARS set "LP_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
call "%LP_VCVARS%" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%
cd /d "%LP_ROOT%\sources\gsplat"
"%LP_PYTHON%" setup.py bdist_wheel --dist-dir "%LP_ROOT%\wheels"
if errorlevel 1 exit /b %errorlevel%
"%LP_ROOT%\tools\uv.exe" pip install --break-system-packages --python "%LP_PYTHON%" --no-index --no-deps --find-links "%LP_ROOT%\wheels" gsplat==1.5.3
if errorlevel 1 exit /b %errorlevel%
exit /b 0
