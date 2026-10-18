@echo off
setlocal
rem Optional CPU compilation only. No runtime installation and no CUDA tensor test.
rem Official source: https://github.com/rahul-goel/fused-ssim.git
rem git clone URL "%LP_ROOT%\sources\fused-ssim", then git checkout the pinned commit.
if not defined LP_ROOT set "LP_ROOT=E:\AI\LocalPostshot"
set "LP_SOURCE=%LP_ROOT%\sources\fused-ssim"
set "LP_PYTHON=%LP_ROOT%\runtime\python-install\cpython-3.10.22-windows-x86_64-none\python.exe"
set "LP_PIN=a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
if not exist "%LP_SOURCE%\setup.py" (
  echo Missing pinned fused-ssim checkout: %LP_SOURCE%
  exit /b 1
)
for /f "delims=" %%G in ('git -C "%LP_SOURCE%" rev-parse HEAD') do set "LP_ACTUAL=%%G"
if not "%LP_ACTUAL%"=="%LP_PIN%" (
  echo Refusing unpinned source: expected %LP_PIN%, found %LP_ACTUAL%
  exit /b 1
)
git -C "%LP_SOURCE%" diff --quiet HEAD --
if errorlevel 1 (
  echo Refusing modified official source.
  exit /b 1
)
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%CUDA_HOME%"
set "CUDA_ARCHITECTURES=89"
set "TORCH_CUDA_ARCH_LIST=8.9"
set "MAX_JOBS=2"
set "PYTORCH_NVML_BASED_CUDA_CHECK=1"
set "TORCH_EXTENSIONS_DIR=%LP_ROOT%\cache\torch_extensions"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "PATH=%CUDA_HOME%\bin;%LP_ROOT%\runtime\python-install\cpython-3.10.22-windows-x86_64-none\Scripts;%PATH%"
if not defined LP_VCVARS set "LP_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
call "%LP_VCVARS%" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%
cd /d "%LP_SOURCE%"
"%LP_PYTHON%" setup.py bdist_wheel --dist-dir "%LP_ROOT%\wheels"
exit /b %errorlevel%
