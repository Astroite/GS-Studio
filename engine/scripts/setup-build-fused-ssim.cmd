@echo off
setlocal
rem Explicit offline build only; shared helper never installs or edits inputs.
rem Set GSSTUDIO_BUILD_PYTHON to an existing Python executable, then pass
rem --output NEW_JOB --vcvars BAT --cuda-home CUDA and pinned source inputs.
rem --plan validates paths without importing Torch or running a compiler.
if not defined GSSTUDIO_BUILD_PYTHON if defined LP_PYTHON set "GSSTUDIO_BUILD_PYTHON=%LP_PYTHON%"
if not defined GSSTUDIO_BUILD_PYTHON (
  echo Set GSSTUDIO_BUILD_PYTHON to the explicit Python executable. 1>&2
  exit /b 2
)
if not exist "%GSSTUDIO_BUILD_PYTHON%" (
  echo Missing Python executable: "%GSSTUDIO_BUILD_PYTHON%" 1>&2
  exit /b 2
)
"%GSSTUDIO_BUILD_PYTHON%" -I -B -X utf8 "%~dp0build_native.py" fused-ssim %*
exit /b %errorlevel%
