param([string]$Architecture = '8.9')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../../scripts/session.ps1')
# Only this child process gets compiler environment changes. No installs or
# existing native source/artifact writes; outputs go to the configured cache.
$gsSelectionScript = Join-Path $PSScriptRoot 'build_studio_selection.py'
$gsSelectionPython = 'E:/AI/GSStudio/runtime/python/python.exe'
$gsSelectionVc = 'C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build/vcvarsall.bat'
$gsSelectionCommand = 'call "{0}" x64 -vcvars_ver=14.44 && set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" && set "DISTUTILS_USE_SDK=1" && "{1}" "{2}" --architecture "{3}"' -f $gsSelectionVc,$gsSelectionPython,$gsSelectionScript,$Architecture
& $env:ComSpec /d /s /c $gsSelectionCommand
if ($LASTEXITCODE -ne 0) { throw "Editor helper build failed: $LASTEXITCODE" }
