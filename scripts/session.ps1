# Dot-source this file from any directory. Nothing is executed until Invoke-Gsstudio.
param([string]$DataRoot, [switch]$DeveloperFallback)
$global:GsstudioAppRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$gsRuntimePath = if ($env:GSSTUDIO_RUNTIME) { $env:GSSTUDIO_RUNTIME } else { 'E:\AI\GSStudio\runtime\runtime.json' }
$gsRuntime = @{}
$gsRuntimePath = [IO.Path]::GetFullPath($gsRuntimePath)
if (Test-Path -LiteralPath $gsRuntimePath -PathType Leaf) {
    $gsRuntimeLoaded = Get-Content -LiteralPath $gsRuntimePath -Raw | ConvertFrom-Json
    foreach ($gsProperty in $gsRuntimeLoaded.PSObject.Properties) { $gsRuntime[$gsProperty.Name] = $gsProperty.Value }
}
foreach ($gsKey in @('preprocess_python','python','data_root','gpu_lock_path','media_helper','mediasdk_root','realityscan','realityscan_export_params','ffmpeg','ffprobe','colmap','torch_home','engine_root','pipeline_root','torch_extensions_dir','cache_root','cuda_path')) {
    if ($gsRuntime[$gsKey] -and -not [IO.Path]::IsPathRooted($gsRuntime[$gsKey])) {
        $gsRuntime[$gsKey] = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $gsRuntimePath) $gsRuntime[$gsKey]))
    }
}
$gsPythonPath = if ($env:GSSTUDIO_PREPROCESS_PYTHON) { $env:GSSTUDIO_PREPROCESS_PYTHON } elseif ($gsRuntime['preprocess_python']) { $gsRuntime['preprocess_python'] } else { 'E:\AI\GSStudio\runtime\preprocess-python\python.exe' }
if (-not [IO.Path]::IsPathRooted($gsPythonPath)) { $gsPythonPath = Join-Path (Split-Path -Parent $gsRuntimePath) $gsPythonPath }
if ($DeveloperFallback) { $gsPythonPath = Join-Path $global:GsstudioAppRoot '.venv\Scripts\python.exe' }
$global:GsstudioPython = $gsPythonPath
if (-not (Test-Path -LiteralPath $global:GsstudioPython -PathType Leaf)) {
    throw 'Private preprocessing runtime missing. Provision GSSTUDIO_RUNTIME or explicitly use -DeveloperFallback for local development.'
}
if ($DataRoot) {
    $env:GSSTUDIO_DATA_ROOT = [IO.Path]::GetFullPath($DataRoot)
}
elseif (-not $env:GSSTUDIO_DATA_ROOT) { $env:GSSTUDIO_DATA_ROOT = if ($gsRuntime['data_root']) { $gsRuntime['data_root'] } else { 'E:\AI\GSStudio\projects' } }
$env:GSSTUDIO_RUNTIME = [IO.Path]::GetFullPath($gsRuntimePath)
if (-not $env:GSSTUDIO_GPU_LOCK_PATH) { $env:GSSTUDIO_GPU_LOCK_PATH = if ($gsRuntime['gpu_lock_path']) { $gsRuntime['gpu_lock_path'] } else { 'E:\AI\GSStudio\cache\gpu.lock' } }
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUNBUFFERED = '1'
foreach ($gsPair in @(@('media_helper','GSSTUDIO_MEDIA_HELPER'), @('mediasdk_root','INSTA360_MEDIA_SDK_ROOT'), @('realityscan','GSSTUDIO_REALITYSCAN_CLI'), @('realityscan_export_params','GSSTUDIO_REALITYSCAN_EXPORT_PARAMS'), @('torch_home','TORCH_HOME'))) {
    $gsKey,$gsEnvName = $gsPair
    if ($gsRuntime[$gsKey] -and -not [Environment]::GetEnvironmentVariable($gsEnvName)) {
        [Environment]::SetEnvironmentVariable($gsEnvName,$gsRuntime[$gsKey],'Process')
    }
}
foreach ($gsKey in @('ffmpeg','ffprobe','colmap')) {
    if ($gsRuntime[$gsKey]) { $env:PATH = (Split-Path -Parent $gsRuntime[$gsKey]) + [IO.Path]::PathSeparator + $env:PATH }
}
# Force UTF-8 for child processes so Windows console code pages cannot
# corrupt tool output capture (GBK vs UTF-8 decode failures).
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
    # Host may not expose a console; Python env vars above still apply.
}
$helper = 'E:\AI\GSStudio\runtime\tools\insta360\bin\gsstudio-media-helper.exe'
if (-not (Test-Path -LiteralPath $helper) -and $DeveloperFallback) {
    $helper = Join-Path $global:GsstudioAppRoot 'tools\mediasdk-helper\build\Release\gsstudio-media-helper.exe'
}
if (-not $env:GSSTUDIO_MEDIA_HELPER -and (Test-Path -LiteralPath $helper)) {
    $env:GSSTUDIO_MEDIA_HELPER = $helper
}
$sdkRoot = Join-Path $global:GsstudioAppRoot 'sdks\Insta360-MediaSDK'
if (-not $env:INSTA360_MEDIA_SDK_ROOT -and (Test-Path -LiteralPath (Join-Path $sdkRoot 'bin\MediaSDK.dll'))) {
    $env:INSTA360_MEDIA_SDK_ROOT = $sdkRoot
}
function global:Invoke-Gsstudio {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GsstudioArguments)
    $gsPreviousPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = (Join-Path $global:GsstudioAppRoot 'src')
    Push-Location -LiteralPath $global:GsstudioAppRoot
    try {
        & $global:GsstudioPython -m gsstudio @GsstudioArguments
        if ($LASTEXITCODE -ne 0) { throw "gsstudio exited with code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
        $env:PYTHONPATH = $gsPreviousPythonPath
    }
}
