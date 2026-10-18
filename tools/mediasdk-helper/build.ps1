[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug')]
    [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
$HelperRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if (-not $env:INSTA360_MEDIA_SDK_ROOT) {
    throw 'Set INSTA360_MEDIA_SDK_ROOT to your approved Desktop MediaSDK directory.'
}
$BuildRoot = Join-Path $HelperRoot 'build'
cmake -S $HelperRoot -B $BuildRoot -A x64
cmake --build $BuildRoot --config $Configuration
if ($LASTEXITCODE -ne 0) { throw 'gsstudio-media-helper build failed' }

$OutputDir = Join-Path $BuildRoot $Configuration
$SdkBin = Join-Path $env:INSTA360_MEDIA_SDK_ROOT 'bin'

# MediaSDK.dll/InsMetaDataSDK.dll and their large CUDA/OpenCV/onnxruntime/MNN
# dependency set are not on PATH by default; the built exe needs them beside
# it (or on PATH) to actually load at runtime.
Get-ChildItem -LiteralPath $SdkBin -Filter '*.dll' | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $OutputDir -Force
}

Write-Host "Helper build complete. Set GSSTUDIO_MEDIA_HELPER to $OutputDir\gsstudio-media-helper.exe."
