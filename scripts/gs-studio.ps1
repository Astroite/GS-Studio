# Launch the accepted native GS Studio directory package.
[CmdletBinding()]
param([string]$AppRoot = 'E:\AI\GSStudio\app')
$ErrorActionPreference = 'Stop'
$gsExecutable = Join-Path $AppRoot 'GS Studio.exe'
$gsRuntimeConfig = Join-Path $AppRoot 'runtime\runtime.json'
if (-not (Test-Path -LiteralPath $gsExecutable -PathType Leaf) -or -not (Test-Path -LiteralPath $gsRuntimeConfig -PathType Leaf)) {
    throw "GS Studio package is missing from $AppRoot. Supply -AppRoot after moving the package."
}
# The desktop resolves its own package runtime and clears inherited Python paths.
Start-Process -FilePath $gsExecutable -WorkingDirectory $AppRoot
