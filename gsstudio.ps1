# Formal CLI delegates to the same accepted directory package as the desktop.
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][AllowEmptyString()][string[]]$GsstudioArgs)
$ErrorActionPreference = 'Stop'
$gsPackageRoot = if ($env:GSSTUDIO_APP_ROOT) { $env:GSSTUDIO_APP_ROOT } else { 'E:\AI\GSStudio\app' }
$gsPackageCli = Join-Path $gsPackageRoot 'gsstudio.ps1'
if (-not (Test-Path -LiteralPath $gsPackageCli -PathType Leaf)) { throw "Missing GS Studio package CLI: $gsPackageCli" }
& $gsPackageCli @GsstudioArgs
if ($MyInvocation.InvocationName -eq '.') { return }
exit $LASTEXITCODE
