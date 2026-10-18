# Prepared source proxy. Parent activates this only after package acceptance.
[CmdletBinding(PositionalBinding=$false)]
param(
    [string]$AppRoot = 'E:/AI/GSStudio/app',
    [Parameter(ValueFromRemainingArguments = $true)][AllowEmptyString()][string[]]$GsstudioArgs
)
$gsProxyDotSourced = $MyInvocation.InvocationName -eq '.'
$gsProxyExitCode = & {
    param($Root, [string[]]$CliArguments)
    try {
        $Entry = Join-Path ([IO.Path]::GetFullPath($Root)) 'gsstudio.ps1'
        if (-not (Test-Path -LiteralPath $Entry -PathType Leaf)) { throw "Accepted package CLI is missing: $Entry" }
        & $Entry @CliArguments
        return $LASTEXITCODE
    } catch {
        Write-Error -Message $_ -ErrorAction Continue
        return 1
    }
} $AppRoot $GsstudioArgs
$global:LASTEXITCODE = [int]$gsProxyExitCode
if ($gsProxyDotSourced) { return }
exit $gsProxyExitCode
