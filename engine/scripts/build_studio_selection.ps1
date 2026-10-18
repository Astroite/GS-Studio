[CmdletBinding()]
param(
    [string]$PythonPath = $env:GSSTUDIO_BUILD_PYTHON,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$SourceRoot = '',
    [string]$VcVars = $env:GSSTUDIO_VCVARS,
    [string]$CudaHome = $env:CUDA_HOME,
    [string]$Architecture = '8.9',
    [string]$VcVarsVersion = '14.44',
    [int]$Jobs = 2,
    [string]$NinjaPath = '',
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Provide -PythonPath (or GSSTUDIO_BUILD_PYTHON) for the explicit build environment.'
}
$gsSelectionArguments = @(
    '-I', '-B', '-X', 'utf8', (Join-Path $PSScriptRoot 'build_studio_selection.py'),
    '--python', $PythonPath, '--output', $Output, '--architecture', $Architecture,
    '--vcvars-version', $VcVarsVersion, '--jobs', "$Jobs"
)
foreach ($gsSelectionOption in @(
    @('--source-root', $SourceRoot), @('--vcvars', $VcVars),
    @('--cuda-home', $CudaHome), @('--ninja', $NinjaPath)
)) {
    if ($gsSelectionOption[1]) { $gsSelectionArguments += $gsSelectionOption }
}
if ($PlanOnly) { $gsSelectionArguments += '--plan' }
& $PythonPath @gsSelectionArguments
if ($LASTEXITCODE -ne 0) { throw "Editor helper build failed (exit $LASTEXITCODE)." }
