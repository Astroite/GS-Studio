# Package-root CLI template. The packager installs this file as gsstudio.ps1.
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][AllowEmptyString()][string[]]$GsstudioArgs)
$gsPackageCliDotSourced = $MyInvocation.InvocationName -eq '.'
$gsPackageCliExitCode = & {
    param($PackageRoot, [string[]]$CliArguments)
    $ErrorActionPreference = 'Stop'
    Set-StrictMode -Version Latest
    function Full-Path([string]$Value, [string]$Base) {
        if ([string]::IsNullOrWhiteSpace($Value)) { throw 'Empty runtime path' }
        if (-not [IO.Path]::IsPathRooted($Value)) { $Value = Join-Path $Base $Value }
        return [IO.Path]::GetFullPath($Value)
    }
    function Assert-Contained([string]$Path, [string]$Root) {
        $Boundary = $Root.TrimEnd('\','/') + [IO.Path]::DirectorySeparatorChar
        if (-not $Path.StartsWith($Boundary, [StringComparison]::OrdinalIgnoreCase)) { throw "Runtime path must be package-contained: $Path" }
        $Entry = Get-Item -LiteralPath $Path -Force
        while ($Entry) {
            if ($Entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse runtime path is unsafe: $($Entry.FullName)" }
            if ($Entry.FullName.TrimEnd('\','/') -eq $Root.TrimEnd('\','/')) { break }
            $Entry = if ($Entry -is [IO.DirectoryInfo]) { $Entry.Parent } else { $Entry.Directory }
        }
    }
    function Native-Argument([AllowEmptyString()][string]$Value) {
        return '"' + [regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
    }
    try {
        $PackageRoot = [IO.Path]::GetFullPath($PackageRoot)
        $RuntimePath = Join-Path $PackageRoot 'runtime\runtime.json'
        if ($env:GSSTUDIO_RUNTIME) { $RuntimePath = [IO.Path]::GetFullPath($env:GSSTUDIO_RUNTIME) }
        elseif ($env:LOCAL_POSTSHOT_RUNTIME) { $RuntimePath = [IO.Path]::GetFullPath($env:LOCAL_POSTSHOT_RUNTIME) }
        $Config = Get-Content -LiteralPath $RuntimePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $Resolved = @{}
        $ExternalRoots = [Collections.Generic.HashSet[string]]::new()
        # Optional commercial dependencies are resolved later by the canonical Python resolver.
        foreach ($Key in @('python','preprocess_python','engine_root','pipeline_root')) {
            if (-not $Config.PSObject.Properties[$Key] -or [string]::IsNullOrWhiteSpace($Config.$Key)) { throw "Package runtime requires explicit $Key" }
            $Resolved[$Key] = Full-Path $Config.$Key (Split-Path -Parent $RuntimePath)
        }
        $OverridePath = if ($env:GSSTUDIO_RUNTIME_OVERRIDE) { [IO.Path]::GetFullPath($env:GSSTUDIO_RUNTIME_OVERRIDE) } else {
            $Local = if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { Join-Path $env:USERPROFILE 'AppData\Local' }
            Join-Path $Local 'GSStudio\runtime.override.json'
        }
        if (Test-Path -LiteralPath $OverridePath -PathType Leaf) {
            $Override = Get-Content -LiteralPath $OverridePath -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($Key in @('python','preprocess_python','engine_root','pipeline_root')) {
                if ($Override.PSObject.Properties[$Key] -and $Override.$Key) {
                    $Resolved[$Key] = Full-Path $Override.$Key (Split-Path -Parent $OverridePath)
                    $null = $ExternalRoots.Add($Key)
                }
            }
        } elseif ($env:GSSTUDIO_RUNTIME_OVERRIDE) { throw "Configured runtime override is missing: $OverridePath" }
        foreach ($Key in @('python','preprocess_python','engine_root','pipeline_root')) {
            if (-not (Test-Path -LiteralPath $Resolved[$Key])) { throw "Missing package $Key" }
            # Bundled config uses bundled modules; external configs/overrides are explicit.
            if (-not $env:GSSTUDIO_RUNTIME -and -not $env:LOCAL_POSTSHOT_RUNTIME -and -not $ExternalRoots.Contains($Key)) { Assert-Contained $Resolved[$Key] $PackageRoot }
        }
        if (-not $CliArguments -or $CliArguments.Count -eq 0) { $CliArguments = @('--help') }
        $Bootstrap = @'
import os,runpy,sys
sys.dont_write_bytecode=True
for stream in (sys.stdin,sys.stdout,sys.stderr):
    if stream is not None and hasattr(stream,'reconfigure'):
        stream.reconfigure(encoding='utf-8')
root,runtime=sys.argv[1:3]
del sys.argv[1:3]
sys.argv[0]='gsstudio'
sys.path.insert(0,root)
os.environ['GSSTUDIO_RUNTIME']=runtime
from gsstudio.infrastructure.runtime.configuration import runtime_config,child_environment
cfg=runtime_config()
env=child_environment(cfg)
os.environ.clear()
os.environ.update(env)
runpy.run_module('gsstudio',run_name='__main__',alter_sys=True)
'@
        $Start = New-Object System.Diagnostics.ProcessStartInfo
        $Start.FileName = $Resolved.preprocess_python
        $Start.UseShellExecute = $false
        $Start.CreateNoWindow = $true
        $Start.RedirectStandardOutput = $true
        $Start.RedirectStandardError = $true
        $Start.WorkingDirectory = $PackageRoot
        $ChildArguments = @('-I','-B','-u','-c',$Bootstrap,$Resolved.pipeline_root,$RuntimePath) + @($CliArguments)
        if ($Start.PSObject.Properties['ArgumentList']) {
            foreach ($Argument in $ChildArguments) { $Start.ArgumentList.Add([string]$Argument) }
        } else {
            $Start.Arguments = ($ChildArguments | ForEach-Object { Native-Argument $_ }) -join ' '
        }
        foreach ($Name in @('PYTHONHOME','PYTHONPATH','PYTHONUSERBASE','LOCAL_POSTSHOT_RUNTIME')) { $Start.EnvironmentVariables.Remove($Name) }
        $Start.EnvironmentVariables['GSSTUDIO_RUNTIME'] = $RuntimePath
        $Start.EnvironmentVariables['PYTHONUTF8'] = '1'
        $Start.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
        $Start.EnvironmentVariables['PYTHONNOUSERSITE'] = '1'
        $Start.EnvironmentVariables['PYTHONDONTWRITEBYTECODE'] = '1'
        $Child = [Diagnostics.Process]::Start($Start)
        try {
            # Pump both byte streams concurrently, including when PowerShell itself
            # is redirected. This preserves UTF-8/JSONL and live CLI progress.
            $OutTask = $Child.StandardOutput.BaseStream.CopyToAsync([Console]::OpenStandardOutput())
            $ErrorTask = $Child.StandardError.BaseStream.CopyToAsync([Console]::OpenStandardError())
            $Child.WaitForExit()
            $null = $OutTask.GetAwaiter().GetResult()
            $null = $ErrorTask.GetAwaiter().GetResult()
            return $Child.ExitCode
        } finally { $Child.Dispose() }
    } catch {
        Write-Error -Message $_ -ErrorAction Continue
        return 1
    }
} $PSScriptRoot $GsstudioArgs
$global:LASTEXITCODE = [int]$gsPackageCliExitCode
if ($gsPackageCliDotSourced) { return }
exit $gsPackageCliExitCode
