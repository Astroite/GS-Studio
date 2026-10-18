# CPU-only template and copy-filter checks. All writes stay in a new temp fixture.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Assert([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$Template = Join-Path $PSScriptRoot 'gsstudio-cli.ps1'
$Proxy = Join-Path $RepoRoot 'scripts/gsstudio-next.ps1'
$Packager = Join-Path $PSScriptRoot 'package-gsstudio.ps1'
foreach ($File in @($Template,$Proxy,$Packager,$PSCommandPath)) {
    $Tokens = $null; $Errors = $null
    $null = [Management.Automation.Language.Parser]::ParseFile($File,[ref]$Tokens,[ref]$Errors)
    Assert ($Errors.Count -eq 0) "PowerShell parse failed: $File : $Errors"
}
$TempRoot = Join-Path ([IO.Path]::GetTempPath()) ('gsstudio-cli-' + [guid]::NewGuid().ToString('N'))
$Fixture = Join-Path $TempRoot ('package ' + [char]0x4e2d + [char]0x6587 + " O'Brien")
New-Item -ItemType Directory -Path $Fixture -Force | Out-Null
# A real native mock interpreter records exactly the argv received by Windows.
$MockSource = Join-Path $TempRoot 'Mock.cs'
@'
using System;
using System.Collections.Generic;
using System.IO;
using System.Web.Script.Serialization;
class Mock {
    static int Main(string[] args) {
        var env = new Dictionary<string,string>();
        foreach (string key in new [] {"GSSTUDIO_RUNTIME","GSSTUDIO_DATA_ROOT","GSSTUDIO_GPU_LOCK_PATH","TORCH_HOME","PYTHONPATH","PATH"})
            env[key] = Environment.GetEnvironmentVariable(key);
        File.WriteAllText(Environment.GetEnvironmentVariable("GSCLI_RECORD"), new JavaScriptSerializer().Serialize(new {argv=args, env=env}));
        return Int32.Parse(Environment.GetEnvironmentVariable("GSCLI_EXIT") ?? "0");
    }
}
'@ | Set-Content -LiteralPath $MockSource -Encoding UTF8
$Compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
$Mock = Join-Path $TempRoot 'mock.exe'
& $Compiler /nologo /target:exe /reference:System.Web.Extensions.dll "/out:$Mock" $MockSource
Assert ($LASTEXITCODE -eq 0) 'Mock compiler failed'
foreach ($Directory in @('runtime/prep','runtime/core','pipeline/gsstudio/infrastructure/runtime','engine','tools/sdk')) {
    New-Item -ItemType Directory -Path (Join-Path $Fixture $Directory) -Force | Out-Null
}
Copy-Item -LiteralPath $Mock -Destination (Join-Path $Fixture 'runtime/prep/python.exe')
Copy-Item -LiteralPath $Mock -Destination (Join-Path $Fixture 'runtime/core/python.exe')
Copy-Item -LiteralPath $Template -Destination (Join-Path $Fixture 'gsstudio.ps1')
foreach ($Name in @('pipeline/gsstudio/__main__.py','pipeline/gsstudio/infrastructure/runtime/configuration.py')) {
    Set-Content -LiteralPath (Join-Path $Fixture $Name) -Value '# Mock module; interpreter never executes it'
}
$Config = [ordered]@{python='core/python.exe';preprocess_python='prep/python.exe';engine_root='../engine';pipeline_root='../pipeline';data_root='../../projects';torch_home='../cache/models';gpu_lock_path=(Join-Path $TempRoot 'shared.lock');cache_root='../cache';torch_extensions_dir='../cache/extensions';ffmpeg='../tools/ffmpeg.exe';ffprobe='../tools/ffprobe.exe';colmap='../tools/colmap.exe';media_helper='../tools/helper.exe';mediasdk_root='../tools/sdk';realityscan=(Join-Path $TempRoot 'external/RealityScan.exe');realityscan_export_params='../tools/export.xml'}
foreach ($Name in @('ffmpeg.exe','ffprobe.exe','colmap.exe','helper.exe','export.xml')) {
    Set-Content -LiteralPath (Join-Path $Fixture "tools/$Name") -Value 'not executed'
}
$RuntimePath = Join-Path $Fixture 'runtime/runtime.json'
function Save-Config { $Config | ConvertTo-Json | Set-Content -LiteralPath $RuntimePath -Encoding UTF8 }
Save-Config
$Record = Join-Path $TempRoot 'record.json'
$SavedEnv = @{}
foreach ($Name in @('GSCLI_RECORD','GSCLI_EXIT','GSSTUDIO_RUNTIME','GSSTUDIO_DATA_ROOT','GSSTUDIO_GPU_LOCK_PATH','PYTHONPATH','PATH','TORCH_HOME')) {
    $SavedEnv[$Name] = [Environment]::GetEnvironmentVariable($Name,'Process')
}
try {
    $env:GSCLI_RECORD = $Record; $env:GSCLI_EXIT = '37'
    $env:GSSTUDIO_RUNTIME = 'source-manifest-must-stay.json'
    $env:GSSTUDIO_DATA_ROOT = 'source-projects-must-stay'
    $env:GSSTUDIO_GPU_LOCK_PATH = 'source-lock-must-stay'
    $env:TORCH_HOME = 'source-cache-must-stay'; $env:PYTHONPATH = 'poison-must-not-propagate'
    $Before = @{}; foreach ($Name in $SavedEnv.Keys) { $Before[$Name] = [Environment]::GetEnvironmentVariable($Name,'Process') }
    $CliArgs = @('external','inspect','--images',('C:\space ' + [char]0x4e2d + [char]0x6587 + "\O'Brien\"),'','embedded "double" quote',"single'quote",'trailing\\')
    . (Join-Path $Fixture 'gsstudio.ps1') @CliArgs
    Assert ($LASTEXITCODE -eq 37) 'Exact nonzero exit code was lost'
    $Captured = Get-Content -LiteralPath $Record -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert (($Captured.argv[0..3] -join ',') -eq '-I,-B,-u,-c') 'Interpreter isolation/bytecode flags missing'
    Assert ($Captured.argv.Count -eq ($CliArgs.Count + 7)) 'Argument count changed'
    for ($Index=0; $Index -lt $CliArgs.Count; $Index++) { Assert ($Captured.argv[$Index+7] -ceq $CliArgs[$Index]) "Argument changed at $Index" }
    Assert ($Captured.env.GSSTUDIO_RUNTIME -eq $RuntimePath) 'Source manifest escaped into package'
    Assert ($Captured.env.GSSTUDIO_DATA_ROOT -eq [IO.Path]::GetFullPath((Join-Path (Split-Path $RuntimePath) $Config.data_root))) 'Package data_root changed'
    Assert ($Captured.env.GSSTUDIO_GPU_LOCK_PATH -eq $Config.gpu_lock_path) 'Shared GPU lock changed'
    Assert (-not $Captured.env.PYTHONPATH) 'PYTHONPATH poison propagated'
    Assert ($Captured.env.PATH -eq $Before.PATH) 'Host PATH changed before bootstrap'
    foreach ($Name in $Before.Keys) { Assert ([Environment]::GetEnvironmentVariable($Name,'Process') -ceq $Before[$Name]) "Dot-source changed environment $Name" }
    . $Proxy -AppRoot $Fixture @CliArgs
    Assert ($LASTEXITCODE -eq 37) 'Source proxy lost child exit code'
    $Captured = Get-Content -LiteralPath $Record -Raw -Encoding UTF8 | ConvertFrom-Json
    for ($Index=0; $Index -lt $CliArgs.Count; $Index++) { Assert ($Captured.argv[$Index+7] -ceq $CliArgs[$Index]) "Proxy argument changed at $Index" }
    # Exercise normal positional CLI commands without an explicit -AppRoot.
    # Only the fixture default path differs from the actual proxy source.
    $DefaultProxy = Join-Path $TempRoot 'default-proxy.ps1'
    (Get-Content -LiteralPath $Proxy -Raw).Replace("'E:/AI/GSStudio/app'", ("'" + $Fixture.Replace("'", "''") + "'")) | Set-Content -LiteralPath $DefaultProxy -Encoding UTF8
    . $DefaultProxy @CliArgs
    Assert ($LASTEXITCODE -eq 37) 'Default-root proxy consumed the first CLI command as AppRoot'
    $Captured = Get-Content -LiteralPath $Record -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert ($Captured.argv.Count -eq ($CliArgs.Count + 7)) 'Default-root proxy argument count changed'
    for ($Index=0; $Index -lt $CliArgs.Count; $Index++) { Assert ($Captured.argv[$Index+7] -ceq $CliArgs[$Index]) "Default-root proxy argument changed at $Index" }
    $HostExecutable = (Get-Process -Id $PID).Path
    & $HostExecutable -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Fixture 'gsstudio.ps1') --help
    Assert ($LASTEXITCODE -eq 37) 'Invoked package script lost exit code'
    & $HostExecutable -NoProfile -ExecutionPolicy Bypass -File $Proxy -AppRoot $Fixture --help
    Assert ($LASTEXITCODE -eq 37) 'Invoked source proxy lost exit code'
    $env:GSCLI_EXIT = '0'
    . (Join-Path $Fixture 'gsstudio.ps1')
    $Captured = Get-Content -LiteralPath $Record -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert ($Captured.argv[-1] -eq '--help') 'No-args must request CPU help'
    foreach ($Case in @('missing','missing-tool','outside-python','outside-module','outside-tool','relative-lock')) {
        Save-Config
        $Bad = Get-Content -LiteralPath $RuntimePath -Raw | ConvertFrom-Json
        switch ($Case) {
            'missing' { $Bad.PSObject.Properties.Remove('preprocess_python') }
            'missing-tool' { $Bad.PSObject.Properties.Remove('ffprobe') }
            'outside-python' { $Bad.preprocess_python = $Mock }
            'outside-module' { $Bad.pipeline_root = $TempRoot }
            'outside-tool' { $Bad.ffmpeg = $Mock }
            'relative-lock' { $Bad.gpu_lock_path = 'gpu.lock' }
        }
        $Bad | ConvertTo-Json | Set-Content -LiteralPath $RuntimePath -Encoding UTF8
        $PriorRecord = (Get-Item -LiteralPath $Record).LastWriteTimeUtc
        . (Join-Path $Fixture 'gsstudio.ps1') --help 2>$null
        Assert ($LASTEXITCODE -eq 1) "Path guard did not fail: $Case"
        Assert ((Get-Item -LiteralPath $Record).LastWriteTimeUtc -eq $PriorRecord) "Guard launched interpreter: $Case"
    }
    Save-Config
} finally {
    foreach ($Name in $SavedEnv.Keys) { [Environment]::SetEnvironmentVariable($Name,$SavedEnv[$Name],'Process') }
}

# Load only pure copy/filter functions, never the packager's full-copy entry.
$Tokens = $null; $Errors = $null
$Ast = [Management.Automation.Language.Parser]::ParseFile($Packager,[ref]$Tokens,[ref]$Errors)
foreach ($Name in @('Absolute-Path','Under','Destination','Skip-File','Copy-SafeTree')) {
    $Definition = $Ast.Find({param($Node) $Node -is [Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq $Name},$true)
    . ([scriptblock]::Create($Definition.Extent.Text))
}
$InputTree = Join-Path $TempRoot 'input'; $PackageRoot = Join-Path $TempRoot 'filtered-copy'
$Excluded = @('Lib/site-packages/tornado/test/test.key','Lib/site-packages/cmake/data/share/cmake-4.4/Templates/Windows/Windows_TemporaryKey.pfx','Lib/site-packages/a1_coverage.pth','Lib/site-packages/pytest-cov.pth','Lib/site-packages/pkg/__pycache__/x.pyc')
$Allowed = @('Lib/site-packages/certifi/cacert.pem','SDK/LICENSE.key','SDK/required.dll','Lib/site-packages/ordinary.pth','Lib/site-packages/torch/LICENSE')
foreach ($Name in $Excluded + $Allowed) {
    $Path = Join-Path $InputTree $Name
    New-Item -ItemType Directory -Path (Split-Path $Path) -Force | Out-Null
    Set-Content -LiteralPath $Path -Value 'synthetic copy-filter fixture'
}
Copy-SafeTree $InputTree 'runtime'
foreach ($Name in $Excluded) { Assert (-not (Test-Path -LiteralPath (Join-Path $PackageRoot "runtime/$Name"))) "Excluded vendor artifact copied: $Name" }
foreach ($Name in $Allowed) { Assert (Test-Path -LiteralPath (Join-Path $PackageRoot "runtime/$Name")) "Required artifact was excluded: $Name" }
foreach ($Name in $Excluded[0..3]) {
    $Refused = $false
    try { Copy-SafeTree (Join-Path $InputTree $Name) ('leaf/' + (Split-Path $Name -Leaf)) } catch { $Refused = $true }
    Assert $Refused "Explicit leaf copy bypassed vendor filter: $Name"
}
$Text = Get-Content -LiteralPath $Packager -Raw
Assert ($Text.IndexOf("'gsstudio-cli.ps1'") -lt $Text.IndexOf('$Inventory =')) 'CLI copy must precede full inventory'
Write-Output "PASS PowerShell $($PSVersionTable.PSVersion): parse, native argv, env, exit, proxy, no-args, path guards, scoped vendor copy filters"
Write-Output "Fixture evidence retained: $TempRoot"
exit 0
