[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RuntimeConfig,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('public','local')][string]$Profile = 'public',
    [string]$Executable = (Join-Path $PSScriptRoot '../src-tauri/target/release/gs-studio.exe'),
    [switch]$RuntimeReady, [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'editor-selection-package.ps1')
# Read-only sources, new output only; no provisioning, downloads or source cleanup.
function Absolute-Path([string]$Value, [string]$Base) {
    if (-not $Value) { throw 'Empty package input path' }
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Base $Value))
}
function Required-Path([object]$Object, [string]$Name, [string]$Base) {
    if (-not $Object.PSObject.Properties[$Name] -or -not $Object.$Name) { throw "runtime config requires explicit $Name" }
    $Resolved = Absolute-Path ([string]$Object.$Name) $Base
    if (-not (Test-Path -LiteralPath $Resolved)) { throw "Missing $Name : $Resolved" }
    return $Resolved
}
function Under([string]$Path, [string]$Root) {
    $Full = [IO.Path]::GetFullPath($Path)
    $Boundary = [IO.Path]::GetFullPath($Root).TrimEnd('\','/')
    return $Full.Equals($Boundary,[StringComparison]::OrdinalIgnoreCase) -or $Full.StartsWith($Boundary + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)
}
function Destination([string]$Relative) {
    if ([IO.Path]::IsPathRooted($Relative)) { throw "Package destination must be relative: $Relative" }
    $Resolved = Absolute-Path $Relative $PackageRoot
    if (-not (Under $Resolved $PackageRoot)) { throw "Package path escapes output: $Relative" }
    return $Resolved
}
function Inside-Relative([string]$Path, [string]$Root) {
    if (-not (Under $Path $Root)) { throw "Executable is outside its configured root: $Path" }
    return $Path.Substring($Root.TrimEnd('\','/').Length).TrimStart('\','/').Replace('\','/')
}
$ConfigPath = [IO.Path]::GetFullPath($RuntimeConfig)
$ConfigBase = Split-Path -Parent $ConfigPath
$Runtime = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$PackageRoot = [IO.Path]::GetFullPath($Output)
$AuditRoot = $PackageRoot + '.audit'
if (Test-Path -LiteralPath $PackageRoot) { throw "Output already exists; choose a new output directory: $PackageRoot" }
$Ancestor = Split-Path -Parent $PackageRoot
while (-not (Test-Path -LiteralPath $Ancestor)) { $Ancestor = Split-Path -Parent $Ancestor }
. (Join-Path $PSScriptRoot 'runtime-pruning.ps1')
. (Join-Path $PSScriptRoot 'package-provenance.ps1')
Assert-NoReparse $Ancestor
Assert-SafeSource $ConfigPath
if (-not $Runtime.PSObject.Properties['package']) { throw 'runtime.json requires package metadata; see desktop/scripts/runtime-package.example.json' }
$Metadata = $Runtime.package
$CoreRoot = Required-Path $Metadata 'core_runtime_root' $ConfigBase
$PreprocessRoot = Required-Path $Metadata 'preprocess_runtime_root' $ConfigBase
$Python = Required-Path $Runtime 'python' $ConfigBase
$PreprocessPython = Required-Path $Runtime 'preprocess_python' $ConfigBase
if ($CoreRoot -ne $PreprocessRoot -or $Python -ne $PreprocessPython -or -not (Under $Python $CoreRoot)) {
    throw 'One locked Python environment is required: both runtime roots must match and python/preprocess_python must name the same executable; overlapping-but-different roots are forbidden'
}
$LockFile = Required-Path $Metadata 'python_lock_file' $ConfigBase
$EngineRoot = Required-Path $Runtime 'engine_root' $ConfigBase
$PipelineRoot = Required-Path $Runtime 'pipeline_root' $ConfigBase
$ExtensionsRoot = Required-Path $Runtime 'torch_extensions_dir' $ConfigBase
$Tools = @{}
foreach ($Key in @('ffmpeg_root','colmap_root')) { $Tools[$Key] = Required-Path $Metadata $Key $ConfigBase }
$ToolExecutables = @{}
foreach ($Key in @('ffmpeg','ffprobe','colmap')) { $ToolExecutables[$Key] = Required-Path $Runtime $Key $ConfigBase }
foreach ($Pair in @(@('ffmpeg','ffmpeg_root'),@('ffprobe','ffmpeg_root'),@('colmap','colmap_root'))) { $null = Inside-Relative $ToolExecutables[$Pair[0]] $Tools[$Pair[1]] }
if (-not $Metadata.PSObject.Properties['model_cache_roots'] -or @($Metadata.model_cache_roots).Count -ne 1) { throw 'package.model_cache_roots must list exactly one verified Torch model root (destination torch)' }
$ModelSource = Required-Path $Metadata.model_cache_roots[0] 'source' $ConfigBase
$TorchHome = Required-Path $Runtime 'torch_home' $ConfigBase
if (-not @(Get-ChildItem -LiteralPath $ModelSource -Recurse -File | Where-Object Extension -in @('.pth','.pt')).Count) { throw 'Verified offline model weights are mandatory; model root has no .pth/.pt files' }
if ($ModelSource -ne $TorchHome -or $Metadata.model_cache_roots[0].destination -ne 'torch') { throw 'Model source must equal torch_home and destination must be torch; weights live in models/torch' }
if (-not $Metadata.PSObject.Properties['notice_files'] -or -not @($Metadata.notice_files).Count) { throw 'package.notice_files must list redistribution notices' }
$NoticeSources = @($Metadata.notice_files | ForEach-Object { $Notice = Absolute-Path ([string]$_) $ConfigBase; if (-not (Test-Path -LiteralPath $Notice -PathType Leaf)) { throw "Missing notice $Notice" }; $Notice })
$MediaSdkRuntime = $null
$MediaHelper = $null
$PublicSkippedSources = @()
if ($Profile -eq 'public') {
    # Optional sources may be absent; present commercial input omissions are audited.
    if ($Metadata.PSObject.Properties['mediasdk_runtime_root'] -and $Metadata.mediasdk_runtime_root) {
        $Candidate = Join-Path (Absolute-Path ([string]$Metadata.mediasdk_runtime_root) $ConfigBase) 'bin'
        if (Test-Path -LiteralPath $Candidate) { $PublicSkippedSources += @{source=$Candidate;destination='tools/insta360/bin'} }
    }
    if ($Runtime.PSObject.Properties['media_helper'] -and $Runtime.media_helper) {
        $Candidate = Absolute-Path ([string]$Runtime.media_helper) $ConfigBase
        if ((Test-Path -LiteralPath $Candidate -PathType Leaf) -and -not @($PublicSkippedSources | Where-Object { Under $Candidate $_.source }).Count) {
            $PublicSkippedSources += @{source=$Candidate;destination=('tools/insta360/bin/' + (Split-Path -Leaf $Candidate))}
        }
    }
}
if ($Profile -eq 'local') {
    $MediaSdkRuntime = Required-Path $Metadata 'mediasdk_runtime_root' $ConfigBase
    $MediaHelper = Required-Path $Runtime 'media_helper' $ConfigBase
    $null = Inside-Relative $MediaHelper (Join-Path $MediaSdkRuntime 'bin')
    if (-not (Test-Path -LiteralPath (Join-Path $MediaSdkRuntime 'bin/models') -PathType Container)) { throw 'Local SDK requires bin/models with stitching models' }
}
$ExportParams = if ($Runtime.PSObject.Properties['realityscan_export_params'] -and $Runtime.realityscan_export_params) { Required-Path $Runtime 'realityscan_export_params' $ConfigBase } else { $null }
$ExePath = [IO.Path]::GetFullPath($Executable)
if (-not (Test-Path -LiteralPath $ExePath -PathType Leaf)) { throw "Build the Tauri release first: $ExePath" }
$Additional = @()
if ($Metadata.PSObject.Properties['redistributable_files']) {
    foreach ($Item in $Metadata.redistributable_files) {
        $Source = Required-Path $Item 'source' $ConfigBase
        $Target = [string]$Item.destination
        if ($Target.Replace('\','/') -notmatch '^notices/[^/]+$') { throw 'redistributable_files destinations are restricted to notices/<filename>' }
        $null = Destination $Target
        $Additional += @{source=$Source;destination=$Target}
    }
}
$Sources = @($CoreRoot,$EngineRoot,$PipelineRoot,$ExtensionsRoot,$LockFile,$ExePath,$ModelSource,$PrunePolicyPath,
             (Join-Path $PSScriptRoot 'gsstudio-cli.ps1'),(Join-Path $PSScriptRoot '../frontend-notices.txt')) + @($Tools.Values) + $NoticeSources
if ($MediaSdkRuntime) { $Sources += Join-Path $MediaSdkRuntime 'bin' }
if ($ExportParams) { $Sources += $ExportParams }
$Sources += @($Additional | ForEach-Object source)
$Sources += @($PublicSkippedSources | ForEach-Object source)
foreach ($Root in @($CoreRoot,$EngineRoot,$PipelineRoot,$ExtensionsRoot,$ModelSource) + @($Tools.Values)) {
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { throw "Package input root must be a directory: $Root" }
}
foreach ($File in @($Python,$LockFile,$ExePath) + @($ToolExecutables.Values) + $NoticeSources) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Package input must be a file: $File" }
}
# Preserve official COLMAP and every linked runtime DLL, including upstream Qt.
foreach ($Source in $Sources) { Assert-SafeSource $Source }
$Selection = Get-EditorSelectionPlan $Python $EngineRoot $ConfigPath $ExtensionsRoot
Assert-EditorSelectionFiles $Selection
$SourceComponents = @(
    @{source=$CoreRoot;component='python';path='runtime/python'},
    @{source=$EngineRoot;component='engine';path='engine'},
    @{source=$PipelineRoot;component='pipeline';path='pipeline'},
    @{source=$ExtensionsRoot;component='extensions';path='engine/extensions'},
    @{source=$Tools.ffmpeg_root;component='ffmpeg';path='tools/ffmpeg'},
    @{source=$Tools.colmap_root;component='colmap';path='tools/colmap'},
    @{source=$ModelSource;component='torch-models';path='models/torch'},
    @{source=$LockFile;component='python-lock';path='runtime/python-requirements.lock.txt'}
)
foreach ($Skipped in $PublicSkippedSources) { $SourceComponents += @{source=$Skipped.source;component='commercial-sdk';path=$Skipped.destination} }
if ($MediaSdkRuntime) { $SourceComponents += @{source=(Join-Path $MediaSdkRuntime 'bin');component='commercial-sdk';path='tools/insta360/bin'} }
for ($i=0; $i -lt $NoticeSources.Count; $i++) { $SourceComponents += @{source=$NoticeSources[$i];component='notices';path=('notices/' + ("{0:D3}-" -f $i) + (Split-Path -Leaf $NoticeSources[$i]))} }
foreach ($Source in $Sources) {
    if ((Under $AuditRoot $Source) -or (Under $Source $AuditRoot)) { throw "Unsafe source/audit overlap: $Source" }
}
$Plan = [ordered]@{schema_version=2;profile=$Profile;output=$PackageRoot;executable=$ExePath;python_runtime=$CoreRoot;
    python_lock_file=$LockFile;python_lock_sha256=(Get-FileHash -LiteralPath $LockFile -Algorithm SHA256).Hash.ToLowerInvariant();
    engine=$EngineRoot;pipeline=$PipelineRoot;editor_selection=$Selection;tools=$Tools;mediasdk=$MediaSdkRuntime;
    audit_output=$AuditRoot;model_cache=$ModelSource;public_excluded_sources=$PublicSkippedSources;notices=$NoticeSources;prune_policy_sha256=(Get-FileHash -LiteralPath $PrunePolicyPath -Algorithm SHA256).Hash.ToLowerInvariant()}
if ($PlanOnly) { $Plan | ConvertTo-Json -Depth 8; return }
if (-not $RuntimeReady) { throw 'Full package copy requires parent readiness: pass -RuntimeReady after provisioning. Use -PlanOnly for inspection.' }
Assert-EditorSelectionFiles $Selection
if (Test-Path -LiteralPath $AuditRoot) { throw "Audit output already exists; choose a new output directory: $AuditRoot" }
New-Item -ItemType Directory -Path $PackageRoot | Out-Null
New-Item -ItemType Directory -Path $AuditRoot | Out-Null
try {
    foreach ($Skipped in $PublicSkippedSources) {
        $IsFile = Test-Path -LiteralPath $Skipped.source -PathType Leaf
        $Files = if ($IsFile) { @(Get-Item -LiteralPath $Skipped.source) } else { @(Get-ChildItem -LiteralPath $Skipped.source -Recurse -File | Sort-Object FullName) }
        foreach ($Entry in $Files) {
            $Name = if ($IsFile) { $Skipped.destination } else { Join-Path $Skipped.destination ($Entry.FullName.Substring($Skipped.source.Length).TrimStart('\','/')) }
            $Exclusions.Add([ordered]@{path=$Name.Replace('\','/');bytes=$Entry.Length;reason='Public profile excludes commercial SDK/helper distribution';rule='public-commercial-sdk';source=$Entry.FullName;sha256=(Get-FileHash -LiteralPath $Entry.FullName -Algorithm SHA256).Hash.ToLowerInvariant()})
        }
    }
    Copy-SafeTree $ExePath 'GS Studio.exe'
    Copy-SafeTree (Join-Path $PSScriptRoot 'gsstudio-cli.ps1') 'gsstudio.ps1'
    Copy-SafeTree (Join-Path $PSScriptRoot '../frontend-notices.txt') 'notices/frontend-notices.txt'
    Copy-SafeTree $PrunePolicyPath 'runtime-prune-policy.json'
    Copy-SafeTree $LockFile 'runtime/python-requirements.lock.txt'
    Copy-SafeTree $CoreRoot 'runtime/python'
    Copy-SafeTree (Join-Path $EngineRoot 'local_postshot') 'engine/local_postshot'
    Copy-EditorSelection $Selection
    Copy-SafeTree (Join-Path $PipelineRoot 'gsstudio') 'pipeline/gsstudio'
    Copy-SafeTree $Tools.ffmpeg_root 'tools/ffmpeg'
    Copy-SafeTree $Tools.colmap_root 'tools/colmap'
    if ($MediaSdkRuntime) { Copy-SafeTree (Join-Path $MediaSdkRuntime 'bin') 'tools/insta360/bin' }
    Copy-SafeTree $ModelSource 'models/torch'
    for ($i=0; $i -lt $NoticeSources.Count; $i++) { Copy-SafeTree $NoticeSources[$i] (Join-Path 'notices' (("{0:D3}-" -f $i) + (Split-Path -Leaf $NoticeSources[$i]))) }
    foreach ($Item in $Additional) { Copy-SafeTree $Item.source $Item.destination }
    # Public behavior is generated from an allowlist, never host config fallbacks.
    $Packaged = [ordered]@{release=$true;package_profile=$Profile;user_data_root='${user_data_root}';
        python=('python/' + (Inside-Relative $Python $CoreRoot));preprocess_python=('python/' + (Inside-Relative $Python $CoreRoot));
        engine_root='../engine';pipeline_root='../pipeline';data_root='${user_data_root}/projects';
        cache_root='${user_data_root}/cache';log_root='${user_data_root}/logs';gpu_lock_path='${user_data_root}/cache/gpu.lock';
        torch_extensions_dir='../engine/extensions';torch_home='../models/torch';
        media_helper=$null;mediasdk_root=$null;realityscan=$null;realityscan_export_params=$null;cuda_path=$null}
    foreach ($Pair in @(@('ffmpeg','ffmpeg_root','../tools/ffmpeg'),@('ffprobe','ffmpeg_root','../tools/ffmpeg'),@('colmap','colmap_root','../tools/colmap'))) {
        $Packaged[$Pair[0]] = $Pair[2] + '/' + (Inside-Relative $ToolExecutables[$Pair[0]] $Tools[$Pair[1]])
    }
    if ($ExportParams) {
        Copy-SafeTree $ExportParams 'tools/realityscan/colmap-export-params.xml'
        $Packaged.realityscan_export_params = '../tools/realityscan/colmap-export-params.xml'
    }
    if ($MediaSdkRuntime) {
        $Packaged.mediasdk_root = '../tools/insta360'
        $Packaged.media_helper = '../tools/insta360/bin/' + (Inside-Relative $MediaHelper (Join-Path $MediaSdkRuntime 'bin'))
    }
    # Explicit local configuration can preserve the existing projects and lock.
    if ($Profile -eq 'local') {
        foreach ($Key in @('data_root','gpu_lock_path','realityscan','cuda_path')) {
            if ($Runtime.PSObject.Properties[$Key] -and $Runtime.$Key) {
                $Value = [string]$Runtime.$Key
                $Packaged[$Key] = if ($Value.Contains('${user_data_root}')) { $Value } else { Absolute-Path $Value $ConfigBase }
            }
        }
    }
    $Packaged | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Destination 'runtime/runtime.json') -Encoding UTF8
    Public-Plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Destination 'package-inputs.json') -Encoding UTF8
    $Plan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $AuditRoot 'package-inputs.json') -Encoding UTF8
} finally {
    # Partial outputs stay reviewable and cannot be overwritten on retry.
    [long]$ExcludedBytes = 0
    foreach ($Record in $Exclusions) { $ExcludedBytes += $Record.bytes }
    [ordered]@{schema_version=1;policy_sha256=$Plan.prune_policy_sha256;
        excluded_bytes=$ExcludedBytes;
        files=@(Public-Exclusions)} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Destination 'package-exclusions.json') -Encoding UTF8
    [ordered]@{schema_version=1;policy_sha256=$Plan.prune_policy_sha256;excluded_bytes=$ExcludedBytes;files=@($Exclusions.ToArray())} |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $AuditRoot 'package-exclusions.json') -Encoding UTF8
    @{schema_version=1;files=@($Transformations.ToArray())} | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $AuditRoot 'package-transformations.json') -Encoding UTF8
}
$Inventory = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
    [ordered]@{path=$_.FullName.Substring($PackageRoot.Length).TrimStart('\','/').Replace('\','/');bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
@{schema_version=2;product='GS Studio';version='0.3.0-rc.1';profile=$Profile;files=$Inventory} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Destination 'package-inventory.json') -Encoding UTF8
Write-Output "GS Studio $Profile package copied with hash inventory and exclusion audit: $PackageRoot"
