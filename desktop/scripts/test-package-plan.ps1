param([string]$RuntimeConfig = 'E:/AI/GSStudio/coordination/validation-runtime.json')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Configured = Get-Content -LiteralPath $RuntimeConfig -Raw | ConvertFrom-Json
$ConfigBase = Split-Path -Parent ([IO.Path]::GetFullPath($RuntimeConfig))
function Configured-Path([string]$Value) { if ([IO.Path]::IsPathRooted($Value)) { return $Value }; return [IO.Path]::GetFullPath((Join-Path $ConfigBase $Value)) }
$Python = Configured-Path $Configured.python
$Preprocess = Configured-Path $Configured.preprocess_python
$Fixture = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ('../.test-output/package-plan-' + [Guid]::NewGuid().ToString('N'))))
New-Item -ItemType Directory -Path $Fixture -Force | Out-Null
foreach ($Folder in @('core','preprocess','engine/local_postshot','engine/editor_native','pipeline/gsstudio','tools/ffmpeg','tools/colmap','tools/insta360/bin/models','cache/torch/hub/checkpoints','cache/extensions/studio-selection','notices')) { New-Item -ItemType Directory -Path (Join-Path $Fixture $Folder) -Force | Out-Null }
# Thin portable test interpreter: executable/DLLs plus a compressed standard
# library, without real site-packages. No installed environment is changed.
$ActualCore = Split-Path -Parent $Python
Copy-Item -LiteralPath $Python -Destination (Join-Path $Fixture 'core/python.exe')
foreach ($Dll in Get-ChildItem -LiteralPath $ActualCore -Filter '*.dll' -File) { Copy-Item -LiteralPath $Dll.FullName -Destination (Join-Path $Fixture 'core') }
$VersionTag = & $Python -I -B -c 'import sys;print(str(sys.version_info.major)+str(sys.version_info.minor))'
if ($LASTEXITCODE -ne 0) { throw 'Could not read configured Python version' }
$MakeStdlib = @'
from pathlib import Path
import sys,zipfile
source,target=map(Path,sys.argv[1:3])
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in source.rglob('*.py'):
        relative=path.relative_to(source)
        if any(part in {'site-packages','__pycache__','test','tests','idlelib','tkinter','turtledemo','ensurepip'} for part in relative.parts): continue
        archive.write(path,relative.as_posix())
'@
& $Python -I -B -c $MakeStdlib (Join-Path $ActualCore 'Lib') (Join-Path $Fixture "core/python$VersionTag.zip")
if ($LASTEXITCODE -ne 0) { throw 'Could not make the portable standard-library fixture' }
Copy-Item -LiteralPath (Join-Path $ActualCore 'DLLs') -Destination (Join-Path $Fixture 'core/DLLs') -Recurse
@("python$VersionTag.zip",'DLLs','.') | Set-Content -LiteralPath (Join-Path $Fixture "core/python$VersionTag._pth")
$Python = Join-Path $Fixture 'core/python.exe'
$Preprocess = $Python
foreach ($File in @('engine/local_postshot/__init__.py','engine/local_postshot/worker.py','pipeline/gsstudio/__init__.py','tools/ffmpeg/ffmpeg.exe','tools/ffmpeg/ffprobe.exe','tools/colmap/colmap.exe','tools/insta360/bin/helper.exe','notices/notice.txt','gs-studio.exe','core/frozen29native-runtime.pyd','tools/insta360/bin/MediaSDK.dll','tools/insta360/bin/models/stitch.dat','requirements.lock.txt','cache/torch/hub/checkpoints/fixture.pth','core/original.psht','core/postshot-original.ptx')) { Set-Content -LiteralPath (Join-Path $Fixture $File) -Value '# synthetic fixture; never executed as a binary' }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../../engine/local_postshot/studio_selection.py') -Destination (Join-Path $Fixture 'engine/local_postshot/studio_selection.py')
@'
__version__ = 'synthetic-package-contract'
class cuda:
    @staticmethod
    def is_initialized(): return False
'@ | Set-Content -LiteralPath (Join-Path $Fixture 'engine/torch.py')
Copy-Item -LiteralPath (Join-Path $Fixture 'engine/torch.py') -Destination (Join-Path $Fixture 'core/torch.py')
@'
import json,os
from pathlib import Path
def config():
    path=Path(os.environ['GSSTUDIO_RUNTIME'])
    value=json.loads(path.read_text(encoding='utf-8-sig'))['torch_extensions_dir']
    return {'torch_extensions_dir': str((path.parent/value).resolve())}
'@ | Set-Content -LiteralPath (Join-Path $Fixture 'engine/local_postshot/studio_bridge.py')
foreach ($Name in @('selection.cpp','selection.cu')) { Set-Content -LiteralPath (Join-Path $Fixture "engine/editor_native/$Name") -Value "// synthetic source $Name" }
$Excluded = @('Lib/site-packages/torch/include/torch.h','Lib/site-packages/torch/lib/torch.lib',
    'Lib/site-packages/pkg/test/test_example.py','Lib/site-packages/pkg/examples/demo.py',
    'Lib/site-packages/__editable___legacy.pth','Lib/site-packages/gsdb/__init__.py',
    'Lib/site-packages/PySide6/Qt6Core.dll','Lib/site-packages/PyQt6/Qt6Widgets.dll','Lib/site-packages/pkg/debug.pdb','Lib/site-packages/pkg/build.exp',
    'Lib/site-packages/pytest-cov.pth','Lib/site-packages/cmake/data/bin/cmake.exe',
    'Lib/site-packages/cmake/data/bin/compiler-support.dll','Lib/site-packages/_pytest/config.py',
    'Lib/site-packages/pytest/__init__.py','Lib/site-packages/scikit_build_core/__init__.py',
    'Lib/site-packages/nanobind/__init__.py','Lib/site-packages/devtools/__init__.py',
    'Lib/site-packages/ninja/data/bin/ninja.exe','Scripts/ninja.exe',
    'Lib/site-packages/pkg-1.0.dist-info/direct_url.json')
$Preserved = @('Lib/site-packages/torch/utils/cpp_extension.py','Lib/site-packages/pkg/kernel.pyd',
    'Lib/site-packages/torch/lib/torch_cuda.dll','Lib/site-packages/pkg-1.0.dist-info/METADATA',
    'Lib/site-packages/pkg-1.0.dist-info/RECORD','Lib/site-packages/pkg/tests/LICENSE',
    'Lib/site-packages/pkg/data/resource.json','Lib/site-packages/ninja/__init__.py',
    'Lib/site-packages/setuptools/__init__.py','Lib/site-packages/tomli/__init__.py',
    'Lib/site-packages/cmake-4.4.4.dist-info/METADATA','Lib/site-packages/cmake/LICENSE')
foreach ($Name in $Excluded + $Preserved) {
    $Path = Join-Path $Fixture ("core/" + $Name)
    New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
    Set-Content -LiteralPath $Path -Value ("synthetic " + $Name)
}
# These are synthetic metadata copies; no real vendor files are changed.
New-Item -ItemType Directory -Path (Join-Path $Fixture 'pipeline/gsstudio/_vendor') -Force | Out-Null
Set-Content -LiteralPath (Join-Path $Fixture 'pipeline/gsstudio/_vendor/provenance.json') -Value '{"baseline_root":"E:/AI/original/nerfstudio","hash":"unchanged"}'
Set-Content -LiteralPath (Join-Path $Fixture 'pipeline/gsstudio/_vendor/PROVENANCE.md') -Value 'Original at E:/AI/original/nerfstudio'
$Config = @{
    python=$Python; preprocess_python=$Preprocess; engine_root='engine';pipeline_root='pipeline';data_root='E:/AI/GSStudio/projects';
    ffmpeg='tools/ffmpeg/ffmpeg.exe';ffprobe='tools/ffmpeg/ffprobe.exe';colmap='tools/colmap/colmap.exe';media_helper='tools/insta360/bin/helper.exe';torch_home='cache/torch';gpu_lock_path='gpu.lock';torch_extensions_dir='cache/extensions';
    package=@{python_lock_file='requirements.lock.txt';mediasdk_runtime_root='tools/insta360';core_runtime_root=(Split-Path -Parent $Python);preprocess_runtime_root=(Split-Path -Parent $Preprocess);ffmpeg_root='tools/ffmpeg';colmap_root='tools/colmap';model_cache_roots=@(@{source='cache/torch';destination='torch'});notice_files=@('notices/notice.txt');redistributable_files=@()}
}
$ConfigPath = Join-Path $Fixture 'runtime.json'
$Config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ConfigPath
# Real isolated interpreter/current loader with a CPU-only torch stub; never
# load the synthetic .pyd or copy either full runtime.
$MakeFixture = @'
import sys,os,json,hashlib
from pathlib import Path
os.environ['GSSTUDIO_RUNTIME']=sys.argv[2]
sys.path.insert(0,sys.argv[1])
from local_postshot.studio_selection import build_location
s,r=build_location();r.mkdir(parents=True)
p=r/'studio_selection_cuda.pyd';p.write_bytes(b'synthetic helper binary; not executable')
proof={'output':str(r),'architecture':'8.9','cuda_initialized':False,'binary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
       'source_sha256':{n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ('selection.cpp','selection.cu')}}
(r/'build-proof.json').write_text(json.dumps(proof),encoding='utf-8')
(r/'build.ninja').write_text('do not ship')
print(r)
'@
$Helper = & $Python -I -B -c $MakeFixture (Join-Path $Fixture 'engine') $ConfigPath
if ($LASTEXITCODE -ne 0) { throw 'Could not create synthetic helper fixture' }
$Helper = [string]$Helper
$Output = Join-Path $Fixture 'app-output'
$Packager = Join-Path $PSScriptRoot 'package-gsstudio.ps1'
function Invoke-Plan { & $Packager -RuntimeConfig $ConfigPath -Output $Output -Executable (Join-Path $Fixture 'gs-studio.exe') -PlanOnly | ConvertFrom-Json }
function Rejected-BeforeWrite([scriptblock]$Action, [string]$Reason) {
    $Rejected = $false
    try { & $Action | Out-Null } catch { if ($_.Exception.Message -match $Reason) { $Rejected=$true } else { throw } }
    if (-not $Rejected -or (Test-Path -LiteralPath $Output)) { throw "Package guard failed before writes: $Reason" }
}
$Plan = Invoke-Plan
if ($Plan.output -ne [IO.Path]::GetFullPath($Output) -or $Plan.editor_selection.cuda_initialized -ne $false) { throw 'Package plan/helper mismatch' }
if (Test-Path -LiteralPath $Output) { throw 'PlanOnly created output' }
Rejected-BeforeWrite { & $Packager -RuntimeConfig $ConfigPath -Output $Output -Executable (Join-Path $Fixture 'gs-studio.exe') } 'requires parent readiness'
$OriginalConfig = Get-Content -LiteralPath $ConfigPath -Raw
$Config.package.preprocess_runtime_root = 'core/Lib'
$Config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ConfigPath
try { Rejected-BeforeWrite { Invoke-Plan } 'overlapping-but-different' } finally {
    $Config.package.preprocess_runtime_root = $Config.package.core_runtime_root
    Set-Content -LiteralPath $ConfigPath -Value $OriginalConfig -NoNewline
}
$Config.preprocess_python = Join-Path $Fixture 'gs-studio.exe'
$Config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ConfigPath
try { Rejected-BeforeWrite { Invoke-Plan } 'same executable' } finally {
    $Config.preprocess_python = $Python
    Set-Content -LiteralPath $ConfigPath -Value $OriginalConfig -NoNewline
}
$Config.package.mediasdk_runtime_root = 'absent-sdk'
$Config.media_helper = 'absent-helper.exe'
$Config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ConfigPath
try {
    $null = Invoke-Plan # public has no SDK dependency
    Rejected-BeforeWrite { & $Packager -RuntimeConfig $ConfigPath -Output $Output -Executable (Join-Path $Fixture 'gs-studio.exe') -Profile local -PlanOnly } 'Missing mediasdk_runtime_root'
} finally {
    $Config.package.mediasdk_runtime_root = 'tools/insta360'
    $Config.media_helper = 'tools/insta360/bin/helper.exe'
    Set-Content -LiteralPath $ConfigPath -Value $OriginalConfig -NoNewline
}
# Keep the synthetic junction outside all source roots for the remaining tests.
$Junction = Join-Path $Fixture 'reparse-source'
New-Item -ItemType Junction -Path $Junction -Target (Join-Path $Fixture 'engine') | Out-Null
$Config.engine_root = $Junction
$Config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ConfigPath
try { Rejected-BeforeWrite { Invoke-Plan } 'Refuse reparse point' } finally {
    $Config.engine_root = 'engine'
    Set-Content -LiteralPath $ConfigPath -Value $OriginalConfig -NoNewline
}
Rejected-BeforeWrite { & $Packager -RuntimeConfig $ConfigPath -Output (Join-Path $Fixture 'core/unsafe-output') -Executable (Join-Path $Fixture 'gs-studio.exe') -PlanOnly } 'source/output overlap'
$Binary = Join-Path $Helper 'studio_selection_cuda.pyd'
$QtFixture = Join-Path $Fixture 'tools/colmap/Qt6Core.dll'
Set-Content -LiteralPath $QtFixture -Value 'synthetic Qt dependency'
$null = Invoke-Plan # Official COLMAP linked Qt DLLs are supported.
New-Item -ItemType Directory -Path (Join-Path $Fixture 'tools/colmap/bin') -Force | Out-Null
Set-Content -LiteralPath (Join-Path $Fixture 'tools/colmap/bin/kernel_test.exe') -Value 'synthetic build test'
Move-Item -LiteralPath $Binary -Destination ($Binary + '.held')
try { Rejected-BeforeWrite { Invoke-Plan } 'Prebuilt editor helper missing' } finally { Move-Item -LiteralPath ($Binary + '.held') -Destination $Binary }
$ProofPath = Join-Path $Helper 'build-proof.json'
$ProofText = Get-Content -LiteralPath $ProofPath -Raw
$Proof = $ProofText | ConvertFrom-Json
$Proof.binary_sha256 = '0' * 64
$Proof | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ProofPath
try { Rejected-BeforeWrite { Invoke-Plan } 'binary hash mismatch' } finally { Set-Content -LiteralPath $ProofPath -Value $ProofText -NoNewline }
$Proof = $ProofText | ConvertFrom-Json
$Proof.source_sha256.'selection.cpp' = '0' * 64
$Proof | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ProofPath
try { Rejected-BeforeWrite { Invoke-Plan } 'source hash mismatch' } finally { Set-Content -LiteralPath $ProofPath -Value $ProofText -NoNewline }
Move-Item -LiteralPath $ProofPath -Destination ($ProofPath + '.held')
try { Rejected-BeforeWrite { Invoke-Plan } 'build-proof.json' } finally { Move-Item -LiteralPath ($ProofPath + '.held') -Destination $ProofPath }
$SourcePath = Join-Path $Fixture 'engine/editor_native/selection.cpp'
$SourceBytes = [IO.File]::ReadAllBytes($SourcePath)
Add-Content -LiteralPath $SourcePath -Value '// stale build'
try { Rejected-BeforeWrite { Invoke-Plan } 'Prebuilt editor helper missing' } finally { [IO.File]::WriteAllBytes($SourcePath,$SourceBytes) }
. (Join-Path $PSScriptRoot 'editor-selection-package.ps1')
$Copied = Join-Path $Fixture 'helper-copy'
function Copy-SafeTree([string]$Source, [string]$Relative) {
    $Target = Join-Path $Copied $Relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target
}
$Plan = Invoke-Plan
Copy-EditorSelection $Plan.editor_selection
foreach ($Pair in @(@('engine/editor_native/selection.cpp',$SourcePath),@('engine/editor_native/selection.cu',(Join-Path $Fixture 'engine/editor_native/selection.cu')),
    @("engine/extensions/studio-selection/$($Plan.editor_selection.key)/studio_selection_cuda.pyd",$Binary),@("engine/extensions/studio-selection/$($Plan.editor_selection.key)/build-proof.json",$ProofPath))) {
    if ((Get-FileHash -LiteralPath (Join-Path $Copied $Pair[0])).Hash -ne (Get-FileHash -LiteralPath $Pair[1]).Hash) { throw "Packaged helper bytes differ: $($Pair[0])" }
}
if (@(Get-ChildItem -LiteralPath $Copied -Recurse -File).Count -ne 4) { throw 'Helper copied build debris or omitted required files' }
# Exercise release wiring and inventory, using only the small synthetic inputs.
& $Packager -RuntimeConfig $ConfigPath -Output $Output -Executable (Join-Path $Fixture 'gs-studio.exe') -RuntimeReady | Out-Null
$Packaged = Get-Content -LiteralPath (Join-Path $Output 'runtime/runtime.json') -Raw | ConvertFrom-Json
if ($Packaged.torch_extensions_dir -ne '../engine/extensions' -or $Packaged.torch_home -ne '../models/torch' -or $Packaged.gpu_lock_path -ne '${user_data_root}/cache/gpu.lock') { throw 'Packaged cache/lock mapping mismatch' }
$Inventory = Get-Content -LiteralPath (Join-Path $Output 'package-inventory.json') -Raw | ConvertFrom-Json
$ExpectedBinary = "engine/extensions/studio-selection/$($Plan.editor_selection.key)/studio_selection_cuda.pyd"
$InventoryBinary = @($Inventory.files | Where-Object path -eq $ExpectedBinary)
if ($InventoryBinary.Count -ne 1 -or $InventoryBinary[0].sha256 -ne $Plan.editor_selection.binary_sha256) { throw 'Full synthetic package omitted helper or inventory hash' }
if ((Get-FileHash -LiteralPath (Join-Path $Output 'tools/colmap/Qt6Core.dll')).Hash -ne (Get-FileHash -LiteralPath $QtFixture).Hash) { throw 'COLMAP linked Qt bytes were changed' }
if (-not (Test-Path -LiteralPath (Join-Path $Output 'runtime/python/frozen29native-runtime.pyd'))) { throw 'Native runtime artifact was incorrectly filtered' }
if (@($Inventory.files | Where-Object { $_.path -match 'original\.(psht|ptx)|build\.ninja$' }).Count) { throw 'Package copied original Postshot artifacts or helper build debris' }
if ($Packaged.python -ne $Packaged.preprocess_python -or $Packaged.python -ne 'python/python.exe') { throw 'Interpreter aliases diverged' }
if ($Packaged.media_helper -or $Packaged.mediasdk_root -or $Packaged.realityscan -or (Test-Path (Join-Path $Output 'tools/insta360'))) { throw 'Public SDK must be optional and absent' }
if ((Get-Content (Join-Path $Output 'runtime/runtime.json') -Raw) -match 'E:/AI|D:/Project') { throw 'Public runtime leaked host fallback' }
foreach ($Name in $Excluded) {
    if (Test-Path -LiteralPath (Join-Path $Output ("runtime/python/" + $Name))) { throw "Not pruned: $Name" }
}
foreach ($Name in $Preserved) {
    if (-not (Test-Path -LiteralPath (Join-Path $Output ("runtime/python/" + $Name)))) { throw "Required runtime file pruned: $Name" }
}
$PublicAudit = Get-Content -LiteralPath (Join-Path $Output 'package-exclusions.json') -Raw | ConvertFrom-Json
$Audit = Get-Content -LiteralPath (Join-Path ($Output + '.audit') 'package-exclusions.json') -Raw | ConvertFrom-Json
if (@($PublicAudit.files | Where-Object { $_.PSObject.Properties['source'] }).Count) { throw 'Public audit contains an absolute source field' }
foreach ($File in Get-ChildItem -LiteralPath $Output -Recurse -File | Where-Object Extension -in @('.json','.md','.txt','._pth')) {
    if ((Get-Content -LiteralPath $File.FullName -Raw) -match '(?i)[A-Z]:[/\\]+(?:Project|AI|Users)[/\\]') { throw "Developer path leaked into public text: $($File.FullName)" }
}
$OriginalProof = Join-Path ($Output + '.audit') ("originals/engine/extensions/studio-selection/$($Plan.editor_selection.key)/build-proof.json")
if ((Get-FileHash -LiteralPath $OriginalProof).Hash -ne (Get-FileHash -LiteralPath $ProofPath).Hash) { throw 'Raw selection proof was not retained outside public output' }
$PublicVendor = Get-Content -LiteralPath (Join-Path $Output 'pipeline/gsstudio/_vendor/provenance.json') -Raw | ConvertFrom-Json
if ($PublicVendor.hash -ne 'unchanged' -or $PublicVendor.baseline_root -ne 'sources/nerfstudio') { throw 'Public provenance sanitization changed non-path evidence' }
if (@($Audit.files | Where-Object rule -eq 'public-commercial-sdk').Count -ne 3) { throw 'Commercial SDK/helper omissions are not audited once each' }
foreach ($Name in $Excluded) {
    $Record = @($Audit.files | Where-Object path -eq ("runtime/python/" + $Name))
    if ($Record.Count -ne 1 -or -not $Record[0].reason -or $Record[0].bytes -le 0 -or
        $Record[0].sha256 -ne (Get-FileHash -LiteralPath $Record[0].source -Algorithm SHA256).Hash.ToLowerInvariant()) { throw "Incomplete exclusion audit: $Name" }
}
$LocalOutput = Join-Path $Fixture 'local-output'
& $Packager -RuntimeConfig $ConfigPath -Output $LocalOutput -Executable (Join-Path $Fixture 'gs-studio.exe') -Profile local -RuntimeReady | Out-Null
$LocalConfig = Get-Content -LiteralPath (Join-Path $LocalOutput 'runtime/runtime.json') -Raw | ConvertFrom-Json
if ($LocalConfig.media_helper -ne '../tools/insta360/bin/helper.exe' -or $LocalConfig.mediasdk_root -ne '../tools/insta360') { throw 'Local SDK paths mismatch' }
if (@(Get-ChildItem -LiteralPath $LocalOutput -Recurse -File -Filter helper.exe).Count -ne 1 -or
    @(Get-ChildItem -LiteralPath $LocalOutput -Recurse -File -Filter MediaSDK.dll).Count -ne 1 -or
    (Test-Path -LiteralPath (Join-Path $LocalOutput 'tools/helper'))) { throw 'Duplicate SDK/helper copied' }
$Rejected = $false
try { & $Packager -RuntimeConfig $ConfigPath -Output $LocalOutput -Executable (Join-Path $Fixture 'gs-studio.exe') -RuntimeReady } catch {
    $Rejected = $_.Exception.Message -match 'Output already exists'
}
if (-not $Rejected) { throw 'Existing output was not protected' }
if (Test-Path -LiteralPath (Join-Path $Output 'tools/colmap/bin/kernel_test.exe')) { throw 'COLMAP build test was not pruned' }
$Config.data_root = '${user_data_root}/projects'
$Config.gpu_lock_path = '${user_data_root}/cache/gpu.lock'
$Config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ConfigPath
$LocalTokenOutput = Join-Path $Fixture 'local-token-output'
try {
    & $Packager -RuntimeConfig $ConfigPath -Output $LocalTokenOutput -Executable (Join-Path $Fixture 'gs-studio.exe') -Profile local -RuntimeReady | Out-Null
    $LocalTokenConfig = Get-Content -LiteralPath (Join-Path $LocalTokenOutput 'runtime/runtime.json') -Raw | ConvertFrom-Json
    if ($LocalTokenConfig.data_root -ne $Config.data_root -or $LocalTokenConfig.gpu_lock_path -ne $Config.gpu_lock_path) { throw 'Local profile corrupted portable tokens' }
} finally { Set-Content -LiteralPath $ConfigPath -Value $OriginalConfig -NoNewline }
$Relocated = & (Join-Path $Output 'runtime/python/python.exe') -I -B -c 'import sys,os;os.environ["GSSTUDIO_RUNTIME"]=sys.argv[2];sys.path.insert(0,sys.argv[1]);from local_postshot.studio_selection import build_location;s,r=build_location();assert (r/"studio_selection_cuda.pyd").is_file();assert (s/"selection.cpp").is_file();print(r)' (Join-Path $Output 'engine') (Join-Path $Output 'runtime/runtime.json')
if ($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath([string]$Relocated) -ne [IO.Path]::GetFullPath((Join-Path $Output (Split-Path -Parent $ExpectedBinary)))) { throw 'Copied helper loader location is not relocation-safe' }
Write-Output "PASS: Single interpreter/public+local SDK/policy audit/new-output/PlanOnly/readiness/missing binary/stale source/proof hash guards; synthetic package wiring, native preservation, inventory and relocated loader (no CUDA/real runtime copy). Evidence: $Fixture"
