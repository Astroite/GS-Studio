[CmdletBinding()]
param(
    [string]$DataRoot = 'E:\AI\LocalPostshot',
    [string]$Proxy = '',
    [string]$UvPath = '',
    [string]$FfmpegPath = '',
    [string]$CudaHome = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9',
    [string]$VcVars = 'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat',
    [switch]$SkipChecks
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($DataRoot)
$scripts = $PSScriptRoot
$savedEnvironment = @{}
foreach ($key in @('UV_CACHE_DIR','UV_PYTHON_INSTALL_DIR','LP_ROOT','LP_VCVARS','CUDA_HOME','TORCH_EXTENSIONS_DIR','LOCAL_POSTSHOT_ROOT','HTTP_PROXY','HTTPS_PROXY')) {
    $savedEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
function Assert-Exit([string]$Action) { if ($LASTEXITCODE -ne 0) { throw "$Action failed (exit $LASTEXITCODE)" } }
function Download-Checked([string]$Url, [string]$File, [string]$Sha256) {
    if ((Test-Path -LiteralPath $File) -and ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -eq $Sha256)) { return }
    $arguments = @('--fail','--location','--retry','4','--retry-all-errors','--connect-timeout','20','--output',"$File.part")
    if ($Proxy) { $arguments += @('--proxy',$Proxy) }
    & curl.exe @arguments $Url
    Assert-Exit "Download $Url"
    if ((Get-FileHash -LiteralPath "$File.part" -Algorithm SHA256).Hash -ne $Sha256) { throw "Checksum mismatch: $Url" }
    Move-Item -LiteralPath "$File.part" -Destination $File -Force
}
try {
    foreach ($directory in @('downloads','tools','sources','wheels','logs','runtime','cache','datasets','evidence/runtime')) { New-Item -ItemType Directory -Path (Join-Path $root $directory) -Force | Out-Null }
    $env:UV_CACHE_DIR = Join-Path $root 'cache/uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $root 'runtime/python-install'
    $env:LP_ROOT = $root
    $env:LOCAL_POSTSHOT_ROOT = $root
    $env:LP_VCVARS = $VcVars
    $env:CUDA_HOME = $CudaHome
    $env:TORCH_EXTENSIONS_DIR = Join-Path $root 'cache/torch_extensions'
    if ($Proxy) { $env:HTTPS_PROXY = $Proxy; $env:HTTP_PROXY = $Proxy }
    $uv = Join-Path $root 'tools/uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        if ($UvPath) { Copy-Item -LiteralPath $UvPath -Destination $uv }
        else {
            $zip = Join-Path $root 'downloads/uv-x86_64-pc-windows-msvc.zip'
            Download-Checked 'https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-pc-windows-msvc.zip' $zip '088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c'
            Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $root 'tools/uv-bootstrap') -Force
            $binary = Get-ChildItem (Join-Path $root 'tools/uv-bootstrap') -Recurse -Filter uv.exe | Select-Object -First 1
            Copy-Item -LiteralPath $binary.FullName -Destination $uv
        }
    }
    & $uv python install 3.10.22 --no-bin
    Assert-Exit 'Install standalone Python'
    $python = Join-Path $root 'runtime/python-install/cpython-3.10.22-windows-x86_64-none/python.exe'
    $vcRoot = Split-Path (Split-Path (Split-Path $VcVars))
    $redist = Get-ChildItem (Join-Path $vcRoot 'Redist/MSVC') -Directory -ErrorAction SilentlyContinue | Where-Object Name -match '^\d+\.' | Sort-Object Name -Descending | Select-Object -First 1
    if ($redist) {
        foreach ($dll in Get-ChildItem (Join-Path $redist.FullName 'x64/Microsoft.VC143.CRT') -Filter '*.dll') {
            $destination = Join-Path (Split-Path $python) $dll.Name
            if (-not (Test-Path -LiteralPath $destination)) { Copy-Item -LiteralPath $dll.FullName -Destination $destination }
        }
    }
    & $uv pip install --break-system-packages --python $python torch==2.8.0+cu129 torchvision==0.23.0+cu129 --index-url https://download.pytorch.org/whl/cu129
    Assert-Exit 'Install PyTorch'
    # Private CUDA modules must come from our reviewed local source/wheels,
    # never a coincidentally named project on a public package index.
    $requirements = Get-Content (Join-Path $scripts 'setup-runtime-requirements.lock.txt') | Where-Object { $_ -notmatch '^(torch|torchvision|gsplat|spz|fused[-_]ssim|local[-_]splat3[-_](stats|render|tiles|projection))==' }
    $requirementsPath = Join-Path $root 'runtime/requirements-base.lock.txt'
    $requirements | Set-Content -LiteralPath $requirementsPath -Encoding utf8
    & $uv pip install --break-system-packages --python $python -r $requirementsPath --index-url https://pypi.org/simple
    Assert-Exit 'Install runtime dependencies'
    $colmapZip = Join-Path $root 'downloads/colmap-x64-windows-cuda.zip'
    Download-Checked 'https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip' $colmapZip 'e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881'
    if (-not (Test-Path (Join-Path $root 'tools/colmap-4.2.1/bin/colmap.exe'))) { Expand-Archive -LiteralPath $colmapZip -DestinationPath (Join-Path $root 'tools/colmap-4.2.1') -Force }
    $ffmpeg = Join-Path $root 'tools/ffmpeg/ffmpeg.exe'
    if (-not (Test-Path -LiteralPath $ffmpeg)) {
        New-Item -ItemType Directory -Path (Split-Path $ffmpeg) -Force | Out-Null
        if ($FfmpegPath) { Copy-Item -LiteralPath $FfmpegPath -Destination $ffmpeg }
        else {
            $ffmpegZip = Join-Path $root 'downloads/ffmpeg-9.0.1-full_build.zip'
            Download-Checked 'https://github.com/GyanD/codexffmpeg/releases/download/9.0.1/ffmpeg-9.0.1-full_build.zip' $ffmpegZip '2e8e28af97c2ae338ccef92e36da9b2a4cd21d0cad9dde093545606cb07f5b00'
            Expand-Archive -LiteralPath $ffmpegZip -DestinationPath (Join-Path $root 'tools/ffmpeg-download') -Force
            $binary = Get-ChildItem (Join-Path $root 'tools/ffmpeg-download') -Recurse -Filter ffmpeg.exe | Select-Object -First 1
            Copy-Item -LiteralPath $binary.FullName -Destination $ffmpeg
            $licenseRoot = Split-Path (Split-Path $binary.FullName)
            Copy-Item -LiteralPath (Join-Path $licenseRoot 'LICENSE') -Destination (Join-Path $root 'tools/ffmpeg/LICENSE')
        }
    }
    $southZip = Join-Path $root 'downloads/south-building.zip'
    Download-Checked 'https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip' $southZip 'd210016bd2de20936a5f02b87fd38a76bf0440c42d045231218372cf9db9a7a1'
    if (-not (Test-Path (Join-Path $root 'datasets/south-building/images'))) { Expand-Archive -LiteralPath $southZip -DestinationPath (Join-Path $root 'datasets') -Force }
    $sources = @(
        @('gsplat','https://github.com/nerfstudio-project/gsplat.git','937e29912570c372bed6747a5c9bf85fed877bae'),
        @('fused-ssim','https://github.com/rahul-goel/fused-ssim.git','a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38'),
        @('spz','https://github.com/nianticlabs/spz.git','affd0ecea7fbb4c265ee119475af7ee5b2997482'),
        @('zstd','https://github.com/facebook/zstd.git','794ea1b0afca0f020f4e57b6732332231fb23c70')
    )
    foreach ($source in $sources) {
        $path = Join-Path $root "sources/$($source[0])"
        if (-not (Test-Path (Join-Path $path '.git'))) {
            & git clone $source[1] $path; Assert-Exit "Clone $($source[0])"
            & git -C $path checkout --detach $source[2]; Assert-Exit "Pin $($source[0])"
        }
    }
    & git -C (Join-Path $root 'sources/gsplat') submodule update --init --recursive
    Assert-Exit 'Checkout GLM submodule'
    $zlibZip = Join-Path $root 'downloads/zlib-1.3.2.tar.gz'
    Download-Checked 'https://zlib.net/zlib-1.3.2.tar.gz' $zlibZip 'bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16'
    if (-not (Test-Path (Join-Path $root 'sources/zlib-1.3.2/CMakeLists.txt'))) { & tar -xzf $zlibZip -C (Join-Path $root 'sources'); Assert-Exit 'Extract zlib' }
    $gsplatWheel = Join-Path $root 'wheels/gsplat-1.5.3-cp310-cp310-win_amd64.whl'
    if (-not (Test-Path -LiteralPath $gsplatWheel)) { & (Join-Path $scripts 'setup-build-native.cmd'); Assert-Exit 'Build gsplat native wheel' }
    & $uv pip install --break-system-packages --python $python --no-index --no-deps $gsplatWheel
    Assert-Exit 'Install compiled gsplat'
    $spzWheel = Join-Path $root 'wheels/spz-1.1.0-cp310-cp310-win_amd64.whl'
    if (Test-Path -LiteralPath $spzWheel) { & $uv pip install --break-system-packages --python $python --no-index --no-deps $spzWheel; Assert-Exit 'Install SPZ wheel' }
    else { & (Join-Path $scripts 'setup-build-spz.cmd'); Assert-Exit 'Build native SPZ module' }
    $ssimWheel = Join-Path $root 'wheels/fused_ssim-1.0.0-cp310-cp310-win_amd64.whl'
    if (-not (Test-Path -LiteralPath $ssimWheel)) { & (Join-Path $scripts 'setup-build-fused-ssim.cmd'); Assert-Exit 'Build fused SSIM wheel' }
    & $uv pip install --break-system-packages --python $python --no-index --no-deps $ssimWheel
    Assert-Exit 'Install fused SSIM wheel'
    $statsWheel = Join-Path $root 'wheels/local_splat3_stats-0.1.1-cp310-cp310-win_amd64.whl'
    if (-not (Test-Path -LiteralPath $statsWheel)) { & (Join-Path $scripts 'setup-build-splat3-stats.cmd'); Assert-Exit 'Build independent contribution statistics wheel' }
    & $uv pip install --break-system-packages --python $python --no-index --no-deps $statsWheel
    Assert-Exit 'Install independent contribution statistics wheel'
    $rendererWheel = Join-Path $root 'wheels/local_splat3_render-0.1.0-cp310-cp310-win_amd64.whl'
    if (-not (Test-Path -LiteralPath $rendererWheel)) { & (Join-Path $scripts 'setup-build-splat3-render.cmd'); Assert-Exit 'Build private RGB compositor wheel' }
    & $uv pip install --break-system-packages --python $python --no-index --no-deps $rendererWheel
    Assert-Exit 'Install private RGB compositor wheel'
    foreach ($NativeName in @('tiles','projection')) {
        $NativeWheel = Join-Path $root "wheels/local_splat3_$NativeName-0.1.0-cp310-cp310-win_amd64.whl"
        if (-not (Test-Path -LiteralPath $NativeWheel)) { & (Join-Path $scripts "setup-build-splat3-$NativeName.cmd"); Assert-Exit "Build private $NativeName wheel" }
        & $uv pip install --break-system-packages --python $python --no-index --no-deps $NativeWheel
        Assert-Exit "Install private $NativeName wheel"
    }
    [ordered]@{python=$python;colmap=(Join-Path $root 'tools/colmap-4.2.1/bin/colmap.exe');ffmpeg=$ffmpeg;data_root=$root;torch_extensions_dir=$env:TORCH_EXTENSIONS_DIR} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'runtime.json') -Encoding utf8
    & $python (Join-Path $scripts 'setup_record_runtime.py'); Assert-Exit 'Record runtime and dataset manifest'
    if (-not $SkipChecks) {
        & $python (Join-Path $scripts 'setup_gpu_preflight.py') --output (Join-Path $root 'evidence/runtime/gpu-preflight.json'); Assert-Exit 'CUDA forward/backward'
        & $python (Join-Path $scripts 'setup_spz_preflight.py'); Assert-Exit 'Official SPZ roundtrip'
    }
    Write-Host "Runtime ready: $(Join-Path $root 'runtime.json')"
} finally {
    foreach ($key in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$savedEnvironment[$key],'Process') }
}
