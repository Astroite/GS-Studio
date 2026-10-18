[CmdletBinding()]
param([string]$PythonPath)

$ErrorActionPreference = 'Stop'
$repo = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repo) { throw 'Run inside the application Git repository.' }
$gitDirectory = (& git -C $repo rev-parse --absolute-git-dir).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve repository Git directory.' }
$required = @('.githooks/pre-commit', '.githooks/pre-push', '.githooks/run-source-guard.sh',
    'scripts/check_source_payload.py', 'scripts/source-fixtures.json')
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $repo $relative) -PathType Leaf)) {
        throw "Missing guard source: $relative"
    }
}
if (-not $PythonPath) {
    $candidate = Get-Command python3, python -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $candidate) { throw 'Pass -PythonPath with a Python 3.10+ executable.' }
    $PythonPath = $candidate.Source
}
$interpreter = (Get-Item -LiteralPath $PythonPath).FullName
if ((Get-Item -LiteralPath $interpreter).PSIsContainer) { throw 'PythonPath must be an executable file.' }
& $interpreter -I -B -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'
if ($LASTEXITCODE -ne 0) { throw 'A working Python 3.10+ interpreter is required.' }

function Read-LocalSetting([string]$Name) {
    $value = @(& git -C $repo config --local --get-all $Name)
    if ($LASTEXITCODE -eq 1) { return @() }
    if ($LASTEXITCODE -ne 0) { throw "Cannot read local setting $Name" }
    return $value
}
function Restore-LocalSetting([string]$Name, [object[]]$Values) {
    & git -C $repo config --local --unset-all $Name 2>$null
    if ($LASTEXITCODE -notin @(0, 5)) { throw "Cannot reset local setting $Name" }
    foreach ($value in $Values) {
        & git -C $repo config --local --add $Name ([string]$value)
        if ($LASTEXITCODE -ne 0) { throw "Cannot restore local setting $Name" }
    }
}

$previousHooks = @(Read-LocalSetting 'core.hooksPath')
$previousPython = @(Read-LocalSetting 'gsstudio.guardPython')
$effectiveHooks = @(& git -C $repo config --show-origin --get core.hooksPath)
if ($LASTEXITCODE -notin @(0, 1)) { throw 'Cannot inspect effective hooks configuration.' }
$activeHooks = (& git -C $repo rev-parse --git-path hooks).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve previous hook directory.' }
if (-not [IO.Path]::IsPathRooted($activeHooks)) { $activeHooks = Join-Path $repo $activeHooks }
$backup = Join-Path $gitDirectory ('source-guard-backups/' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $backup | Out-Null
$copied = @()
if (Test-Path -LiteralPath $activeHooks -PathType Container) {
    $directory = Get-Item -LiteralPath $activeHooks
    if ($directory.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw 'Previous hook directory is a reparse point; inspect it before installing.'
    }
    $savedHooks = Join-Path $backup 'previous-hooks'
    New-Item -ItemType Directory -Path $savedHooks | Out-Null
    foreach ($file in Get-ChildItem -LiteralPath $activeHooks -File -Force) {
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'Previous hook file is a reparse point; inspect it before installing.'
        }
        $savedFile = Join-Path $savedHooks $file.Name
        Copy-Item -LiteralPath $file.FullName -Destination $savedFile
        $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $savedHash = (Get-FileHash -LiteralPath $savedFile -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($sourceHash -ne $savedHash) { throw 'Previous hook backup hash differs; settings were not changed.' }
        $copied += [ordered]@{ name = $file.Name; sha256 = $sourceHash }
    }
}
$receipt = [ordered]@{
    schema_version = 1
    repository = $repo
    previous_local_hooks_path = $previousHooks
    previous_local_guard_python = $previousPython
    previous_effective_hooks_setting = $effectiveHooks
    previous_hook_directory = $activeHooks
    preserved_hook_files = $copied
    configured_python = $interpreter
}
$json = $receipt | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $backup 'settings.json'), $json, [Text.UTF8Encoding]::new($false))
try {
    & git -C $repo config --local --replace-all gsstudio.guardPython $interpreter
    if ($LASTEXITCODE -ne 0) { throw 'Cannot configure guard Python.' }
    & git -C $repo config --local --replace-all core.hooksPath .githooks
    if ($LASTEXITCODE -ne 0) { throw 'Cannot configure repository hooks.' }
} catch {
    Restore-LocalSetting 'core.hooksPath' $previousHooks
    Restore-LocalSetting 'gsstudio.guardPython' $previousPython
    throw
}
Write-Host "Source-only hooks installed in this repository. Backup: $backup"
Write-Host 'No global settings, Python environment, media or LFS objects were changed.'
