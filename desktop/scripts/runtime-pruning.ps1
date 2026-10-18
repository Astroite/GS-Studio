# Explicit, versioned policy. Every excluded file is hashed before skipping.
$PrunePolicyPath = Join-Path $PSScriptRoot 'runtime-prune-policy.json'
$PrunePolicy = Get-Content -LiteralPath $PrunePolicyPath -Raw | ConvertFrom-Json
$Exclusions = [Collections.Generic.List[object]]::new()
function Exclusion-Rule([string]$Relative) {
    $Path = $Relative.Replace('\','/')
    foreach ($Rule in @($PrunePolicy.rules | Where-Object phase -eq 'forbidden')) {
        if ($Path -match $Rule.pattern) { return $Rule }
    }
    foreach ($Rule in @($PrunePolicy.rules | Where-Object phase -eq 'provenance')) {
        if ($Path -match $Rule.pattern) { return $Rule }
    }
    # Licenses and metadata survive payload pruning for dependency/attribution audit.
    if ($Path -match '(?i)(^|/)[^/]+\.(dist-info|egg-info)(/|$)|(^|/)(licenses?[^/]*|copying[^/]*|notice[^/]*)(/|$)') { return $null }
    foreach ($Rule in @($PrunePolicy.rules | Where-Object phase -eq 'payload')) {
        if ($Path -match $Rule.pattern) { return $Rule }
    }
    # No blanket DLL/PYD removal from runtime packages or upstream COLMAP tools.
    if ($Path -match '(?i)\.(pyd|dll)$') { return $null }
    foreach ($Rule in @($PrunePolicy.rules | Where-Object phase -eq 'standard')) {
        if ($Path -match $Rule.pattern) { return $Rule }
    }
    return $null
}
function Assert-NoReparse([string]$Path) {
    $Entry = Get-Item -LiteralPath $Path -Force
    while ($Entry) {
        if ($Entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refuse reparse point: $($Entry.FullName)" }
        $Entry = if ($Entry -is [IO.DirectoryInfo]) { $Entry.Parent } else { $Entry.Directory }
    }
}
function Assert-SafeSource([string]$Source) {
    if ((Under $PackageRoot $Source) -or (Under $Source $PackageRoot)) { throw "Unsafe source/output overlap: $Source" }
    Assert-NoReparse $Source
    if (Test-Path -LiteralPath $Source -PathType Container) {
        foreach ($Entry in Get-ChildItem -LiteralPath $Source -Recurse -Force) {
            if ($Entry.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refuse reparse point: $($Entry.FullName)" }
        }
    }
}
function Copy-SafeTree([string]$Source, [string]$Relative) {
    Assert-SafeSource $Source
    $IsFile = Test-Path -LiteralPath $Source -PathType Leaf
    $Entries = if ($IsFile) { @(Get-Item -LiteralPath $Source) } else { @(Get-ChildItem -LiteralPath $Source -Recurse -File -Force | Sort-Object FullName) }
    foreach ($Entry in $Entries) {
        $Child = if ($IsFile) { '' } else { $Entry.FullName.Substring($Source.TrimEnd('\','/').Length).TrimStart('\','/') }
        $Name = if ($Child) { Join-Path $Relative $Child } else { $Relative }
        $Dest = Destination $Name
        $Rule = Exclusion-Rule $Name
        if ($Rule) {
            $Exclusions.Add([ordered]@{path=$Name.Replace('\','/');bytes=$Entry.Length;reason=$Rule.reason;rule=$Rule.id;source=$Entry.FullName;sha256=(Get-FileHash -LiteralPath $Entry.FullName -Algorithm SHA256).Hash.ToLowerInvariant()})
            if ($Rule.phase -eq 'provenance') { Save-AuditOriginal $Entry.FullName $Name }
            continue
        }
        if (Test-Path -LiteralPath $Dest) { throw "Duplicate package destination: $Name" }
        New-Item -ItemType Directory -Path (Split-Path -Parent $Dest) -Force | Out-Null
        Copy-Item -LiteralPath $Entry.FullName -Destination $Dest
        if ($Profile -eq 'public') { Protect-PublicText $Dest $Entry.FullName $Name }
    }
}
