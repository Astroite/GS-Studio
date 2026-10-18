# Detailed machine provenance belongs in the sibling audit, never public text.
$Transformations = [Collections.Generic.List[object]]::new()
$DeveloperPathPattern = '(?i)(?:file:/+)?[a-z]:[/\\]+(?:Project|AI|Users)[/\\][^\r\n"''<>\x60]*'
function Logical-Source([string]$Source) {
    foreach ($Component in @($SourceComponents | Sort-Object { $_.source.Length } -Descending)) {
        if (Under $Source $Component.source) {
            $Tail = $Source.Substring($Component.source.TrimEnd('\','/').Length).TrimStart('\','/').Replace('\','/')
            return @{component=$Component.component;relative=$(if ($Tail) { $Tail } else { Split-Path -Leaf $Source });path=$(if ($Tail) { $Component.path.TrimEnd('/') + '/' + $Tail } else { $Component.path })}
        }
    }
    return @{component='external-source';relative=(Split-Path -Leaf $Source);path=('sources/' + (Split-Path -Leaf $Source))}
}
function Save-AuditOriginal([string]$Source, [string]$Relative) {
    $Target = [IO.Path]::GetFullPath((Join-Path $AuditRoot ('originals/' + $Relative)))
    if (-not (Under $Target $AuditRoot)) { throw "Unsafe audit path: $Relative" }
    if (-not (Test-Path -LiteralPath $Target)) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Target
    }
}
function Protect-PublicText([string]$Target, [string]$Source, [string]$Relative) {
    $Extension = [IO.Path]::GetExtension($Target).ToLowerInvariant()
    if ($Extension -notin @('.json','.md','.txt','.xml','.ini','.cfg','.csv','.html','.yaml','.yml','.pth','._pth','.pc','.cmake','')) { return }
    try { $Content = [Text.UTF8Encoding]::new($false,$true).GetString([IO.File]::ReadAllBytes($Target)) } catch { return }
    if ($Content -notmatch $DeveloperPathPattern) { return }
    Save-AuditOriginal $Source $Relative
    $OriginalHash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash.ToLowerInvariant()
    $Evaluator = [Text.RegularExpressions.MatchEvaluator]{
        param($Match)
        $Path = $Match.Value -replace '^file:/+', ''
        $Path = $Path -replace '[\\/]+', '/'
        return (Logical-Source $Path).path
    }
    $Sanitized = [regex]::Replace($Content, $DeveloperPathPattern, $Evaluator)
    [IO.File]::WriteAllText($Target,$Sanitized,[Text.UTF8Encoding]::new($false))
    $Transformations.Add([ordered]@{path=$Relative.Replace('\','/');reason='Replace developer-local provenance paths with logical component paths';source=$Source;source_sha256=$OriginalHash;sha256=(Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLowerInvariant()})
}
function Public-Exclusions {
    foreach ($Record in $Exclusions) {
        $Logical = Logical-Source $Record.source
        [ordered]@{component=$Logical.component;path=$Record.path;source_relative=$Logical.relative;bytes=$Record.bytes;reason=$Record.reason;rule=$Record.rule;sha256=$Record.sha256}
    }
}
function Public-Plan {
    $SelectionPath = 'engine/extensions/studio-selection/' + $Selection.key
    return [ordered]@{schema_version=3;profile=$Profile;product='GS Studio';version='0.3.0-rc.1';
        components=@($SourceComponents | ForEach-Object { [ordered]@{component=$_.component;path=$_.path} });
        python_lock=[ordered]@{path='runtime/python-requirements.lock.txt';sha256=$Plan.python_lock_sha256};
        prune_policy=[ordered]@{path='runtime-prune-policy.json';sha256=$Plan.prune_policy_sha256};
        editor_selection=[ordered]@{key=$Selection.key;binary=($SelectionPath + '/' + $Selection.binary_name);binary_sha256=$Selection.binary_sha256;
            source_proof_sha256=$Selection.proof_sha256;architecture=$Selection.architecture;
            sources=@($Selection.sources | ForEach-Object { [ordered]@{path=('engine/editor_native/' + $_.name);sha256=$_.sha256} })}}
}
