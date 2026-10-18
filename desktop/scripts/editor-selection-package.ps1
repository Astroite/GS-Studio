function Get-EditorSelectionPlan([string]$Python, [string]$EngineRoot, [string]$ConfigPath, [string]$ExtensionsRoot) {
    $Start = [Diagnostics.ProcessStartInfo]::new()
    $Start.FileName = $Python
    $Start.UseShellExecute = $false
    $Start.CreateNoWindow = $true
    $Start.RedirectStandardOutput = $true
    $Start.RedirectStandardError = $true
    $Start.Environment.Clear()
    foreach ($Name in @('SystemRoot','WINDIR','TEMP','TMP','USERPROFILE','LOCALAPPDATA','APPDATA','PROGRAMDATA','COMSPEC','NUMBER_OF_PROCESSORS')) {
        $Value = [Environment]::GetEnvironmentVariable($Name)
        if ($Value) { $Start.Environment[$Name] = $Value }
    }
    $Start.Environment['PATH'] = (Split-Path -Parent $Python) + [IO.Path]::PathSeparator + (Join-Path $env:SystemRoot 'System32')
    $Start.Environment['CUDA_VISIBLE_DEVICES'] = ''
    $Start.Environment['PYTHONUTF8'] = '1'
    $Start.Environment['PYTHONIOENCODING'] = 'utf-8'
    foreach ($Argument in @('-I','-B',(Join-Path $PSScriptRoot 'inspect-editor-selection.py'),$EngineRoot,$ConfigPath,$ExtensionsRoot)) { $Start.ArgumentList.Add($Argument) }
    $Process = [Diagnostics.Process]::new()
    $Process.StartInfo = $Start
    try {
        $null = $Process.Start()
        $Stdout = $Process.StandardOutput.ReadToEndAsync()
        $Stderr = $Process.StandardError.ReadToEndAsync()
        if (-not $Process.WaitForExit(45000)) { $Process.Kill($true); throw 'Editor helper discovery timed out (no runtime compilation is allowed)' }
        $OutputText = $Stdout.GetAwaiter().GetResult()
        $ErrorText = $Stderr.GetAwaiter().GetResult()
        if ($Process.ExitCode -ne 0) { throw "Editor helper packaging preflight failed: $ErrorText" }
        return $OutputText | ConvertFrom-Json
    } finally { $Process.Dispose() }
}

function Assert-EditorSelectionFiles([object]$Selection) {
    foreach ($File in @($Selection.sources) + @(@{path=$Selection.binary;sha256=$Selection.binary_sha256},@{path=$Selection.proof;sha256=$Selection.proof_sha256})) {
        if (-not (Test-Path -LiteralPath $File.path -PathType Leaf) -or (Get-FileHash -LiteralPath $File.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $File.sha256) { throw "Editor helper changed or disappeared after preflight: $($File.path)" }
    }
}

function Copy-EditorSelection([object]$Selection) {
    Assert-EditorSelectionFiles $Selection
    foreach ($Source in $Selection.sources) { Copy-SafeTree $Source.path (Join-Path 'engine/editor_native' $Source.name) }
    $Relative = Join-Path 'engine/extensions/studio-selection' $Selection.key
    Copy-SafeTree $Selection.binary (Join-Path $Relative $Selection.binary_name)
    Copy-SafeTree $Selection.proof (Join-Path $Relative 'build-proof.json')
}
