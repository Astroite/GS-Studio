"""Exact runtime pruning policy; use the actual PowerShell evaluator, no GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name == "nt" and shutil.which("pwsh"), "Native PowerShell 7 policy contract")
class RuntimePruningContracts(unittest.TestCase):
    def test_only_eight_named_development_executables_are_removed_and_runtime_resources_survive(self):
        remove = [f"runtime/python/Scripts/{name}.exe" for name in (
            "ccmake", "isympy", "markdown-it", "pygmentize", "scikit-build-core", "scikit-build", "typer")]
        remove.append("runtime/python/Lib/site-packages/torch/bin/protoc.exe")
        keep = [
            "runtime/python/Scripts/gsstudio.exe", "runtime/python/Scripts/typer.py",
            "runtime/python/Scripts/ccmake.exe.old", "runtime/python/Scripts/scikit-build-core.exe.config",
            "runtime/python/Lib/site-packages/typer/main.py",
            "runtime/python/Lib/site-packages/pygments/lexers/__init__.py",
            "runtime/python/Lib/site-packages/sympy/__init__.py",
            "runtime/python/Lib/site-packages/scikit_build_core-1.1.1.dist-info/METADATA",
            "runtime/python/Lib/site-packages/scikit_build_core-1.1.1.dist-info/licenses/LICENSE",
            "runtime/python/Lib/site-packages/typer-0.27.2.dist-info/entry_points.txt",
            "runtime/python/Lib/site-packages/torch/bin/torch_shm_manager.exe",
            "runtime/python/Lib/site-packages/torch/lib/torch_cuda.dll",
            "runtime/python/Lib/site-packages/other_plugin/protoc.exe",
            "tools/ffmpeg/ffmpeg.exe", "tools/ffmpeg/ffprobe.exe",
            "tools/colmap/bin/colmap.exe", "tools/colmap/bin/Qt6Core.dll",
            "tools/insta360/bin/gsstudio-media-helper.exe",
        ]
        alias = r"RUNTIME\PYTHON\SCRIPTS\TYPer.EXE"
        with tempfile.TemporaryDirectory(prefix="GS policy 中文 ") as temporary:
            parent = Path(temporary)
            cases = parent / "cases.json"
            cases.write_text(json.dumps(remove + keep + [alias]), encoding="utf-8")
            script = parent / "evaluate-policy.ps1"
            script.write_text(r'''
param([string]$PolicyEvaluator, [string]$Cases)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
. $PolicyEvaluator
$Names=Get-Content -LiteralPath $Cases -Raw | ConvertFrom-Json
$Results=@(foreach($Name in $Names) {
    $Rule=Exclusion-Rule $Name
    [pscustomobject]@{path=$Name;rule=$(if($Rule){$Rule.id}else{$null})}
})
ConvertTo-Json -InputObject $Results -Compress
''', encoding="utf-8")
            proc = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File", str(script),
                                   str(ROOT / "desktop/scripts/runtime-pruning.ps1"), str(cases)],
                                  capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            results = {row["path"]: row["rule"] for row in json.loads(proc.stdout)}
        for path in remove + [alias]:
            with self.subTest(remove=path):
                self.assertEqual(results[path], "unused-development-executables")
        for path in keep:
            with self.subTest(keep=path):
                self.assertIsNone(results[path])


if __name__ == "__main__":
    unittest.main()
