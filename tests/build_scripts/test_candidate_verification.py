"""Real temporary package manifests; no installed application or GPU required."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/release/verify_candidate.py"
SPEC = importlib.util.spec_from_file_location("candidate_integrity", SCRIPT)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


class CandidateVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="GS Studio 验收 ")
        self.root = Path(self.temp.name)
        self.package = self.root / "目录 package"
        self.package.mkdir()
        self.report = self.root / "result.json"
        self.source = self.root / "source"
        self.source.mkdir()
        self.key = "0123456789abcdef0123"
        self.prefix = f"engine/extensions/studio-selection/{self.key}/"
        self.write("GS Studio.exe", b"synthetic fixture only")
        self.write("runtime/python-requirements.lock.txt", b"torch==2.8.0+cu129\n")
        self.write("runtime-prune-policy.json", b"{}\n")
        self.write("engine/local_postshot/model.py", b"value = 1\n")
        self.write("pipeline/gsstudio/main.py", b"value = 2\n")
        self.write("engine/editor_native/selection.cpp", b"// synthetic selection C++\n")
        self.write("engine/editor_native/selection.cu", b"// synthetic selection CUDA\n")
        self.write(self.prefix + "studio_selection_cuda.pyd", b"synthetic binary fixture only")
        self.write_json(self.prefix + "build-proof.json", {"architecture": "8.9", "cuda_initialized": False})
        self.inputs = {
            "schema_version": 3, "profile": "public", "product": "GS Studio", "version": "test",
            "components": [{"component": "engine", "path": "engine"},
                           {"component": "pipeline", "path": "pipeline"},
                           {"component": "commercial-sdk", "path": "tools/insta360/bin"}],
            "python_lock": self.binding("runtime/python-requirements.lock.txt"),
            "prune_policy": self.binding("runtime-prune-policy.json"),
            "editor_selection": {"key": self.key, "architecture": "8.9",
                "binary": self.prefix + "studio_selection_cuda.pyd",
                "binary_sha256": self.hash(self.prefix + "studio_selection_cuda.pyd"),
                "source_proof_sha256": self.hash(self.prefix + "build-proof.json"),
                "sources": [self.binding("engine/editor_native/selection.cpp"), self.binding("engine/editor_native/selection.cu")]},
        }
        self.write_json("package-inputs.json", self.inputs)
        self.write_json("package-exclusions.json", {"schema_version": 1,
            "policy_sha256": self.hash("runtime-prune-policy.json"), "files": [], "excluded_bytes": 0})
        self.reinventory()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, data):
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_json(self, relative, value):
        self.write(relative, (json.dumps(value, indent=2) + "\n").encode())

    def hash(self, relative):
        return hashlib.sha256((self.package / relative).read_bytes()).hexdigest()

    def binding(self, relative):
        return {"path": relative, "sha256": self.hash(relative)}

    def reinventory(self):
        self.inventory = {"schema_version": 2, "profile": "public", "product": "GS Studio", "version": "test",
                          "files": [{"path": path.relative_to(self.package).as_posix(), "bytes": path.stat().st_size,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                    for path in sorted(self.package.rglob("*"))
                                    if path.is_file() and path.name != "package-inventory.json"]}
        self.write_json("package-inventory.json", self.inventory)

    def assert_failed(self, text, **kwargs):
        result = candidate.verify(self.package, **kwargs)
        self.assertEqual(result["status"], "failed", result)
        self.assertTrue(any(text in message for message in result["errors"]), result)

    def test_valid_manifest_binds_inputs_but_never_claims_runtime_legal_or_commit(self):
        result = candidate.verify(self.package)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["inventory"]["payload_files"], len(self.inventory["files"]))
        self.assertEqual(result["inputs_and_provenance"]["public_optional_omissions"], ["tools/insta360/bin"])
        self.assertFalse(result["runtime_executed"])
        self.assertEqual(result["gpu_acceptance"], "not-performed")
        self.assertEqual(result["legal_acceptance"], "not-performed")
        self.assertEqual(result["full_artifact_commit_binding"], "not-established")

    def test_size_and_same_size_hash_tampering_are_both_detected(self):
        self.write("engine/local_postshot/model.py", b"value = 9\n")
        self.assert_failed("SHA256 mismatch")
        self.write("engine/local_postshot/model.py", b"different size\n")
        self.assert_failed("size mismatch")

    def test_unlisted_and_missing_files_are_detected(self):
        self.write("unexpected.txt", b"not listed")
        self.assert_failed("unlisted=")
        (self.package / "unexpected.txt").unlink()
        (self.package / "GS Studio.exe").unlink()
        self.assert_failed("missing=['GS Studio.exe']")

    def test_manifest_rejects_parent_escape_absolute_drive_backslash_and_windows_aliases(self):
        for path in ("../outside.txt", "/absolute", "C:/outside", "engine\\bad.py", "engine/../file",
                     "engine/NUL.txt", "engine/x:stream", "engine/dot.", "engine/space "):
            with self.subTest(path=path):
                self.inventory["files"][0]["path"] = path
                self.write_json("package-inventory.json", self.inventory)
                self.assert_failed("path")

    def test_duplicate_and_case_aliased_manifest_paths_are_rejected(self):
        self.inventory["files"].append(dict(self.inventory["files"][0]))
        self.write_json("package-inventory.json", self.inventory)
        self.assert_failed("Duplicate/self")
        self.inventory["files"][-1]["path"] = self.inventory["files"][0]["path"].upper()
        self.write_json("package-inventory.json", self.inventory)
        self.assert_failed("Duplicate/self")

    def test_duplicate_json_keys_are_rejected(self):
        self.write("package-inventory.json", b'{"schema_version":2,"schema_version":2}')
        self.assert_failed("Duplicate JSON key")

    def test_updated_inventory_does_not_hide_broken_lock_binding(self):
        self.write("runtime/python-requirements.lock.txt", b"changed==1\n")
        self.reinventory()
        self.assert_failed("provenance hash mismatch")

    def test_updated_inventory_does_not_hide_broken_editor_binary_binding(self):
        self.write(self.prefix + "studio_selection_cuda.pyd", b"tampered synthetic binary")
        self.reinventory()
        self.assert_failed("provenance hash mismatch")

    @unittest.skipUnless(os.name == "nt" and shutil.which("pwsh"), "Native PowerShell 7 package metadata contract")
    def test_sanitized_editor_proof_binds_shipped_bytes_and_retains_original_audit(self):
        original_proof = {
            "output": f"E:/AI/development/compiled/studio-selection/{self.key}",
            "architecture": "8.9", "cuda_initialized": False,
            "binary_sha256": self.hash(self.prefix + "studio_selection_cuda.pyd"),
            "source_sha256": {name: self.hash("engine/editor_native/" + name)
                              for name in ("selection.cpp", "selection.cu")},
        }
        original = (json.dumps(original_proof, indent=2) + "\n").encode()
        self.write(self.prefix + "build-proof.json", original)
        source_proof = self.root / "original-build-proof.json"
        source_proof.write_bytes(original)
        audit = self.root / "private-audit"
        selection = dict(self.inputs["editor_selection"])
        selection.update(binary_name="studio_selection_cuda.pyd", proof_sha256=hashlib.sha256(original).hexdigest(),
                         sources=[{"name": name, "sha256": self.hash("engine/editor_native/" + name)}
                                  for name in ("selection.cpp", "selection.cu")])
        arguments = self.root / "fixture.json"
        arguments.write_text(json.dumps({"package": str(self.package), "audit": str(audit),
            "source_proof": str(source_proof), "relative_proof": self.prefix + "build-proof.json",
            "selection": selection, "lock_hash": self.hash("runtime/python-requirements.lock.txt"),
            "policy_hash": self.hash("runtime-prune-policy.json"),
            "source_components": [{"source": "E:/AI/development/compiled", "component": "extensions",
                                   "path": "engine/extensions"}]}), encoding="utf-8")
        script = self.root / "sanitize-fixture.ps1"
        script.write_text(r'''
param([string]$FixtureInputs, [string]$ProvenanceScript)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Fixture = Get-Content -LiteralPath $FixtureInputs -Raw | ConvertFrom-Json
$PackageRoot = $Fixture.package
$AuditRoot = $Fixture.audit
$SourceComponents = @($Fixture.source_components)
$Selection = $Fixture.selection
$Profile = 'public'
$Plan = @{python_lock_sha256=$Fixture.lock_hash;prune_policy_sha256=$Fixture.policy_hash;editor_selection=$Selection}
function Under([string]$Path, [string]$Root) {
    $Full = [IO.Path]::GetFullPath($Path)
    $Boundary = [IO.Path]::GetFullPath($Root).TrimEnd('\','/')
    return $Full.Equals($Boundary,[StringComparison]::OrdinalIgnoreCase) -or $Full.StartsWith($Boundary + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)
}
function Destination([string]$Relative) { return Join-Path $PackageRoot $Relative }
. $ProvenanceScript
Protect-PublicText (Destination $Fixture.relative_proof) $Fixture.source_proof $Fixture.relative_proof
Public-Plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Destination 'package-inputs.json') -Encoding UTF8
$Plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $AuditRoot 'package-inputs.json') -Encoding UTF8
@($Transformations.ToArray()) | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $AuditRoot 'transformations.json') -Encoding UTF8
''', encoding="utf-8")
        proc = subprocess.run([shutil.which("pwsh"), "-NoProfile", "-File", str(script), str(arguments),
                               str(REPO / "desktop/scripts/package-provenance.ps1")],
                              capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        shipped = candidate.read_json(self.package / (self.prefix + "build-proof.json"))
        self.assertEqual(shipped["output"], self.prefix.rstrip("/"))
        for field in ("architecture", "cuda_initialized", "binary_sha256", "source_sha256"):
            self.assertEqual(shipped[field], original_proof[field])
        self.assertEqual((audit / "originals" / (self.prefix + "build-proof.json")).read_bytes(), original)
        private_inputs = candidate.read_json(audit / "package-inputs.json")
        self.assertEqual(private_inputs["editor_selection"]["proof_sha256"], hashlib.sha256(original).hexdigest())
        self.assertNotEqual(self.hash(self.prefix + "build-proof.json"), hashlib.sha256(original).hexdigest())

        def refresh_manifest():
            self.reinventory()
            self.inventory["version"] = "0.3.0-rc.1"
            self.write_json("package-inventory.json", self.inventory)

        refresh_manifest()
        result = candidate.verify(self.package)
        self.assertEqual(result["status"], "passed", result)
        # Even a matching regenerated inventory must not conceal a changed proof.
        shipped["output"] = "changed-after-packaging"
        self.write_json(self.prefix + "build-proof.json", shipped)
        refresh_manifest()
        self.assert_failed("provenance hash mismatch")

    def test_editor_proof_assertions_and_source_hashes_are_checked(self):
        self.write_json(self.prefix + "build-proof.json", {"architecture": "9.0", "cuda_initialized": True})
        self.inputs["editor_selection"]["source_proof_sha256"] = self.hash(self.prefix + "build-proof.json")
        self.write_json("package-inputs.json", self.inputs)
        self.reinventory()
        self.assert_failed("architecture/offline")

    def test_exclusion_policy_binding_is_checked(self):
        self.write_json("package-exclusions.json", {"schema_version": 1, "policy_sha256": "0" * 64,
                                                  "files": [], "excluded_bytes": 0})
        self.reinventory()
        self.assert_failed("Exclusion/prune-policy")

    def populate_source(self):
        for record in self.inventory["files"]:
            relative = record["path"]
            if relative.startswith("engine/") and Path(relative).suffix in candidate.SOURCE_SUFFIXES:
                target = self.source / relative
            elif relative.startswith("pipeline/") and Path(relative).suffix in candidate.SOURCE_SUFFIXES:
                target = self.source / "src" / relative.removeprefix("pipeline/")
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.package / relative).read_bytes())

    def test_source_snapshot_compares_real_bytes_without_claiming_compiled_origin(self):
        self.populate_source()
        result = candidate.verify(self.package, self.source)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["source_snapshot"]["compared_files"], 4)
        self.assertEqual(result["source_snapshot"]["status"], "matched-source-files-only")
        self.assertFalse(result["source_snapshot"]["commit_bound"])
        (self.source / "src/gsstudio/main.py").write_bytes(b"changed = True\n")
        self.assert_failed("source snapshot differs", source_root=self.source)

    def test_expected_commit_is_a_strict_gate_not_an_unsupported_metadata_claim(self):
        self.inputs["source_commit"] = "a" * 40
        self.write_json("package-inputs.json", self.inputs)
        self.reinventory()
        self.assert_failed("source commit cannot be established", expected_source_commit="a" * 40)

    def test_cli_creates_new_report_and_refuses_overwrite_or_package_writes(self):
        arguments = [sys.executable, "-I", "-B", "-X", "utf8", str(SCRIPT),
                     "--package", str(self.package), "--report", str(self.report)]
        proc = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        before = self.report.read_bytes()
        proc = subprocess.run(arguments, capture_output=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(self.report.read_bytes(), before)
        arguments[-1] = str(self.package / "forbidden-report.json")
        proc = subprocess.run(arguments, capture_output=True)
        self.assertEqual(proc.returncode, 2)
        self.assertFalse((self.package / "forbidden-report.json").exists())

    def test_symlinked_payload_is_rejected_without_following_it(self):
        target = self.root / "outside.txt"
        target.write_bytes(b"external private fixture")
        link = self.package / "linked.txt"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("symlink creation unavailable")
        self.assert_failed("Symlink/reparse")
        self.assertEqual(target.read_bytes(), b"external private fixture")


if __name__ == "__main__":
    unittest.main()
