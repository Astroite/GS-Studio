"""Real temporary package manifests; no installed application or GPU required."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
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
