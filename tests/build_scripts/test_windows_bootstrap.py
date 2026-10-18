"""Public-input bootstrap contracts; no network, installation or CUDA import."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("windows_bootstrap", ROOT / "scripts/runtime/bootstrap_windows.py")
BOOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOT)


class WindowsBootstrapContracts(unittest.TestCase):
    def args(self, workspace, **changes):
        values = dict(workspace=workspace, download_cache=None, architecture="8.9",
                      vcvars=None, vcvars_version="14.44", cuda_home=None,
                      download_cuda=False, jobs=2, git=None)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_public_inputs_cover_all_frozen_non_native_packages(self):
        inputs = BOOT.load_inputs()
        self.assertEqual(len(inputs["wheels"]), 44)
        self.assertEqual(len(inputs["sdists"]), 1)
        self.assertEqual(inputs["python_version"], "3.10.22")
        self.assertEqual(inputs["cuda"]["version"], "12.9.0")
        nvcc = next(x for x in inputs["cuda"]["components"] if x["name"] == "cuda_nvcc")
        self.assertEqual(nvcc["version"], "12.9.41")

    def test_torch_and_sources_are_original_versions(self):
        inputs = BOOT.load_inputs()
        packages = {BOOT.canonical(w["name"]): w["version"] for w in inputs["wheels"]}
        self.assertEqual(packages["torch"], "2.8.0+cu129")
        self.assertEqual(packages["torchvision"], "0.23.0+cu129")
        gsplat = next(s for s in inputs["sources"] if s["name"] == "gsplat")
        self.assertEqual(gsplat["revision"], "937e29912570c372bed6747a5c9bf85fed877bae")
        self.assertEqual(gsplat["glm_revision"], "33b4a621a697a305bc3a7610d290677b96beb181")

    def test_plan_unicode_and_spaces_creates_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "中文 cold build"
            with mock.patch.object(BOOT.subprocess, "run", side_effect=AssertionError("No subprocess in plan")):
                plan = BOOT.make_plan(self.args(workspace))
            self.assertEqual(plan["architecture"], "8.9")
            self.assertEqual(plan["git_checkout_policy"], {"core.symlinks": False})
            self.assertFalse(workspace.exists())
            self.assertIn("engine/native/splat3_tiles/tiles.cu", plan["numerical_sources"])

    def test_workspace_cannot_be_repository_or_parent(self):
        for directory in (ROOT, ROOT.parent, ROOT / "not-created-bootstrap"):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "outside"):
                BOOT.make_plan(self.args(directory))

    def test_workspace_cannot_adopt_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "not owned"):
            BOOT.make_plan(self.args(Path(temporary)))

    def test_workspace_cannot_contain_download_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "new"
            with self.assertRaisesRegex(ValueError, "outside source and workspace"):
                BOOT.make_plan(self.args(workspace, download_cache=workspace / "cache"))

    def test_plan_rejects_malformed_architecture_jobs_and_toolset(self):
        with tempfile.TemporaryDirectory() as temporary:
            for changes in (dict(architecture="native"), dict(jobs=0), dict(vcvars_version="latest")):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    BOOT.make_plan(self.args(Path(temporary) / "new", **changes))

    def test_readonly_download_cache_copies_exact_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            cache = parent / "cache"
            cache.mkdir()
            (cache / "input.whl").write_bytes(b"public input")
            expected = hashlib.sha256(b"public input").hexdigest()
            item = {"filename": "input.whl", "sha256": expected, "url": "https://example.invalid"}
            with mock.patch.object(BOOT, "run", side_effect=AssertionError("No network")):
                BOOT.checked_download(item, parent / "copy.whl", cache)
            self.assertEqual(BOOT.digest(parent / "copy.whl"), expected)
            self.assertEqual((cache / "input.whl").read_bytes(), b"public input")

    @unittest.skipUnless(shutil.which("git"), "Git fixture requires the build tool")
    def test_source_checkout_uses_local_false_despite_global_true_and_keeps_link_blob(self):
        git = shutil.which("git")
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            origin, checkout = parent / "origin", parent / "Windows checkout"
            global_config = parent / "isolated-global.gitconfig"
            global_config.write_text("[core]\n\tsymlinks = true\n", encoding="utf-8")
            before = global_config.read_bytes()
            with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(global_config), "GIT_CONFIG_NOSYSTEM": "1"}):
                subprocess.run([git, "init", "-q", str(origin)], check=True, capture_output=True)
                (origin / "target.txt").write_text("real target contents", encoding="utf-8")
                subprocess.run([git, "-C", str(origin), "add", "target.txt"], check=True, capture_output=True)
                link_blob = subprocess.check_output([git, "-C", str(origin), "hash-object", "-w", "--stdin"],
                                                    input=b"target.txt").decode().strip()
                subprocess.run([git, "-C", str(origin), "update-index", "--add", "--cacheinfo",
                                f"120000,{link_blob},link.txt"], check=True, capture_output=True)
                subprocess.run([git, "-C", str(origin), "-c", "user.name=Fixture", "-c",
                                "user.email=fixture@example.invalid", "commit", "-q", "--no-gpg-sign", "-m", "link fixture"],
                               check=True, capture_output=True)
                revision = subprocess.check_output([git, "-C", str(origin), "rev-parse", "HEAD"]).decode().strip()
                with mock.patch("builtins.print"):
                    proof = BOOT.checkout_source(git, {"name": "fixture", "url": str(origin), "revision": revision}, checkout)
                self.assertEqual(proof["core.symlinks"], False)
                self.assertFalse((checkout / "link.txt").is_symlink())
                self.assertEqual((checkout / "link.txt").read_bytes(), b"target.txt")
                self.assertEqual(subprocess.check_output([git, "-C", str(checkout), "config", "--local", "--get", "core.symlinks"]).decode().strip(), "false")
                files = BOOT.native_helper().source_files(checkout)
                self.assertIn("link.txt", files)
                self.assertEqual(files["link.txt"], hashlib.sha256(b"target.txt").hexdigest())
            self.assertEqual(global_config.read_bytes(), before)

    def test_bad_cache_is_retained_and_not_downloaded_over(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent / "input.whl").write_bytes(b"changed")
            item = {"filename": "input.whl", "sha256": "0" * 64, "url": "https://example.invalid"}
            with mock.patch.object(BOOT, "run", side_effect=AssertionError("No network")):
                with self.assertRaisesRegex(ValueError, "cache hash mismatch"):
                    BOOT.checked_download(item, parent / "copy.whl", parent)
            self.assertFalse((parent / "copy.whl").exists())
            self.assertEqual((parent / "input.whl").read_bytes(), b"changed")

    def test_existing_download_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "file"
            target.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "Existing download"):
                BOOT.checked_download({"sha256": "0" * 64}, target)
            self.assertEqual(target.read_bytes(), b"keep")

    def test_archive_path_traversal_rejected_before_extract(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / "input.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                info = tarfile.TarInfo("../escape.txt")
                info.size = 4
                stream.addfile(info, io.BytesIO(b"bad!"))
            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                BOOT.extract_archive(archive, parent / "out")
            self.assertFalse((parent / "escape.txt").exists())

    def test_archive_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / "input.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                info = tarfile.TarInfo("link")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                stream.addfile(info)
            with self.assertRaisesRegex(ValueError, "nonregular"):
                BOOT.extract_archive(archive, parent / "out")

    def test_zip_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / "input.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                stream.writestr(info, "../../outside")
            with self.assertRaisesRegex(ValueError, "symlink"):
                BOOT.extract_archive(archive, parent / "out")

    def test_windows_archive_drive_or_ads_rejected(self):
        for name in ("C:/Windows/file", "folder/file:stream", "\\\\host\\share\\x", "/absolute"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                BOOT.safe_member(name)

    def test_valid_archive_retains_legal_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / "input.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("component/LICENSE", "license")
            BOOT.extract_archive(archive, parent / "out")
            self.assertEqual((parent / "out/component/LICENSE").read_text(), "license")

    def test_offline_process_environment_excludes_global_python_cuda(self):
        with mock.patch.dict(BOOT.os.environ, {"PATH": "C:/unsafe/python;C:/cuda/bin",
                         "PYTHONPATH": "C:/unsafe/site-packages", "CUDA_HOME": "C:/cuda"}, clear=True):
            env = BOOT.offline_env(Path("new-runtime"))
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("CUDA_HOME", env)
        self.assertNotIn("unsafe", env["PATH"])
        self.assertEqual(env["PIP_NO_INDEX"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")

    def test_lock_include_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            (parent / "requirements.txt").write_text("-r ../outside.txt\n")
            with self.assertRaisesRegex(ValueError, "escapes"):
                BOOT.read_pins(parent / "requirements.txt")

    def test_changed_owned_identity_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            args = self.args(parent / "owned")
            plan = BOOT.make_plan(args)
            Path(plan["workspace"]).mkdir()
            BOOT.write_json(Path(plan["workspace"]) / "bootstrap-inputs.json", plan)
            plan["architecture"] = "9.0"
            with self.assertRaisesRegex(ValueError, "identity changed"):
                BOOT.owned_plan(plan)

    def test_verify_does_not_pass_without_completed_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary) / "owned")
            plan = BOOT.make_plan(args)
            Path(plan["workspace"]).mkdir()
            BOOT.write_json(Path(plan["workspace"]) / "bootstrap-inputs.json", plan)
            with self.assertRaisesRegex(ValueError, "has not completed"):
                BOOT.verify(plan)


if __name__ == "__main__":
    unittest.main()
