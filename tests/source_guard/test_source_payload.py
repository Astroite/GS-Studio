"""Pure stdlib, isolated Git repositories; never reads user media or runs LFS."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "scripts/check_source_payload.py"
ZERO = "0" * 40


class SourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gsstudio-source-guard-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repository"
        self.repo.mkdir()
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
        # Do not inherit an outer Git hook's index/worktree/config injection.
        for key in list(self.env):
            if key in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"} or key.startswith("GIT_CONFIG_COUNT"):
                self.env.pop(key)
        self.git("init", "-q", "--initial-branch=main")
        self.git("config", "user.name", "Source Guard Test")
        self.git("config", "user.email", "source-guard@example.invalid")
        self.git("config", "core.hooksPath", ".disabled-test-hooks")
        self.git("config", "core.autocrlf", "false")
        self.write("README.md", b"synthetic source\n")
        self.git("add", "README.md")
        self.git("commit", "-qm", "initial source")

    def git(self, *args, cwd=None):
        result = subprocess.run(["git", *args], cwd=cwd or self.repo, env=self.env,
                                capture_output=True, check=True)
        return result.stdout.decode("utf-8", "replace").strip()

    def write(self, path, payload):
        location = self.repo / path
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_bytes(payload)

    def stage(self, path, payload):
        self.write(path, payload)
        self.git("add", "-f", "--", path)

    def scan(self, *args, stdin="", repo=None):
        result = subprocess.run([sys.executable, "-I", "-B", str(SCANNER), "--repo",
                                 str(repo or self.repo), *args], cwd=self.repo, env=self.env,
                                input=stdin, text=True, encoding="utf-8", capture_output=True)
        return result

    def assertAllowed(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assertRejected(self, result, reason):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(reason, result.stderr)

    def test_clean_source_and_typescript_allowed(self):
        self.stage("desktop/src/app.ts", b"export const fixture = true;\n")
        self.assertAllowed(self.scan("staged"))

    def test_forced_ignored_video_is_rejected(self):
        self.stage(".gitignore", b"*.mp4\n")
        self.stage("private-movie.mp4", b"not even a valid movie")
        self.assertRejected(self.scan("staged"), ".mp4")

    def test_uppercase_model_and_video_extensions_rejected(self):
        for path in ("render.MP4", "model.PLY", "model.SPZ", "weights.PTH"):
            with self.subTest(path=path):
                self.stage(path, b"synthetic rejected artifact")
                self.assertRejected(self.scan("staged"), "forbidden source payload")
                self.git("rm", "--cached", "--", path)

    def test_disguised_lfs_pointer_rejected(self):
        self.stage("notes.txt", b"version https://git-lfs.github.com/spec/v1\noid sha256:"
                   + b"a" * 64 + b"\nsize 123\n")
        self.assertRejected(self.scan("staged"), "Git LFS pointer")

    def test_oversized_blob_rejected_without_worktree_file(self):
        self.stage("notes.txt", b"x" * (5 * 1024 * 1024 + 1))
        (self.repo / "notes.txt").unlink()
        self.assertRejected(self.scan("staged"), "exceeds 5 MiB")

    def test_dirty_worktree_cannot_hide_staged_pointer(self):
        self.stage("notes.txt", b"version https://git-lfs.github.com/spec/v1\n")
        self.write("notes.txt", b"clean working tree copy\n")
        self.assertRejected(self.scan("staged"), "Git LFS pointer")

    def test_unstaged_pointer_does_not_replace_safe_staged_blob(self):
        self.stage("notes.txt", b"staged source\n")
        self.write("notes.txt", b"version https://git-lfs.github.com/spec/v1\n")
        self.assertAllowed(self.scan("staged"))

    def test_current_local_data_directory_casefolded(self):
        self.stage("DATA/capture.json", b"{}\n")
        self.assertRejected(self.scan("staged"), "local data/runtime/output")

    def test_safe_deletion_staged_but_contaminated_ancestor_rejected(self):
        self.stage("scene.ply", b"ply\nsynthetic old artifact\n")
        self.git("commit", "-qm", "old contaminated source")
        self.git("rm", "scene.ply")
        self.assertAllowed(self.scan("staged"))
        self.git("commit", "-qm", "clean current source")
        self.assertRejected(self.scan("history", "--ref", "HEAD"), "scene.ply")

    def test_same_blob_safe_name_and_historical_video_alias(self):
        payload = b"the identical harmless blob has two paths\n"
        self.stage("notes.txt", payload)
        self.stage("exports/preview.mp4", payload)
        self.git("commit", "-qm", "old aliased artifact")
        self.git("rm", "exports/preview.mp4")
        self.git("commit", "-qm", "remove alias")
        self.assertRejected(self.scan("history"), "exports/preview.mp4")

    def test_legacy_metadata_ancestor_retained_but_tip_rejected(self):
        self.stage("locations/legacy/metadata.json", b"{}\n")
        self.git("commit", "-qm", "legacy metadata only")
        self.assertRejected(self.scan("history"), "local data/runtime/output")
        self.git("rm", "locations/legacy/metadata.json")
        self.git("commit", "-qm", "source-only current tree")
        self.assertAllowed(self.scan("history"))

    def test_pre_push_rejects_private_ref_even_clean(self):
        oid = self.git("rev-parse", "HEAD")
        line = f"refs/heads/main {oid} refs/codex/checkpoints/run-1 {ZERO}\n"
        self.assertRejected(self.scan("pre-push", stdin=line), "disallows remote ref")

    def test_pre_push_deletion_of_private_ref_allowed(self):
        oid = self.git("rev-parse", "HEAD")
        line = f"(delete) {ZERO} refs/codex/checkpoints/run-1 {oid}\n"
        self.assertAllowed(self.scan("pre-push", stdin=line))

    def test_pre_push_safe_branch_and_annotated_tag(self):
        self.git("tag", "-a", "preview", "-m", "source preview")
        oid = self.git("rev-parse", "HEAD")
        tag = self.git("rev-parse", "refs/tags/preview")
        lines = f"refs/heads/main {oid} refs/heads/main {ZERO}\nrefs/tags/preview {tag} refs/tags/preview {ZERO}\n"
        self.assertAllowed(self.scan("pre-push", stdin=lines))

    def test_pre_push_scans_ancestor_not_only_clean_tip(self):
        self.stage("old.MOV", b"old synthetic media")
        self.git("commit", "-qm", "contaminated ancestor")
        self.git("rm", "old.MOV")
        self.git("commit", "-qm", "clean tip")
        oid = self.git("rev-parse", "HEAD")
        self.assertRejected(self.scan("pre-push", stdin=f"refs/heads/main {oid} refs/heads/main {ZERO}\n"), "old.MOV")

    def test_merge_of_clean_tip_with_contaminated_side_ancestor_rejected(self):
        self.git("checkout", "-qb", "feature")
        self.stage("side-history.mp4", b"synthetic historical side-branch video")
        self.git("commit", "-qm", "contaminated side ancestor")
        self.git("rm", "side-history.mp4")
        self.git("commit", "-qm", "clean side tip")
        self.git("checkout", "-q", "main")
        self.stage("source.py", b"answer = 42\n")
        self.git("commit", "-qm", "main source advance")
        self.git("merge", "--no-ff", "-qm", "merge clean side tip", "feature")
        self.assertAllowed(self.scan("staged"))
        self.assertRejected(self.scan("history"), "side-history.mp4")

    def test_staged_and_historical_symlinks_rejected_without_os_privilege(self):
        blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], input=b"../private-data",
                              cwd=self.repo, env=self.env, capture_output=True, check=True).stdout.decode().strip()
        self.git("update-index", "--add", "--cacheinfo", f"120000,{blob},source-link")
        self.assertRejected(self.scan("staged"), "unsupported index entry")
        self.git("commit", "-qm", "synthetic symlink tree")
        self.assertRejected(self.scan("history"), "unsupported mode/non-blob tree entry")

    def test_real_synthetic_fixture_exact_hash_allowed(self):
        for path in ("tests/fixtures/nerfstudio_vendor/parsing/cameras.bin",
                     "tests/fixtures/nerfstudio_vendor/projection-input.npy",
                     "tests/fixtures/nerfstudio_vendor/projection-results.npz",
                     "tests/fixtures/nerfstudio_vendor/conversion/single_default/expected/sparse-ply.txt"):
            self.stage(path, (ROOT/path).read_bytes())
        icon = "desktop/src-tauri/icons/icon.ico"
        self.stage(icon, (ROOT/icon).read_bytes())
        self.assertAllowed(self.scan("staged"))

    def test_small_private_photos_and_unreviewed_image_fixtures_rejected(self):
        for path,payload in (("private.JPG", b"\xff\xd8\xffminimal jpeg marker"),
                             ("tiny.PnG", b"\x89PNG\r\n\x1a\nminimal png marker"),
                             ("tests/fixtures/unreviewed.png", b"\x89PNG\r\n\x1a\nsmall image")):
            with self.subTest(path=path):
                self.stage(path,payload)
                self.assertRejected(self.scan("staged"), "forbidden source payload")
                self.git("rm", "--cached", "--", path)

    def test_fixture_allowlist_does_not_allow_modified_or_renamed_binary(self):
        path = "tests/fixtures/nerfstudio_vendor/parsing/cameras.bin"
        self.stage(path, (ROOT/path).read_bytes() + b"changed")
        self.assertRejected(self.scan("staged"), "forbidden source payload")
        self.git("rm", "--cached", path)
        self.stage("another.bin", (ROOT/path).read_bytes())
        self.assertRejected(self.scan("staged"), "forbidden source payload")

    def test_renamed_executable_video_and_ply_content_rejected(self):
        for payload in (b"MZsynthetic executable", b"\x00\x00\x00\x18ftypisom", b"ply\nformat ascii 1.0\n"):
            self.stage("notes.txt", payload)
            self.assertRejected(self.scan("staged"), "content is forbidden")

    def test_invalid_ref_and_malformed_input_fail_closed(self):
        self.assertRejected(self.scan("history", "--ref", "does-not-exist"), "failed closed")
        self.assertRejected(self.scan("pre-push", stdin="bad hook input\n"), "failed closed")

    def test_shallow_history_fails_closed(self):
        clone = Path(self.temporary.name) / "shallow"
        self.git("clone", "-q", "--depth", "1", self.repo.as_uri(), str(clone))
        self.assertRejected(self.scan("history", repo=clone), "complete history required")

    def test_bare_mirror_supported(self):
        bare = Path(self.temporary.name) / "clean.git"
        self.git("clone", "-q", "--bare", str(self.repo), str(bare))
        self.assertAllowed(self.scan("history", "--all", repo=bare))

    def test_installer_preserves_old_hook_and_sets_only_local_settings(self):
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("Native Windows PowerShell installer requires PowerShell")
        shutil.copytree(ROOT/".githooks", self.repo/".githooks")
        (self.repo/"scripts").mkdir()
        for name in ("check_source_payload.py", "source-fixtures.json", "install-source-guards.ps1"):
            shutil.copy2(ROOT/"scripts"/name, self.repo/"scripts"/name)
        previous = self.repo/".disabled-test-hooks/pre-push"
        previous.parent.mkdir()
        previous.write_bytes(b"#!/bin/sh\necho preserved old hook\n")
        result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                 str(self.repo/"scripts/install-source-guards.ps1"), "-PythonPath", sys.executable],
                                cwd=self.repo, env=self.env, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout.decode("utf-8", "replace") + result.stderr.decode("utf-8", "replace"))
        self.assertEqual(self.git("config", "--local", "--get", "core.hooksPath"), ".githooks")
        self.assertEqual(Path(self.git("config", "--local", "--get", "gsstudio.guardPython")), Path(sys.executable))
        receipts = list((self.repo/".git/source-guard-backups").glob("*/settings.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text("utf-8"))
        self.assertEqual(receipt["previous_local_hooks_path"], [".disabled-test-hooks"])
        backup_hook = receipts[0].parent/"previous-hooks/pre-push"
        self.assertEqual(backup_hook.read_bytes(), previous.read_bytes())
        self.assertEqual(receipt["preserved_hook_files"][0]["name"], "pre-push")

    def test_real_commit_and_local_push_use_configured_guard(self):
        shutil.copytree(ROOT/".githooks", self.repo/".githooks")
        for hook in (self.repo/".githooks").iterdir():
            hook.chmod(0o755)
        (self.repo/"scripts").mkdir()
        for name in ("check_source_payload.py", "source-fixtures.json"):
            shutil.copy2(ROOT/"scripts"/name, self.repo/"scripts"/name)
        self.git("config", "core.hooksPath", ".githooks")
        self.git("config", "gsstudio.guardPython", sys.executable)
        self.stage("source.py", b"answer = 42\n")
        self.git("commit", "-qm", "real configured pre-commit hook")
        destination = Path(self.temporary.name)/"local destination.git"
        self.git("init", "--bare", "-q", str(destination))
        # Only a synthetic local bare destination is written; no network.
        self.git("push", str(destination), "main:refs/heads/main")
        self.assertEqual(self.git("rev-parse", "HEAD"),
                         self.git("rev-parse", "refs/heads/main", cwd=destination))

    def test_local_hook_interpreter_failure_is_closed_and_lfs_not_invoked(self):
        git_executable = Path(shutil.which("git"))
        shell = shutil.which("sh")
        if not shell and os.name == "nt":
            shell_candidate = git_executable.parent.parent / "usr/bin/sh.exe"
            if shell_candidate.is_file():
                shell = str(shell_candidate)
        self.assertIsNotNone(shell, "Git's POSIX shell is required to exercise installed hooks")
        shutil.copytree(ROOT/".githooks", self.repo/".githooks")
        (self.repo/"scripts").mkdir()
        shutil.copy2(SCANNER, self.repo/"scripts/check_source_payload.py")
        self.git("config", "gsstudio.guardPython", str(self.repo/"missing-python.exe"))
        result = subprocess.run([shell, str(self.repo/".githooks/pre-push")], cwd=self.repo,
                                env=self.env, input="", text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("git-lfs", result.stdout+result.stderr)


if __name__ == "__main__":
    unittest.main()
