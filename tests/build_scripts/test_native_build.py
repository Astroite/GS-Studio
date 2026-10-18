"""Portable build entrypoints: real CLI/path checks, no Torch, CUDA or compiler."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "engine/scripts/build_native.py"
SPEC = importlib.util.spec_from_file_location("native_build_contract", SCRIPT)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="GS Studio 构建 ")
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / "输出 jobs" / "new-job"

    def tearDown(self):
        self.temp.cleanup()

    def args(self, module="splat3-render", *extra):
        return native.parse_args([module, "--python", sys.executable,
                                  "--output", str(self.out), "--plan", *extra])

    def fixture(self, name, files):
        root = self.root / name
        for file in files:
            path = root / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        return root

    def invoke(self, script, args, env=None):
        return subprocess.run([sys.executable, "-I", "-B", "-X", "utf8", str(script), *args],
                              cwd=self.root, env=env, capture_output=True, text=True, encoding="utf-8")

    def test_internal_mapping_is_engine_native_from_unrelated_cwd(self):
        for module in ("splat3-stats", "splat3-render", "splat3-tiles"):
            with self.subTest(module=module):
                result = self.invoke(SCRIPT, [module, "--output", str(self.out), "--plan"])
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                root = Path(plan["sources"][module]["root"])
                self.assertEqual(root, REPO / "engine/native" / module.replace("-", "_"))
                self.assertNotIn("engine/engine", root.as_posix())
        self.assertFalse(self.out.exists())

    def test_selection_plan_never_imports_torch_or_engine_runtime(self):
        before = set(sys.modules)
        plan = native.make_plan(self.args("studio-selection"))
        self.assertEqual(Path(plan["sources"]["studio-selection"]["root"]), REPO / "engine/editor_native")
        self.assertNotIn("torch", set(sys.modules) - before)
        self.assertFalse(any(name.startswith("local_postshot") for name in set(sys.modules) - before))
        self.assertFalse(self.out.exists())

    def test_projection_requires_and_records_pinned_glm_source(self):
        with self.assertRaisesRegex(ValueError, "GLM"):
            native.make_plan(self.args("splat3-projection"))
        gsplat = self.fixture("gsplat 中文", [native.GLM + "/glm/glm.hpp", "setup.py"])
        plan = native.make_plan(self.args("splat3-projection", "--gsplat-source", str(gsplat)))
        self.assertEqual(plan["sources"]["gsplat"]["revision"], native.PINS["gsplat"])
        self.assertEqual(plan["glm_revision"], native.GLM_PIN)

    def test_external_sources_must_be_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            for module in ("gsplat", "fused-ssim", "spz"):
                with self.subTest(module=module), self.assertRaisesRegex(ValueError, "Explicit"):
                    native.make_plan(self.args(module))

    def test_spz_requires_local_compression_sources_and_preserves_build_cmake(self):
        spz = self.fixture("spz", ["pyproject.toml"])
        zlib = self.fixture("zlib", ["CMakeLists.txt"])
        zstd = self.fixture("zstd", ["build/cmake/CMakeLists.txt", "lib/zstd.h"])
        with self.assertRaisesRegex(ValueError, "zlib-source"):
            native.make_plan(self.args("spz", "--spz-source", str(spz)))
        plan = native.make_plan(self.args("spz", "--spz-source", str(spz),
                                "--zlib-source", str(zlib), "--zstd-source", str(zstd)))
        self.assertIn("build/cmake/CMakeLists.txt", plan["sources"]["zstd"]["files"])
        self.assertFalse(plan["downloads"])

    def test_plan_preserves_space_unicode_arguments_and_omits_compiler_requirement(self):
        source = self.fixture("源码 含空格", ["setup.py", "binding.cpp"])
        result = self.invoke(SCRIPT, ["splat3-render", "--source-root", str(source),
                                     "--output", str(self.out), "--plan"])
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["sources"]["splat3-render"]["root"], str(source))
        self.assertEqual(plan["output"], str(self.out))
        self.assertFalse(self.out.exists())

    def test_output_must_be_new_external_and_not_runtime_ancestor(self):
        self.out.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            native.make_plan(self.args())
        self.out.rmdir()
        for output in (REPO / "new-build-output", Path(sys.executable).parent / "new-build-output"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overlap"):
                native.make_plan(native.parse_args(["splat3-render", "--output", str(output), "--plan"]))

    def test_missing_source_python_and_invalid_architecture_are_rejected(self):
        for extra, error in [(["--source-root", str(self.root / "missing")], "entry missing"),
                             (["--python", str(self.root / "missing.exe")], "Python executable missing"),
                             (["--architecture", "8.9 & cmd"], "Architecture")]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, error):
                native.make_plan(self.args("splat3-render", *extra))

    def test_source_snapshot_omits_build_products_and_keeps_text_sources(self):
        source = self.fixture("snapshot", ["setup.py", "native/kernel.cu", "build/temp.obj",
                                         "cache.pyc", "dist/old.whl", "old.egg-info/PKG-INFO"])
        self.assertEqual(set(native.source_files(source)), {"setup.py", "native/kernel.cu"})

    def test_child_environment_drops_host_compiler_flags_and_preserves_windows_context(self):
        unsafe = {name: "injected-host-setting" for name in (
            "CC", "CXX", "CL", "_CL_", "CFLAGS", "CPPFLAGS", "CXXFLAGS", "LDFLAGS",
            "NVCC_FLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CUDACXX",
            "CMAKE_ARGS", "SKBUILD_CONFIGURE_OPTIONS", "BUILD_NO_CUDA", "WITH_SYMBOLS",
            "LINE_INFO", "CUDA_HOME", "CUDA_PATH", "MAX_JOBS", "TORCH_CUDA_ARCH_LIST",
            "LP_GSPLAT_SOURCE", "PYTHONPATH", "INCLUDE", "LIB", "LIBPATH", "VSCMD_VER", "VSINSTALLDIR")}
        original = {"Path": "existing-system-path", "SystemRoot": "C:/Windows", "TEMP": str(self.root), **unsafe}
        with patch.dict(os.environ, original, clear=True):
            before = dict(os.environ)
            result = native.base_build_environment()
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(result, {"PATH": "existing-system-path", "SYSTEMROOT": "C:/Windows", "TEMP": str(self.root)})

    @unittest.skipUnless(os.name == "nt", "Windows compiler initialization contract")
    def test_compiler_environment_uses_explicit_plan_after_discarding_host_settings(self):
        vcvars = self.root / "explicit vcvarsall.bat"
        vcvars.write_text("fixture only\n")
        cuda = self.fixture("explicit cuda", ["bin/nvcc.exe"])
        plan = native.make_plan(self.args("splat3-render", "--vcvars", str(vcvars), "--cuda-home", str(cuda),
                                         "--architecture", "8.9", "--jobs", "2"))
        captured = []
        def initialize(command, **kwargs):
            captured.append(dict(kwargs["env"]))
            return subprocess.CompletedProcess(command, 0, stdout="PATH=explicit-msvc-path\nINCLUDE=explicit-msvc-headers\n".encode("utf-16-le"), stderr=b"")
        with patch.dict(os.environ, {"PATH": "old-path", "SystemRoot": "C:/Windows", "CC": "wrong-cc",
                                    "CL": "/fp:fast", "NVCC_FLAGS": "--fmad=false", "CUDA_HOME": "wrong-cuda",
                                    "TORCH_CUDA_ARCH_LIST": "7.5", "MAX_JOBS": "99"}, clear=True), \
                patch.object(native.subprocess, "run", side_effect=initialize), \
                patch.object(native.shutil, "which", return_value=str(self.root / "ninja.exe")):
            env = native.compiler_environment(plan)
        self.assertNotIn("CC", captured[0])
        self.assertNotIn("CL", env)
        self.assertNotIn("NVCC_FLAGS", env)
        self.assertEqual(env["CUDA_HOME"], str(cuda))
        self.assertEqual(env["CUDA_PATH"], str(cuda))
        self.assertEqual(env["TORCH_CUDA_ARCH_LIST"], "8.9")
        self.assertEqual(env["CUDA_ARCHITECTURES"], "89")
        self.assertEqual(env["MAX_JOBS"], "2")
        self.assertEqual(env["INCLUDE"], "explicit-msvc-headers")

    def test_archive_receipt_binds_every_source_file_and_revision(self):
        source = self.fixture("fused source", ["setup.py", "kernel.cu"])
        plan = native.make_plan(self.args("fused-ssim", "--fused-ssim-source", str(source)))
        with self.assertRaisesRegex(ValueError, "archive needs"):
            native.verify_external_sources(plan)
        receipts = {"fused-ssim": {"revision": native.PINS["fused-ssim"],
                                   "files": dict(plan["sources"]["fused-ssim"]["files"])}}
        receipt = self.root / "receipts.json"
        receipt.write_text(json.dumps(receipts))
        native.verify_external_sources(plan, receipt)
        receipts["fused-ssim"]["files"]["extra.py"] = "0" * 64
        receipt.write_text(json.dumps(receipts))
        with self.assertRaisesRegex(ValueError, "archive needs"):
            native.verify_external_sources(plan, receipt)

    def test_git_commit_mismatch_and_dirty_trees_are_rejected(self):
        source = self.fixture("fused git", ["setup.py", ".git/HEAD"])
        plan = native.make_plan(self.args("fused-ssim", "--fused-ssim-source", str(source)))
        with patch.object(native, "git_output", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "Unpinned"):
                native.verify_external_sources(plan)
        with patch.object(native, "git_output", side_effect=[native.PINS["fused-ssim"], " M kernel.cu"]):
            with self.assertRaisesRegex(ValueError, "Modified"):
                native.verify_external_sources(plan)

    def test_symlink_source_is_rejected_when_supported(self):
        source = self.fixture("source", ["setup.py"])
        link = self.root / "linked"
        try:
            link.symlink_to(source, target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation is unavailable")
        with self.assertRaisesRegex(ValueError, "symlink/reparse"):
            native.make_plan(self.args("splat3-render", "--source-root", str(link)))

    @unittest.skipUnless(os.name == "nt", "Windows 8.3 overlap boundary")
    def test_short_alias_cannot_bypass_source_or_repository_output_overlap(self):
        import ctypes
        from ctypes import wintypes
        get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        get_short_path.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
        get_short_path.restype = wintypes.DWORD
        source = self.fixture("long source repository", ["setup.py", "kernel.cu"])
        size = get_short_path(str(source), None, 0)
        if not size:
            self.skipTest("GetShortPathNameW unavailable on the fixture volume")
        buffer = ctypes.create_unicode_buffer(size)
        if not get_short_path(str(source), buffer, size):
            self.skipTest("GetShortPathNameW did not return an alias")
        alias = Path(buffer.value)
        if str(alias).casefold() == str(source).casefold():
            self.skipTest("8.3 alias generation is disabled on the fixture volume")
        with self.assertRaisesRegex(ValueError, "overlap"):
            native.make_plan(native.parse_args(["splat3-render", "--python", sys.executable,
                "--source-root", str(source), "--output", str(alias / "new-output"), "--plan"]))
        unrelated_source = self.fixture("unrelated module", ["setup.py"])
        with patch.object(native, "REPO", source), self.assertRaisesRegex(ValueError, "overlap"):
            native.make_plan(native.parse_args(["splat3-render", "--python", sys.executable,
                "--source-root", str(unrelated_source), "--output", str(alias / "new-output"), "--plan"]))
        self.assertFalse((source / "new-output").exists())

    def test_real_build_failure_propagates_and_keeps_failure_proof(self):
        plan = native.make_plan(self.args())
        fake_result = subprocess.CompletedProcess([], 7)
        with patch.object(native, "compiler_environment", return_value=dict(os.environ)), \
                patch.object(native.subprocess, "run", return_value=fake_result):
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                native.build(plan)
        proof = json.loads((self.out / "build-proof.json").read_text())
        self.assertFalse(proof["success"])
        self.assertEqual(proof["exit_code"], 7)
        self.assertTrue((self.out / "sources/splat3-render/setup.py").is_file())
        self.assertFalse((REPO / "engine/native/splat3_render/build").exists())

    def test_main_failure_has_nonzero_status_without_output_mutation(self):
        with contextlib.redirect_stderr(io.StringIO()) as error:
            status = native.main(["splat3-render", "--python", "missing-python", "--output", str(self.out), "--plan"])
        self.assertEqual(status, 1)
        self.assertIn("Python executable missing", error.getvalue())
        self.assertFalse(self.out.exists())

    def test_spz_wheel_metadata_must_match_frozen_version_and_actual_cp310_abi(self):
        plan = {"distribution": "spz", "version": "1.1.0"}
        wheel = self.root / "fixture.whl"
        for version, tag, valid in (("1.1.0", "cp310-cp310-win_amd64", True),
                                    ("1.1.0", "cp312-abi3-win_amd64", False),
                                    ("1.2.0", "cp310-cp310-win_amd64", False)):
            with self.subTest(version=version, tag=tag):
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("spz.dist-info/METADATA", f"Name: spz\nVersion: {version}\n")
                    archive.writestr("spz.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
                if valid:
                    native.validate_wheel(wheel, plan)
                else:
                    with self.assertRaises(RuntimeError):
                        native.validate_wheel(wheel, plan)

    def test_spz_uses_ninja_and_initialized_msvc_not_host_generator_or_compilers(self):
        env = {"PATH": "explicit-vcvars-path", "CC": "wrong-c-compiler", "CXX": "wrong-cxx-compiler",
               "CMAKE_GENERATOR": "Visual Studio 17 2022", "CMAKE_GENERATOR_PLATFORM": "x64",
               "CMAKE_GENERATOR_TOOLSET": "old", "CMAKE_GENERATOR_INSTANCE": "old-vs",
               "CMAKE_TOOLCHAIN_FILE": "other-toolchain.cmake", "CMAKE_ARGS": "-G other"}
        compiler = str(self.root / "MSVC 含空格" / "cl.exe")
        with patch.object(native.shutil, "which", return_value=compiler) as lookup:
            native.configure_spz_environment(env, {"jobs": 2})
        lookup.assert_called_once_with("cl.exe", path="explicit-vcvars-path")
        self.assertEqual(env["CMAKE_GENERATOR"], "Ninja")
        self.assertEqual(env["CMAKE_BUILD_PARALLEL_LEVEL"], "2")
        self.assertEqual(env["CC"], compiler)
        self.assertEqual(env["CXX"], compiler)
        self.assertFalse(set(env) & {"CMAKE_GENERATOR_PLATFORM", "CMAKE_GENERATOR_TOOLSET",
                                    "CMAKE_GENERATOR_INSTANCE", "CMAKE_TOOLCHAIN_FILE", "CMAKE_ARGS"})

    def test_spz_missing_initialized_msvc_is_an_explicit_failure(self):
        with patch.object(native.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "cl.exe is missing"):
                native.configure_spz_environment({"PATH": "isolated"}, {"jobs": 2})

    @unittest.skipUnless(os.name == "nt", "Windows entrypoints")
    def test_every_cmd_delegates_to_same_relative_helper_and_propagates_bad_plan(self):
        mapping = {"native": "gsplat", "fused-ssim": "fused-ssim", "spz": "spz",
                   **{name: name for name in ("splat3-stats", "splat3-render", "splat3-tiles", "splat3-projection")}}
        env = dict(os.environ, GSSTUDIO_BUILD_PYTHON=sys.executable)
        for name, module in mapping.items():
            with self.subTest(script=name):
                source = self.fixture("CMD " + name, ["setup.py", "pyproject.toml", native.GLM + "/glm/glm.hpp"])
                command = f'"{REPO / ("engine/scripts/setup-build-" + name + ".cmd")}" --source-root "{source}" --output "{self.out}" --plan'
                if module == "spz":
                    zlib = self.fixture("CMD zlib", ["CMakeLists.txt"])
                    zstd = self.fixture("CMD zstd", ["build/cmake/CMakeLists.txt"])
                    command += f' --zlib-source "{zlib}" --zstd-source "{zstd}"'
                if module == "splat3-projection":
                    command += f' --gsplat-source "{source}"'
                proc = subprocess.run(native.cmd_command(command),
                                      cwd=self.root, env=env, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["module"], module)
        command = f'"{REPO / "engine/scripts/setup-build-splat3-render.cmd"}" --source-root "{self.root / "missing"}" --output "{self.out}" --plan'
        proc = subprocess.run(native.cmd_command(command), env=env, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.out.exists())

    @unittest.skipUnless(os.name == "nt", "Windows entrypoints")
    def test_selection_powershell_plan_is_explicit_and_unicode_safe(self):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        proc = subprocess.run([shell, "-NoProfile", "-File", str(REPO / "engine/scripts/build_studio_selection.ps1"),
                               "-PythonPath", sys.executable, "-Output", str(self.out), "-PlanOnly"],
                              cwd=self.root, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["module"], "studio-selection")
        self.assertFalse(self.out.exists())

    @unittest.skipUnless(os.name == "nt", "Windows entrypoints")
    def test_retired_runtime_entrypoint_exits_two_without_writing(self):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        proc = subprocess.run([shell, "-NoProfile", "-File", str(REPO / "engine/scripts/setup-runtime.ps1"),
                               "-DataRoot", str(self.out)], cwd=self.root,
                              capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn("bootstrap_windows.py", proc.stderr)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
