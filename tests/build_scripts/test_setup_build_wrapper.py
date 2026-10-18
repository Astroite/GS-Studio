"""Long staged path normalization using an executed setup fixture, no compiler."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "engine/scripts/run_setup_build.py"
SPEC = importlib.util.spec_from_file_location("source_relative_setup", SCRIPT)
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


class SetupBuildWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="GS Studio 编译 path ")
        self.parent = Path(self.temp.name)
        self.root = self.parent / ("workspace " + "x" * 40) / ("中文 long build " + "y" * 30) / "sources/module"
        self.root.mkdir(parents=True)
        for name in ("bindings.cpp", "stats.cu", "nested/extra.cpp"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("// synthetic source\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_execute_setup_fixture_keeps_metadata_and_flags_and_bounds_object_names(self):
        fixture = '''from pathlib import Path
from types import SimpleNamespace
from setuptools import setup
ROOT = Path(__file__).resolve().parent
setup(name="local_splat3_stats", version="0.1.1", ext_modules=[SimpleNamespace(
    sources=[str(ROOT/"bindings.cpp"), str(ROOT/"stats.cu"), "nested/extra.cpp"],
    include_dirs=[str(ROOT), "include/unchanged"],
    extra_compile_args={"cxx": ["/O2"], "nvcc": ["-O3"]})])
'''
        (self.root / "setup.py").write_text(fixture, encoding="utf-8")
        captured = []
        setup = lambda *args, **kwargs: captured.append(kwargs)
        fake_setuptools = SimpleNamespace(setup=setup)
        original_cwd, original_argv = Path.cwd(), sys.argv
        with patch.dict(sys.modules, {"setuptools": fake_setuptools}):
            result = wrapper.main(["--source-root", str(self.root), "--", "build", "--build-base", "external-build"])
        self.assertEqual(result, 0)
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertIs(sys.argv, original_argv)
        self.assertIs(fake_setuptools.setup, setup)
        self.assertEqual(len(captured), 1)
        metadata = captured[0]
        self.assertEqual((metadata["name"], metadata["version"]), ("local_splat3_stats", "0.1.1"))
        extension = metadata["ext_modules"][0]
        self.assertEqual(extension.sources, ["bindings.cpp", "stats.cu", "nested/extra.cpp"])
        self.assertEqual(extension.include_dirs, [str(self.root), "include/unchanged"])
        self.assertEqual(extension.extra_compile_args, {"cxx": ["/O2"], "nvcc": ["-O3"]})
        self.assertLess(max(map(len, extension.sources)), 30)
        self.assertFalse(any(Path(source).is_absolute() for source in extension.sources))

    def test_absolute_or_relative_source_outside_stage_is_rejected(self):
        outside = self.parent / "outside.cu"
        outside.write_text("// outside fixture\n")
        for source in (str(outside), os.path.relpath(outside, self.root)):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "escapes staged"):
                wrapper.normalize_extensions([SimpleNamespace(sources=[source])], self.root)

    def test_missing_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source missing"):
            wrapper.normalize_extensions([SimpleNamespace(sources=["missing.cu"])], self.root)

    def test_setup_error_restores_process_state_and_original_function(self):
        (self.root / "setup.py").write_text('raise RuntimeError("fixture failure")\n')
        setup = lambda **kwargs: None
        fake_setuptools = SimpleNamespace(setup=setup)
        original_cwd, original_argv = Path.cwd(), sys.argv
        with patch.dict(sys.modules, {"setuptools": fake_setuptools}):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                wrapper.main(["--source-root", str(self.root), "--", "build"])
        self.assertEqual(Path.cwd(), original_cwd)
        self.assertIs(sys.argv, original_argv)
        self.assertIs(fake_setuptools.setup, setup)

    def test_source_symlink_is_rejected(self):
        link = self.root / "linked.cu"
        try:
            link.symlink_to(self.root / "stats.cu")
        except OSError:
            self.skipTest("symlink creation unavailable")
        with self.assertRaisesRegex(ValueError, "symlink/reparse"):
            wrapper.normalize_extensions([SimpleNamespace(sources=["linked.cu"])], self.root)

    @unittest.skipUnless(os.name == "nt", "Windows 8.3 path aliases")
    def test_short_staged_root_accepts_canonical_long_source_paths(self):
        import ctypes
        from ctypes import wintypes
        get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
        get_short_path.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
        get_short_path.restype = wintypes.DWORD
        canonical_root = self.root.resolve()
        size = get_short_path(str(canonical_root), None, 0)
        if not size:
            self.skipTest("GetShortPathNameW unavailable for this fixture volume")
        buffer = ctypes.create_unicode_buffer(size)
        if not get_short_path(str(canonical_root), buffer, size):
            self.skipTest("GetShortPathNameW did not return a fixture alias")
        short_root = Path(buffer.value)
        if str(short_root).casefold() == str(canonical_root).casefold():
            self.skipTest("8.3 alias generation is disabled on this fixture volume")
        extension = SimpleNamespace(sources=[str(canonical_root / "bindings.cpp"),
                                             str(canonical_root / "stats.cu")])
        wrapper.normalize_extensions([extension], short_root)
        self.assertEqual(extension.sources, ["bindings.cpp", "stats.cu"])
        self.assertEqual(wrapper.plain(short_root), canonical_root)


if __name__ == "__main__":
    unittest.main()
