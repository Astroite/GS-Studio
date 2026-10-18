"""SPZ dependency colocation and wheel RECORD integrity, no native execution."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_spz_packaging", ROOT / "engine/scripts/build_native.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class SpzWheelPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="GS Studio SPZ 包 ")
        self.root = Path(self.temp.name)
        (self.root / "wheels").mkdir()
        self.wheel = self.root / "wheels/spz-1.1.0-cp310-cp310-win_amd64.whl"
        (self.root / "plan.json").write_text(json.dumps({"module": "spz"}))
        self.dependency = self.root / "build/_deps/zlib-build/z.dll"
        self.dependency.parent.mkdir(parents=True)
        self.dll = b"synthetic newly built zlib DLL fixture"
        self.dependency.write_bytes(self.dll)
        self.payload = {
            "bin/z.dll": self.dll, "spz/__init__.py": b"# synthetic wrapper\n",
            "spz/spz.cp310-win_amd64.pyd": b"synthetic extension fixture",
            "spz-1.1.0.dist-info/METADATA": b"Name: spz\nVersion: 1.1.0\n",
            "spz-1.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: cp310-cp310-win_amd64\n",
            "spz-1.1.0.dist-info/licenses/LICENSE": b"SPZ license fixture\n",
            "share/doc/spz/zlib/LICENSE": b"zlib license fixture\n",
            "include/zlib.h": b"// original header payload fixture\n",
        }
        self.record = "spz-1.1.0.dist-info/RECORD"
        self.write_wheel()

    def tearDown(self):
        self.temp.cleanup()

    def write_wheel(self):
        rows = io.StringIO(newline="")
        writer = csv.writer(rows, lineterminator="\n")
        for name, value in self.payload.items():
            encoded = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + encoded, str(len(value))))
        writer.writerow((self.record, "", ""))
        with zipfile.ZipFile(self.wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.comment = b"preserve wheel comment fixture"
            for name, value in self.payload.items():
                archive.writestr(name, value)
            archive.writestr(self.record, rows.getvalue())

    def test_colocates_own_build_dependency_preserves_payload_and_validates_every_record(self):
        original = self.wheel.read_bytes()
        proof = native.colocate_spz_zlib(self.wheel, self.root)
        self.assertEqual(proof["dependency_source"], "build/_deps/zlib-build/z.dll")
        self.assertEqual(proof["dependency_sha256"], hashlib.sha256(self.dll).hexdigest())
        self.assertEqual(proof["removed_redundant_entries"], ["bin/z.dll"])
        self.assertEqual((self.root / proof["original_wheel"]).read_bytes(), original)
        native.validate_wheel(self.wheel, {"distribution": "spz", "version": "1.1.0"})
        with zipfile.ZipFile(self.wheel) as archive:
            self.assertNotIn("bin/z.dll", archive.namelist())
            self.assertEqual(archive.read("spz/z.dll"), self.dll)
            self.assertEqual(archive.comment, b"preserve wheel comment fixture")
            for name, value in self.payload.items():
                if name != "bin/z.dll":
                    self.assertEqual(archive.read(name), value)
            rows = list(csv.reader(io.StringIO(archive.read(self.record).decode())))
            self.assertEqual({row[0] for row in rows}, set(archive.namelist()))
            for name, encoded, size in rows:
                if name == self.record:
                    self.assertEqual((encoded, size), ("", ""))
                else:
                    value = archive.read(name)
                    expected = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
                    self.assertEqual(encoded, "sha256=" + expected)
                    self.assertEqual(int(size), len(value))

    def test_missing_new_build_dependency_fails_without_using_wheel_or_global_dll(self):
        original = self.wheel.read_bytes()
        self.dependency.unlink()
        with self.assertRaisesRegex(ValueError, "missing its own"):
            native.colocate_spz_zlib(self.wheel, self.root)
        self.assertEqual(self.wheel.read_bytes(), original)

    def test_mismatched_existing_dll_is_not_silently_overwritten(self):
        self.payload["bin/z.dll"] = b"different DLL fixture"
        self.write_wheel()
        original = self.wheel.read_bytes()
        with self.assertRaisesRegex(ValueError, "differs from this job"):
            native.colocate_spz_zlib(self.wheel, self.root)
        self.assertEqual(self.wheel.read_bytes(), original)

    def test_dependency_can_be_added_when_original_wheel_omitted_it(self):
        del self.payload["bin/z.dll"]
        self.write_wheel()
        proof = native.colocate_spz_zlib(self.wheel, self.root)
        self.assertEqual(proof["removed_redundant_entries"], [])
        with zipfile.ZipFile(self.wheel) as archive:
            self.assertEqual(archive.read("spz/z.dll"), self.dll)

    def test_transform_refuses_artifact_outside_owned_job(self):
        with self.assertRaisesRegex(ValueError, "owned job/wheels"):
            native.colocate_spz_zlib(self.root / "outside.whl", self.root)
        (self.root / "plan.json").write_text(json.dumps({"module": "other"}))
        with self.assertRaisesRegex(ValueError, "matching job plan"):
            native.colocate_spz_zlib(self.wheel, self.root)


if __name__ == "__main__":
    unittest.main()
