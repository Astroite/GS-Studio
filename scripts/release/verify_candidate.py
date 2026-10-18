"""Read-only directory-package integrity checks; no runtime, GPU or legal gate.

The package inventory is an integrity manifest, not a trusted signature. Schema
3 package inputs bind lock/policy/editor hashes but do not bind a source commit
or prove that executable/native binaries were built from the compared sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys


HASH = re.compile(r"[a-fA-F0-9]{64}\Z")
RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)
SOURCE_SUFFIXES = {".py", ".cpp", ".cu", ".h", ".hpp", ".cuh"}


def plain(path):
    path = Path(os.path.abspath(path))
    for node in (path, *path.parents):
        if node.is_symlink() or (node.exists() and getattr(node.lstat(), "st_file_attributes", 0)
                                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"Symlink/reparse point rejected: {node}")
    return path


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\0" in value:
        raise ValueError(f"Invalid relative package path: {value!r}")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"Absolute package path rejected: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or part[-1:] in {".", " "} or RESERVED.fullmatch(part)
           or any(char in part for char in '<>"|?*\r\n') for part in parts):
        raise ValueError(f"Unsafe package path: {value!r}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def package_files(root):
    result, folded = {}, set()
    for parent, directories, names in os.walk(root, followlinks=False):
        for name in directories + names:
            plain(Path(parent) / name)
        for name in names:
            path = Path(parent) / name
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError(f"Non-regular package file rejected: {path}")
            relative = safe_relative(path.relative_to(root).as_posix())
            if relative.casefold() in folded:
                raise ValueError(f"Case-aliased package path: {relative}")
            folded.add(relative.casefold())
            result[relative] = path
    return result


def checked_hash(value, label):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"Invalid SHA256 for {label}")
    return value.lower()


def verify_inventory(root, actual):
    inventory_path = root / "package-inventory.json"
    inventory = read_json(inventory_path)
    if inventory.get("schema_version") != 2 or inventory.get("product") != "GS Studio":
        raise ValueError("Expected GS Studio inventory schema_version 2")
    records, folded = {}, set()
    if not isinstance(inventory.get("files"), list):
        raise ValueError("Inventory files must be a list")
    for record in inventory["files"]:
        relative = safe_relative(record.get("path"))
        if relative == "package-inventory.json" or relative.casefold() in folded:
            raise ValueError(f"Duplicate/self inventory entry: {relative}")
        folded.add(relative.casefold())
        size = record.get("bytes")
        if type(size) is not int or size < 0:
            raise ValueError(f"Invalid byte count: {relative}")
        records[relative] = {"bytes": size, "sha256": checked_hash(record.get("sha256"), relative)}
    unexpected = sorted(set(actual) - set(records) - {"package-inventory.json"})
    missing = sorted(set(records) - set(actual))
    if unexpected or missing:
        raise ValueError(f"Inventory membership mismatch; unlisted={unexpected[:20]}, missing={missing[:20]}")
    for relative, record in records.items():
        path = actual[relative]
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"File size mismatch: {relative}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"File SHA256 mismatch: {relative}")
    return inventory, records, {
        "status": "passed", "payload_files": len(records),
        "payload_bytes": sum(record["bytes"] for record in records.values()),
        "inventory_sha256": sha256(inventory_path),
        "inventory_bytes": inventory_path.stat().st_size,
        "all_package_files": len(actual), "manifest_is_not_a_signature": True,
    }


def bind_record(record, records, hash_key="sha256"):
    relative = safe_relative(record.get("path"))
    expected = checked_hash(record.get(hash_key), relative)
    if relative not in records or records[relative]["sha256"] != expected:
        raise ValueError(f"Package input/provenance hash mismatch: {relative}")
    return relative


def verify_inputs(root, inventory, records):
    inputs = read_json(root / "package-inputs.json")
    if inputs.get("schema_version") != 3:
        raise ValueError("Expected package-inputs schema_version 3")
    for field in ("profile", "product", "version"):
        if inputs.get(field) != inventory.get(field):
            raise ValueError(f"Input/inventory {field} mismatch")
    if inputs.get("profile") not in {"public", "local"}:
        raise ValueError("Unknown package profile")
    for key in ("python_lock", "prune_policy"):
        bind_record(inputs[key], records)
    components = inputs.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Expected nonempty component provenance")
    component_paths, optional_omissions = set(), []
    for component in components:
        relative = safe_relative(component.get("path"))
        if relative.casefold() in component_paths:
            raise ValueError(f"Duplicate component path: {relative}")
        component_paths.add(relative.casefold())
        if not (root / relative).exists():
            if component.get("component") == "commercial-sdk" and inputs["profile"] == "public":
                optional_omissions.append(relative)
            else:
                raise ValueError(f"Missing declared package component: {relative}")
    selection = inputs["editor_selection"]
    key = selection.get("key")
    if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{20}", key):
        raise ValueError("Invalid editor selection key")
    binary = safe_relative(selection.get("binary"))
    expected_parent = f"engine/extensions/studio-selection/{key}/"
    if not binary.startswith(expected_parent) or not PurePosixPath(binary).name.startswith("studio_selection_cuda") or not binary.endswith(".pyd"):
        raise ValueError("Editor binary/key layout mismatch")
    bind_record({"path": binary, "sha256": selection.get("binary_sha256")}, records)
    proof_path = expected_parent + "build-proof.json"
    bind_record({"path": proof_path, "sha256": selection.get("source_proof_sha256")}, records)
    proof = read_json(root / proof_path)
    if proof.get("architecture") != selection.get("architecture") or proof.get("cuda_initialized") is not False:
        raise ValueError("Editor build proof architecture/offline assertion mismatch")
    sources = selection.get("sources")
    expected_sources = {"engine/editor_native/selection.cpp", "engine/editor_native/selection.cu"}
    if not isinstance(sources, list) or {source.get("path") for source in sources} != expected_sources or len(sources) != 2:
        raise ValueError("Editor proof must bind exactly selection.cpp and selection.cu")
    for source in sources:
        bind_record(source, records)
        if "source_sha256" in proof:
            if proof["source_sha256"].get(PurePosixPath(source["path"]).name) != source["sha256"]:
                raise ValueError("Editor proof source hash mismatch")
    exclusions = read_json(root / "package-exclusions.json")
    policy_hash = inputs["prune_policy"]["sha256"]
    if exclusions.get("schema_version") != 1 or exclusions.get("policy_sha256") != policy_hash:
        raise ValueError("Exclusion/prune-policy provenance mismatch")
    excluded_bytes, excluded_paths = 0, set()
    included_paths = {path.casefold() for path in records}
    if not isinstance(exclusions.get("files"), list):
        raise ValueError("Exclusions files must be a list")
    for excluded in exclusions["files"]:
        relative = safe_relative(excluded.get("path"))
        if relative.casefold() in excluded_paths or relative.casefold() in included_paths:
            raise ValueError(f"Duplicate or included excluded path: {relative}")
        excluded_paths.add(relative.casefold())
        if type(excluded.get("bytes")) is not int or excluded["bytes"] < 0:
            raise ValueError(f"Invalid exclusion byte count: {relative}")
        checked_hash(excluded.get("sha256"), relative)
        if not excluded.get("reason") or not excluded.get("rule"):
            raise ValueError(f"Missing exclusion reason/rule: {relative}")
        excluded_bytes += excluded["bytes"]
    if exclusions.get("excluded_bytes") != excluded_bytes:
        raise ValueError("Exclusion byte total mismatch")
    return {
        "status": "passed", "profile": inputs["profile"], "version": inputs["version"],
        "bindings": ["python_lock", "prune_policy", "editor_binary", "editor_build_proof", "editor_sources", "exclusion_policy"],
        "public_optional_omissions": optional_omissions,
        "excluded_files_recorded": len(excluded_paths), "excluded_bytes_recorded": excluded_bytes,
        "exclusions_are_recorded_not_independently_reconstructed": True,
    }


def source_snapshot(root, source_root, records):
    if source_root is None:
        return {"status": "not-requested", "commit_bound": False}
    source_root = plain(source_root)
    if not source_root.is_dir():
        raise ValueError("Source root must be an existing directory")
    compared, differences = [], []
    for relative in sorted(records):
        if Path(relative).suffix not in SOURCE_SUFFIXES:
            continue
        if relative.startswith("engine/local_postshot/") or relative.startswith("engine/editor_native/"):
            source_relative = relative
        elif relative.startswith("pipeline/gsstudio/"):
            source_relative = "src/" + relative.removeprefix("pipeline/")
        else:
            continue
        source = plain(source_root / source_relative)
        expected = records[relative]["sha256"]
        actual = sha256(source) if source.is_file() else None
        compared.append({"package_path": relative, "source_path": source_relative,
                         "package_sha256": expected, "source_sha256": actual})
        if expected != actual:
            differences.append(relative)
    if not compared:
        raise ValueError("No packaged application/editor source matched the supplied source root")
    if differences:
        raise ValueError(f"Packaged/current source snapshot differs: {differences[:20]}")
    digest = hashlib.sha256(json.dumps(compared, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"status": "matched-source-files-only", "commit_bound": False,
            "compared_files": len(compared), "snapshot_sha256": digest, "files": compared,
            "scope": "Packaged Python/editor source files only; excludes metadata text, frontend EXE and compiled native build origin"}


def verify(package, source_root=None, expected_source_commit=None):
    report = {
        "schema_version": 1, "status": "failed", "scope": "static-directory-package-integrity",
        "runtime_executed": False, "gpu_acceptance": "not-performed", "legal_acceptance": "not-performed",
        "full_artifact_commit_binding": "not-established", "errors": [],
    }
    try:
        root = plain(package)
        if not root.is_dir():
            raise ValueError("Package must be an existing directory")
        report["package"] = str(root)
        actual = package_files(root)
        inventory, records, summary = verify_inventory(root, actual)
        report["inventory"] = summary
        report["inputs_and_provenance"] = verify_inputs(root, inventory, records)
        report["source_snapshot"] = source_snapshot(root, source_root, records)
        if expected_source_commit is not None:
            report["expected_source_commit"] = expected_source_commit
            raise ValueError("Expected source commit cannot be established: package-inputs schema 3 has no source-commit/build-origin binding; use independent build receipts")
        report["status"] = "passed"
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        report["errors"].append(str(exc))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path, help="New output file outside the package")
    parser.add_argument("--source-root", type=Path, help="Optional current source snapshot to compare (no Git/network access)")
    parser.add_argument("--expected-source-commit", help="Strict gate: schema 3 cannot establish a commit binding and will fail")
    args = parser.parse_args(argv)
    try:
        package, output = plain(args.package), plain(args.report)
        if not args.package.is_absolute() or not args.report.is_absolute():
            raise ValueError("Package/report paths must be absolute")
        if output == package or output.is_relative_to(package):
            raise ValueError("Report must be outside the read-only package")
        if output.exists():
            raise ValueError("Report already exists; choose a new file")
        if not output.parent.is_dir():
            raise ValueError("Report parent must already exist")
        result = verify(package, args.source_root, args.expected_source_commit)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "report": str(output), "errors": result["errors"]}, ensure_ascii=False))
        return 0 if result["status"] == "passed" else 1
    except (ValueError, OSError) as exc:
        print(f"Candidate verification refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
