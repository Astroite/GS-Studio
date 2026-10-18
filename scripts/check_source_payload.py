"""Fail-closed Git-object guard for this source-only repository (stdlib only).

This reads the index/commit objects, never substitutes working-tree content, and
never invokes Git LFS. Run from any directory inside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

MAX_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = frozenset("""
.mp4 .mov .mkv .avi .webm .insv .lrv .m4v .wmv .flv .mpg .mpeg .mts .m2ts
.m2v .3gp .3g2 .mxf .vob .rm .rmvb .f4v .360
.ply .spz .splat .ksplat .psht .ckpt .pt .pth .onnx .safetensors .h5 .hdf5
.pkl .pickle .joblib .las .laz .e57 .pcd .glb .gltf .fbx .blend .usd .usdz
.usda .usdc .bin .npy .npz
.exe .dll .pyd .lib .a .so .dylib .o .obj .class .jar .whl .zip .7z .rar
.tar .gz .bz2 .xz .iso .msi .msix .dmg .deb .rpm .pdb
.pem .key .pfx
.jpg .jpeg .png .webp .tif .tiff .heic .heif .dng .exr .hdr .bmp .gif .ico
""".split())
REVIEWABLE_SUFFIXES = frozenset("""
.bin .npy .npz .jpg .jpeg .png .webp .tif .tiff .heic .heif .dng .exr .hdr
.bmp .gif .ico
""".split())
LOCAL_ROOTS = frozenset({
    "data", "locations", "projects", "runtime", "cache", ".cache", "private",
    "env", "sdks", "wheels", "dist", "build", "studio-projects", "models", "experiments",
})
LOCAL_COMPONENTS = frozenset({
    "node_modules", "__pycache__", ".pytest_cache", ".venv", ".venv-gsplat",
    "captures", "runs", "checkpoints",
})
RAW_PREFIXES = ("engine/evidence/", "engine/reference/evidence/", "tests/output/")
OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class GuardError(RuntimeError):
    pass


def git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "--no-replace-objects", *args], input=data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise GuardError(f"git {args[0]} failed: {detail}")
    return result.stdout


def decode_path(raw: bytes) -> str:
    # Surrogate escapes keep non-UTF8 Git paths distinct; diagnostics use repr().
    return raw.decode("utf-8", "surrogateescape")


def path_errors(path: str, *, current: bool) -> list[str]:
    normalized = path.replace("\\", "/").casefold()
    parts = PurePosixPath(normalized).parts
    errors = []
    if not parts or normalized.startswith("/") or ".." in parts:
        errors.append("unsafe repository path")
    if current and parts:
        if (parts[0] in LOCAL_ROOTS or LOCAL_COMPONENTS.intersection(parts)
                or any(part.startswith(".venv") for part in parts)
                or normalized.startswith(RAW_PREFIXES)):
            errors.append("local data/runtime/output directory is not source")
        if parts[-1] == ".env" or (parts[-1].startswith(".env.")
                                   and parts[-1] != ".env.example"):
            errors.append("local environment/secret file is not source")
    return errors


def fixture_allowlist() -> dict[str, list[str]]:
    location = Path(__file__).with_name("source-fixtures.json")
    manifest = json.loads(location.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise GuardError("invalid source-fixtures.json schema")
    entries = manifest["sha256_by_path"]
    if not isinstance(entries, dict) or any(
            not isinstance(path, str) or not isinstance(digests, list) or not digests
            or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                   for digest in digests)
            for path, digests in entries.items()):
        raise GuardError("invalid source fixture allowlist")
    return entries


def index_entries() -> set[tuple[str, str]]:
    entries = set()
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, oid, stage = header.decode("ascii").split()
        if stage != "0" or mode not in {"100644", "100755"}:
            raise GuardError(f"unmerged or unsupported index entry: {decode_path(raw_path)!r}")
        entries.add((decode_path(raw_path), oid))
    return entries


def complete_history_required() -> None:
    if git("rev-parse", "--is-shallow-repository").strip() != b"false":
        raise GuardError("complete history required; fetch with --unshallow before scanning/pushing")
    grafts = Path(decode_path(git("rev-parse", "--git-path", "info/grafts").strip()))
    if grafts.is_file() and grafts.read_bytes().strip():
        raise GuardError("Git grafts can hide ancestors; remove grafts before scanning/pushing")


def tree_entries(tree: str) -> set[tuple[str, str]]:
    entries = set()
    for record in git("ls-tree", "-r", "-z", "--full-tree", tree).split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise GuardError(f"unsupported mode/non-blob tree entry: {decode_path(raw_path)!r}")
        entries.add((decode_path(raw_path), oid))
    return entries


def history_entries(refs: list[str]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    complete_history_required()
    tips = set()
    trees = set()
    for ref in refs:
        commit = git("rev-parse", "--verify", f"{ref}^{{commit}}").strip().decode("ascii")
        if not OID.fullmatch(commit):
            raise GuardError(f"not a commit reference: {ref!r}")
        trees.add(git("rev-parse", f"{commit}^{{tree}}").strip().decode("ascii"))
        tips.update(tree_entries(commit))
        # Every commit tree is inspected. rev-list --objects alone loses aliases:
        # the same safe blob can also be stored under a forbidden media filename.
        trees.update(git("rev-list", "--format=%T", "--no-commit-header", commit).decode("ascii").splitlines())
    entries = set(tips)
    for tree in sorted(trees):
        entries.update(tree_entries(tree))
    return entries, tips


def pushed_refs(lines: str) -> list[str]:
    refs = []
    for line in lines.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise GuardError("malformed pre-push input; expected four fields per ref")
        local_ref, local_oid, remote_ref, remote_oid = fields
        if not OID.fullmatch(local_oid) or not OID.fullmatch(remote_oid):
            raise GuardError("malformed pre-push object ID")
        if set(local_oid) == {"0"}:
            continue  # Ref deletion does not upload objects, including private refs.
        if not remote_ref.startswith(("refs/heads/", "refs/tags/")):
            raise GuardError(f"source-only push disallows remote ref {remote_ref!r}")
        if not (local_ref.startswith(("refs/heads/", "refs/tags/"))
                or local_ref == "HEAD" or OID.fullmatch(local_ref)):
            raise GuardError(f"source-only push disallows local ref {local_ref!r}")
        refs.append(local_oid)
    return refs


def scan(entries: set[tuple[str, str]], current: set[tuple[str, str]]) -> list[str]:
    errors = []
    if not entries:
        return errors
    allowlist = fixture_allowlist()
    oids = sorted({oid for _, oid in entries})
    sizes = {}
    metadata = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                   data=("\n".join(oids)+"\n").encode("ascii"))
    for line in metadata.decode("ascii").splitlines():
        oid, kind, size = line.split()
        if kind != "blob":
            raise GuardError(f"expected blob, got {kind}: {oid}")
        sizes[oid] = int(size)
    if set(sizes) != set(oids):
        raise GuardError("incomplete Git object metadata")
    readable = [oid for oid in oids if sizes[oid] <= MAX_BYTES]
    raw = git("cat-file", "--batch", data=("\n".join(readable)+"\n").encode("ascii")) if readable else b""
    payloads = {}
    offset = 0
    for oid in readable:
        end = raw.find(b"\n", offset)
        header = raw[offset:end].decode("ascii").split()
        if end < 0 or header != [oid, "blob", str(sizes[oid])]:
            raise GuardError("malformed Git object response")
        start = end+1
        payloads[oid] = raw[start:start+sizes[oid]]
        offset = start+sizes[oid]+1
        if raw[offset-1:offset] != b"\n" or len(payloads[oid]) != sizes[oid]:
            raise GuardError("truncated Git blob response")
    if offset != len(raw):
        raise GuardError("unexpected trailing Git object response")
    for path, oid in sorted(entries):
        reasons = path_errors(path, current=(path, oid) in current)
        if sizes[oid] > MAX_BYTES:
            reasons.append(f"blob exceeds 5 MiB ({sizes[oid]} bytes)")
        payload = payloads.get(oid, b"")
        canonical = payload.lstrip(b"\xef\xbb\xbf \t\r\n")
        if canonical.startswith(b"version https://git-lfs.github.com/spec/v1"):
            reasons.append("Git LFS pointer is forbidden regardless of filename")
        suffix = PurePosixPath(path.replace("\\", "/")).suffix.casefold()
        allowed_fixture = (path in allowlist and payload
                           and hashlib.sha256(payload).hexdigest() in allowlist[path])
        if suffix in FORBIDDEN_SUFFIXES and not (allowed_fixture and suffix in REVIEWABLE_SUFFIXES):
            reasons.append(f"forbidden source payload extension {suffix}")
        # Byte signatures also catch a renamed executable/archive/video/model.
        if not allowed_fixture:
            binary = payload.startswith((b"MZ", b"\x7fELF", b"PK\x03\x04", b"7z\xbc\xaf\x27\x1c", b"Rar!"))
            video = len(payload) >= 8 and payload[4:8] == b"ftyp"
            ply = payload.startswith((b"ply\n", b"ply\r\n"))
            if binary or video or ply:
                reasons.append("executable/archive/video/PLY content is forbidden")
        if reasons:
            errors.append(f"{path!r} [{oid[:12]}]: "+"; ".join(reasons))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="repository directory (including a bare mirror)")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("staged", help="check the complete staged snapshot, not working files")
    history = sub.add_parser("history", help="check every ancestor and path/blob of refs")
    history.add_argument("--ref", action="append", default=[])
    history.add_argument("--all", action="store_true", help="HEAD, local/remote heads and tags")
    sub.add_parser("pre-push", help="read Git pre-push ref updates on stdin")
    args = parser.parse_args()
    try:
        if args.repo is not None:
            os.chdir(args.repo)
        git("rev-parse", "--git-dir")
        if args.mode == "staged":
            entries = index_entries()
            current = entries
        else:
            if args.mode == "pre-push":
                refs = pushed_refs(sys.stdin.buffer.read().decode("utf-8"))
            elif args.all:
                refs = ["HEAD", *git("for-each-ref", "--format=%(refname)", "refs/heads/", "refs/remotes/", "refs/tags/").decode("utf-8").splitlines()]
                refs.extend(args.ref)
            else:
                refs = args.ref or ["HEAD"]
            entries, current = history_entries(refs) if refs else (set(), set())
        errors = scan(entries, current)
        if errors:
            print("Source-only publication rejected:", file=sys.stderr)
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"Source-only guard passed: {len(entries)} path/blob combinations ({args.mode}).")
        return 0
    except (GuardError, OSError, ValueError, KeyError, TypeError, UnicodeError) as exc:
        print(f"Source-only guard failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
