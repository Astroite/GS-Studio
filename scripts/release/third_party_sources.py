"""Fetch, retain and verify the public release's third-party evidence (stdlib only).

Examples, from the repository root:
  python scripts/release/third_party_sources.py fetch --source-dir distribution-sources
  python scripts/release/third_party_sources.py materialize --source-dir distribution-sources --package-root package
  python scripts/release/third_party_sources.py verify --source-dir distribution-sources --evidence-only
  python scripts/release/third_party_sources.py verify --source-dir distribution-sources --package-root package --report release-evidence.json
  python scripts/release/third_party_sources.py bundle --source-dir distribution-sources --output-dir source-bundle
  python scripts/release/third_party_sources.py self-test

The default verify mode is a release gate. An intact evidence cache does not
clear documented licensing/source gaps. --evidence-only checks evidence integrity
and reports those gaps without representing the application as release ready.
No package binaries are executed, imported, changed or deleted. Reports contain
relative artifact names, never the caller's local paths or environment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import email.parser
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import unittest
import urllib.parse
import urllib.request


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2]/'docs/distribution/third-party-sources.json'
HASH = re.compile(r'[0-9a-f]{64}\Z')
PRIVATE_PATH = re.compile(r'(?i)(?:\b[A-Z]:[\\/]|file://|\\\\[^\\]+\\|/(?:Users|home)/)')
NOTICE_NAME = re.compile(r'(?i)^(?:licen[cs]e|copying|notice|copyright|authors)(?:[._-].*)?$')


def sha256(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def relative_name(name):
    """Validate a portable name without letting Windows reinterpret a drive/ADS."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Artifact names must be portable relative paths')
    parts = PurePosixPath(name).parts
    if name.startswith('/') or any(part in ('..', '.') for part in parts) or name != PurePosixPath(name).as_posix():
        raise ValueError('Artifact path traversal or noncanonical path')
    return name


def inside(root, name):
    target = root/relative_name(name)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes the selected root')
    return target


def public_strings(value):
    if isinstance(value, dict):
        for item in value.values():
            public_strings(item)
    elif isinstance(value, list):
        for item in value:
            public_strings(item)
    elif isinstance(value, str):
        if PRIVATE_PATH.search(value):
            raise ValueError('Public manifest/report contains a local path')
        if value.startswith(('https://', 'http://')):
            parsed = urllib.parse.urlsplit(value)
            if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.query:
                raise ValueError('Public URLs must use HTTPS without credentials or query tokens')


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    public_strings(manifest)
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported third-party manifest schema')
    ids, names = set(), set()
    for item in manifest['artifacts']:
        if item['id'] in ids:
            raise ValueError('Duplicate artifact identifier')
        ids.add(item['id'])
        relative_name(item['file'])
        if item['file'] in names or not HASH.fullmatch(item['sha256']) or item['bytes'] < 0:
            raise ValueError('Duplicate artifact path or invalid artifact fingerprint')
        names.add(item['file'])
        if item['retrieval']['kind'] == 'url':
            parsed = urllib.parse.urlsplit(item['retrieval']['url'])
            if parsed.hostname not in manifest['allowed_download_hosts']:
                raise ValueError('Source URL host is not approved in the manifest')
        elif item['retrieval']['kind'] == 'archive-member':
            relative_name(item['retrieval']['member'])
        elif item['retrieval']['kind'] == 'package-resource':
            relative_name(item['retrieval']['package_path'])
        elif item['retrieval']['kind'] != 'observed-snapshot':
            raise ValueError('Unknown evidence retrieval method')
    for item in manifest['artifacts']:
        if item['retrieval']['kind'] == 'archive-member' and item['retrieval']['archive_id'] not in ids:
            raise ValueError('Notice references an unknown source archive')
    for binary in manifest['binary_fingerprints']:
        relative_name(binary['package_path'])
        if not HASH.fullmatch(binary['sha256']):
            raise ValueError('Invalid binary fingerprint')
    for dependency in manifest['python_distributions']:
        for resource in dependency['notices']:
            if resource not in ids:
                raise ValueError('Python distribution references an unknown notice')
    for gap in manifest['release_gaps']:
        if gap['status'] == 'resolved':
            resolution = gap.get('resolution', {})
            references = resolution.get('evidence_artifact_ids', [])
            if not references or any(reference not in ids for reference in references) or not resolution.get('public_summary'):
                raise ValueError('Resolved release gaps require hashed evidence references and a public summary')
    return manifest


def emit(report, output=None):
    public_strings(report)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)+'\n'
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding='utf-8', newline='\n')
    print(encoded, end='')


def matches(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and sha256(path) == item['sha256']


def fetch_one(item, root, hosts):
    target = inside(root, item['file'])
    if matches(target, item):
        return dict(id=item['id'], status='verified-cache')
    if target.exists():
        return dict(id=item['id'], status='error', code='existing-artifact-hash-mismatch')
    if item['retrieval']['kind'] != 'url':
        return dict(id=item['id'], status='deferred', code=item['retrieval']['kind'])
    target.parent.mkdir(parents=True, exist_ok=True)
    part = None
    try:
        request = urllib.request.Request(item['retrieval']['url'], headers={'User-Agent':'GSStudio-release-sources/0.3.0-rc.1'})
        with urllib.request.urlopen(request, timeout=40) as response:
            final = urllib.parse.urlsplit(response.url)
            if final.scheme != 'https' or final.hostname not in hosts or final.username or final.password or final.query:
                raise ValueError('Unexpected source redirect')
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.source-', suffix='.part', delete=False) as stream:
                part = Path(stream.name)
                size = 0
                for block in iter(lambda: response.read(1024*1024), b''):
                    size += len(block)
                    if size > item['bytes']:
                        raise ValueError('Source exceeds pinned byte count')
                    stream.write(block)
        if not matches(part, item):
            raise ValueError('Source does not match the pinned SHA256/size')
        part.rename(target)
        return dict(id=item['id'], status='downloaded-and-verified')
    except Exception as error:
        # Do not serialize exception text: urllib/filesystem errors can expose
        # local paths, signed redirects, proxy credentials or private settings.
        return dict(id=item['id'], status='error', code='fetch-failed', error_type=type(error).__name__)
    finally:
        if part and part.is_file():
            part.unlink()  # Only this command's uniquely created partial file.


def materialize(manifest, root, package=None):
    """Restore notice bytes from pinned archives or the caller's package; read only."""
    by_id = {item['id']: item for item in manifest['artifacts']}
    results = []
    for item in manifest['artifacts']:
        recipe = item['retrieval']
        if recipe['kind'] not in ('archive-member', 'package-resource'):
            continue
        target = inside(root, item['file'])
        if matches(target, item):
            results.append(dict(id=item['id'], status='verified-cache'))
            continue
        if target.exists():
            results.append(dict(id=item['id'], status='error', code='existing-artifact-hash-mismatch'))
            continue
        try:
            if recipe['kind'] == 'archive-member':
                archive_item = by_id[recipe['archive_id']]
                archive = inside(root, archive_item['file'])
                if not matches(archive, archive_item):
                    raise ValueError('Source archive hash mismatch')
                with tarfile.open(archive) as source:
                    member = source.getmember(relative_name(recipe['member']))
                    if not member.isfile() or member.size > 8*1024*1024:
                        raise ValueError('Notice must be a bounded regular archive member')
                    content = source.extractfile(member).read()
            elif package:
                source = inside(package, recipe['package_path'])
                if not matches(source, item):
                    raise ValueError('Package resource hash mismatch')
                content = source.read_bytes()
            else:
                results.append(dict(id=item['id'], status='deferred', code='package-resource'))
                continue
            if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('Retained resource hash mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation protects existing evidence, including partial runs.
            with target.open('xb') as stream:
                stream.write(content)
            results.append(dict(id=item['id'], status='materialized-and-verified'))
        except Exception as error:
            results.append(dict(id=item['id'], status='error', code='materialize-failed', error_type=type(error).__name__))
    return results


def canonical_distribution(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def package_inventory(package):
    """Observe real installed metadata; no imports or binary probing."""
    distributions = []
    site = package/'runtime/python/Lib/site-packages'
    for directory in sorted(site.glob('*.dist-info')):
        metadata = directory/'METADATA'
        if not metadata.is_file():
            distributions.append(dict(name=directory.name, version=None, notices=[], error='missing-METADATA'))
            continue
        meta = email.parser.Parser().parsestr(metadata.read_text(encoding='utf-8'))
        notices = [dict(path=path.relative_to(package).as_posix(), sha256=sha256(path), bytes=path.stat().st_size)
                   for path in sorted(directory.rglob('*')) if path.is_file() and NOTICE_NAME.match(path.name)]
        distributions.append(dict(name=meta.get('Name', directory.name), version=meta.get('Version'), notices=notices))
    return distributions


def inspect_package(manifest, package):
    issues, verified = [], []
    observed = package_inventory(package)
    expected = {canonical_distribution(item['name']): item for item in manifest['python_distributions']}
    present = {canonical_distribution(item['name']): item for item in observed}
    for name in manifest['required_python_distributions']:
        if canonical_distribution(name) not in present:
            issues.append(dict(code='missing-required-python-distribution', component=name))
    for name, item in present.items():
        known = expected.get(name)
        if known is None:
            issues.append(dict(code='unreviewed-python-distribution', component=item['name']))
        elif item['version'] != known['version']:
            issues.append(dict(code='python-version-mismatch', component=item['name'], expected=known['version'], actual=item['version']))
        elif not known['notices'] and name not in manifest['self_code_distributions']:
            issues.append(dict(code='python-distribution-license-resource-gap', component=item['name']))
    for item in manifest['binary_fingerprints']:
        path = inside(package, item['package_path'])
        if not path.is_file():
            if item.get('required', False):
                issues.append(dict(code='required-binary-missing', path=item['package_path']))
        elif matches(path, item):
            verified.append(item['package_path'])
        else:
            issues.append(dict(code='binary-hash-mismatch', path=item['package_path']))
    # Retained notices can appear with an existing packager's numeric prefix.
    # Match exact content; matching a filename alone never satisfies a notice.
    actual_notice_hashes = set()
    for folder in ('notices', 'tools/ffmpeg', 'tools/colmap'):
        base = package/folder
        if not base.is_dir():
            continue
        for path in base.rglob('*'):
            if path.is_file() and path.suffix.lower() not in ('.dll', '.exe', '.pyd') and path.stat().st_size <= 8*1024*1024:
                actual_notice_hashes.add(sha256(path))
    for item in manifest['artifacts']:
        if not item.get('package_notice', False):
            continue
        condition = item.get('when_distribution')
        if condition and canonical_distribution(condition) not in present:
            continue
        direct = item['retrieval'].get('package_path')
        if direct and matches(inside(package, direct), item):
            continue
        if item['sha256'] not in actual_notice_hashes:
            issues.append(dict(code='retained-notice-missing-from-package', artifact=item['id']))
    approved_models = {item['package_path'] for item in manifest['binary_fingerprints'] if item.get('role') == 'pretrained-weight'}
    unknown_models, unclassified_nvidia = [], []
    known_nvidia = {item['package_path'] for item in manifest['binary_fingerprints'] if item.get('component') in ('nvidia-cuda', 'nvidia-cudnn')}
    for path in package.rglob('*'):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            issues.append(dict(code='public-package-reparse-point', path=path.relative_to(package).as_posix()))
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        for rule in manifest['package_exclusions']:
            if any(fnmatch.fnmatchcase(relative.lower(), pattern.lower()) for pattern in rule['patterns']):
                issues.append(dict(code=rule['id'], path=relative))
        if path.suffix.lower() in ('.pth', '.pt', '.onnx', '.safetensors') and '/site-packages/' not in '/'+relative and relative not in approved_models:
            unknown_models.append(relative)
        if path.suffix.lower() == '.dll' and re.match(r'(?i)^(?:cublas|cudart|cudnn|cufft|cupti|curand|cusolver|cusparse|nvrtc|nvjitlink)', path.name) and relative not in known_nvidia:
            unclassified_nvidia.append(relative)
    issues += [dict(code='unreviewed-pretrained-weight', path=name) for name in unknown_models]
    issues += [dict(code='unclassified-nvidia-runtime', path=name) for name in unclassified_nvidia]
    return dict(distributions=observed, verified_binaries=verified, issues=issues)


def verify(manifest, source_dir, package=None, evidence_only=False):
    artifact_results, issues = [], []
    for item in manifest['artifacts']:
        valid = matches(inside(source_dir, item['file']), item)
        artifact_results.append(dict(id=item['id'], file=item['file'], verified=valid))
        if not valid and item.get('required_local', True):
            issues.append(dict(code='missing-or-corrupt-source-evidence', artifact=item['id']))
    source_integrity = not issues
    package_result = inspect_package(manifest, package) if package else None
    if package_result:
        issues += package_result['issues']
    blockers = [dict(id=gap['id'], component=gap['component'], detail=gap['detail'], alternatives=gap['alternatives'])
                for gap in manifest['release_gaps'] if gap['status'] != 'resolved']
    if not package:
        blockers.append(dict(id='final-public-package-not-inspected', component='release',
                             detail='The final public package must be supplied to verify; no binary assets were available when this manifest was assembled.'))
    ready = not issues and not blockers and package is not None
    result = dict(schema_version=1, release=manifest['release'], evidence_integrity=not issues,
                  source_evidence_integrity=source_integrity,
                  package_integrity=None if package_result is None else not package_result['issues'],
                  release_ready=ready, scope='evidence-only' if evidence_only else 'release',
                  artifacts=artifact_results, package=package_result, issues=issues, release_blockers=blockers)
    return result, 0 if (not issues if evidence_only else ready) else 1


def bundle(manifest, manifest_path, source_dir, output):
    """Export only manifest-listed public evidence, excluding local working notes."""
    report, code = verify(manifest, source_dir, evidence_only=True)
    if code:
        raise ValueError('Cannot bundle corrupt or incomplete evidence')
    if output.resolve().is_relative_to(source_dir.resolve()) or source_dir.resolve().is_relative_to(output.resolve()):
        raise ValueError('Source and bundle directories must not overlap')
    output.mkdir(parents=True, exist_ok=False)
    for item in manifest['artifacts']:
        target = inside(output, item['file'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(source_dir, item['file']), target)
        if not matches(target, item):
            raise ValueError('Source changed while exporting the bundle')
    shutil.copyfile(manifest_path, output/'third-party-sources.json')
    shutil.copyfile(Path(__file__), output/'third_party_sources.py')
    (output/'evidence-verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (output/'SOURCE-BUNDLE-README.txt').write_text(
        'Third-party source/notice evidence for '+manifest['release']+'\n\n'
        'Verify offline with:\n'
        'python third_party_sources.py verify --manifest third-party-sources.json --source-dir . --evidence-only\n\n'
        'This bundle preserves the recorded sources and notices. Open release gaps remain\n'
        'in the manifest; an intact bundle is not approval to publish application binaries.\n'
        'Component licenses and model/runtime terms remain separate from self-authored code.\n', encoding='utf-8', newline='\n')
    files = sorted(path for path in output.rglob('*') if path.is_file())
    sums = ''.join(sha256(path)+'  '+path.relative_to(output).as_posix()+'\n' for path in files)
    (output/'SHA256SUMS').write_text(sums, encoding='utf-8', newline='\n')
    return dict(schema_version=1, release=manifest['release'], exported_files=len(files)+1,
                evidence_integrity=True, release_ready=False,
                note='Only manifest-listed resources and the public verifier/report were exported; review open release gaps.')


def self_test():
    class Contract(unittest.TestCase):
        def test_rejects_paths_and_private_metadata(self):
            for name in ('../secret', '/absolute', 'a/../b', 'a//b', 'a\\b', 'C:/secret', 'file:stream'):
                with self.assertRaises(ValueError):
                    relative_name(name)
            for value in ('C:/Users/name', 'file:///private', 'https://example.org/a?token=secret', 'https://user:pass@example.org/a'):
                with self.assertRaises(ValueError):
                    public_strings(value)

        def test_hash_checks_actual_bytes(self):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                path = root/'source.tar'
                path.write_bytes(b'known original bytes')
                item = dict(bytes=len(b'known original bytes'), sha256=hashlib.sha256(b'known original bytes').hexdigest())
                self.assertTrue(matches(path, item))
                path.write_bytes(b'known modified bytes')
                self.assertFalse(matches(path, item))

        def test_evidence_integrity_does_not_clear_release_gap(self):
            with tempfile.TemporaryDirectory() as name:
                manifest = dict(release='test', artifacts=[], release_gaps=[dict(id='weight-grant', component='weights', status='open', detail='Missing model redistribution grant', alternatives=['defer binary'])])
                report, code = verify(manifest, Path(name), evidence_only=True)
                self.assertEqual(code, 0)
                self.assertFalse(report['release_ready'])
                self.assertEqual(verify(manifest, Path(name))[1], 1)

        def test_materialize_rejects_bad_notice_and_preserves_existing(self):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                (root/'notice').write_bytes(b'old evidence')
                item = dict(id='license', file='notice', bytes=3, sha256=hashlib.sha256(b'new').hexdigest(), retrieval=dict(kind='package-resource', package_path='LICENSE'))
                result = materialize(dict(artifacts=[item]), root, root)
                self.assertEqual(result[0]['code'], 'existing-artifact-hash-mismatch')
                self.assertEqual((root/'notice').read_bytes(), b'old evidence')

        def test_materialize_extracts_exact_member_and_rejects_links(self):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                content = b'upstream copyright and license\n'
                with tarfile.open(root/'source.tar', 'w') as archive:
                    member = tarfile.TarInfo('upstream/LICENSE')
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
                    link = tarfile.TarInfo('upstream/LINK')
                    link.type = tarfile.SYMTYPE
                    link.linkname = '../../outside'
                    archive.addfile(link)
                source = dict(id='source', file='source.tar', bytes=(root/'source.tar').stat().st_size, sha256=sha256(root/'source.tar'), retrieval=dict(kind='observed-snapshot'))
                notice = dict(id='license', file='retained/LICENSE', bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), retrieval=dict(kind='archive-member', archive_id='source', member='upstream/LICENSE'))
                result = materialize(dict(artifacts=[source, notice]), root)
                self.assertEqual(result[0]['status'], 'materialized-and-verified')
                self.assertEqual((root/'retained/LICENSE').read_bytes(), content)
                bad = dict(notice, id='link', file='retained/LINK', retrieval=dict(kind='archive-member', archive_id='source', member='upstream/LINK'))
                result = materialize(dict(artifacts=[source, bad]), root)
                self.assertEqual(result[0]['status'], 'error')
                self.assertFalse((root/'retained/LINK').exists())

        def test_actual_package_forbidden_sdk_and_unknown_model(self):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                (root/'tools/insta360').mkdir(parents=True)
                (root/'tools/insta360/MediaSDK.dll').write_bytes(b'synthetic excluded payload')
                (root/'unreviewed.onnx').write_bytes(b'synthetic unknown weight')
                manifest = dict(python_distributions=[], required_python_distributions=[], self_code_distributions=[], binary_fingerprints=[], artifacts=[],
                    package_exclusions=[dict(id='private-sdk-excluded', patterns=['tools/insta360/*'])])
                codes = {item['code'] for item in inspect_package(manifest, root)['issues']}
                self.assertEqual(codes, {'private-sdk-excluded', 'unreviewed-pretrained-weight'})

        def test_bundle_exports_only_reviewed_resources_and_rejects_corruption(self):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                source = root/'cache'
                source.mkdir()
                (source/'LICENSE').write_bytes(b'public upstream notice')
                (source/'private-working-notes.txt').write_bytes(b'must never be exported')
                artifact = dict(id='license', file='LICENSE', bytes=22,
                    sha256=hashlib.sha256(b'public upstream notice').hexdigest(), retrieval=dict(kind='observed-snapshot'))
                manifest = dict(release='test', artifacts=[artifact], release_gaps=[])
                manifest_path = root/'manifest.json'
                manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
                result = bundle(manifest, manifest_path, source, root/'public-bundle')
                self.assertFalse(result['release_ready'])
                self.assertFalse((root/'public-bundle/private-working-notes.txt').exists())
                self.assertTrue((root/'public-bundle/SHA256SUMS').is_file())
                (source/'LICENSE').write_bytes(b'corrupted')
                with self.assertRaises(ValueError):
                    bundle(manifest, manifest_path, source, root/'must-not-exist')
                self.assertFalse((root/'must-not-exist').exists())

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['fetch', 'materialize', 'verify', 'inventory', 'bundle', 'self-test'])
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--package-root', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--evidence-only', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'self-test':
        return self_test()
    try:
        manifest = load_manifest(args.manifest)
        if args.command == 'inventory':
            if not args.package_root or not args.package_root.is_dir():
                parser.error('inventory requires an existing --package-root')
            emit(dict(schema_version=1, release=manifest['release'], inventory=inspect_package(manifest, args.package_root)), args.report)
            return 0
        if not args.source_dir:
            parser.error('--source-dir is required')
        if args.command == 'fetch':
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda item: fetch_one(item, args.source_dir, manifest['allowed_download_hosts']), manifest['artifacts']))
            emit(dict(schema_version=1, artifacts=results), args.report)
            return int(any(item['status'] == 'error' for item in results))
        if args.command == 'materialize':
            results = materialize(manifest, args.source_dir, args.package_root)
            emit(dict(schema_version=1, artifacts=results), args.report)
            return int(any(item['status'] == 'error' for item in results))
        if args.command == 'bundle':
            if not args.output_dir:
                parser.error('bundle requires a new --output-dir')
            emit(bundle(manifest, args.manifest, args.source_dir, args.output_dir), args.report)
            return 0
        if args.package_root and not args.package_root.is_dir():
            raise ValueError('Selected package root does not exist')
        report, code = verify(manifest, args.source_dir, args.package_root, args.evidence_only)
        emit(report, args.report)
        return code
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        emit(dict(schema_version=1, release_ready=False, error='invalid-or-unreadable-evidence', error_type=type(error).__name__), args.report)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
