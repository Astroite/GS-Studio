"""Read-only source hygiene, local doc links and archive provenance checks.

Uses only the standard library. Does not import the app, Torch, native extensions
or runtime configuration, and never touches local data or ignored environments.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATH = re.compile(
    r'(?i)(?:[DE]:[/\\]+(?:AI[/\\]+(?:GSStudio|LocalPostshot)|'
    r'Project[/\\]+3DGS|PlayGround[/\\]+local-postshot)|'
    r'C:[/\\]+Users[/\\]+astro)'
)
LOCAL_LINK = re.compile(r'!?\[[^\]\n]*\]\(([^)\n]+)\)')
SHA256 = re.compile(r'[0-9a-f]{64}')


def source_paths() -> list[str]:
    raw = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT,
    ).decode('utf-8')
    return sorted({p for p in raw.split('\0') if p and (ROOT/p).is_file()})


def public_document(name: str) -> bool:
    # Every tracked/unignored Markdown file can enter the public source snapshot.
    return name.endswith('.md')


def main() -> int:
    errors: list[str] = []
    paths = source_paths()
    documents = 0
    python_files = 0
    for name in paths:
        p = PurePosixPath(name)
        forbidden = {'.claude', 'node_modules', '.venv', '.venv-gsplat',
                     '__pycache__', '.pytest_cache', 'sdks'}
        if forbidden.intersection(p.parts) or p.parts[0] in {
            'runtime', 'cache', '.cache', 'data', 'Data', 'env', 'projects',
            'locations', 'prototypes',
        }:
            errors.append(f'{name}: local/private/retired payload in source')
        if name.startswith(('engine/reference/evidence/', 'engine/evidence/')):
            errors.append(f'{name}: raw evidence belongs outside public source')
        if p.name == '.env' or p.suffix.lower() in {
            '.dll', '.pyd', '.exe', '.obj', '.lib', '.insv', '.mp4', '.pt',
            '.pth', '.psht', '.pfx', '.pem', '.key',
        }:
            errors.append(f'{name}: runtime, raw media or secret-like artifact in source')
        if (ROOT/name).stat().st_size > 5*1024*1024:
            errors.append(f'{name}: source file exceeds 5 MiB; review external storage')
        if p.suffix == '.py':
            python_files += 1
            try:
                tree = ast.parse((ROOT/name).read_text(encoding='utf-8-sig'), filename=name)
                for node in ast.walk(tree):
                    modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                               else [a.name for a in node.names] if isinstance(node, ast.Import)
                               else [])
                    if any(m == 'gsdb' or m.startswith('gsdb.') for m in modules):
                        errors.append(f'{name}:{node.lineno}: retired gsdb import')
                    if any(m == 'gsstudio.interfaces.desktop' or m.startswith('gsstudio.interfaces.desktop.') for m in modules):
                        errors.append(f'{name}:{node.lineno}: retired Qt frontend import')
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f'{name}: {exc}')
        if not public_document(name):
            continue
        documents += 1
        content = (ROOT/name).read_text(encoding='utf-8-sig')
        for number, line in enumerate(content.splitlines(), 1):
            if PRIVATE_PATH.search(line) or 'file:///' in line:
                errors.append(f'{name}:{number}: private machine path in public documentation')
        # Ignore executable examples when interpreting Markdown link syntax.
        content = re.sub(r'(?ms)^(```|~~~).*?^\1[^\n]*', '', content)
        for match in LOCAL_LINK.finditer(content):
            target = match.group(1).strip()
            target = target[1:target.find('>')] if target.startswith('<') else target.split()[0]
            parsed = urlsplit(target)
            if parsed.scheme in {'http', 'https', 'mailto'} or target.startswith('#'):
                continue
            if parsed.scheme or target.startswith(('/', '\\')):
                errors.append(f'{name}: nonportable link {target}')
                continue
            linked = (ROOT/name).parent / unquote(parsed.path)
            resolved = linked.resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f'{name}: missing or escaping local link {target}')

    manifests = sorted((ROOT/'docs/archive').glob('source-retirement-*.json'))
    if not manifests:
        errors.append('docs/archive: missing source retirement manifest')
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            seen = set()
            if manifest['schema_version'] != 1 or not SHA256.fullmatch(manifest['source_snapshot_sha256']):
                raise ValueError('invalid manifest schema/snapshot hash')
            for entry in manifest['entries']:
                name = entry['original_relative_path']
                relative = PurePosixPath(name)
                archive = PurePosixPath(entry['archive_relative_path'])
                if name in seen or relative.is_absolute() or '..' in relative.parts or ':' in name:
                    raise ValueError(f'duplicate or unsafe source path: {name}')
                if archive != PurePosixPath('files')/relative:
                    raise ValueError(f'invalid archive path: {name}')
                if not SHA256.fullmatch(entry['sha256']) or entry['bytes'] < 0 or not entry['reason']:
                    raise ValueError(f'invalid hash/size/reason: {name}')
                if entry['action'] not in {'archive', 'backup_before_edit'}:
                    raise ValueError(f'invalid action: {name}')
                if entry['action'] == 'archive' and (ROOT/name).exists():
                    errors.append(f'{name}: retired source reintroduced; review provenance first')
                seen.add(name)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            errors.append(f'{manifest_path.relative_to(ROOT)}: {exc}')

    try:
        frontend = json.loads((ROOT/'desktop/package.json').read_text(encoding='utf-8'))
        lock = json.loads((ROOT/'desktop/package-lock.json').read_text(encoding='utf-8'))
        for key in ('version', 'dependencies', 'devDependencies'):
            if lock['packages'][''].get(key) != frontend.get(key):
                errors.append(f'desktop/package-lock.json: root {key} differs from package.json')
        if lock['lockfileVersion'] != 3:
            errors.append('desktop/package-lock.json: expected lockfileVersion 3')
        if not (ROOT/'desktop/src-tauri/Cargo.lock').is_file():
            errors.append('desktop/src-tauri/Cargo.lock: missing Rust dependency lock')
    except (KeyError, ValueError, OSError) as exc:
        errors.append(f'frontend lock: {exc}')

    protected = [
        '.claude/settings.local.json', '.venv/keep.txt', '.venv-gsplat/keep.txt',
        'env/keep.txt', 'sdks/keep.txt', 'runtime/keep.txt', 'cache/keep.txt',
        'Data/keep.txt', 'projects/keep.txt', '.env',
    ]
    ignored = subprocess.run(
        ['git', 'check-ignore', '--no-index', '-z', '--stdin'], cwd=ROOT,
        input=('\0'.join(protected)+'\0').encode('utf-8'), capture_output=True,
    )
    missing = set(protected)-set(ignored.stdout.decode('utf-8').split('\0'))
    if ignored.returncode not in {0, 1}:
        errors.append('git check-ignore failed: '+ignored.stderr.decode('utf-8', errors='replace'))
    if missing:
        errors.append('local directories must be ignored, never deleted: '+', '.join(sorted(missing)))
    for required in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'CHANGELOG.md', 'CONTRIBUTING.md'):
        if not (ROOT/required).is_file():
            errors.append(f'{required}: missing release source document')
    if (ROOT/'LICENSE').is_file() and hashlib.sha256((ROOT/'LICENSE').read_bytes()).hexdigest() != (
        'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'
    ):
        errors.append('LICENSE: expected the complete, unmodified official Apache 2.0 text')

    print(f'Checked {len(paths)} source files, {python_files} Python syntax trees, '
          f'{documents} public documents, archive metadata and dependency locks.')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print('Source checks passed. No runtime, GPU or release acceptance was performed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
