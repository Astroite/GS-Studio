"""Read-only offline helper discovery; never import/load/compile the extension."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def inspect(engine, runtime, extensions):
    engine, runtime, extensions = (Path(p).resolve() for p in (engine, runtime, extensions))
    os.environ['GSSTUDIO_RUNTIME'] = str(runtime)
    sys.path.insert(0, str(engine))
    import torch
    from local_postshot.studio_selection import build_location

    source, root = (Path(p).resolve() for p in build_location())
    if source != engine / 'editor_native':
        raise ValueError('Editor helper source is outside configured engine/editor_native')
    key = hashlib.sha256((torch.__version__ + sys.version).encode())
    sources = []
    for name in ('selection.cpp', 'selection.cu'):
        path = source / name
        data = path.read_bytes()
        if not data:
            raise ValueError(f'Empty editor helper source: {path}')
        key.update(data)
        sources.append({'path': str(path), 'name': name, 'sha256': hashlib.sha256(data).hexdigest()})
    source_key = key.hexdigest()[:20]
    if root != extensions / 'studio-selection' / source_key:
        raise ValueError('Editor helper loader location does not match Python/torch/source hash or configured cache')
    candidates = list(root.glob('studio_selection_cuda*.pyd'))
    if len(candidates) != 1 or candidates[0].stat().st_size == 0:
        raise ValueError(f'Prebuilt editor helper missing or ambiguous: {root}; offline build required before packaging')
    binary = candidates[0]
    proof_path = root / 'build-proof.json'
    proof = json.loads(proof_path.read_text(encoding='utf-8-sig'))
    if (Path(proof['output']).resolve() != root or proof.get('cuda_initialized') is not False
            or not re.fullmatch(r'\d+\.\d+(\+PTX)?', str(proof.get('architecture', '')))):
        raise ValueError('Editor helper build proof does not match loader location/offline architecture')
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    # Newer core proofs may additionally attest file bytes. Older offline proofs
    # bind output to the loader's Python/torch/source hash directory.
    if 'binary_sha256' in proof and proof['binary_sha256'] != binary_hash:
        raise ValueError('Editor helper build proof binary hash mismatch')
    if 'source_sha256' in proof and proof['source_sha256'] != {s['name']: s['sha256'] for s in sources}:
        raise ValueError('Editor helper build proof source hash mismatch')
    if torch.cuda.is_initialized():
        raise ValueError('Editor helper discovery must not initialize CUDA')
    return dict(source=str(source), root=str(root), key=source_key, sources=sources,
                binary=str(binary), binary_name=binary.name, binary_sha256=binary_hash,
                proof=str(proof_path), proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
                architecture=proof['architecture'], cuda_initialized=False)


if __name__ == '__main__':
    try:
        print(json.dumps(inspect(*sys.argv[1:]), allow_nan=False))
    except Exception as exc:
        sys.exit(f'Editor helper packaging preflight failed: {exc}')
