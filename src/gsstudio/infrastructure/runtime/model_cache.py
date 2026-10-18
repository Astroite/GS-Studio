"""Pinned pretrained models shared by all executables in an offline release."""
from __future__ import annotations

import hashlib
from pathlib import Path


MODEL_SHA256 = {
    "maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth":
        "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e",
    "alexnet-owt-7be5be79.pth":
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
}


def require_model_cache(root: Path, *, verify_hashes: bool = False) -> list[Path]:
    """Check required hub checkpoints without importing Torch or downloading."""
    models = []
    for filename, expected in MODEL_SHA256.items():
        path = root / "hub" / "checkpoints" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required offline model is missing: {path}")
        if verify_hashes:
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(8 * 1024 * 1024):
                    sha.update(block)
            if sha.hexdigest() != expected:
                raise RuntimeError(f"Offline model SHA-256 does not match: {path}")
        models.append(path)
    return models
