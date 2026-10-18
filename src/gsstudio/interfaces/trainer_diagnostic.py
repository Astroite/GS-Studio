"""Standalone health check shared by source and frozen gsplat trainers."""
from __future__ import annotations

import json
import hashlib
from importlib.metadata import version
from pathlib import Path

from gsstudio.infrastructure.runtime.gpu_lock import gpu_session


def main() -> int:
    with gpu_session():
        return _diagnose()


def _diagnose() -> int:
    import torch
    import gsplat
    from gsplat import csrc  # noqa: F401 - proves that the native extension loads

    assert torch.__version__.split("+")[0] == "2.8.0", "Expected isolated PyTorch 2.8.0"
    assert torch.version.cuda == "12.9", "Expected isolated cu129 PyTorch build"
    assert version("gsplat").split("+")[0] == "1.5.3", "Expected gsplat 1.5.3"
    assert torch.cuda.is_available(), "CUDA unavailable in native training environment"
    means = torch.tensor([[0., 0., 3.]], device="cuda", requires_grad=True)
    quats = torch.tensor([[1., 0., 0., 0.]], device="cuda", requires_grad=True)
    scales = torch.full((1, 3), .2, device="cuda", requires_grad=True)
    opacities = torch.full((1,), .5, device="cuda", requires_grad=True)
    colors = torch.full((1, 3), .5, device="cuda", requires_grad=True)
    rgb, alpha, _ = gsplat.rasterization(
        means, quats, scales, opacities, colors,
        torch.eye(4, device="cuda")[None],
        torch.tensor([[[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]]], device="cuda"), 8, 8,
    )
    (rgb.sum() + alpha.sum()).backward()
    assert torch.isfinite(rgb).all(), "Nonfinite CUDA output"
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in (means, quats, scales, opacities, colors)), "Invalid CUDA gradients"
    torch.cuda.synchronize()
    print(json.dumps({"torch": torch.__version__, "gsplat": version("gsplat"),
                      "cuda": torch.version.cuda, "forward_backward": "passed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
