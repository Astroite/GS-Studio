"""Small, real CUDA forward/backward test for the isolated gsplat runtime."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import gsplat
from gsplat import rasterization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    torch.manual_seed(1234)
    started = time.perf_counter()
    device = "cuda"
    count = 128
    means = torch.randn(count, 3, device=device) * 0.2
    means[:, 2] += 2.5
    means.requires_grad_(True)
    quats = torch.zeros(count, 4, device=device)
    quats[:, 0] = 1
    quats.requires_grad_(True)
    scales = torch.full((count, 3), 0.08, device=device, requires_grad=True)
    opacities = torch.full((count,), 0.5, device=device, requires_grad=True)
    colors = torch.rand(count, 3, device=device, requires_grad=True)
    views = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor([[[70., 0., 32.], [0., 70., 32.], [0., 0., 1.]]], device=device)
    params = [means, quats, scales, opacities, colors]
    optimizer = torch.optim.Adam(params, lr=0.001)
    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        rgb, alpha, info = rasterization(means, quats, scales, opacities, colors,
                                         views, intrinsics, 64, 64, packed=False)
        loss = (rgb - 0.25).square().mean()
        loss.backward()
        for param in params:
            assert param.grad is not None and torch.isfinite(param.grad).all()
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    assert losses[-1] < losses[0], losses
    result = {
        "success": True,
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gsplat": gsplat.__version__,
        "gpu": torch.cuda.get_device_name(),
        "render_shape": list(rgb.shape),
        "losses": losses,
        "seconds": time.perf_counter() - started,
        "allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
