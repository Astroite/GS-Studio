"""Bounded image caches and opt-in mathematically equivalent training helpers.

This module does not change a recipe, checkpoint schema, sampler, or evaluation
metric. RGB caches retain uint8 pixels; each returned tensor is float32 NHWC in
the same [0, 1] convention as the original image loader. Cache byte limits cover
retained pixels, not a decoder's workspace or the caller's current float target.
"""
from collections import OrderedDict
import operator
from pathlib import Path

import numpy as np
from PIL import Image

MIB = 1024 * 1024
GIB = 1024 * MIB


class ImageTensorCache:
    """RGB8 CPU/GPU LRU caches with an uncached image path for small budgets.

    A 1612x1204 RGB image costs about5.55MiB per cache. The transient float32
    target costs22.21MiB. The default1GiB host budget fits South Building's112
    training images;256MiB on the GPU is conservative. A caller may explicitly
    choose1GiB GPU cache to fit that entire training set. No RGB float tensors,
    camera matrices, RNG state, or optimizer state are retained here.

    GPU cache allocation OOM clears/disables that cache and retries one transient
    RGB8 upload. An OOM making the required float target also releases/disables
    the cache and retries once. A genuine OOM after that propagates normally.
    GPU caching is restricted to the first CUDA device used by this instance.
    """

    def __init__(self, root, cameras, host_cache_bytes=GIB,
                 gpu_cache_bytes=256 * MIB, pin_memory=False,
                 gpu_min_free_bytes=GIB):
        self.root, self.cameras = Path(root), cameras
        self.host_cache_bytes = self._budget(host_cache_bytes)
        self.gpu_cache_bytes = self._budget(gpu_cache_bytes)
        self.gpu_min_free_bytes = self._budget(gpu_min_free_bytes)
        self.pin_memory = bool(pin_memory)
        self._host, self._gpu = OrderedDict(), OrderedDict()
        self._host_bytes = self._gpu_bytes = 0
        self._gpu_device = None
        self._gpu_disabled = False
        self._pinned_disabled = False
        self._counts = dict(decodes=0, host_hits=0, gpu_hits=0, uploads=0,
                            host_evictions=0, gpu_evictions=0, cache_ooms=0)

    @staticmethod
    def _budget(value):
        value = operator.index(value)
        if value < 0:
            raise ValueError("Image cache byte budgets must be nonnegative")
        return value

    def _index(self, index):
        index = operator.index(index)
        if not 0 <= index < len(self.cameras):
            raise IndexError("Image index outside the reconstructed camera list")
        return index

    def _rgb8(self, index):
        index = self._index(index)
        if index in self._host:
            self._counts["host_hits"] += 1
            self._host.move_to_end(index)
            return self._host[index]
        camera = self.cameras[index]
        with Image.open(self.root / camera["name"]) as image:
            if image.size != (camera["width"], camera["height"]):
                raise ValueError("Image size differs from reconstructed camera")
            pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        self._counts["decodes"] += 1
        if pixels.nbytes <= self.host_cache_bytes:
            while self._host and self._host_bytes + pixels.nbytes > self.host_cache_bytes:
                _, evicted = self._host.popitem(last=False)
                self._host_bytes -= evicted.nbytes
                self._counts["host_evictions"] += 1
            self._host[index] = pixels
            self._host_bytes += pixels.nbytes
        return pixels

    def get(self, index):
        """Legacy loader API: independent float32 HWC NumPy pixels."""
        return self._rgb8(index).astype(np.float32) / np.float32(255)

    def warm_cpu(self, indices=None):
        """Decode a specified training subset without allocating CUDA memory."""
        for index in range(len(self.cameras)) if indices is None else indices:
            self._rgb8(index)
        return self.stats()

    def _disable_gpu_cache(self):
        self._gpu.clear()
        self._gpu_bytes = 0
        self._gpu_disabled = True

    def _upload(self, index, device):
        import torch
        cpu = torch.from_numpy(self._rgb8(index))
        # A single transient pinned RGB8 image avoids pinning the whole host
        # cache. The caller uses the same stream: no unsynchronized prefetch.
        if self.pin_memory and not self._pinned_disabled:
            try:
                cpu = cpu.pin_memory()
            except RuntimeError:
                self._pinned_disabled = True
        self._counts["uploads"] += 1
        return cpu.to(device=device, non_blocking=cpu.is_pinned())

    def _cuda_rgb8(self, index, device):
        import torch
        index = self._index(index)
        if self._gpu_device is None:
            self._gpu_device = device
        cacheable_device = device == self._gpu_device
        if cacheable_device and index in self._gpu:
            self._counts["gpu_hits"] += 1
            self._gpu.move_to_end(index)
            return self._gpu[index]
        camera = self.cameras[index]
        size = int(camera["width"]) * int(camera["height"]) * 3
        cacheable = (cacheable_device and not self._gpu_disabled
                     and 0 < size <= self.gpu_cache_bytes)
        if cacheable:
            while self._gpu and self._gpu_bytes + size > self.gpu_cache_bytes:
                _, evicted = self._gpu.popitem(last=False)
                self._gpu_bytes -= evicted.numel() * evicted.element_size()
                self._counts["gpu_evictions"] += 1
                del evicted
            free, _ = torch.cuda.mem_get_info(device)
            cacheable = free >= self.gpu_min_free_bytes + size
        try:
            pixels = self._upload(index, device)
        except torch.cuda.OutOfMemoryError:
            self._counts["cache_ooms"] += 1
            self._disable_gpu_cache()
            pixels = self._upload(index, device)
            cacheable = False
        if cacheable:
            self._gpu[index] = pixels
            self._gpu_bytes += pixels.numel() * pixels.element_size()
        return pixels

    def get_tensor(self, index, device="cuda"):
        """Return float32 [1,H,W,3], retaining only RGB8 cache entries.

        Division uses float32 /255. CUDA may lower scalar division to reciprocal
        multiplication (at most one float32 ULP for RGB8 here); the legacy get()
        method retains NumPy's exact historical rounding for old recipes.
        """
        import torch
        index = self._index(index)
        device = torch.device(device)
        if device.type != "cuda":
            return torch.from_numpy(self.get(index)).to(device=device)[None]
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        pixels = self._cuda_rgb8(index, device)
        try:
            return pixels.to(dtype=torch.float32).div_(255)[None]
        except torch.cuda.OutOfMemoryError:
            # Release cache references before retrying the caller's required
            # target. Reuse the current RGB8 upload, avoiding a second decode.
            self._counts["cache_ooms"] += 1
            self._disable_gpu_cache()
            return pixels.to(dtype=torch.float32).div_(255)[None]

    def clear(self):
        self._host.clear()
        self._gpu.clear()
        self._host_bytes = self._gpu_bytes = 0

    def stats(self):
        return {**self._counts, "host_bytes": self._host_bytes,
                "gpu_bytes": self._gpu_bytes, "host_entries": len(self._host),
                "gpu_entries": len(self._gpu),
                "host_budget_bytes": self.host_cache_bytes,
                "gpu_budget_bytes": self.gpu_cache_bytes,
                "gpu_cache_disabled": self._gpu_disabled,
                "pinned_upload_disabled": self._pinned_disabled}


def uniform_ssim(x, y, method="separable"):
    """Existing NHWC uniform11 SSIM, including zero padding in the divisor.

    ``method='direct'`` keeps the original pool operators and arithmetic order.
    ``separable`` factors the same box kernel into two1D average pools. It has
    the same mathematics and gradients but floating-point summation differs;
    callers resuming an exact legacy recipe should explicitly retain direct.
    Gaussian/valid-window SSIM is deliberately outside this metric helper.
    """
    import torch.nn.functional as F
    if x.ndim != 4 or x.shape != y.shape:
        raise ValueError("SSIM requires equal-shaped NHWC image batches")
    if method not in ("direct", "separable"):
        raise ValueError("Unknown uniform SSIM pooling method")
    x, y = x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)
    size = min(11, x.shape[-1], x.shape[-2])
    if size < 1:
        raise ValueError("SSIM images must have nonempty dimensions")
    if size % 2 == 0:
        size -= 1
    def mean(t):
        if method == "direct":
            return F.avg_pool2d(t, size, 1, size // 2)
        return F.avg_pool2d(F.avg_pool2d(t, (1, size), 1, (0, size // 2)),
                            (size, 1), 1, (size // 2, 0))
    a, b = mean(x), mean(y)
    varx, vary, cov = mean(x*x)-a*a, mean(y*y)-b*b, mean(x*y)-a*b
    return (((2*a*b+.01**2)*(2*cov+.03**2))/
            ((a*a+b*b+.01**2)*(varx+vary+.03**2))).mean()


def _adam_backend(param, backend):
    if backend == "single":
        backend = "forloop"
    if backend == "auto":
        backend = "fused" if param.device.type == "cuda" else "forloop"
    if backend not in ("fused", "foreach", "forloop"):
        raise ValueError("Adam backend must be auto, fused, foreach, or single/forloop")
    if backend == "fused" and (param.device.type != "cuda" or param.is_complex()):
        raise ValueError("The fused training backend requires real CUDA parameters")
    return backend


def make_adam(param, lr, eps=1e-15, backend="auto", **kwargs):
    """Construct ordinary torch Adam with a chosen implementation backend.

    State remains standard Adam state (step, exp_avg, exp_avg_sq). Fused changes
    floating-point operation order: choose it only as an explicit new recipe
    option, not by changing a legacy checkpoint's restored param-group flags.
    """
    import torch
    backend = _adam_backend(param, backend)
    if any(name in kwargs for name in ("fused", "foreach", "differentiable", "capturable")):
        raise ValueError("Adam execution flags are managed by the backend choice")
    return torch.optim.Adam([param], lr=lr, eps=eps, fused=backend == "fused",
                            foreach=backend == "foreach", **kwargs)


def configure_adam_backend(optimizer, backend="preserve"):
    """Explicitly select a backend after load_state_dict, or preserve old flags.

    Old state_dicts overwrite execution flags, so preservation is the default.
    Explicit switching migrates only scalar step counters to the location that
    torch Adam expects; moments, parameter values, hyperparameters and steps are
    otherwise unchanged. Do not use explicit switching for exact legacy resume.
    """
    import torch
    if not isinstance(optimizer, torch.optim.Adam):
        raise TypeError("Expected a torch.optim.Adam instance")
    if backend == "preserve":
        return optimizer
    choices = []
    for group in optimizer.param_groups:
        if group.get("differentiable") or group.get("capturable"):
            raise ValueError("This training backend helper requires noncapturable, nondifferentiable Adam")
        for param in group["params"]:
            choices.append(_adam_backend(param, backend))
    if len(set(choices)) > 1:
        raise ValueError("Explicit auto switching of mixed-device Adam is unsupported")
    chosen = choices[0] if choices else "forloop"
    for group in optimizer.param_groups:
        group["fused"], group["foreach"] = chosen == "fused", chosen == "foreach"
        for param in group["params"]:
            step = optimizer.state.get(param, {}).get("step")
            if isinstance(step, torch.Tensor):
                device = param.device if chosen == "fused" else torch.device("cpu")
                dtype = torch.float32 if chosen == "fused" else step.dtype
                optimizer.state[param]["step"] = step.to(device=device, dtype=dtype)
    optimizer.defaults.update(fused=chosen == "fused", foreach=chosen == "foreach")
    return optimizer
