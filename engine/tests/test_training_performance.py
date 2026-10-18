import copy
import os

import numpy as np
from PIL import Image
import pytest
import torch

from local_postshot.training_performance import (
    ImageTensorCache, configure_adam_backend, make_adam, uniform_ssim,
)


def make_images(tmp_path, sizes=((8, 6), (8, 6), (5, 4))):
    rng = np.random.default_rng(23)
    cameras, pixels = [], []
    for i, (w, h) in enumerate(sizes):
        array = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        name = f"photo{i}.png"
        Image.fromarray(array).save(tmp_path/name)
        cameras.append(dict(name=name, width=w, height=h))
        pixels.append(array)
    return cameras, pixels


def test_rgb8_cache_is_bounded_and_legacy_pixels_exact(tmp_path):
    cameras, pixels = make_images(tmp_path)
    size = pixels[0].nbytes
    cache = ImageTensorCache(tmp_path, cameras, host_cache_bytes=size, gpu_cache_bytes=0)
    expected = pixels[0].astype(np.float32)/255
    np.testing.assert_array_equal(cache.get(0), expected)
    np.testing.assert_array_equal(cache.get_tensor(0, "cpu")[0].numpy(), expected)
    assert cache.stats()["decodes"] == 1
    assert cache.stats()["host_bytes"] == size
    cache.get(1)
    assert cache.stats()["host_entries"] == 1
    assert cache.stats()["host_evictions"] == 1
    cache.get(0)
    assert cache.stats()["decodes"] == 3
    cache.clear()
    assert cache.stats()["host_bytes"] == cache.stats()["gpu_bytes"] == 0


def test_oversized_and_zero_budget_images_are_uncached(tmp_path):
    cameras, _ = make_images(tmp_path)
    for budget in (0, 10):
        cache = ImageTensorCache(tmp_path, cameras, host_cache_bytes=budget)
        cache.warm_cpu([0, 1])
        cache.get(0)
        assert cache.stats()["decodes"] == 3
        assert cache.stats()["host_bytes"] == 0
    with pytest.raises(ValueError):
        ImageTensorCache(tmp_path, cameras, host_cache_bytes=-1)


def test_cache_enforces_camera_size_and_index(tmp_path):
    cameras, _ = make_images(tmp_path)
    cameras[0]["width"] += 1
    cache = ImageTensorCache(tmp_path, cameras)
    with pytest.raises(ValueError, match="Image size"):
        cache.get(0)
    with pytest.raises(IndexError):
        cache.get(-1)


def test_gpu_lru_and_oom_fallback_without_cuda_allocation(tmp_path, monkeypatch):
    cameras, pixels = make_images(tmp_path)
    size = pixels[0].nbytes
    cache = ImageTensorCache(tmp_path, cameras, gpu_cache_bytes=size * 2,
                             gpu_min_free_bytes=0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (10**9, 10**9))
    monkeypatch.setattr(cache, "_upload", lambda i, _: torch.from_numpy(pixels[i]))
    device = torch.device("cuda:0")
    cache._cuda_rgb8(0, device)
    cache._cuda_rgb8(1, device)
    cache._cuda_rgb8(2, device)
    assert cache.stats()["gpu_bytes"] <= size * 2
    assert cache.stats()["gpu_evictions"] == 1
    assert cache.stats()["gpu_entries"] == 2
    assert 0 not in cache._gpu
    calls = []
    def fail_once(i, _):
        calls.append(i)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("Injected allocation failure")
        return torch.from_numpy(pixels[i])
    monkeypatch.setattr(cache, "_upload", fail_once)
    actual = cache._cuda_rgb8(0, device)
    np.testing.assert_array_equal(actual.numpy(), pixels[0])
    assert calls == [0, 0]
    assert cache.stats()["cache_ooms"] == 1
    assert cache.stats()["gpu_cache_disabled"]
    assert cache.stats()["gpu_bytes"] == 0


@pytest.mark.parametrize("shape", [(1, 19, 23, 3), (2, 4, 8, 3), (1, 1, 3, 1)])
def test_uniform_ssim_separable_matches_legacy_values_and_gradients(shape):
    from local_postshot.training import ssim
    gen = torch.Generator().manual_seed(133)
    a = torch.rand(shape, generator=gen, dtype=torch.float64, requires_grad=True)
    b = torch.rand(shape, generator=gen, dtype=torch.float64, requires_grad=True)
    old = ssim(a, b)
    direct = uniform_ssim(a, b, "direct")
    actual = uniform_ssim(a, b)
    torch.testing.assert_close(direct, old, rtol=0, atol=0)
    torch.testing.assert_close(actual, old, rtol=1e-12, atol=1e-12)
    grad_old = torch.autograd.grad(old, (a, b), retain_graph=True)
    grad_new = torch.autograd.grad(actual, (a, b))
    for left, right in zip(grad_old, grad_new):
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)


def test_adam_standard_state_load_preserves_checkpoint_flags():
    a = torch.nn.Parameter(torch.tensor([.4, -.2], dtype=torch.float64))
    old = torch.optim.Adam([a], lr=.01, eps=1e-15)
    a.grad = torch.tensor([.3, -.1], dtype=torch.float64)
    old.step()
    saved = copy.deepcopy(old.state_dict())
    b = torch.nn.Parameter(a.detach().clone())
    new = make_adam(b, lr=.01, backend="foreach")
    new.load_state_dict(saved)
    configure_adam_backend(new)
    assert new.param_groups[0]["foreach"] == saved["param_groups"][0]["foreach"]
    for parameter, optimizer in ((a, old), (b, new)):
        parameter.grad = torch.tensor([.2, .6], dtype=torch.float64)
        optimizer.step()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    configure_adam_backend(new, "foreach")
    assert new.param_groups[0]["foreach"] is True
    assert new.param_groups[0]["fused"] is False
    assert new.state[b]["step"].device.type == "cpu"


@pytest.mark.skipif(os.environ.get("LOCAL_POSTSHOT_GPU_TESTS") != "1", reason="Opt-in CUDA performance contract")
def test_cuda_rgb8_cache_and_fused_adam_legacy_state(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cameras, pixels = make_images(tmp_path, ((256, 1),))
    # Include every possible8-bit value to check GPU normalization rounding.
    pixels[0][:, :, :] = np.arange(256, dtype=np.uint8)[None, :, None]
    Image.fromarray(pixels[0]).save(tmp_path/cameras[0]["name"])
    size = pixels[0].nbytes
    cache = ImageTensorCache(tmp_path, cameras, gpu_cache_bytes=size, gpu_min_free_bytes=0)
    expected = torch.from_numpy(pixels[0].astype(np.float32)/255)[None].cuda()
    torch.testing.assert_close(cache.get_tensor(0), expected, rtol=0, atol=6e-8)
    torch.testing.assert_close(cache.get_tensor(0), expected, rtol=0, atol=6e-8)
    assert cache.stats()["gpu_hits"] == 1
    assert cache.stats()["gpu_bytes"] == size
    a = torch.nn.Parameter(torch.linspace(-1, 1, 128, device="cuda"))
    old = make_adam(a, lr=.01, backend="forloop")
    a.grad = torch.full_like(a, .3)
    old.step()
    b = torch.nn.Parameter(a.detach().clone())
    new = make_adam(b, lr=.01)
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    configure_adam_backend(new, "fused")
    assert new.state[b]["step"].device.type == "cuda"
    for parameter, optimizer in ((a, old), (b, new)):
        parameter.grad = torch.linspace(.1, .3, 128, device="cuda")
        optimizer.step()
    torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
