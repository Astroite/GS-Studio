import math

import numpy as np
import pytest
import torch

from local_postshot.training_recipe import (initial_arrays, make_v2_strategy,
    recipe_config, recipe_fingerprint, recipe_from_checkpoint, resolve_ssim_backend,
    require_training_recipe_ready, training_camera_guard, training_loss, training_ssim,
    trim_cuda_cache_if_needed)


def parameters(n=4):
    p = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(.001))),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]]).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.full((n,), torch.logit(torch.tensor(.2)))),
        "sh0": torch.nn.Parameter(torch.full((n, 1, 3), .1)),
        "shN": torch.nn.Parameter(torch.zeros((n, 15, 3)))})
    optimizers = {k: torch.optim.Adam([value], lr=.001, eps=1e-15) for k, value in p.items()}
    sum(value.sum() for value in p.values()).backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    state = {"grad2d": torch.zeros(n), "count": torch.ones(n), "radii": torch.zeros(n), "scene_scale": 1.}
    return p, optimizers, state


def assert_consistent_optimizer_shapes(p, optimizers, n):
    for key, value in p.items():
        assert len(value) == n
        assert optimizers[key].param_groups[0]["params"][0] is value
        for field in ("exp_avg", "exp_avg_sq"):
            assert optimizers[key].state[value][field].shape == value.shape


def test_v2_fixed_schedule_and_legacy_scaled_schedule_are_distinct():
    v2, legacy = recipe_config("quality_v2", 10000), recipe_config("legacy_adc", 10000)
    assert v2["strategy"]["refine_start_iter"] == 500
    assert v2["strategy"]["refine_every"] == 100
    assert v2["strategy"]["reset_every"] == 3000
    assert v2["strategy"]["pause_refine_after_reset"] == 500
    assert v2["strategy"]["refine_stop_iter"] == 10000
    assert legacy["strategy"]["refine_start_iter"] == 167
    assert legacy["strategy"]["refine_every"] == 33
    assert legacy["strategy"]["reset_every"] == 1000
    assert legacy["ssim_kind"] == "uniform11" and legacy["ssim_padding"] == "same"
    assert v2["ssim_kind"] == "gaussian11" and v2["ssim_padding"] == "valid"
    assert recipe_config("quality_v2", 30000)["strategy"]["refine_every"] == 100


@pytest.mark.parametrize("count,interval,margin", [(112, 3000, 1500), (2000, 4000, 2000)])
def test_observed_v3_preset_binds_actual_training_image_count(count, interval, margin):
    config = recipe_config("observed_v3", 10000, train_image_count=count)
    assert config["name"] == "observed_v3" and config["train_image_count"] == count
    assert config["train_image_count_source"] == "caller_training_split"
    assert config["training_binding_status"] == "resolved"
    assert config["strategy"]["pause_refine_after_reset"] == count
    assert config["strategy"]["reset_every"] == interval
    assert config["reset_stop_margin"] == margin and config["reset_interval_requested"] == 3000
    assert config["strategy"]["prune_opa"] == 5 / 256
    assert config["opacity_reset_multiplier"] == 1.5
    assert config["adam_beta2"] == .997 and config["adam_beta1"] == .9
    assert config["init_mode"] == "splat3_observed" and config["init_rotation"] == "shoemake"
    assert config["init_opacity"] == .1 and config["init_scale_floor"] == 1e-7
    assert config["ssim_kind"] == "gaussian11" and config["ssim_padding"] == "same"
    assert config["ssim_backend"] == "auto"
    assert any("scale_rel_l2reg=.05" in item for item in config["unrecovered_original_behavior"])
    require_training_recipe_ready(config)


def test_observed_v3_override_order_and_effective_interval_floor_are_explicit():
    config = recipe_config("observed_v3", 10000, {"reset_every": 1000}, train_image_count=2000)
    assert config["reset_interval_requested"] == 1000
    assert config["strategy"]["reset_every"] == 4000 and config["reset_stop_margin"] == 2000
    longer = recipe_config("observed_v3", 10000, {"reset_every": 6001}, train_image_count=112)
    assert longer["reset_stop_margin"] == 3000  # Native half-period uses integer floor.
    overrides = {"reset_stop_margin": 99, "reset_every": 6001, "pause_refine_after_reset": 17,
                 "init_rotation": "identity", "adam_beta2": .998}
    a = recipe_config("observed_v3", 10000, overrides, train_image_count=112)
    b = recipe_config("observed_v3", 10000, dict(reversed(list(overrides.items()))), train_image_count=112)
    assert a == b
    assert a["strategy"]["pause_refine_after_reset"] == 17 and a["reset_stop_margin"] == 99
    assert a["init_rotation"] == "identity" and a["adam_beta2"] == .998
    with pytest.raises(ValueError, match="immutable"):
        recipe_config("observed_v3", 10000, {"train_image_count": 10}, train_image_count=112)


def test_observed_v3_n112_numerical_configuration_equals_frozen_r3_explicit_recipe():
    r3 = recipe_config("quality_v2", 10000, {
        "init_mode": "splat3_observed", "ssim_padding": "same", "prune_opa": 5 / 256,
        "adam_beta2": .997, "opacity_reset_multiplier": 1.5,
        "reset_stop_margin": 1500, "pause_refine_after_reset": 112})
    preset = recipe_config("observed_v3", 10000, train_image_count=112)
    metadata = {"name", "train_image_count", "train_image_count_source", "training_binding_status",
                "reset_interval_requested", "unrecovered_original_behavior"}
    assert {k: v for k, v in preset.items() if k not in metadata} == {
        k: v for k, v in r3.items() if k not in metadata}


def test_observed_v3_unbound_sentinel_validates_parameters_but_cannot_train_or_resume():
    config = recipe_config("observed_v3", 10000)
    assert config["train_image_count"] is None
    assert config["train_image_count_source"] == "unresolved_validation_only"
    assert config["training_binding_status"] == "validation_only"
    for operation in (lambda: require_training_recipe_ready(config),
                      lambda: initial_arrays(np.zeros((4, 3)), np.zeros((4, 3)), 3, 10, config),
                      lambda: make_v2_strategy(config, 10),
                      lambda: recipe_from_checkpoint({"training_recipe": config})):
        with pytest.raises(ValueError, match="explicit train_image_count"):
            operation()
    with pytest.raises(ValueError, match="adam_"):
        recipe_config("observed_v3", 10000, {"adam_beta2": 1})


@pytest.mark.parametrize("count", [0, -1, True, 112.5])
def test_observed_v3_rejects_invalid_training_split_sizes(count):
    with pytest.raises(ValueError, match="train_image_count"):
        recipe_config("observed_v3", 10000, train_image_count=count)


def test_new_preset_never_changes_v2_legacy_or_saved_observed_recipe():
    for name in ("legacy_adc", "quality_v2"):
        assert recipe_config(name, 10000) == recipe_config(name, 10000, train_image_count=2000)
        assert "train_image_count" not in recipe_config(name, 10000)
    assert recipe_from_checkpoint({"config": {"training_recipe": "observed_v3"}}) is None
    saved = recipe_config("observed_v3", 10000, {"reset_every": 6001}, train_image_count=112)
    state = {"training_recipe": saved, "config": {"steps": 30000}}
    restored = recipe_from_checkpoint(state)
    assert restored == saved and restored["schedule_steps"] == 10000
    assert restored["strategy"]["reset_every"] == 6001 and restored["reset_stop_margin"] == 3000
    restored["strategy"]["reset_every"] = 9000
    assert saved["strategy"]["reset_every"] == 6001
    incomplete = dict(saved)
    del incomplete["train_image_count"]
    with pytest.raises(ValueError, match="incomplete"):
        recipe_from_checkpoint({"training_recipe": incomplete})


def test_observed_v3_cpu_strategy_uses_n112_pause_and_skips_terminal_reset(monkeypatch):
    import gsplat.strategy.ops as ops
    calls = []
    monkeypatch.setattr(ops, "reset_opa", lambda **kw: calls.append(("reset", current[0], kw["value"])))
    strategy = make_v2_strategy(recipe_config("observed_v3", 10000, train_image_count=112), 10)
    monkeypatch.setattr(strategy, "_update_state", lambda *a, **kw: None)
    monkeypatch.setattr(strategy, "_prune_gs", lambda *a: 0)
    monkeypatch.setattr(strategy, "_grow_gs", lambda *a: calls.append(("refine", current[0])) or (0, 0))
    p, optimizers, state = parameters()
    current = [0]
    for step in [3000, 3100, 3200, 6000, 6100, 6200, 9000]:
        current[0] = step
        strategy.step_post_backward(p, optimizers, state, step, {})
    assert calls == [("refine", 3000), ("reset", 3000, 7.5 / 256), ("refine", 3200),
                     ("refine", 6000), ("reset", 6000, 7.5 / 256), ("refine", 6200), ("refine", 9000)]


def test_recipe_configuration_is_explicit_and_fingerprint_stable():
    a = recipe_config("v2", 10000)
    assert recipe_fingerprint(a) == recipe_fingerprint(recipe_config("quality_v2", 10000))
    b = recipe_config("v2", 10000, {"absgrad": True})
    assert b["strategy"]["grow_grad2d"] == .0008
    assert recipe_fingerprint(a) != recipe_fingerprint(b)
    assert recipe_config("v2", 10000, {"refine_scale2d_stop_iter": 0})["strategy"]["refine_scale2d_stop_iter"] == 0
    for overrides in ({"unknown": 1}, {"reset_every": 0}, {"opacity_reg": -1}, {"ssim_lambda": float("nan")}):
        with pytest.raises(ValueError):
            recipe_config("v2", 10000, overrides)


def test_adam_recipe_defaults_preserve_legacy_and_allow_explicit_observed_beta2():
    public = recipe_config("quality_v2", 10000)
    assert (public["adam_beta1"], public["adam_beta2"], public["adam_epsilon"]) == (.9, .999, 1e-15)
    observed = recipe_config("quality_v2", 10000, {"adam_beta2": .997})
    assert observed["adam_beta2"] == .997
    for overrides in ({"adam_beta1": -1}, {"adam_beta2": 1}, {"adam_epsilon": 0}, {"adam_epsilon": float("nan")}):
        with pytest.raises(ValueError, match="adam_"):
            recipe_config("quality_v2", 10000, overrides)


def test_missing_checkpoint_recipe_never_reconstructs_a_new_schedule():
    old = {"strategy_config": {"refine_stop_iter": 2700, "reset_every": 3000}}
    assert recipe_from_checkpoint(old) is None
    assert old["strategy_config"]["reset_every"] == 3000
    state = {"training_recipe": recipe_config("v2", 10000)}
    restored = recipe_from_checkpoint(state)
    restored["strategy"]["reset_every"] = 6000
    assert state["training_recipe"]["strategy"]["reset_every"] == 3000
    with pytest.raises(ValueError, match="Unsupported"):
        recipe_from_checkpoint({"training_recipe": {"schema": 999}})


def test_initialization_preserves_public_3nn_rms_sh_opacity_and_identity():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    rgb = np.array([[0, .5, 1], [.2, .3, .4], [.6, .7, .8], [1, 1, 1]], np.float32)
    arrays = initial_arrays(points, rgb, 3, 100, recipe_config("v2", 10000))
    np.testing.assert_array_equal(arrays["means"], points)
    assert np.exp(arrays["scales"][0, 0]) == pytest.approx(1)
    assert np.exp(arrays["scales"][1, 0]) == pytest.approx(math.sqrt(5 / 3))
    np.testing.assert_allclose(arrays["sh0"][:, 0] * .28209479177387814 + .5, rgb, atol=1e-7)
    np.testing.assert_array_equal(arrays["shN"], 0)
    np.testing.assert_array_equal(arrays["quats"], [[1, 0, 0, 0]] * 4)
    np.testing.assert_allclose(1 / (1 + np.exp(-arrays["opacities"])), .1)
    with pytest.raises(ValueError, match="normalized"):
        initial_arrays(points, rgb * 255, 3, 100, recipe_config("v2", 10000))


def test_observed_initialization_is_arithmetic_mean_not_rms():
    points = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [9, 0, 0]], np.float32)
    rgb = np.full((4, 3), .5, np.float32)
    public = initial_arrays(points, rgb, 3, 100, recipe_config("v2", 10000))
    config = recipe_config("v2", 10000, {"init_mode": "splat3_observed"})
    observed = initial_arrays(points, rgb, 3, 100, config, rng=np.random.default_rng(42))
    assert np.exp(public["scales"][0, 0]) == pytest.approx(math.sqrt((1 + 9 + 81) / 3))
    assert np.exp(observed["scales"][0, 0]) == pytest.approx((1 + 3 + 9) / 3 / 1.5)
    assert config["init_scale_floor"] == 1e-7 and config["init_rotation"] == "shoemake"
    assert config["init_opacity"] == .1  # Also matches the recovered shared static default.


def test_observed_scale_floor_is_before_division_and_shoemake_covariance_is_isotropic():
    from scipy.spatial.transform import Rotation
    points = np.zeros((4, 3), np.float32)
    config = recipe_config("v2", 10000, {"init_mode": "splat3_observed"})
    values = initial_arrays(points, np.full((4, 3), .5), 3, 100, config, rng=np.random.default_rng(42))
    scales = np.exp(values["scales"])
    np.testing.assert_allclose(scales, 1e-7 / 1.5, rtol=1e-6, atol=0)
    q = values["quats"]
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-7)
    rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    covariance = (rotations * scales[:, None, :] ** 2) @ rotations.transpose(0, 2, 1)
    np.testing.assert_allclose(covariance, np.eye(3)[None] * scales[:, :1, None] ** 2, atol=1e-28, rtol=1e-6)
    assert not np.allclose(q[:, 1:], 0)


def gaussian_reference(x, y, padding="valid"):
    from scipy.signal import convolve2d
    axis = np.arange(-5, 6)
    weights = np.exp(-.5 * (axis / 1.5) ** 2)
    weights /= weights.sum()
    window = np.outer(weights, weights)
    scores = []
    for channel in range(3):
        a, b = x[0, :, :, channel], y[0, :, :, channel]
        blur = lambda z: convolve2d(z, window, mode=padding)
        ma, mb = blur(a), blur(b)
        va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
        scores.append(((2 * ma * mb + .01 ** 2) * (2 * cov + .03 ** 2)) /
                      ((ma ** 2 + mb ** 2 + .01 ** 2) * (va + vb + .03 ** 2)))
    return np.mean(scores)


def test_gaussian_ssim_matches_independent_2d_convolution_and_gradient():
    rng = np.random.default_rng(81)
    x, y = rng.random((1, 13, 17, 3)), rng.random((1, 13, 17, 3))
    prediction = torch.tensor(x, dtype=torch.float64, requires_grad=True, device="cpu")
    target = torch.tensor(y, dtype=torch.float64, device="cpu")
    score = training_ssim(prediction, target, backend="torch")
    assert score.item() == pytest.approx(gaussian_reference(x, y), abs=2e-13)
    score.backward()
    assert torch.isfinite(prediction.grad).all()
    index, epsilon = (0, 6, 8, 1), 1e-5
    plus, minus = x.copy(), x.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    numerical = (gaussian_reference(plus, y) - gaussian_reference(minus, y)) / (2 * epsilon)
    assert prediction.grad[index].item() == pytest.approx(numerical, abs=1e-8)


def test_native_same_padding_ssim_and_recovered_reverse_chain_rule():
    from scipy.signal import convolve2d
    rng = np.random.default_rng(58)
    x, y = rng.random((1, 17, 19, 3)), rng.random((1, 17, 19, 3))
    prediction = torch.tensor(x, dtype=torch.float64, requires_grad=True, device="cpu")
    target = torch.tensor(y, dtype=torch.float64, device="cpu")
    score = training_ssim(prediction, target, backend="torch", padding="same")
    assert score.item() == pytest.approx(gaussian_reference(x, y, "same"), abs=2e-13)
    score.backward()
    axis = np.arange(-5, 6)
    weights = np.exp(-.5 * (axis / 1.5) ** 2)
    weights /= weights.sum()
    window = np.outer(weights, weights)
    blur = lambda z: convolve2d(z, window, mode="same")
    gradient = np.zeros_like(x)
    for channel in range(3):
        a, b = x[0, :, :, channel], y[0, :, :, channel]
        ma, mb = blur(a), blur(b)
        aa = 2 * ma * mb + .01 ** 2
        bb = 2 * (blur(a * b) - ma * mb) + .03 ** 2
        cc = ma ** 2 + mb ** 2 + .01 ** 2
        dd = blur(a * a + b * b) - ma ** 2 - mb ** 2 + .03 ** 2
        # Direct algebra of current-cache i_749's three derivative outputs and
        # i_J69's reverse Gaussian/chain-rule assembly. Not autodifferentiation
        # of a duplicate Torch forward implementation.
        dmu = 2 * ((bb - aa) * mb / (cc * dd) - aa * bb * ma / (cc ** 2 * dd) + aa * bb * ma / (cc * dd ** 2))
        dxx = -aa * bb / (cc * dd ** 2)
        dxy = 2 * aa / (cc * dd)
        gradient[0, :, :, channel] = (blur(dmu) + 2 * a * blur(dxx) + b * blur(dxy)) / a.size / 3
    np.testing.assert_allclose(prediction.grad.numpy(), gradient, atol=1e-13, rtol=1e-10)


def test_legacy_uniform_metric_is_unchanged_and_loss_has_no_render_clamp():
    from local_postshot.training import ssim
    x = torch.rand((1, 17, 21, 3), device="cpu") * 1.2
    y = torch.rand_like(x)
    assert training_ssim(x, y, backend="torch", kind="uniform11", padding="same").item() == pytest.approx(ssim(x, y).item(), abs=1e-7)
    config = recipe_config("v2", 10000, {"ssim_backend": "torch"})
    result = training_loss(x, y, {}, config)
    assert result["l1"].item() == pytest.approx((x - y).abs().mean().item())
    assert result["loss"].item() == pytest.approx(.8 * result["l1"].item() + .2 * (1 - result["ssim"].item()))
    assert resolve_ssim_backend("cpu", "auto") == "torch"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_ssim_backend("cpu", "fused")


def test_explicit_torch_backend_does_not_lazy_import_an_optional_native_extension(monkeypatch):
    import local_postshot.training_recipe as module
    def forbidden():
        raise AssertionError("An explicitly selected Torch backend must not import fused_ssim")
    monkeypatch.setattr(module, "_fused_ssim", forbidden)
    assert resolve_ssim_backend("cuda", "torch") == "torch"


@pytest.mark.parametrize("gradient", [0., 1.])
def test_screen_split_is_budgeted_even_with_zero_gradient_or_clone_overlap(gradient):
    p, optimizers, state = parameters()
    state["grad2d"][0] = gradient
    state["radii"][0] = .08
    strategy = make_v2_strategy(recipe_config("v2", 10000), max_splats=5)
    before = p["means"].detach().clone()
    clones, splits = strategy._grow_gs(p, optimizers, state, 3500)
    assert (clones, splits) == (0, 1)
    assert len(p["means"]) == 5
    assert_consistent_optimizer_shapes(p, optimizers, 5)
    assert len(state["radii"]) == 5
    torch.testing.assert_close(p["means"][:3], before[1:])


def test_all_screen_growth_is_blocked_at_cap():
    p, optimizers, state = parameters()
    state["radii"][:] = .08
    state["grad2d"][:] = 1
    strategy = make_v2_strategy(recipe_config("v2", 10000), max_splats=4)
    assert strategy._grow_gs(p, optimizers, state, 3500) == (0, 0)
    assert_consistent_optimizer_shapes(p, optimizers, 4)


def test_pruning_frees_budget_before_real_clone_and_optimizer_resize(monkeypatch):
    p, optimizers, state = parameters()
    with torch.no_grad():
        p["scales"][0] = math.log(.2)
    state["grad2d"][1] = 1
    config = recipe_config("v2", 10000, {"pause_refine_after_reset": 0})
    strategy = make_v2_strategy(config, max_splats=4)
    monkeypatch.setattr(strategy, "_update_state", lambda *args, **kwargs: None)
    strategy.step_post_backward(p, optimizers, state, 3500, {})
    assert state["last_refinement"] == {"step": 3500, "pruned": 1, "cloned": 1, "split": 0, "count": 4}
    assert_consistent_optimizer_shapes(p, optimizers, 4)
    assert not state["grad2d"].any() and not state["radii"].any()


def test_runtime_informed_schedule_has_explicit_reset_step_order(monkeypatch):
    import gsplat.strategy.ops as ops
    events = []
    monkeypatch.setattr(ops, "reset_opa", lambda **kwargs: events.append(("reset", current[0])))
    strategy = make_v2_strategy(recipe_config("v2", 10000), 10)
    monkeypatch.setattr(strategy, "_update_state", lambda *a, **kw: None)
    monkeypatch.setattr(strategy, "_prune_gs", lambda *a: 0)
    monkeypatch.setattr(strategy, "_grow_gs", lambda *a: events.append(("refine", current[0])) or (0, 0))
    p, optimizers, state = parameters()
    current = [0]
    for step in [499, 500, 600, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 10000]:
        current[0] = step
        strategy.step_post_backward(p, optimizers, state, step, {})
    assert events == [("refine", 500), ("refine", 600), ("refine", 2900),
                      ("refine", 3000), ("reset", 3000), ("refine", 3500)]


def test_screen_pruning_can_be_disabled_explicitly():
    for enabled in (False, True):
        p, optimizers, state = parameters()
        state["radii"][0] = .2
        config = recipe_config("v2", 10000, {"refine_scale2d_stop_iter": 10000 if enabled else 0})
        strategy = make_v2_strategy(config, 10)
        count = strategy._prune_gs(p, optimizers, state, 3500)
        assert count == (1 if enabled else 0)


def test_observed_reset_cap_clamps_alpha_and_clears_only_opacity_moments(monkeypatch):
    p, optimizers, state = parameters()
    alpha = torch.tensor([.8, .01, .6, .001])
    with torch.no_grad():
        p["opacities"].copy_(torch.logit(alpha))
    means_moment = optimizers["means"].state[p["means"]]["exp_avg"].clone()
    config = recipe_config("v2", 10000, {"prune_opa": .02, "opacity_reset_multiplier": 1.5,
        "reset_stop_margin": 1500, "refine_start_iter": 99999, "refine_stop_iter": 15000})
    strategy = make_v2_strategy(config, 10)
    monkeypatch.setattr(strategy, "_update_state", lambda *a, **kw: None)
    strategy.step_post_backward(p, optimizers, state, 6000, {})
    torch.testing.assert_close(p["opacities"].sigmoid(), alpha.clamp_max(.03))
    opacity_state = optimizers["opacities"].state[p["opacities"]]
    assert not opacity_state["exp_avg"].any() and not opacity_state["exp_avg_sq"].any()
    assert int(opacity_state["step"]) == 1
    torch.testing.assert_close(optimizers["means"].state[p["means"]]["exp_avg"], means_moment)
    with torch.no_grad():
        p["opacities"].copy_(torch.logit(alpha))
    strategy.step_post_backward(p, optimizers, state, 9000, {})
    torch.testing.assert_close(p["opacities"].sigmoid(), alpha)  # Too close to the10k horizon.


def test_zero_reset_margin_preserves_v2_extended_horizon_behavior(monkeypatch):
    import gsplat.strategy.ops as ops
    calls = []
    monkeypatch.setattr(ops, "reset_opa", lambda **kwargs: calls.append(kwargs["value"]))
    config = recipe_config("v2", 10000, {"refine_start_iter": 99999, "refine_stop_iter": 15000})
    strategy = make_v2_strategy(config, 10)
    monkeypatch.setattr(strategy, "_update_state", lambda *a, **kw: None)
    p, optimizers, state = parameters()
    strategy.step_post_backward(p, optimizers, state, 12000, {})
    assert calls == [.01]


def test_reset_tail_margin_includes_exact_half_period_boundary(monkeypatch):
    import gsplat.strategy.ops as ops
    calls = []
    monkeypatch.setattr(ops, "reset_opa", lambda **kwargs: calls.append(kwargs["value"]))
    config = recipe_config("v2", 4500, {"refine_start_iter": 99999, "reset_stop_margin": 1500})
    strategy = make_v2_strategy(config, 10)
    monkeypatch.setattr(strategy, "_update_state", lambda *a, **kw: None)
    p, optimizers, state = parameters()
    strategy.step_post_backward(p, optimizers, state, 3000, {})
    assert calls == [.01]  # 3000 + 1500 == 4500 is allowed by the native caller.


@pytest.mark.parametrize("reserved_gib,allocated_gib,expected", [(4, 1, False), (5, 4, False), (5, 3, True)])
def test_cache_trim_is_pressure_only_and_preserves_rng(monkeypatch, reserved_gib, allocated_gib, expected):
    calls = []
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: reserved_gib * 2 ** 30)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: allocated_gib * 2 ** 30)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(True))
    before = torch.get_rng_state().clone()
    config = recipe_config("v2", 10000)
    result = trim_cuda_cache_if_needed(config, "cuda:0")
    assert bool(calls) is expected and (result is not None) is expected
    torch.testing.assert_close(torch.get_rng_state(), before)
    config["cache_trim_enabled"] = False
    assert trim_cuda_cache_if_needed(config, "cuda:0") is None
    assert trim_cuda_cache_if_needed({}, "cpu") is None


def test_global_guard_uses_only_explicit_training_geometry_and_never_mutates():
    p, _, _ = parameters()
    before = {key: value.detach().clone() for key, value in p.items()}
    calls = []
    camera = {"K": np.eye(3).tolist(), "w2c": np.eye(4).tolist(), "width": 100, "height": 80}
    cameras = [dict(camera), dict(camera)]
    cameras[0]["w2c"] = (np.eye(4) * 2).tolist()
    def projection(**kwargs):
        calls.append(kwargs)
        return (torch.tensor([[[20, 20], [0, 0], [5, 5], [10, 10]]]),)
    mask, diagnostics = training_camera_guard(p, cameras, [1], projector=projection)
    assert mask.tolist() == [True, False, False, False]
    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["viewmats"][0], torch.eye(4))
    assert diagnostics["training_camera_indices"] == [1]
    assert diagnostics["heldout_RGB_used"] is False
    for key in p:
        torch.testing.assert_close(p[key], before[key])
