"""CPU mathematical verification; no original CUDA kernel is executed."""
import numpy as np
import pytest
import torch

from local_postshot.splat3_regularization import (add_regularization_gradient_, compute_min_scale_anchors,
    fast_add_regularization_gradient_,
    regularization_gradient, regularization_loss, scale_anchor_gradient, scale_anchor_potential)


def test_asymmetric_scale_anchor_gradient_matches_observed_piecewise_chain_rule():
    physical = torch.tensor([[.5, 1., 2.], [.25, .5, .75], [1., 2., 3.]], dtype=torch.float64)
    logs = physical.log().requires_grad_()
    anchor = torch.tensor([1., .5, -1.], dtype=torch.float64, requires_grad=True)
    visible = torch.tensor([True, True, True])
    loss = scale_anchor_potential(logs, anchor, visible)
    loss.backward()
    expected = physical * torch.tensor([[-.05, 0, .05], [-.025, 0, .0125], [0., 0, 0]], dtype=torch.float64)
    torch.testing.assert_close(logs.grad, expected, atol=1e-16, rtol=1e-15)
    torch.testing.assert_close(scale_anchor_gradient(logs, anchor, visible), expected, atol=1e-16, rtol=1e-15)
    assert anchor.grad is None


def test_potential_reverse_gradient_matches_independent_central_finite_differences():
    rng = np.random.default_rng(847)
    logs = torch.tensor(rng.normal(-1, .5, (7, 3)), dtype=torch.float64, requires_grad=True)
    anchors = torch.tensor(rng.uniform(.05, .2, 7), dtype=torch.float64)
    visible = torch.tensor([True, True, False, True, False, True, True])
    loss = scale_anchor_potential(logs, anchors, visible, upper_weight=.05)
    loss.backward()
    finite = torch.zeros_like(logs)
    step = 1e-6
    for row in range(len(logs)):
        for column in range(3):
            plus, minus = logs.detach().clone(), logs.detach().clone()
            plus[row, column] += step
            minus[row, column] -= step
            finite[row, column] = (scale_anchor_potential(plus, anchors, visible)
                                   - scale_anchor_potential(minus, anchors, visible)) / (2 * step)
    torch.testing.assert_close(logs.grad, finite, atol=2e-11, rtol=1e-7)
    torch.testing.assert_close(logs.grad, scale_anchor_gradient(logs, anchors, visible), atol=1e-16, rtol=1e-15)


def test_no_count_normalization_and_unobserved_rows_have_zero_gradient():
    logs = torch.tensor([[0., -.5, -1.]], dtype=torch.float64, requires_grad=True)
    first = scale_anchor_potential(logs, torch.tensor([.2]), torch.tensor([True]))
    repeated = logs.detach().repeat(2, 1).requires_grad_()
    doubled = scale_anchor_potential(repeated, torch.tensor([.2, .2]), torch.tensor([True, True]))
    assert doubled.item() == pytest.approx(2 * first.item(), abs=1e-16)
    first.backward();doubled.backward()
    torch.testing.assert_close(repeated.grad, logs.grad.repeat(2, 1), atol=0, rtol=0)
    empty = scale_anchor_potential(repeated, torch.tensor([.2, -1.]), torch.tensor([False, True]))
    assert empty.item() == 0
    torch.testing.assert_close(scale_anchor_gradient(repeated, torch.tensor([.2, -1.]), torch.tensor([False, True])), torch.zeros_like(repeated))


def test_detached_anchor_does_not_introduce_position_or_opacity_derivatives():
    geometry = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
    anchor = geometry / 100
    logs = torch.tensor([[-3., -4., -5.]], dtype=torch.float64, requires_grad=True)
    scale_anchor_potential(logs, anchor, torch.tensor([True])).backward()
    assert geometry.grad is None and anchor.grad_fn is not None
    assert logs.grad is not None and torch.isfinite(logs.grad).all()


def test_zero_upper_weight_keeps_the_observed_fixed_lower_anchor_pressure():
    logs = torch.tensor([[np.log(.5), np.log(1.), np.log(2.)]], dtype=torch.float64)
    gradient = scale_anchor_gradient(logs, torch.tensor([1.]), torch.tensor([True]), upper_weight=0)
    torch.testing.assert_close(gradient, torch.tensor([[-.025, 0, 0]], dtype=torch.float64), atol=0, rtol=0)


def test_explicit_gradient_injection_equals_loss_backward_without_mutating_values_or_rng():
    logs = torch.nn.Parameter(torch.tensor([[-1., -2., -3.], [-4., -5., -6.]], dtype=torch.float64))
    anchors = torch.tensor([.05, .02], dtype=torch.float64)
    visible = torch.tensor([True, False])
    before = logs.detach().clone()
    rng = torch.get_rng_state().clone()
    regularization_loss(logs, anchors, visible).backward()
    expected = logs.grad.clone()
    torch.testing.assert_close(expected, regularization_gradient(logs, anchors, visible), atol=1e-17, rtol=1e-15)
    logs.grad = torch.full_like(logs, .2)
    add_regularization_gradient_(logs, anchors, visible)
    torch.testing.assert_close(logs.grad, expected + .2, atol=0, rtol=0)
    logs.grad = None
    add_regularization_gradient_(logs, anchors, visible)
    torch.testing.assert_close(logs.grad, expected, atol=1e-17, rtol=1e-15)
    torch.testing.assert_close(logs, before, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_fast_gradient_injection_never_reads_a_tensor_scalar(monkeypatch):
    logs = torch.nn.Parameter(torch.tensor([[-1., -2., -3.], [-4., -5., -6.]], dtype=torch.float64))
    anchors = torch.tensor([.05, .02], dtype=torch.float64)
    visible = torch.tensor([True, False])
    expected = regularization_gradient(logs, anchors, visible)
    def forbidden(*args, **kwargs):
        raise AssertionError("Fast production gradient must not synchronize by reading tensor scalars")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        fast_add_regularization_gradient_(logs, anchors, visible)
        fast_add_regularization_gradient_(logs, anchors, visible)
    torch.testing.assert_close(logs.grad, 2 * expected, atol=1e-17, rtol=1e-15)


def test_anchor_minimum_uses_only_training_geometry_with_explicit_public_visibility_stub():
    means = torch.tensor([[0, 0, 1.], [0, 0, 2.], [0, 0, .05], [0, 0, -1.], [0, 0, 3.]], dtype=torch.float64, requires_grad=True)
    params = {"means": means, "scales": torch.full((5, 3), -3., dtype=torch.float64),
        "quats": torch.tensor([[1., 0, 0, 0]], dtype=torch.float64).repeat(5, 1),
        "opacities": torch.zeros(5, dtype=torch.float64)}
    base = {"w2c": torch.eye(4).tolist(), "width": 100, "height": 80}
    cameras = [{**base, "K": [[999999, 0, 50], [0, 999999, 40], [0, 0, 1]], "name": "HOLDOUT_NEVER_USED"},
               {**base, "K": [[100, 0, 50], [0, 200, 40], [0, 0, 1]]},
               {**base, "K": [[400, 0, 50], [0, 100, 40], [0, 0, 1]]}]
    cameras[2]["w2c"] = torch.eye(4).tolist()
    cameras[2]["w2c"][2][3] = 1.
    calls = []
    def projection(**kwargs):
        calls.append(kwargs)
        focal = kwargs["Ks"][0, 0, 0].item()
        assert focal in (100, 400)
        radius = [[1, 1], [1, 1], [0, 0], [1, 1], [0, 0] if focal == 100 else [1, 1]]
        return (torch.tensor([radius]),)
    anchors = compute_min_scale_anchors(params, cameras, [1, 2], projector=projection)
    torch.testing.assert_close(anchors, torch.tensor([.00125, .001875, -1, -1, .0025], dtype=torch.float64), atol=1e-18, rtol=1e-15)
    assert len(calls) == 2 and anchors.grad_fn is None
    assert all(call["near_plane"] == .1 and call["eps2d"] == .3 for call in calls)
    assert all(call["packed"] is False for call in calls)
    torch.testing.assert_close(means, params["means"], atol=0, rtol=0)
    with pytest.raises(ValueError, match="training-camera"):
        compute_min_scale_anchors(params, cameras, [1, 1], projector=projection)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        compute_min_scale_anchors(params, cameras, [1, 2])


@pytest.mark.parametrize("anchors,visible", [([.1, .2], [True]), ([-.5], [True]), ([float('nan')], [True]), ([.1], [1])])
def test_invalid_or_implicit_visibility_is_rejected(anchors, visible):
    with pytest.raises(ValueError):
        scale_anchor_potential(torch.zeros((1, 3), dtype=torch.float64), anchors, visible)
