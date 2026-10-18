import copy
import os

import pytest
import torch

from local_postshot.training_state_performance import update_single_camera_state


def metadata(n, visible, *, absgrad=False, device="cpu", seed=17):
    generator = torch.Generator().manual_seed(seed)
    screen = torch.rand((1, n, 2), generator=generator).to(device).requires_grad_()
    gradient = torch.randn((1, n, 2), generator=generator).to(device)
    # Invisible values are ignored by the original gather and must remain so.
    gradient[0, ~visible.to(device)] = float("nan")
    screen.grad = gradient
    if absgrad:
        screen.absgrad = gradient.abs()
    radius = torch.rand((1, n, 2), generator=generator).to(device) * 32 + .1
    radius[0, ~visible.to(device), 0] = 0
    return dict(width=1614, height=1205, n_cameras=1, radii=radius,
                gaussian_ids=None, means2d=screen)


def assert_upstream_equal(device="cpu", *, absgrad=False, track_radii=True, n=79):
    from gsplat import DefaultStrategy
    strategy = DefaultStrategy(absgrad=absgrad, refine_scale2d_stop_iter=1000 if track_radii else 0)
    old, new = strategy.initialize_state(), strategy.initialize_state()
    params = {"means": torch.zeros((n,3), device=device)}
    for step in range(9):
        visible = torch.arange(n) % (step % 4 + 2) != 0
        if step == 0:
            visible[:] = False
        elif step == 1:
            visible[:] = True
        info = metadata(n, visible, absgrad=absgrad, device=device, seed=step)
        before = info["means2d"].grad.clone()
        strategy._update_state(params, old, info)
        assert update_single_camera_state(new, info, absgrad=absgrad, track_radii=track_radii)
        for key in ("grad2d", "count") + (("radii",) if track_radii else ()):
            torch.testing.assert_close(new[key],old[key],rtol=0,atol=0,equal_nan=True)
        torch.testing.assert_close(info["means2d"].grad,before,rtol=0,atol=0,equal_nan=True)
    # State is portable through ordinary checkpoint serialization; no new fields.
    assert set(new)==set(old)
    return new


@pytest.mark.parametrize("absgrad",[False,True])
@pytest.mark.parametrize("track_radii",[False,True])
@pytest.mark.parametrize("n",[0,1,79,1024])
def test_single_camera_dense_update_matches_pinned_upstream_exactly(absgrad,track_radii,n):
    assert_upstream_equal(absgrad=absgrad,track_radii=track_radii,n=n)


def test_unsupported_metadata_falls_back_without_mutation():
    info=metadata(7,torch.ones(7,dtype=torch.bool))
    state={"grad2d":None,"count":None,"radii":None,"scene_scale":1.}
    original=copy.deepcopy(state)
    assert not update_single_camera_state(state,info,packed=True)
    info["n_cameras"]=2
    assert not update_single_camera_state(state,info)
    info["n_cameras"]=1
    info["means2d"]=info["means2d"].double().detach().requires_grad_()
    info["means2d"].grad=torch.ones_like(info["means2d"])
    assert not update_single_camera_state(state,info)
    assert state==original


def test_stale_state_shape_fails_before_mutation():
    info=metadata(7,torch.ones(7,dtype=torch.bool))
    state={"grad2d":torch.zeros(6),"count":None,"radii":None}
    with pytest.raises(ValueError,match="differs"):
        update_single_camera_state(state,info)
    assert state["count"] is state["radii"] is None


def test_integer_raster_radii_and_resumed_state_match_upstream_exactly():
    from gsplat import DefaultStrategy
    strategy=DefaultStrategy(refine_scale2d_stop_iter=1000)
    visible=torch.tensor([True,False,True,True,False,True,False])
    info=metadata(7,visible)
    info["radii"]=info["radii"].ceil().to(torch.int32)
    old={"grad2d":torch.linspace(.1,.7,7),"count":torch.arange(7,dtype=torch.float32),
         "radii":torch.linspace(.001,.007,7),"scene_scale":5.6}
    new=copy.deepcopy(old)
    identities={key:id(new[key]) for key in ("grad2d","count","radii")}
    strategy._update_state({"means":torch.zeros(7,3)},old,info)
    assert update_single_camera_state(new,info)
    for key in identities:
        assert id(new[key])==identities[key]
        torch.testing.assert_close(new[key],old[key],rtol=0,atol=0)


@pytest.mark.skipif(os.environ.get("LOCAL_POSTSHOT_GPU_TESTS")!="1", reason="Opt-in CUDA equivalence audit")
@pytest.mark.parametrize("absgrad",[False,True])
def test_cuda_single_camera_dense_update_matches_upstream_exactly(absgrad):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    assert_upstream_equal("cuda",absgrad=absgrad,n=100003)
