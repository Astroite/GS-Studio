import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from local_postshot.splat3_aa import compensation, AA_EPSILON, NATIVE_BLUR

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('round4_aa_oracle',ROOT/'tests/aa_oracle.py')
oracle=importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


CASES=[
    (2.,.3,1.),          # regular positive definite
    (.001,0.,.001),      # numerator near its exact binary32 floor
    (0.,0.,0.),          # zero covariance, numerator floor
    (1e-8,0.,1e-8),      # PSD, numerator floor
    (100.,0.,0.),        # rank one PSD, ratio floor
    (1.,1.,1.),          # rotated rank one PSD
    (-.299,0.,-.299),    # nonphysical: denominator near floor, diagnostic only
    (-.3,0.,-.3),        # nonphysical: both determinants not PSD, denominator floor
    (-.4,0.,1.),         # nonphysical: both determinant floors
]


@pytest.mark.parametrize('raw',CASES)
def test_compensation_and_all_floor_vjps_match_literal_ptx_algebra(raw):
    x=torch.tensor(raw,dtype=torch.float64,requires_grad=True)
    y=compensation(x)
    y.backward(torch.tensor(.7,dtype=x.dtype))
    assert float(y.detach())==pytest.approx(oracle.forward(raw)['compensation'],rel=2e-14)
    np.testing.assert_allclose(x.grad.numpy(),oracle.vjp(raw,.7),rtol=1e-12,atol=2e-8)
    np.testing.assert_allclose(oracle.analytic(raw,.7),oracle.vjp(raw,.7),rtol=1e-12,atol=2e-8)


def test_regular_region_matches_mathematical_finite_difference():
    for raw in ([2.,.3,1.],[.004,.001,.006],[1.2,-.8,2.]):
        raw=np.array(raw)
        step=1e-7
        numeric=np.array([(oracle.forward(raw+np.eye(3)[i]*step)['compensation']-
                           oracle.forward(raw-np.eye(3)[i]*step)['compensation'])/(2*step) for i in range(3)])
        np.testing.assert_allclose(numeric,oracle.vjp(raw),rtol=5e-7,atol=1e-8)


def test_floored_backward_intentionally_differs_from_clamp_autograd():
    raw=torch.tensor([1e-8,0.,1e-8],dtype=torch.float64,requires_grad=True)
    comp=compensation(raw);comp.backward();native=raw.grad.clone()
    raw.grad=None
    a,b,c=raw
    ordinary=((a*c-b*b).clamp_min(AA_EPSILON)/((a+NATIVE_BLUR)*(c+NATIVE_BLUR)-b*b).clamp_min(AA_EPSILON)).clamp_min(AA_EPSILON).sqrt()
    ordinary.backward()
    assert not torch.allclose(native,raw.grad,rtol=1e-4,atol=1e-6)
    # Even the final ratio floor is straight through (ordinary gradient is 0).
    rank_one=torch.tensor([100.,0.,0.],dtype=torch.float64,requires_grad=True)
    compensation(rank_one).backward()
    assert abs(float(rank_one.grad[2]))>100


def test_regular_native_vjp_has_no_public_extra_epsilon():
    for raw in ([2.,.3,1.],[.004,.001,.006]):
        public_comp,public=oracle.public_vjp(raw)
        native=oracle.vjp(raw)
        np.testing.assert_allclose(public,native*public_comp/(public_comp+1e-6),rtol=1e-11,atol=1e-12)


def test_psd_determinant_floors_cannot_reach_standard_alpha_support():
    # detBlur >= .3^2 for PSD raw C. With alpha<=1, detRaw-floor
    # and ratio-floor regions cannot exceed 1/255. Never infer this for
    # global opacity multipliers>1, debug modes or nonphysical covariance.
    assert np.sqrt(AA_EPSILON)/(NATIVE_BLUR)<np.float32(1/255)
    assert np.sqrt(AA_EPSILON)<np.float32(1/255)
    for raw in ([0,0,0],[1e-8,0,1e-8],[100,0,0],[1,1,1]):
        assert oracle.forward(raw)['compensation']<np.float32(1/255)


def test_batched_tensor_gradients_and_no_cuda_initialization():
    before=torch.cuda.is_initialized()
    raw=torch.tensor([[2.,.3,1.],[100.,0.,0.]],dtype=torch.float64,requires_grad=True)
    compensation(raw).sum().backward()
    assert raw.grad.shape==raw.shape and torch.isfinite(raw.grad).all()
    assert torch.cuda.is_initialized()==before


@pytest.mark.parametrize('shape,dtype',[((),torch.float32),((2,2),torch.float32),((3,),torch.int32)])
def test_reject_invalid_covariance_metadata(shape,dtype):
    with pytest.raises(ValueError):compensation(torch.zeros(shape,dtype=dtype))


def test_float32_witness_forward_and_backward_remain_finite_on_physical_cases():
    for raw in CASES[:6]:
        assert np.isfinite(oracle.forward(raw,binary32=True)['compensation'])
        assert np.isfinite(oracle.vjp(raw,binary32=True)).all()
