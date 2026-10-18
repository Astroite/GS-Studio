"""Private fused native AA CUDA checks. CPU oracle is independently PTX-bound."""
import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from local_postshot.splat3_projection import fully_fused_projection, _extension, projection_identity

GPU=pytest.mark.skipif(os.environ.get('LOCAL_POSTSHOT_GPU_TESTS')!='1',reason='Opt-in private native AA CUDA audit')

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('round4_aa_gpu_oracle',ROOT/'tests/aa_oracle.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)


def parameters(n=3):
    means=torch.tensor([[.2,.1,3.],[-.3,.15,4.],[.1,-.2,2.]],device='cuda',dtype=torch.float32)[:n].clone().requires_grad_()
    quats=torch.tensor([[1.,.1,.05,.02],[1.,0.,.1,.03],[1.,.04,.03,0.]],device='cuda',dtype=torch.float32)[:n].clone().requires_grad_()
    scales=torch.tensor([[.04,.02,.03],[.03,.06,.05],[.02,.025,.03]],device='cuda',dtype=torch.float32)[:n].clone().requires_grad_()
    K=torch.tensor([[[100.,0,32.],[0,105.,32.],[0,0,1.]]],device='cuda',dtype=torch.float32)
    view=torch.eye(4,device='cuda',dtype=torch.float32)[None]
    opacity=torch.tensor([.6,.7,.8],device='cuda',dtype=torch.float32)[:n].clone().requires_grad_()
    return dict(means=means,covars=None,quats=quats,scales=scales,opacities=opacity,
                viewmats=view,Ks=K,width=64,height=64,eps2d=.3,calc_compensations=True)


@GPU
def test_native_covariance_kernel_matches_original_scalar_ptx_including_all_floors():
    rows=[[2.,.3,1.],[.001,0.,.001],[0.,0.,0.],[1e-8,0.,1e-8],[100.,0.,0.],
          [1.,1.,1.],[-.299,0.,-.299],[-.3,0.,-.3],[-.4,0.,1.]]
    raw=torch.tensor(rows,device='cuda',dtype=torch.float32)
    upstream=torch.full((len(rows),),.7,device='cuda')
    comp,gradient=_extension().aa_covariance(raw,upstream,.3)
    expected_c=[oracle.forward(row,binary32=True)['compensation'] for row in raw.cpu().numpy()]
    expected_g=[oracle.vjp(row,float(np.float32(.7)),binary32=True) for row in raw.cpu().numpy()]
    np.testing.assert_allclose(comp.cpu().numpy(),expected_c,rtol=2e-6,atol=1e-9)
    np.testing.assert_allclose(gradient.cpu().numpy(),expected_g,rtol=3e-6,atol=1e-6)


@GPU
def test_regular_fused_forward_geometry_matches_public_and_compensation_oracle():
    from gsplat.cuda._wrapper import fully_fused_projection as public
    params=parameters()
    observed=fully_fused_projection(**params)
    reference=public(**params)
    assert torch.equal(observed[0],reference[0])
    assert bool((observed[0]>0).all())
    for i in range(1,4):torch.testing.assert_close(observed[i],reference[i],rtol=2e-6,atol=3e-6)
    torch.testing.assert_close(observed[4],reference[4],rtol=2e-6,atol=1e-7)
    # Recover only regular raw C from its well-conditioned conic for this
    # independent forward test. Production VJP does not use this shortcut.
    for conic,comp in zip(observed[3][0].detach().cpu().numpy(),observed[4][0].detach().cpu().numpy()):
        q=np.array([[conic[0],conic[1]],[conic[1],conic[2]]],dtype=np.float64)
        covariance=np.linalg.inv(q)-np.eye(2)*oracle.BLUR
        expected=oracle.forward((covariance[0,0],covariance[0,1],covariance[1,1]))['compensation']
        assert comp==pytest.approx(expected,rel=2e-6,abs=1e-7)


@GPU
def test_backward_all_parameter_blocks_match_finite_difference_in_regular_region():
    params=parameters(n=1)
    def loss():
        _,xy,z,conic,comp=fully_fused_projection(**params)
        # GPU projection remains float32. Accumulate the scalar test loss in
        # float64 so subtracting +/- h does not measure float32 sum rounding.
        return (xy.double()*.003).sum()+z.double().sum()*.04+conic.double().sum()*.01+comp.double().sum()*.2
    value=loss();value.backward()
    for name,step,rtol,atol in [('means',1e-3,.006,1e-5),('scales',1e-4,.006,.002),('quats',2e-3,.02,1e-5)]:
        tensor=params[name]
        saved=tensor.detach().clone();gradient=tensor.grad.detach().clone()
        for component in range(tensor.shape[-1]):
            with torch.no_grad():tensor.copy_(saved);tensor[0,component]+=step
            plus=float(loss().detach())
            with torch.no_grad():tensor.copy_(saved);tensor[0,component]-=step
            minus=float(loss().detach())
            numeric=(plus-minus)/(2*step)
            assert float(gradient[0,component])==pytest.approx(numeric,rel=rtol,abs=atol),(name,component)
        with torch.no_grad():tensor.copy_(saved)


@GPU
def test_compensation_upstream_not_materialized_and_invalid_rows_have_zero_gradient():
    params=parameters()
    with torch.no_grad():params['means'][1,2]=-.5;params['scales'][2]=1e-6
    radii,xy,z,conic,comp=fully_fused_projection(**params)
    assert torch.equal(radii[0,1:],torch.zeros_like(radii[0,1:]))
    assert torch.equal(comp[0,1:],torch.zeros_like(comp[0,1:]))
    # Gradient only for compensation; means/conics/depth upstream=None.
    comp.sum().backward()
    for name in ('means','quats','scales'):
        assert torch.isfinite(params[name].grad).all()
        assert torch.equal(params[name].grad[1:],torch.zeros_like(params[name].grad[1:]))


@GPU
def test_empty_projection_roundtrip():
    params=parameters(0)
    radii,xy,z,conic,comp=fully_fused_projection(**params)
    assert radii.shape==(1,0,2) and comp.shape==(1,0)
    (xy.sum()+z.sum()+conic.sum()+comp.sum()).backward()
    assert params['means'].grad.shape==(0,3)


@GPU
def test_explicit_opacity_chain_has_finite_gradient_and_preserves_support_semantics():
    params=parameters()
    radii,xy,z,conic,comp=fully_fused_projection(**params)
    (params['opacities'][None]*comp).sum().backward()
    torch.testing.assert_close(params['opacities'].grad,comp.detach()[0])
    assert torch.isfinite(params['scales'].grad).all()
    assert bool((radii>0).all())


@GPU
def test_native_boundary_rejects_shape_dtype_and_camera_gradient():
    params=parameters()
    with pytest.raises(ValueError):fully_fused_projection(**{**params,'packed':True})
    with pytest.raises(ValueError):fully_fused_projection(**{**params,'calc_compensations':False})
    with pytest.raises(ValueError):fully_fused_projection(**{**params,'viewmats':params['viewmats'].requires_grad_()})
    params['viewmats'].requires_grad_(False)
    with pytest.raises(RuntimeError):fully_fused_projection(**{**params,'scales':params['scales'].double()})
    with pytest.raises(RuntimeError):fully_fused_projection(**{**params,'quats':params['quats'][:,:3]})


def test_projection_identity_does_not_initialize_cuda():
    before=torch.cuda.is_initialized()
    identity=projection_identity()
    assert identity['distribution']=='local-splat3-projection' and len(identity['binary_sha256'])==64
    assert torch.cuda.is_initialized()==before
