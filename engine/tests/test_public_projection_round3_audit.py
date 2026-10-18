import importlib.util
from pathlib import Path

import torch

path=Path(__file__).resolve().parents[1]/'scripts/audit_public_projection_round3.py'
spec=importlib.util.spec_from_file_location('projection_audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_compensation_formula_normal_regime_matches_public_torch_source():
    from gsplat.cuda._torch_impl import _fully_fused_projection
    cov=torch.diag_embed(torch.tensor([[.01,.02,.03]],dtype=torch.float64))
    means=torch.tensor([[0.,0.,3.]],dtype=torch.float64)
    K=torch.tensor([[[100.,0.,50.],[0.,100.,40.],[0.,0.,1.]]],dtype=torch.float64)
    output=_fully_fused_projection(means,cov,torch.eye(4,dtype=torch.float64)[None],K,100,80,
                                  eps2d=.3,calc_compensations=True)
    raw=torch.diag_embed(torch.tensor([[(100/3)**2*.01,(100/3)**2*.02]],dtype=torch.float64))
    torch.testing.assert_close(audit.compensation2d(raw),output[4][0],rtol=1e-12,atol=1e-12)


def test_floor_and_public_compensation_differ_for_tiny_gaussian():
    raw=torch.eye(2,dtype=torch.float64)*1e-10
    public=audit.compensation2d(raw)
    floor=audit.compensation2d(raw,policy='observed_floor')
    assert public<1e-8 and floor>.003
    assert floor/public>1e6


def test_conic_reconstruction_has_correct_values_and_gradients_in_double():
    raw=torch.tensor([[.02,.003],[.003,.04]],dtype=torch.float64,requires_grad=True)
    blurred=raw+torch.eye(2,dtype=torch.float64)*.3
    inverse=torch.linalg.inv(blurred)
    conic=torch.stack([inverse[0,0],inverse[0,1],inverse[1,1]])
    recovered=audit.floor_compensation_from_conics(conic)
    expected=audit.compensation2d(raw,policy='observed_floor')
    # Symmetric independent entries: tie the two off-diagonal values so the
    # conic3 representation and full matrix have the same degrees of freedom.
    torch.testing.assert_close(recovered,expected,rtol=1e-12,atol=1e-12)
    def function(values):
        x,y,z=values.unbind()
        covariance=torch.stack([x,y,y,z]).reshape(2,2)
        inv=torch.linalg.inv(covariance+torch.eye(2,dtype=torch.float64)*.3)
        return audit.floor_compensation_from_conics(torch.stack([inv[0,0],inv[0,1],inv[1,1]]))
    assert torch.autograd.gradcheck(function,(torch.tensor([.02,.003,.04],dtype=torch.float64,requires_grad=True),))


def test_dense_hidden_rows_move_and_projected_support_is_not_contribution():
    before=torch.cuda.is_initialized()
    result=audit.cpu_diagnostics()
    assert result['cuda_initialized']==before
    assert result['cuda_state_unchanged'] is True
    assert result['dense_adam_zero_gradient']['hidden_parameter_delta']<0
    assert abs(result['dense_adam_zero_gradient']['hidden_moment_decay_ratio']-.9)<1e-12
    assert result['selective_adam_all_visible_first_step']['ratio']>1
    example=result['support_not_contribution_example']
    assert example['bbox_overlaps_image'] and example['pixel_contribution_zero']


def test_recovered_near_uses_width_for_both_focal_axes():
    camera={'width':1000,'height':500,'K':[[500,0,500],[0,1500,250],[0,0,1]]}
    assert abs(audit.observed_near_double_reference(camera)-.216)<1e-15
    camera['K'][1][1]=50;camera['K'][0][0]=50
    assert audit.observed_near_double_reference(camera)==.1
    camera['K'][0][0]=10000
    assert audit.observed_near_double_reference(camera)==.3
