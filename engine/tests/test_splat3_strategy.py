"""Policy contracts from retained current-cache native dataflow, CPU only."""
import math
import numpy as np
import pytest
import torch
from local_postshot.splat3_strategy import growth_budget,select_parents,split_selected,clear_completed_windows,compact_dead,reset_opacity_before_adam


def parameters():
    values={"means":torch.tensor([[0.,0.,0.],[4.,5.,6.],[9.,8.,7.]]),
            "scales":torch.tensor([[2.,3.,4.],[1.,2.,3.],[3.,2.,1.]]).log(),
            "quats":torch.tensor([[1.,0.,0.,0.],[math.sqrt(.5),0.,0.,math.sqrt(.5)],[1.,0.,0.,0.]]),
            "opacities":torch.logit(torch.tensor([.1,.5,.9])),
            "sh0":torch.arange(9,dtype=torch.float32).reshape(3,1,3),
            "shN":torch.ones((3,3,3))}
    return torch.nn.ParameterDict({k:torch.nn.Parameter(v) for k,v in values.items()})


def test_split_preserves_original_child_rotates_noise_and_resets_both_moments():
    p=parameters();before={k:v.detach().clone() for k,v in p.items()}
    opts={k:torch.optim.Adam([v],lr=.01) for k,v in p.items()}
    for name,opt in opts.items():
        opt.state[p[name]]={"step":torch.tensor(7.),"exp_avg":torch.ones_like(p[name]),"exp_avg_sq":torch.full_like(p[name],2.)}
        p[name].grad=torch.full_like(p[name],1.25)
    state={"sum1":torch.tensor([5.,6.,7.]),"sum2":torch.tensor([12.,15.,19.]),
           "max_weight":torch.tensor([.1,.2,.3]),"min_scale":torch.tensor([.01,.02,.03]),"scene_scale":4.}
    split_selected(p,opts,state,torch.tensor([1]),noise=torch.tensor([[1.,0.,0.]]))
    torch.testing.assert_close(p["means"][1],torch.tensor([4.,6.,6.]),atol=1e-6,rtol=0)
    torch.testing.assert_close(p["means"][3],before["means"][1],atol=0,rtol=0)
    for name in ("scales","opacities"):
        torch.testing.assert_close(p[name][1],p[name][3],atol=0,rtol=0)
    torch.testing.assert_close(p["scales"][[1,3]].exp(),torch.tensor([[.625,1.25,1.875]]).expand(2,3))
    torch.testing.assert_close(p["opacities"][[1,3]].sigmoid(),torch.tensor([.4,.4]))
    for name in ("quats","sh0","shN"):
        assert torch.equal(p[name][1],before[name][1]) and torch.equal(p[name][3],before[name][1])
    for name,opt in opts.items():
        assert opt.param_groups[0]["params"][0] is p[name]
        assert opt.state[p[name]]["step"]==7
        for field,value in (("exp_avg",1.),("exp_avg_sq",2.)):
            moment=opt.state[p[name]][field]
            assert torch.all(moment[[0,2]]==value) and torch.all(moment[[1,3]]==0)
        assert torch.equal(p[name][[0,2]],before[name][[0,2]])
        assert torch.all(p[name].grad[[0,2]]==1.25) and torch.all(p[name].grad[[1,3]]==0)
    assert state["sum2"].tolist()==[12.,-20.,19.,-20.]
    assert state["min_scale"].tolist()==pytest.approx([.01,.02,.03,.02])
    assert state["max_weight"].tolist()==pytest.approx([.1,0,.3,0])


@pytest.mark.parametrize("active,maximum",[(0,1000),(81615,1000000),(500000,1000000),(999999,1000000),(1000000,1000000),(1000001,1000000)])
def test_growth_budget_has_hard_cap_and_known_saturation(active,maximum):
    b=growth_budget(active,maximum)
    assert 0<=b<=max(maximum-active,0)
    if active<maximum:assert b>=min(maximum//200,maximum-active)
    if active==0:assert b==19  # host float32 1.01 widened, then truncation
    if active==81615:assert b==19066


def test_maturity_and_cdf_sampling_are_distinct_from_gradient_topk():
    # Uniform .5 opacity: draws at the CDF boundaries select IDs0 and8,
    # not a forced top-k. New/warming-up ages and dead anchors are ineligible.
    n=10
    args=[torch.full((n,),2.),torch.full((n,),10.),torch.zeros(n),torch.ones(n)]
    budget=growth_budget(n,100)
    assert budget==2
    selected,info=select_parents(*args,maximum=100,uniforms=torch.tensor([.05,.85]))
    assert selected.tolist()==[0,8] and info["draws"]==2
    args[0].fill_(2.)  # A fresh mature statistic; selection normalizes in place.
    args[1][8]=-20;args[3][9]=-1
    selected,info=select_parents(*args,maximum=100,uniforms=torch.tensor([.99,.99]))
    assert selected.tolist()==[7]


def test_stat_window_preserves_only_short_positive_window_and_negative_cooldown():
    state={"sum1":torch.arange(6,dtype=torch.float32)+1,"sum2":torch.tensor([-20.,-1.,0.,1.,9.,10.])}
    clear_completed_windows(state)
    assert state["sum1"].tolist()==[0.,0.,0.,4.,5.,0.]
    assert state["sum2"].tolist()==[-20.,-1.,0.,1.,9.,0.]
    clear_completed_windows(state,reset=True)
    assert state["sum1"].sum()==0 and state["sum2"].tolist()==[-20.,-1.,0.,0.,0.,0.]


def test_pre_adam_compaction_keeps_live_gradients_and_reset_keeps_step():
    p=parameters();opts={k:torch.optim.Adam([v]) for k,v in p.items()}
    for name,opt in opts.items():
        p[name].grad=torch.full_like(p[name],3.)
        opt.state[p[name]]={"step":torch.tensor(7.),"exp_avg":torch.ones_like(p[name]),"exp_avg_sq":torch.full_like(p[name],2.)}
    state={"min_scale":torch.tensor([1.,-1.,2.])}
    compact_dead(p,opts,state,torch.tensor([False,True,False]))
    assert len(p["means"])==2 and torch.all(p["means"].grad==3.)
    assert state["min_scale"].tolist()==[1.,2.]
    reset_opacity_before_adam(p,opts,1.5*5/256)
    assert torch.all(p["opacities"].sigmoid()<=1.5*5/256+1e-7)
    assert torch.all(p["opacities"].grad==0)
    assert opts["opacities"].state[p["opacities"]]["step"]==7
    assert opts["opacities"].state[p["opacities"]]["exp_avg"].sum()==0
