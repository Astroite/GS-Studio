import pytest


def test_local_strategy_repairs_upstream_opacity_reset_condition(monkeypatch):
    pytest.importorskip("torch")
    import gsplat.strategy.ops as ops
    from local_postshot.training import make_strategy
    calls=[]
    monkeypatch.setattr(ops,"reset_opa",lambda **kwargs:calls.append(kwargs["value"]))
    strategy=make_strategy({"refine_start_iter":100,"refine_stop_iter":20,
                            "refine_every":1000,"reset_every":3},100)
    monkeypatch.setattr(strategy,"_update_state",lambda *a,**kw:None)
    for step in range(10):strategy.step_post_backward({}, {}, {}, step, {})
    assert calls==[.01,.01,.01]


def test_growth_budget_selects_largest_projected_gradients(monkeypatch):
    torch=pytest.importorskip("torch")
    from gsplat import DefaultStrategy
    from local_postshot.training import make_strategy
    seen=[]
    def capture(self,params,optimizers,state,step):
        seen.append(state["grad2d"].clone())
        return 0,0
    monkeypatch.setattr(DefaultStrategy,"_grow_gs",capture)
    strategy=make_strategy({},7)
    state={"grad2d":torch.tensor([1.,4.,2.,5.,3.]),"count":torch.ones(5)}
    strategy._grow_gs({"means":torch.zeros((5,3))},{},state,100)
    torch.testing.assert_close(seen[0],torch.tensor([0.,4.,0.,5.,0.]))
