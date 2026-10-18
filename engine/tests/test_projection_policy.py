import math
import numpy as np
import pytest
from local_postshot.projection_policy import (validated_projection,native_training_near,
    position_spatial_factor,projection_kwargs,position_scheduler)
from local_postshot.training_recipe import recipe_config,recipe_from_checkpoint


def camera(x=0.,fx=800.,fy=800.,width=1000,height=800):
    w2c=np.eye(4);w2c[0,3]=-x
    return {"w2c":w2c.tolist(),"K":[[fx,0,width/2],[0,fy,height/2],[0,0,1]],"width":width,"height":height}


def test_focal_near_uses_width_for_both_axes_and_native_clamp():
    a=camera(fx=800,fy=1200)
    near=native_training_near(a)
    assert near==pytest.approx(.1728,abs=2e-8)
    assert native_training_near({**a,"height":1200})==pytest.approx(near,abs=2e-8)
    assert native_training_near(camera(fx=10,fy=10))==float(np.float32(.1))
    assert native_training_near(camera(fx=10000,fy=10000))==float(np.float32(.3))


def test_training_camera_clip_and_gui_clip_are_separate_and_finite_json_policy():
    p=validated_projection({"train_near_mode":"native_focal_v1","eval_near_plane":.1,
                            "far_mode":"unbounded_perspective"})
    a=camera(fx=1200,fy=1200)
    assert projection_kwargs(a,p,training=True)["near_plane"]==native_training_near(a)
    assert projection_kwargs(a,p)["near_plane"]==.1
    assert math.isinf(projection_kwargs(a,p)["far_plane"])
    assert all(not isinstance(v,float) or math.isfinite(v) for v in p.values())
    assert projection_kwargs(a,None)=={"near_plane":.01,"far_plane":1e8}


def test_native_position_context_uses_train_mean_and_normalized_focal():
    cams=[camera(-1),camera(1),camera(1000)]
    expected=float(np.float32(1.1))
    assert position_spatial_factor(cams,[0,1],"native_train_mean_v1")==expected
    cams[2]=camera(-1000)
    assert position_spatial_factor(cams,[0,1],"native_train_mean_v1")==expected
    assert position_spatial_factor(cams,[0,1])>900
    cams=[camera(-1,fx=2000),camera(1,fx=2000)]
    assert position_spatial_factor(cams,[0,1],"native_train_mean_v1")==expected/2


def test_explicit_recipe_freezes_projection_and_rate_context_without_enriching_old_checkpoint():
    config=recipe_config("observed_v4",10000,{"projection":{"train_near_mode":"native_focal_v1","eval_near_plane":.1},
        "position_lr_horizon":30000,"position_lr_spatial":"native_train_mean_v1"},train_image_count=112)
    assert config["position_lr_horizon"]==30000
    assert config["projection"]["train_near_mode"]=="native_focal_v1"
    assert config["projection"]["rasterize_mode"]=="classic"
    old={key:value for key,value in config.items() if key not in ("projection","position_lr_horizon","position_lr_spatial")}
    restored=recipe_from_checkpoint({"training_recipe":old})
    assert restored==old and "projection" not in restored and "position_lr_horizon" not in restored


def test_native_position_schedule_first_update_horizon_clamp_and_state_restore():
    import torch
    p=torch.nn.Parameter(torch.tensor([0.]))
    opt=torch.optim.Adam([p],lr=.01)
    sched=position_scheduler(opt,3,"completed_step_native")
    rates=[]
    for _ in range(5):
        rates.append(opt.param_groups[0]["lr"])
        p.grad=torch.ones_like(p);opt.step();sched.step()
        if len(rates)==2:
            import copy
            saved=(copy.deepcopy(opt.state_dict()),copy.deepcopy(sched.state_dict()))
    assert rates==pytest.approx([.01*(.01**(1/3)),.01*(.01**(2/3)),.0001,.0001,.0001])
    q=torch.nn.Parameter(torch.tensor([0.]))
    opt2=torch.optim.Adam([q],lr=.01);sched2=position_scheduler(opt2,3,"completed_step_native")
    opt2.load_state_dict(saved[0]);sched2.load_state_dict(saved[1])
    assert opt2.param_groups[0]["lr"]==rates[2]
    q.grad=torch.ones_like(q);opt2.step();sched2.step()
    assert opt2.param_groups[0]["lr"]==rates[3]


def test_published_v5_adaptation_has_explicit_r13_train_eval_boundary_and_budget():
    cfg=recipe_config("observed_v5",1234,train_image_count=112)
    assert cfg["position_lr_horizon"]==1234
    assert cfg["position_lr_spatial"]=="public_all_median" and cfg["position_lr_phase"]=="post_step_legacy"
    assert cfg["contribution_backend"]=="native_renderer_v1"
    assert cfg["projection"]["train_near_plane"]==.01 and cfg["projection"]["eval_near_plane"]==.1
    assert cfg["projection"]["train_background"]=="black" and cfg["projection"]["compositor"]=="native_v1"
    assert cfg["projection"]["train_rasterize_mode"]==cfg["projection"]["eval_rasterize_mode"]=="classic"
    assert cfg["scale_rel_l2reg"]==.00001 and cfg["scale_lower_l2reg"]==.00002


def test_earlier_v5_full_native_context_checkpoint_is_not_regenerated_from_adapted_defaults():
    old=recipe_config("observed_v5",10000,{"position_lr_horizon":30000,
        "position_lr_spatial":"native_train_mean_v1","position_lr_phase":"completed_step_native",
        "projection":{"train_near_mode":"native_focal_v1","eval_near_plane":.1,
            "far_mode":"unbounded_perspective","train_rasterize_mode":"antialiased",
            "eval_rasterize_mode":"classic","train_background":"random_rgb_v1","compositor":"native_v1"}},train_image_count=112)
    restored=recipe_from_checkpoint({"training_recipe":old})
    assert restored==old and restored!=recipe_config("observed_v5",10000,train_image_count=112)
    assert restored["projection"]["train_background"]=="random_rgb_v1" and restored["position_lr_horizon"]==30000


@pytest.mark.parametrize("bad",[{"projection":{"magic":1}},{"projection":{"far_plane":float("inf")}},
    {"projection":{"train_near_mode":"guess"}},{"projection":{"rasterize_mode":"wrong"}},
    {"projection":{"eval_near_plane":0}},{"position_lr_horizon":0},{"position_lr_horizon":True},
    {"position_lr_spatial":"guess"}])
def test_rejects_ambiguous_or_nonfinite_new_configuration(bad):
    with pytest.raises(ValueError):recipe_config("quality_v2",10000,bad)
