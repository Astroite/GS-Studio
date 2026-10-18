"""Opt-in actual CUDA rasterization/training/resume contract test.

Set LOCAL_POSTSHOT_GPU_TESTS=1. Synthetic ground-truth views are explicitly test
fixtures; public-scene acceptance is recorded in the benchmark reports (see
docs/development/BENCHMARKS.md).
"""
import os
import copy
import json
import hashlib
from pathlib import Path
import random
import shutil
import threading
import numpy as np
import pytest
from PIL import Image
from local_postshot import project
from local_postshot.formats import write_ply
from local_postshot.reconstruction import import_media,reconstruction_key
from local_postshot.training import train,tensors_from_cloud,render_parameters

pytestmark=pytest.mark.skipif(os.environ.get("LOCAL_POSTSHOT_GPU_TESTS")!="1",reason="Opt-in real CUDA test")
OBSERVED_RECIPES=("observed_v4","observed_v5","observed_v6")


def assert_exact_tree(actual,expected,path):
    """Checkpoint restoration uses no floating tolerance, including moments."""
    import torch
    if isinstance(expected,torch.Tensor):
        assert isinstance(actual,torch.Tensor),path
        assert actual.dtype==expected.dtype and actual.shape==expected.shape,path
        assert torch.equal(actual.detach().cpu(),expected.detach().cpu()),path
    elif isinstance(expected,np.ndarray):
        assert np.array_equal(actual,expected),path
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys(),path
        for key in expected:assert_exact_tree(actual[key],expected[key],f"{path}.{key}")
    elif isinstance(expected,(tuple,list)):
        assert type(actual)==type(expected) and len(actual)==len(expected),path
        for index,(left,right) in enumerate(zip(actual,expected)):
            assert_exact_tree(left,right,f"{path}[{index}]")
    else:
        assert actual==expected,path


def without_strategy_timing(state):
    """Only these two measured wall-time fields are nonalgorithmic metadata."""
    result=dict(state)
    result.pop("anchor_wall_seconds_total",None)
    if "last_anchor_refresh" in result:
        result["last_anchor_refresh"]=dict(result["last_anchor_refresh"])
        result["last_anchor_refresh"].pop("wall_seconds",None)
    return result


def fixture_project(root,recipe="legacy_adc"):
    import torch
    root=Path(root)
    p=project.create("Synthetic CUDA contract fixture",root)
    points=np.array([[x,y,-3] for x in np.linspace(-.8,.8,5) for y in np.linspace(-.6,.6,4)],np.float32)
    n=len(points)
    cloud={"means":points,"scales":np.full((n,3),-2,np.float32),
           "quats":np.tile([1.,0,0,0],(n,1)).astype(np.float32),
           "opacities":np.ones(n,np.float32),"sh":np.zeros((n,1,3),np.float32)}
    cloud["sh"][:,0]=(np.stack([points[:,0]*.25+.5,points[:,1]*.25+.5,np.full(n,.6)],-1)-.5)/.28209479177387814
    parameters=tensors_from_cloud(cloud)
    cameras=[]
    originals=root/"fixture_inputs"; originals.mkdir()
    for i in range(6):
        w2c=np.diag([1.,-1.,-1.,1.]); w2c[0,3]=(i-2.5)*.035
        camera={"id":i,"name":f"view{i}.png","K":[[48,0,24],[0,48,24],[0,0,1]],
                "width":48,"height":48,"model":1,"w2c":w2c.tolist()}
        with torch.no_grad():
            image,_,_=render_parameters(parameters,camera,0)
        Image.fromarray((image[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(originals/camera["name"])
        cameras.append(camera)
    p=import_media(root,[str(originals)],"images",lambda *_:None,threading.Event())
    for c,asset in zip(cameras,p["assets"]["images"]):
        c["name"]=Path(asset["path"]).name
    sparse=root/"reconstruction"/"fixture"; sparse.mkdir(parents=True)
    np.savez(sparse/"sparse_points.npz",points=points,colors=cloud["sh"][:,0]*.28209479177387814+.5)
    key=reconstruction_key(p)
    p["reconstruction"]={"key":key,"path":str(sparse),"image_path":str(root/"images"),
                         "camera_count":len(cameras),"point_count":n,"cameras":cameras}
    p["pipeline"]["reconstruction"]={"key":key,"status":"complete","fixture":True}
    p["status"]="reconstructed"; project.save(p)
    recipe_overrides={"pause_refine_after_reset":0} if recipe!="legacy_adc" else {}
    if recipe in OBSERVED_RECIPES:recipe_overrides["native_min_error"]=0.
    if recipe=="observed_v5":
        # Exercise the complete recovered context even when the published
        # quality preset uses a conservative independent training context.
        recipe_overrides.update(position_lr_horizon=30000,position_lr_spatial="native_train_mean_v1",
            position_lr_phase="completed_step_native",projection={"train_near_mode":"native_focal_v1",
                "eval_near_plane":.1,"far_mode":"unbounded_perspective",
                "train_rasterize_mode":"antialiased","eval_rasterize_mode":"classic",
                "train_background":"random_rgb_v1","compositor":"native_v1"})
    project.update(root,{"steps":12,"max_splats":100 if recipe in OBSERVED_RECIPES else 24,"sh_degree":0,"training_recipe":recipe,
                         "training_backend":"optimized" if recipe!="legacy_adc" else "legacy",
                         "training_recipe_overrides":recipe_overrides,
                         "refine_start":0,"refine_every":1,"refine_stop":12,
                         "reset_every":3000,
                         "checkpoint_every":6,"preview_every":6})
    return root


@pytest.mark.parametrize("recipe",["legacy_adc","quality_v2","observed_v3","observed_v4","observed_v5","observed_v6"])
def test_cuda_pause_resume_full_optimizer_state_equivalence(tmp_path,recipe,monkeypatch,record_property):
    import torch
    import local_postshot.training as module
    if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    maximum=100 if recipe in OBSERVED_RECIPES else 24
    recipe_tag=recipe.rsplit("_",1)[-1]
    a=fixture_project(tmp_path/"continuous",recipe)
    b=fixture_project(tmp_path/"resumed",recipe)
    if recipe in OBSERVED_RECIPES:
        # Branch after a real split from ONE computed prefix. Two separately
        # executed isotropic prefixes confound resume with CUDA backward drift.
        for root in (a,b):project.update(root,{"checkpoint_every":10})
    branch=["continuous"]
    saved=None
    strategy_restore_checks=[]
    anchor_snapshots={"continuous":{},"paused":{},"resumed":{}}
    lineages={name:[f"initial:{i}" for i in range(20)] for name in anchor_snapshots}
    split_history={name:[] for name in anchor_snapshots}
    iteration_snapshots={name:{} for name in anchor_snapshots}
    shared_prefix=[]
    background_draws={name:{} for name in anchor_snapshots}
    active_iteration=[None]
    if recipe in OBSERVED_RECIPES:
        import local_postshot.splat3_strategy as strategy_module
        from local_postshot.splat3_strategy import ObservedStrategy,growth_budget
        assert growth_budget(20,maximum)>0
        actual_split=strategy_module.split_selected
        def inspect_split(params,optimizers,state,selected,**kwargs):
            ids=selected.detach().cpu().tolist()
            previous=lineages[branch[0]]
            assert len(previous)==len(params["means"])
            previous_anchor=state["min_scale"].detach().cpu().clone()
            actual_split(params,optimizers,state,selected,**kwargs)
            assert_exact_tree(state["min_scale"],torch.cat([previous_anchor,previous_anchor[ids]]),"split.inherited_min_scale")
            lineages[branch[0]]=previous+[f"{previous[i]}/child@{active_iteration[0]}" for i in ids]
            split_history[branch[0]].append({"step":active_iteration[0],"selected":ids,"count":len(params["means"])})
        monkeypatch.setattr(strategy_module,"split_selected",inspect_split)
        actual_prepare=ObservedStrategy.prepare_step
        def inspect_prepare(self,params,optimizers,state,step,cameras,train_ids):
            active_iteration[0]=int(step)
            if branch[0]=="resumed" and not strategy_restore_checks:
                assert step==saved["step"]
                for name in ("sum1","sum2","max_weight","min_scale"):
                    assert isinstance(saved["strategy_state"][name],torch.Tensor),name
                assert_exact_tree(state,saved["strategy_state"],"restored.strategy_state.before_prepare")
                assert_exact_tree(dict(params.items()),saved["parameters"],"restored.parameters.before_prepare")
                for name,optimizer in optimizers.items():
                    assert_exact_tree(optimizer.state_dict(),saved["optimizers"][name],f"restored.optimizers.{name}.before_prepare")
                for actual,expected,label in ((torch.get_rng_state(),saved["rng_torch"],"torch"),
                        (torch.cuda.get_rng_state_all(),saved["rng_cuda"],"cuda"),
                        (random.getstate(),saved["rng_python"],"python"),
                        (np.random.get_state(),saved["rng_numpy"],"numpy")):
                    assert_exact_tree(actual,expected,f"restored.rng_{label}.before_prepare")
                strategy_restore_checks.append("complete_strategy_bit_exact_before_prepare")
            # Test-only CPU traces retain the incoming state at EVERY recorded
            # iteration, so an eventual failure need not be attributed to a
            # guessed CUDA instruction. No trace is substituted for live state.
            iteration_snapshots[branch[0]][int(step)]={
                "parameters":{name:value.detach().cpu().clone() for name,value in params.items()},
                "strategy_state":copy.deepcopy({name:value.detach().cpu().clone() if isinstance(value,torch.Tensor) else value
                                                for name,value in state.items()})}
            previous_anchor_step=state.get("last_anchor_step")
            actual_prepare(self,params,optimizers,state,step,cameras,train_ids)
            if state.get("last_anchor_step")==step and previous_anchor_step!=step:
                assert state["last_anchor_refresh"]["pruned_invalid"]==0,"Controlled fixture should retain all supported rows"
                assert len(lineages[branch[0]])==len(params["means"])
                anchor_snapshots[branch[0]][int(step)]={
                    "means":params["means"].detach().cpu().clone(),
                    "min_scale":state["min_scale"].detach().cpu().clone(),
                    "pruned_invalid":state["last_anchor_refresh"]["pruned_invalid"],
                    "lineage":lineages[branch[0]].copy()}
        monkeypatch.setattr(ObservedStrategy,"prepare_step",inspect_prepare)
        actual_checkpoint=module.atomic_checkpoint
        prefix_file=tmp_path/"shared-prefix-step10.pt"
        def branch_after_written_checkpoint(file,state):
            actual_checkpoint(file,state)
            if Path(file)==Path(a)/"checkpoints/latest.pt" and state["step"]==10:
                count_at_prefix=len(state["parameters"]["means"])
                assert count_at_prefix>20
                assert any(item["selected"] for item in split_history["continuous"])
                assert not shared_prefix,"Exactly one step10 prefix must be captured"
                # Copy the written artifact bytes, never rebuild/re-serialize
                # the branch's source state or generate another prefix.
                shutil.copyfile(file,prefix_file)
                shutil.copyfile(prefix_file,Path(b)/"checkpoints/latest.pt")
                digest=hashlib.sha256(Path(file).read_bytes()).hexdigest()
                assert hashlib.sha256(prefix_file.read_bytes()).hexdigest()==digest
                assert hashlib.sha256((Path(b)/"checkpoints/latest.pt").read_bytes()).hexdigest()==digest
                lineages["resumed"]=lineages["continuous"].copy()
                split_history["resumed"]=copy.deepcopy(split_history["continuous"])
                anchor_snapshots["resumed"]=copy.deepcopy(anchor_snapshots["continuous"])
                shared_prefix.append({"recipe":recipe,"step":10,"count":count_at_prefix,"sha256":digest,
                    "path":str(prefix_file),"source":"one live continuous run after its real split",
                    "resumed_history_before_step10":"copied provenance of the shared prefix, not a second execution"})
        monkeypatch.setattr(module,"atomic_checkpoint",branch_after_written_checkpoint)
        if recipe=="observed_v5":
            actual_rand=torch.rand
            def inspect_background_draw(*args,**kwargs):
                shape=tuple(args[0]) if len(args)==1 and isinstance(args[0],(tuple,list)) else tuple(args)
                is_background=shape==(1,3) and torch.device(kwargs.get("device","cpu")).type=="cuda"
                if is_background and branch[0]=="resumed" and not background_draws["resumed"]:
                    # render_parameters owns this draw. The first-forward hook
                    # must check restored RNG BEFORE entering it.
                    assert restoration_checks,"RNG restoration must be checked before random background sampling"
                    assert_exact_tree(torch.cuda.get_rng_state_all(),saved["rng_cuda"],"restored.rng_before_first_background")
                result=actual_rand(*args,**kwargs)
                if is_background:
                    step=int(active_iteration[0])
                    assert step not in background_draws[branch[0]],"Exactly one RGB background draw per training forward"
                    background_draws[branch[0]][step]=result.detach().cpu().clone()
                return result
            monkeypatch.setattr(torch,"rand",inspect_background_draw)
    noop=lambda *_:None
    full=train(a,noop,threading.Event(),threading.Event(),steps=12,max_splats=maximum)
    if recipe not in OBSERVED_RECIPES:
        stop=threading.Event();pause=threading.Event()
        def callback(event,data):
            if event=="metric" and data["step"]==1:
                pause.set();stop.set()
        branch[0]="paused"
        paused=train(b,callback,stop,pause,steps=12,max_splats=maximum)
        assert paused["status"]=="paused" and paused["training"]["step"]==1
    else:
        assert len(shared_prefix)==1
        assert project.load(a)["reconstruction"]["key"]==project.load(b)["reconstruction"]["key"]
        record_property("shared_prefix_checkpoint",json.dumps(shared_prefix[0]))
    # Preserve the actual source boundary before final training overwrites latest.
    # Real pause-step1 control has separate coverage for observed4/5 below.
    saved=torch.load(Path(b)/"checkpoints"/"latest.pt",map_location="cpu",weights_only=False)
    assert saved["step"]==(10 if recipe in OBSERVED_RECIPES else 1)
    if recipe=="observed_v6":
        from local_postshot.projection_policy import projection_artifact_identity
        assert saved['training_recipe']['projection']['tile_culling']=='conservative_v1'
        assert saved['training_recipe']['projection_artifacts']==projection_artifact_identity(saved['training_recipe']['projection'])
    if recipe=="observed_v5":
        frozen=saved["training_recipe"]
        assert frozen["name"]==recipe and frozen["contribution_backend"]=="native_renderer_v1"
        assert frozen["projection"]["compositor"]=="native_v1"
        assert frozen["projection"]["train_background"]=="random_rgb_v1"
        assert frozen["projection"]["train_near_mode"]=="native_focal_v1"
        assert frozen["projection"]["train_rasterize_mode"]=="antialiased"
        assert frozen["position_lr_horizon"]==30000
        assert frozen["position_lr_spatial"]=="native_train_mean_v1"
        assert frozen["position_lr_phase"]=="completed_step_native"
    if recipe!="legacy_adc":
        original_overrides=project.load(b)["settings"]["training_recipe_overrides"].copy()
        changed_key="native_min_error" if recipe in OBSERVED_RECIPES else "prune_scale2d"
        project.update(b,{"training_recipe_overrides":{**original_overrides,changed_key:.2}})
        checkpoint_digest=hashlib.sha256((Path(b)/"checkpoints/latest.pt").read_bytes()).hexdigest()
        with pytest.raises(ValueError,match="thresholds"):
            train(b,noop,threading.Event(),threading.Event(),steps=12,max_splats=maximum,resume=True)
        assert hashlib.sha256((Path(b)/"checkpoints/latest.pt").read_bytes()).hexdigest()==checkpoint_digest
        project.update(b,{"training_recipe_overrides":original_overrides})
    constructed_optimizers={}
    original_factory=module.optimizers_for
    original_render=module.render_parameters
    restoration_checks=[]
    def capture_optimizers(*args,**kwargs):
        result=original_factory(*args,**kwargs)
        constructed_optimizers.clear();constructed_optimizers.update(result)
        return result
    def first_resume_forward(parameters,*args,**kwargs):
        if not restoration_checks:
            assert constructed_optimizers.keys()==saved["optimizers"].keys()
            assert_exact_tree(dict(parameters.items()),saved["parameters"],"restored.parameters")
            for name,optimizer in constructed_optimizers.items():
                # Includes moments, step, LR, betas, eps, fused/capturable flags
                # and all other groups; no optimizer-field tolerance is allowed.
                assert_exact_tree(optimizer.state_dict(),saved["optimizers"][name],f"restored.optimizers.{name}")
            assert_exact_tree(torch.get_rng_state(),saved["rng_torch"],"restored.rng_torch")
            assert_exact_tree(torch.cuda.get_rng_state_all(),saved["rng_cuda"],"restored.rng_cuda")
            assert_exact_tree(random.getstate(),saved["rng_python"],"restored.rng_python")
            assert_exact_tree(np.random.get_state(),saved["rng_numpy"],"restored.rng_numpy")
            restoration_checks.append("all_saved_parameters_optimizer_fields_and_rng_bit_exact_before_forward")
        return original_render(parameters,*args,**kwargs)
    monkeypatch.setattr(module,"optimizers_for",capture_optimizers)
    monkeypatch.setattr(module,"render_parameters",first_resume_forward)
    branch[0]="resumed"
    final=train(b,noop,threading.Event(),threading.Event(),steps=12,max_splats=maximum,resume=True)
    assert len(restoration_checks)==1
    if recipe in OBSERVED_RECIPES:assert len(strategy_restore_checks)==1
    assert full["training"]["step"]==final["training"]["step"]==12
    x=torch.load(Path(a)/"checkpoints"/"latest.pt",map_location="cpu",weights_only=False)
    y=torch.load(Path(b)/"checkpoints"/"latest.pt",map_location="cpu",weights_only=False)
    for key in ("parameters","optimizers","schedulers","strategy_state","rng_python","rng_numpy",
                "rng_torch","rng_cuda","sampler","dataset_key","config"):
        assert key in x and key in y
    assert x["sampler"]==y["sampler"]
    assert x["training_recipe"]==y["training_recipe"]
    assert x["schedulers"]==y["schedulers"]
    assert x["config"]==y["config"] and x["strategy_config"]==y["strategy_config"]
    assert x["dataset_key"]==y["dataset_key"] and x["scene_scale"]==y["scene_scale"]
    for field in ("rng_python","rng_numpy","rng_torch","rng_cuda"):
        assert_exact_tree(x[field],y[field],f"final.{field}")
    assert x["strategy_state"].keys()==y["strategy_state"].keys()
    diagnostics={"recipe":recipe,"restore":restoration_checks,"strategy_restore":strategy_restore_checks,
                 "parameter_max_abs":{},"moment_max_abs":{},"views":[],"strategy_max_abs":{}}
    for field in ("grad2d","count","radii"):
        a_state,b_state=x["strategy_state"].get(field),y["strategy_state"].get(field)
        if a_state is not None:
            torch.testing.assert_close(a_state,b_state,atol=3e-5,rtol=3e-5)
    omitted={"grad2d","count","radii"}
    if recipe in OBSERVED_RECIPES:
        # Retain evidence BEFORE any topology/statistics assertion can fail.
        # A true CDF/visibility drift is reported, not hidden by the render gate.
        def json_trace(value):
            if isinstance(value,torch.Tensor):return value.tolist()
            if isinstance(value,dict):return {name:json_trace(item) for name,item in value.items()}
            if isinstance(value,(tuple,list)):return [json_trace(item) for item in value]
            return value
        state_receipt={"recipe":recipe,"restore_checks":strategy_restore_checks,"split_history":split_history,
            "experiment":"shared real post-split step10 checkpoint; continuous10-to12 versus resumed10-to12",
            "shared_prefix":shared_prefix,"incoming_iteration_snapshots":json_trace(iteration_snapshots),
            "random_background_draws":json_trace(background_draws),
            "lineages":lineages,"counts":{"continuous":len(x["parameters"]["means"]),"resumed":len(y["parameters"]["means"])},
            "final_strategy":{label:{name:(value.tolist() if isinstance(value,torch.Tensor) else value)
                for name,value in state["strategy_state"].items()} for label,state in (("continuous",x),("resumed",y))},
            "anchor_snapshots":{label:{step:{name:(value.tolist() if isinstance(value,torch.Tensor) else value)
                for name,value in snapshot.items()} for step,snapshot in snapshots.items()} for label,snapshots in anchor_snapshots.items()},
            "final_timing_exclusions":["anchor_wall_seconds_total","last_anchor_refresh.wall_seconds"]}
        state_receipt_path=tmp_path/f"{recipe_tag}-state-diagnostics.json"
        state_receipt_path.write_text(json.dumps(state_receipt,indent=2),encoding="utf-8")
        record_property(f"{recipe_tag}_state_diagnostics",str(state_receipt_path))
        assert len(x["parameters"]["means"])>20 and len(y["parameters"]["means"])>20,f"{recipe} must exercise a real split"
        assert any(item["selected"] for item in split_history["continuous"])
        assert any(item["selected"] for item in split_history["resumed"])
        assert lineages["continuous"]==lineages["resumed"],{"topology_drift":split_history}
        assert split_history["continuous"]==split_history["resumed"],split_history
        if recipe=="observed_v5":
            assert set(background_draws["continuous"])==set(range(12))
            assert set(background_draws["resumed"])=={10,11}
            for step in (10,11):
                assert_exact_tree(background_draws["continuous"][step],background_draws["resumed"][step],f"resumed.background[{step}]")
            assert not torch.equal(background_draws["resumed"][10],background_draws["resumed"][11]),"Random RGB background must advance the restored RNG stream"
        for field in ("sum1","max_weight"):
            left,right=x["strategy_state"][field],y["strategy_state"][field]
            assert left.shape==right.shape and torch.isfinite(left).all() and torch.isfinite(right).all()
            diagnostics["strategy_max_abs"][field]=float((left-right).abs().max())
            # Same original bound as other accumulated floating strategy fields;
            # CUDA atomic contributions are not assumed to be bit deterministic.
            torch.testing.assert_close(left,right,atol=3e-5,rtol=3e-5)
        assert_exact_tree(x["strategy_state"]["sum2"],y["strategy_state"]["sum2"],"final.strategy.sum2_same_topology")
        refresh=x["strategy_state"]["last_anchor_step"]
        assert refresh==y["strategy_state"]["last_anchor_step"]==10
        snapshot_x=anchor_snapshots["continuous"][refresh]
        snapshot_y=anchor_snapshots["resumed"][refresh]
        assert snapshot_x["lineage"]==snapshot_y["lineage"]
        for camera in project.load(a)["reconstruction"]["cameras"]:
            np.testing.assert_array_equal(np.asarray(camera["w2c"])[2],[0,0,-1,0])
            assert camera["K"][0][0]==camera["K"][1][1]==48
        anchors_by_branch={}
        for label,state,snapshot in (("continuous",x,snapshot_x),("resumed",y,snapshot_y)):
            # Every fixture camera has the identical z row and focal48. X-only
            # camera translation makes m=-z*.25/48 independent of which of these
            # cameras support a row. Test the stored refresh, not post-split means.
            expected=-snapshot["means"][:,2]*(state["training_recipe"]["min_scale_px"]/48)
            assert (snapshot["min_scale"]>=0).all()
            rounding=4*torch.finfo(expected.dtype).eps*float(expected.abs().max())
            torch.testing.assert_close(snapshot["min_scale"],expected,atol=rounding,rtol=0)
            lookup={tag:i for i,tag in enumerate(snapshot["lineage"])}
            def source_index(tag):
                while tag not in lookup:
                    assert "/child@" in tag,("Missing anchor ancestor",tag)
                    tag=tag.rsplit("/child@",1)[0]
                return lookup[tag]
            inherited=torch.stack([snapshot["min_scale"][source_index(tag)] for tag in lineages[label]])
            assert_exact_tree(state["strategy_state"]["min_scale"],inherited,f"final.{label}.min_scale_inheritance")
            anchors_by_branch[label]=inherited
        # The min operator is1-Lipschitz in each shared candidate. This fixture's
        # common depth/focal yields a physical drift bound from the SAME refresh
        # geometry, plus4float32 eps for the two rounded computations. No blanket
        # all-tensor tolerance or quaternion/child-row relaxation is used here.
        depth_difference=float((snapshot_x["means"][:,2]-snapshot_y["means"][:,2]).abs().max())
        magnitude=max(float(v.abs().max()) for v in anchors_by_branch.values())
        anchor_bound=depth_difference*(x["training_recipe"]["min_scale_px"]/48)+4*torch.finfo(snapshot_x["min_scale"].dtype).eps*magnitude
        torch.testing.assert_close(anchors_by_branch["continuous"],anchors_by_branch["resumed"],atol=anchor_bound,rtol=0)
        diagnostics["strategy_max_abs"]["min_scale"]=float((anchors_by_branch["continuous"]-anchors_by_branch["resumed"]).abs().max())
        diagnostics["anchor_depth_derived_bound"]=anchor_bound
        diagnostics["split_history"]=split_history
        diagnostics["timing_fields_excluded_from_final_comparison"]=["anchor_wall_seconds_total","last_anchor_refresh.wall_seconds"]
        for state in (x,y):
            assert state["strategy_state"]["anchor_wall_seconds_total"]>=0
            assert state["strategy_state"]["last_anchor_refresh"]["wall_seconds"]>=0
        left_state=without_strategy_timing(x["strategy_state"])
        right_state=without_strategy_timing(y["strategy_state"])
        omitted|={"sum1","sum2","max_weight","min_scale"}
    else:
        left_state,right_state=x["strategy_state"],y["strategy_state"]
    for field in set(left_state)-omitted:
        assert_exact_tree(left_state[field],right_state[field],f"final.strategy.{field}")
    for key in x["parameters"]:
        left,right=x["parameters"][key],y["parameters"][key]
        assert left.shape==right.shape and torch.isfinite(left).all() and torch.isfinite(right).all()
        diagnostics["parameter_max_abs"][key]=float((left-right).abs().max()) if left.numel() else 0.
    for name in x["optimizers"]:
        xs,ys=x["optimizers"][name]["state"],y["optimizers"][name]["state"]
        assert xs.keys()==ys.keys()
        diagnostics["moment_max_abs"][name]={}
        for index in xs:
            for key in xs[index]:
                left,right=xs[index][key],ys[index][key]
                assert left.shape==right.shape and torch.isfinite(left).all() and torch.isfinite(right).all()
                if key=="step":assert_exact_tree(left,right,f"final.optimizer.{name}.step")
                diagnostics["moment_max_abs"][name][key]=float((left-right).abs().max()) if left.numel() else 0.
    # Quaternions of nearly isotropic GSs are gauge-sensitive: tiny backward
    # rounding can be amplified by eps1e-15 Adam and then by stochastic splitting.
    # Exact restoration is asserted above. End-to-end equivalence is judged in
    # rendered image space, not by arbitrary orientation/child-row coordinates.
    # Bounds are fixed before GPU revalidation: RMS below a quarter RGB8 code
    # value (mutual PSNR >=60.17dB), plus <=.01dB quality change against each GT.
    mse_limit=(.25/255)**2
    psnr_delta_limit=.01
    parameters_x={key:value.to("cuda") for key,value in x["parameters"].items()}
    parameters_y={key:value.to("cuda") for key,value in y["parameters"].items()}
    test_project=project.load(a)
    cameras=test_project["reconstruction"]["cameras"]
    train_ids,test_ids=module.split_indices(cameras,x["config"]["test_every"])
    assert train_ids and test_ids
    with torch.no_grad():
        for index,camera in enumerate(cameras):
            # V5 equivalence must use its frozen native evaluation compositor;
            # the prior recipes keep precisely their original render route.
            render_options={"projection_policy":x["training_recipe"]["projection"],"training":False} if recipe in ("observed_v5","observed_v6") else {}
            left=original_render(parameters_x,camera,0,**render_options)[0]
            right=original_render(parameters_y,camera,0,**render_options)[0]
            source=Path(test_project["reconstruction"]["image_path"])/camera["name"]
            with Image.open(source) as image:
                gt=torch.as_tensor(np.asarray(image.convert("RGB")).copy(),device="cuda",dtype=torch.float32)[None]/255
            mse=float(((left-right)**2).mean())
            mse_left=float(((left-gt)**2).mean())
            mse_right=float(((right-gt)**2).mean())
            psnr_left=-10*np.log10(max(mse_left,1e-12))
            psnr_right=-10*np.log10(max(mse_right,1e-12))
            view={"camera":camera["name"],"split":"holdout" if index in test_ids else "train",
                  "mutual_mse":mse,"mutual_psnr":-10*np.log10(max(mse,1e-20)),
                  "continuous_psnr_vs_gt":psnr_left,"resumed_psnr_vs_gt":psnr_right,
                  "psnr_delta_vs_gt":abs(psnr_left-psnr_right)}
            diagnostics["views"].append(view)
    diagnostics["thresholds"]={"mutual_mse":mse_limit,"psnr_delta_vs_gt":psnr_delta_limit,
        "basis":"Quarter RGB8-code RMS and one-hundredth dB; float32 outputs, no clamp/round"}
    (tmp_path/"resume-diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    record_property("restoration_bit_exact",True)
    record_property("resume_diagnostics",json.dumps(diagnostics))
    for view in diagnostics["views"]:
        assert view["mutual_mse"]<=mse_limit,view
        assert view["psnr_delta_vs_gt"]<=psnr_delta_limit,view
    assert final["training"]["count"]<=maximum
    if recipe in OBSERVED_RECIPES:
        record_property("strategy_restoration_bit_exact",True)
        record_property("native_threshold_change_rejected_without_checkpoint_change",True)


@pytest.mark.parametrize("recipe",OBSERVED_RECIPES)
def test_observed_real_pause_step1_resume_step2_restores_full_state(tmp_path,recipe,monkeypatch,record_property):
    """Keep actual pause control separate from shared-prefix equivalence."""
    import torch
    import local_postshot.training as module
    from local_postshot.splat3_strategy import ObservedStrategy
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    recipe_tag=recipe.rsplit("_",1)[-1]
    root=fixture_project(tmp_path/f"{recipe_tag}-real-pause",recipe)
    stop=threading.Event();pause=threading.Event()
    def request_pause(event,data):
        if event=="metric" and data["step"]==1:
            pause.set();stop.set()
    paused=train(root,request_pause,stop,pause,steps=12,max_splats=100)
    assert paused["status"]=="paused" and paused["training"]["step"]==1
    checkpoint=Path(root)/"checkpoints/latest.pt"
    retained=tmp_path/"real-pause-step1.pt"
    shutil.copyfile(checkpoint,retained)
    source_digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert hashlib.sha256(retained.read_bytes()).hexdigest()==source_digest
    saved=torch.load(retained,map_location="cpu",weights_only=False)
    assert saved["step"]==1 and saved["sampler"]["position"]==1
    assert all(isinstance(saved["strategy_state"][name],torch.Tensor)
               for name in ("sum1","sum2","max_weight","min_scale"))
    rejected_drift=[]
    if recipe=="observed_v5":
        original_overrides=project.load(root)["settings"]["training_recipe_overrides"].copy()
        altered_projection={**saved["training_recipe"]["projection"],"train_background":"black"}
        # Each requested change is valid for fresh training, but cannot alter
        # the numerical context of a full checkpoint. Preserve its actualbytes.
        changes=[("recipe",{"training_recipe":"observed_v4","training_recipe_overrides":{}},"recipe"),
            ("position_lr_horizon",{"training_recipe_overrides":{**original_overrides,
                "position_lr_horizon":saved["training_recipe"]["position_lr_horizon"]+1}},"thresholds"),
            ("projection_train_background",{"training_recipe_overrides":{**original_overrides,
                "projection":altered_projection}},"thresholds")]
        for label,update,message in changes:
            project.update(root,update)
            with pytest.raises(ValueError,match=message):
                train(root,lambda *_:None,threading.Event(),threading.Event(),steps=2,max_splats=100,resume=True)
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source_digest
            rejected_drift.append(label)
            project.update(root,{"training_recipe":recipe,"training_recipe_overrides":original_overrides})
    restored=[]
    def exact_rng(label):
        for actual,expected,name in ((torch.get_rng_state(),saved["rng_torch"],"torch"),
                (torch.cuda.get_rng_state_all(),saved["rng_cuda"],"cuda"),
                (random.getstate(),saved["rng_python"],"python"),
                (np.random.get_state(),saved["rng_numpy"],"numpy")):
            assert_exact_tree(actual,expected,f"real_pause.{label}.rng.{name}")
    actual_prepare=ObservedStrategy.prepare_step
    def first_prepare(self,params,optimizers,state,step,cameras,train_ids):
        if not restored:
            assert step==saved["step"]==1
            assert_exact_tree(dict(params.items()),saved["parameters"],"real_pause.before_prepare.parameters")
            assert_exact_tree(state,saved["strategy_state"],"real_pause.before_prepare.complete_strategy")
            for name,optimizer in optimizers.items():
                assert_exact_tree(optimizer.state_dict(),saved["optimizers"][name],f"real_pause.before_prepare.optimizer.{name}")
            exact_rng("before_prepare")
            restored.append("all_parameters_full_adam_rng_complete_strategy_bit_exact_before_prepare")
        return actual_prepare(self,params,optimizers,state,step,cameras,train_ids)
    monkeypatch.setattr(ObservedStrategy,"prepare_step",first_prepare)
    actual_factory=module.optimizers_for
    constructed={}
    def capture_optimizers(*args,**kwargs):
        result=actual_factory(*args,**kwargs)
        constructed.clear();constructed.update(result)
        return result
    monkeypatch.setattr(module,"optimizers_for",capture_optimizers)
    actual_render=module.render_parameters
    def first_forward(parameters,*args,**kwargs):
        if len(restored)==1:
            assert_exact_tree(dict(parameters.items()),saved["parameters"],"real_pause.before_forward.parameters")
            assert constructed.keys()==saved["optimizers"].keys()
            for name,optimizer in constructed.items():
                assert_exact_tree(optimizer.state_dict(),saved["optimizers"][name],f"real_pause.before_forward.optimizer.{name}")
            exact_rng("before_forward")
            restored.append("all_parameters_full_adam_rng_bit_exact_before_forward")
        return actual_render(parameters,*args,**kwargs)
    monkeypatch.setattr(module,"render_parameters",first_forward)
    final=train(root,lambda *_:None,threading.Event(),threading.Event(),steps=2,max_splats=100,resume=True)
    assert final["status"]=="trained" and final["training"]["step"]==2
    assert len(restored)==2
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert state["step"]==2 and state["sampler"]["position"]==2
    assert state["training_recipe"]==saved["training_recipe"]
    assert state["config"]==saved["config"] and state["strategy_config"]==saved["strategy_config"]
    assert len(state["parameters"]["means"])==len(saved["parameters"]["means"])==20
    receipt={"recipe":recipe,"actual_pause_step":1,"resumed_final_step":2,"source_checkpoint":str(retained),
             "source_checkpoint_sha256":source_digest,"checks":restored,
             "rejected_configuration_drift":rejected_drift,
             "purpose":"real pause/control and exact restore; image/statistical equivalence uses the separate shared post-split prefix test"}
    (tmp_path/f"{recipe_tag}-real-pause-receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    record_property(f"{recipe_tag}_real_pause_receipt",json.dumps(receipt))


def test_injected_cuda_oom_preserves_last_complete_checkpoint(tmp_path,monkeypatch):
    import torch
    import local_postshot.training as module
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    root=fixture_project(tmp_path/"oom")
    actual=module.render_parameters
    calls=[]
    def fail_second(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise torch.cuda.OutOfMemoryError("Injected test OOM; no physical memory exhaustion")
        return actual(*args,**kwargs)
    monkeypatch.setattr(module,"render_parameters",fail_second)
    with pytest.raises(RuntimeError,match="out of memory"):
        train(root,lambda *_:None,threading.Event(),threading.Event(),steps=12,max_splats=24)
    p=project.load(root)
    assert p["status"]=="error" and "out of memory" in p["last_error"]
    state=torch.load(Path(root)/"checkpoints"/"latest.pt",map_location="cpu",weights_only=False)
    assert state["step"]==0 and state["schema"]==1
    monkeypatch.setattr(module,"render_parameters",actual)
    final=train(root,lambda *_:None,threading.Event(),threading.Event(),steps=12,max_splats=24,resume=True)
    assert final["training"]["step"]==12 and final["status"]=="trained"


@pytest.mark.parametrize("recipe",OBSERVED_RECIPES)
def test_observed_anchor_oom_preserves_pre_anchor_checkpoint_and_resumes(tmp_path,recipe,monkeypatch,record_property):
    import torch
    import local_postshot.splat3_regularization as regularization
    from local_postshot.splat3_strategy import ObservedStrategy
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    recipe_tag=recipe.rsplit("_",1)[-1]
    root=fixture_project(tmp_path/f"{recipe_tag}-anchor-oom",recipe)
    checkpoint=Path(root)/"checkpoints/latest.pt"
    actual_anchor=regularization.compute_min_scale_anchors
    receipts=[]
    def fail_before_anchor(*args,**kwargs):
        assert checkpoint.is_file(),"A valid initial checkpoint must precede anchor allocation"
        receipts.append({"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"injected":True})
        raise torch.cuda.OutOfMemoryError(f"Injected before {recipe} anchor production; no physical memory exhaustion")
    monkeypatch.setattr(regularization,"compute_min_scale_anchors",fail_before_anchor)
    with pytest.raises(RuntimeError,match="out of memory"):
        train(root,lambda *_:None,threading.Event(),threading.Event(),steps=12,max_splats=100)
    assert len(receipts)==1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==receipts[0]["checkpoint_sha256"]
    project_state=project.load(root)
    assert project_state["status"]=="error" and "out of memory" in project_state["last_error"]
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert saved["step"]==0 and saved["schema"]==1
    retained=tmp_path/f"{recipe_tag}-anchor-oom-step0.pt"
    shutil.copyfile(checkpoint,retained)
    assert hashlib.sha256(retained.read_bytes()).hexdigest()==receipts[0]["checkpoint_sha256"]
    assert saved["sampler"]["position"]==0 and saved["sampler"]["order"]==[]
    for name in ("sum1","sum2","max_weight","min_scale"):
        assert saved["strategy_state"][name] is None
    assert saved["strategy_state"]["last_anchor_step"]==-1
    assert saved["strategy_state"]["anchor_refreshes"]==0
    monkeypatch.setattr(regularization,"compute_min_scale_anchors",actual_anchor)
    actual_prepare=ObservedStrategy.prepare_step
    restored=[]
    def first_prepare(self,params,optimizers,state,step,cameras,train_ids):
        if not restored:
            assert step==0
            assert_exact_tree(dict(params.items()),saved["parameters"],"anchor_oom.restored.parameters")
            assert_exact_tree(state,saved["strategy_state"],"anchor_oom.restored.strategy_before_anchor")
            for name,optimizer in optimizers.items():
                assert_exact_tree(optimizer.state_dict(),saved["optimizers"][name],f"anchor_oom.restored.optimizer.{name}")
            for actual,expected,label in ((torch.get_rng_state(),saved["rng_torch"],"torch"),
                    (torch.cuda.get_rng_state_all(),saved["rng_cuda"],"cuda"),
                    (random.getstate(),saved["rng_python"],"python"),
                    (np.random.get_state(),saved["rng_numpy"],"numpy")):
                assert_exact_tree(actual,expected,f"anchor_oom.restored.rng.{label}")
            restored.append("pre_anchor_step0_state_bit_exact")
        return actual_prepare(self,params,optimizers,state,step,cameras,train_ids)
    monkeypatch.setattr(ObservedStrategy,"prepare_step",first_prepare)
    final=train(root,lambda *_:None,threading.Event(),threading.Event(),steps=12,max_splats=100,resume=True)
    assert restored==["pre_anchor_step0_state_bit_exact"]
    assert final["status"]=="trained" and final["training"]["step"]==12
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert state["strategy_state"]["last_anchor_step"]==10
    assert state["strategy_state"]["anchor_refreshes"]==3
    assert all(isinstance(state["strategy_state"][name],torch.Tensor) for name in ("sum1","sum2","max_weight","min_scale"))
    receipt={"recipe":recipe,"failure_stage":"before_first_anchor_allocation","preserved_step":0,
             "source_checkpoint":str(retained),
             "preserved_checkpoint_sha256":receipts[0]["checkpoint_sha256"],"restore_checks":restored,
             "final_step":state["step"],"anchor_refreshes":state["strategy_state"]["anchor_refreshes"]}
    (tmp_path/f"{recipe_tag}-anchor-oom-receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    record_property(f"{recipe_tag}_anchor_oom_receipt",json.dumps(receipt))


def test_actual_screen_clean_uses_training_geometry_without_reading_rgb(tmp_path,monkeypatch):
    import torch
    from local_postshot.cleaning import clean
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    root=fixture_project(tmp_path/"clean")
    p=project.load(root)
    c={"means":np.array([[0,0,-.4],[0,0,-3]],np.float32),
       "scales":np.array([[np.log(.2)]*3,[-4.]*3],np.float32),
       "quats":np.array([[1,0,0,0],[1,0,0,0]],np.float32),
       "opacities":np.ones(2,np.float32),"sh":np.zeros((2,1,3),np.float32)}
    source=root/"models"/"raw.ply";write_ply(source,c)
    raw_hash=project.digest(source)
    p["models"]=[project.model_record(source,2)];project.save(p)
    def forbid_rgb(*args,**kwargs):raise AssertionError("Cleaning must not read source/heldout images")
    monkeypatch.setattr(Image,"open",forbid_rgb)
    result=clean(root,p["models"][0]["id"],lambda *_:None,threading.Event(),.15)
    assert result["models"][-1]["count"]==1 and result["models"][-1]["visible"]
    assert not result["models"][0]["visible"] and project.digest(source)==raw_hash
    import json
    provenance=result["models"][-1]["provenance"]
    report=json.loads(Path(provenance["report"]).read_text())
    assert report["heldout_rgb_used"] is False and report["training_camera_indices"]==[1,2,3,4,5]
    with np.load(provenance["mask"]) as mask:
        assert mask["remove"].tolist()==[True,False]
