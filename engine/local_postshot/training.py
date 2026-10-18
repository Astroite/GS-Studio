"""Own training orchestration over gsplat rasterization and DefaultStrategy.

This module contains no original Postshot code or binary dependency.
"""
from collections import OrderedDict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from . import project
from .formats import load_cloud, write_ply
from .geometry import SH0, apply_edits
from .reconstruction import Cancelled, reconstruction_key
from .locks import operation_locked


class Sampler:
    def __init__(self, indices, seed=42):
        self.indices = list(indices)
        self.rng = random.Random(seed)
        self.order = []
        self.position = 0

    def next(self):
        if self.position >= len(self.order):
            self.order = self.indices.copy()
            self.rng.shuffle(self.order)
            self.position = 0
        value = self.order[self.position]
        self.position += 1
        return value

    def state_dict(self):
        return {"indices":self.indices,"rng":self.rng.getstate(),"order":self.order,"position":self.position}

    def load_state_dict(self, state):
        if self.indices != state["indices"]:
            raise ValueError("Checkpoint sampler dataset differs")
        self.rng.setstate(state["rng"])
        self.order, self.position = state["order"],state["position"]


class Images:
    def __init__(self, root, cameras, limit=8):
        self.root, self.cameras, self.limit = Path(root),cameras,limit
        self.cache = OrderedDict()

    def get(self, index):
        if index not in self.cache:
            c = self.cameras[index]
            with Image.open(self.root/c["name"]) as im:
                im = im.convert("RGB")
                if im.size != (c["width"],c["height"]):
                    raise ValueError("Image size differs from reconstructed camera")
                self.cache[index] = np.asarray(im,dtype=np.float32)/255
            while len(self.cache)>self.limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def split_indices(cameras, every=8):
    if len(cameras)<4:
        return list(range(len(cameras))), []
    test = list(range(0,len(cameras),max(2,int(every))))
    train = [i for i in range(len(cameras)) if i not in test]
    return train,test


def ssim(x,y):
    import torch.nn.functional as F
    # Inputs NHWC; local statistics with the same differentiable metric on eval.
    x,y = x.permute(0,3,1,2),y.permute(0,3,1,2)
    size = min(11,x.shape[-1],x.shape[-2])
    if size%2==0:
        size-=1
    mean = lambda t:F.avg_pool2d(t,size,1,size//2)
    a,b = mean(x),mean(y)
    varx,vary,cov = mean(x*x)-a*a,mean(y*y)-b*b,mean(x*y)-a*b
    return (((2*a*b+.01**2)*(2*cov+.03**2))/
            ((a*a+b*b+.01**2)*(varx+vary+.03**2))).mean()


def optimizers_for(params, scene_scale, backend="legacy", recipe=None):
    import torch
    lrs = {"means":1.6e-4*scene_scale,"scales":5e-3,"quats":1e-3,
           "opacities":5e-2,"sh0":2.5e-3,"shN":1.25e-4}
    kwargs = {"eps":recipe.get("adam_epsilon",1e-15),
              "betas":(recipe.get("adam_beta1",.9),recipe.get("adam_beta2",.999))} if recipe else {"eps":1e-15}
    if backend == "legacy":
        return {name:torch.optim.Adam([param],lr=lrs[name],**kwargs) for name,param in params.items()}
    from .training_performance import make_adam
    return {name:make_adam(param,lr=lrs[name],backend=backend,**kwargs) for name,param in params.items()}


def initial_parameters(points, colors, degree, max_splats, recipe=None):
    import torch
    if recipe is not None:
        from .training_recipe import initial_arrays
        values = initial_arrays(points,colors,degree,max_splats,recipe)
        return torch.nn.ParameterDict({key:torch.nn.Parameter(torch.as_tensor(value,dtype=torch.float32,device="cuda"))
                                       for key,value in values.items()})
    if len(points)>max_splats:
        idx = np.random.choice(len(points),max_splats,replace=False)
        points,colors = points[idx],colors[idx]
    if len(points)<4:
        raise ValueError("At least four sparse points required for Gaussian initialization")
    distances = cKDTree(points).query(points,k=4,workers=-1)[0][:,1:]
    scales = np.log(np.sqrt((distances**2).mean(-1)).clip(1e-6))
    n = len(points)
    sh = np.zeros((n,(degree+1)**2,3),np.float32)
    sh[:,0] = (colors-.5)/SH0
    q = np.zeros((n,4),np.float32); q[:,0] = 1
    values = {"means":points,"scales":np.repeat(scales[:,None],3,axis=1),"quats":q,
              "opacities":np.full(n,math.log(.1/.9),np.float32),"sh0":sh[:,:1],"shN":sh[:,1:]}
    return torch.nn.ParameterDict({key:torch.nn.Parameter(torch.as_tensor(value,dtype=torch.float32,device="cuda"))
                                   for key,value in values.items()})


def cloud_from_parameters(params):
    import torch
    return {"means":params["means"].detach().cpu().numpy(),
            "scales":params["scales"].detach().cpu().numpy(),
            "quats":params["quats"].detach().cpu().numpy(),
            "opacities":params["opacities"].detach().cpu().numpy(),
            "sh":torch.cat([params["sh0"],params["shN"]],1).detach().cpu().numpy()}


def atomic_checkpoint(file, state):
    import torch
    import os
    file = Path(file)
    file.parent.mkdir(parents=True,exist_ok=True)
    temp = file.with_name(file.name+".tmp")
    with temp.open("wb") as f:
        torch.save(state,f)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(file)


def checkpoint_state(params, optimizers, schedulers, strategy_state, strategy_config,
                     sampler, step, config, dataset_key, scene_scale, cap_policy="gradient_budget", recipe=None,
                     auxiliary=None, experiment_identity=None):
    import torch
    state = {"schema":1,"step":int(step),"parameters":{k:v.detach().clone() for k,v in params.items()},
            "optimizers":{k:o.state_dict() for k,o in optimizers.items()},
            "schedulers":{k:s.state_dict() for k,s in schedulers.items()},
            "strategy_state":strategy_state,"strategy_config":strategy_config,
            "sampler":sampler.state_dict(),"rng_python":random.getstate(),"rng_numpy":np.random.get_state(),
            "rng_torch":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all(),
            "config":config,"dataset_key":dataset_key,"scene_scale":scene_scale,
            "training_recipe":recipe,
            "cap_policy":cap_policy,"opacity_reset_fix":True,
            "implementation":"local-postshot independent public ADC; " + (recipe["name"] if recipe else "legacy_adc")}
    if experiment_identity is not None:
        state["studio_experiment"] = experiment_identity
        state["studio_auxiliary"] = auxiliary.state_dict() if auxiliary is not None else None
    return state


def make_strategy(strategy_config, max_splats, cap_policy="gradient_budget"):
    """Use upstream DefaultStrategy with two explicit, testable local changes.

    gsplat 1.5.3's chained `== 0 & step > 0` condition never resets opacity.
    Keep the installed package untouched and repair it in this subclass.
    A growth budget ranks candidates by their screen gradient and avoids the
    legacy post-growth opacity truncation that favored opaque floaters.
    """
    import torch
    from gsplat import DefaultStrategy
    from gsplat.strategy.ops import reset_opa
    class BoundedDefaultStrategy(DefaultStrategy):
        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            if cap_policy=="gradient_budget":
                budget=max(0,max_splats-len(params["means"]))
                scores=state["grad2d"]/state["count"].clamp_min(1)
                eligible=scores>self.grow_grad2d
                if int(eligible.sum())>budget:
                    mask=torch.zeros_like(eligible)
                    if budget:
                        selected=scores.masked_fill(~eligible,-float("inf")).topk(budget).indices
                        mask[selected]=True
                    state["grad2d"]=state["grad2d"]*mask
            return super()._grow_gs(params,optimizers,state,step)

        def step_post_backward(self,params,optimizers,state,step,info,packed=False):
            super().step_post_backward(params,optimizers,state,step,info,packed=packed)
            if step<self.refine_stop_iter and step>0 and step%self.reset_every==0:
                reset_opa(params=params,optimizers=optimizers,state=state,value=self.prune_opa*2)
    return BoundedDefaultStrategy(**strategy_config)


def load_checkpoint(file, map_location="cuda"):
    import torch
    # These checkpoints are written only into the local project's checkpoint folder.
    return torch.load(file,map_location=map_location,weights_only=False)


def render_parameters(params, camera, degree, absgrad=False, camera_tensors=None, projection_policy=None, training=False, with_depth=False):
    import torch
    from gsplat import rasterization
    from .projection_policy import projection_kwargs
    background=None
    if training and projection_policy and projection_policy.get("train_background")=="random_rgb_v1":
        background=torch.rand((1,3),device=params["means"].device,dtype=params["means"].dtype)
    if projection_policy and projection_policy.get("compositor")=="native_v1":
        if absgrad:raise ValueError("Native compositor does not implement absgrad")
        from .splat3_renderer import rasterize_parameters
        image,alpha,info = rasterize_parameters(params,camera,degree,camera_tensors=camera_tensors,
                                               policy=projection_policy,training=training,background=background)
        if with_depth:
            # Paired native compositor, same projections/visibility/AA/edits.
            # RGB3 depth replication preserves its frozen native RGB kernel.
            from .splat3_renderer import rasterize_projected
            colors = info["depths"][...,None].expand(-1,-1,3).contiguous()
            depth,_ = rasterize_projected(info["means2d"],info["conics"],colors,info["opacities"],
                info["isect_offsets"],info["flatten_ids"],camera["width"],camera["height"],info["tile_size"])
            image = torch.cat([image,depth[...,:1]/alpha.clamp_min(1e-10)],-1)
        return image,alpha,info
    return rasterization(means=params["means"],quats=params["quats"],
                         scales=torch.exp(params["scales"]),opacities=torch.sigmoid(params["opacities"]),
                         colors=torch.cat([params["sh0"],params["shN"]],dim=1),
                         viewmats=(camera_tensors["w2c"] if camera_tensors is not None else torch.tensor(camera["w2c"],dtype=torch.float32,device="cuda")[None]),
                         Ks=(camera_tensors["K"] if camera_tensors is not None else torch.tensor(camera["K"],dtype=torch.float32,device="cuda")[None]),
                         width=int(camera["width"]),height=int(camera["height"]),
                         sh_degree=degree,packed=False,absgrad=absgrad,
                         backgrounds=background,
                         **({"render_mode":"RGB+ED"} if with_depth else {}),
                         **projection_kwargs(camera,projection_policy,training=training))


@operation_locked("train")
def train(path, emit, control, pause_requested, steps=None, max_splats=None, resume=False, observer=None,
          checkpoint_observer=None):
    # Direct worker requests for a GS project must produce the same authoritative
    # experiment records as its CLI. run() re-enters this loop with an observer.
    if observer is None:
        prepared = project.load(path)
        if prepared.get("studio_dataset"):
            from .studio_training import run
            settings = prepared["settings"].copy()
            settings["steps"] = int(steps if steps is not None else settings["steps"])
            settings["max_splats"] = int(max_splats if max_splats is not None else settings["max_splats"])
            extra = prepared.get("studio_training",{})
            run(prepared["studio_dataset"]["package"],prepared["path"],steps=settings["steps"],resume=resume,
                recipe=settings["training_recipe"],photo_comp=extra.get("photo_comp",True),
                use_bilateral_grid=extra.get("use_bilateral_grid",True),
                use_sparse_depth=extra.get("sparse_depth_requested",extra.get("use_sparse_depth",True)),
                preview_every=extra.get("preview_every",0),emit=emit,settings=settings,
                external_control=control,external_pause=pause_requested)
            return project.load(path)
    import torch
    from gsplat.strategy.ops import remove
    from .training_recipe import recipe_config, recipe_from_checkpoint, resolve_ssim_backend, make_v2_strategy, training_loss
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; real gsplat training requires the isolated CUDA runtime")
    p = project.load(path)
    if "reconstruction" not in p or p.get("pipeline",{}).get("reconstruction",{}).get("status")!="complete":
        raise ValueError("Run camera reconstruction before training")
    studio = p.get("studio_dataset")
    if studio:
        from .studio_dataset import validate_project
        validate_project(p)
    elif reconstruction_key(p)!=p["reconstruction"]["key"]:
        raise ValueError("Input photos changed; reconstruct cameras again before training")
    cfg = p["settings"].copy()
    cfg["steps"] = int(steps if steps is not None else cfg["steps"])
    cfg["max_splats"] = int(max_splats if max_splats is not None else cfg["max_splats"])
    if cfg["steps"]<=0 or cfg["max_splats"]<4:
        raise ValueError("Steps must be positive and max_splats at least four")
    root = Path(p["path"])
    checkpoint = root/"checkpoints"/"latest.pt"
    reconstruction = p["reconstruction"]
    cameras = reconstruction["cameras"]
    # Preserve the public dataset's original filename order even though copied
    # assets include a content-hash prefix for duplicate-name safety.
    sources = {Path(a["path"]).name:a for a in p["assets"]["images"]}
    for camera in cameras:
        asset = sources.get(camera["name"],{})
        camera["source_name"] = Path(asset.get("source",camera["name"])).name
        camera["source_sha256"] = asset.get("source_sha256")
    if not studio:
        cameras.sort(key=lambda c:(c["source_name"],c["name"]))
    dataset_key = project.identity({"reconstruction":reconstruction["key"],
                                    "cameras":cameras,"test_every":cfg["test_every"]})
    train_ids,test_ids = (studio["train_indices"],studio["test_indices"]) if studio else split_indices(cameras,cfg["test_every"])
    if studio:
        dataset_key = studio["identity"]
    auxiliary = None
    if studio:
        from .studio_auxiliary import StudioAuxiliary
        auxiliary = StudioAuxiliary(p)
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    sampler = Sampler(train_ids,cfg["seed"])
    from .projection_policy import position_spatial_factor
    scene_scale = position_spatial_factor(cameras,train_ids)
    start = 0
    recipe = None
    if resume:
        if not checkpoint.is_file():
            raise ValueError("No complete checkpoint available to resume")
        state = load_checkpoint(checkpoint)
        if state["schema"]!=1 or state["dataset_key"]!=dataset_key:
            raise ValueError("Checkpoint belongs to a different dataset or engine schema")
        if studio:
            from .studio_training import validate_checkpoint
            validate_checkpoint(state,p["studio_training"],dataset_key,cfg,train_ids)
            auxiliary.load_state_dict(state.get("studio_auxiliary"))
        target_steps = cfg["steps"]
        # Preserve the original schedule horizon/algorithm configuration exactly.
        # Increasing target iterations extends the same optimizer and schedule.
        for key in ("max_splats","sh_degree","sh_degree_interval","seed","test_every"):
            if cfg[key]!=state["config"][key]:
                raise ValueError(f"Cannot change {key} when resuming; start new training")
        if cfg.get("training_recipe","legacy_adc") != state["config"].get("training_recipe","legacy_adc"):
            raise ValueError("Cannot change training recipe when resuming; start new training")
        if cfg.get("training_recipe_overrides",{}) != state["config"].get("training_recipe_overrides",{}):
            raise ValueError("Cannot change training recipe thresholds when resuming; start new training")
        cfg = state["config"]
        recipe = recipe_from_checkpoint(state)
        if recipe is not None:
            # Require the saved numerical backend, not whatever is available now.
            resolve_ssim_backend("cuda",recipe["ssim_backend"],torch.float32)
            if recipe["name"] in ("observed_v4","observed_v5","observed_v6"):
                from .splat3_stats import contribution_identity
                if recipe.get("contribution_artifact")!=contribution_identity(recipe):
                    raise ValueError("Saved contribution statistics binary differs; restore the pinned wheel before resuming")
            from .projection_policy import projection_artifact_identity
            projection_artifacts=projection_artifact_identity(recipe.get("projection"))
            if recipe.get("projection_artifacts",{})!=projection_artifacts:
                raise ValueError("Saved tile/AA projection binary differs; restore the pinned wheels before resuming")
        scene_scale = state["scene_scale"]
        params = torch.nn.ParameterDict({k:torch.nn.Parameter(v) for k,v in state["parameters"].items()})
        start = state["step"]
        strategy_config = state["strategy_config"]
        cap_policy = state.get("cap_policy","opacity_topk_legacy")
    else:
        target_steps = cfg["steps"]
        name = cfg.get("training_recipe","legacy_adc")
        if name not in ("legacy_adc","quality_v2","observed_v3","observed_v4","observed_v5","observed_v6"):
            raise ValueError("Unknown training recipe")
        if name in ("quality_v2","observed_v3","observed_v4","observed_v5","observed_v6"):
            overrides = dict(cfg.get("training_recipe_overrides",{}))
            for source,dest in (("refine_start","refine_start_iter"),("refine_stop","refine_stop_iter"),
                                ("refine_every","refine_every"),("reset_every","reset_every"),("ssim_lambda","ssim_lambda")):
                if source in cfg:
                    overrides[dest] = cfg[source]
            recipe = recipe_config(name,cfg["steps"],overrides,
                                   **({"train_image_count":len(train_ids)} if name in ("observed_v3","observed_v4","observed_v5","observed_v6") else {}))
            recipe["ssim_backend"] = resolve_ssim_backend("cuda",recipe["ssim_backend"],torch.float32)
            if name in ("observed_v4","observed_v5","observed_v6"):
                from .splat3_stats import contribution_identity
                recipe["contribution_artifact"]=contribution_identity(recipe)
            from .projection_policy import projection_artifact_identity
            projection_artifacts=projection_artifact_identity(recipe.get("projection"))
            if projection_artifacts:
                recipe["projection_artifacts"]=projection_artifacts
        with np.load(Path(reconstruction["path"])/"sparse_points.npz") as sparse:
            params = initial_parameters(sparse["points"],sparse["colors"],cfg["sh_degree"],cfg["max_splats"],recipe)
        factor=cfg["steps"]/30000
        strategy_config = {"refine_start_iter":int(cfg.get("refine_start",round(500*factor))),
                           "refine_stop_iter":int(cfg.get("refine_stop",round(15000*factor))),
                           "refine_every":int(cfg.get("refine_every",max(1,round(100*factor)))),
                           "reset_every":int(cfg.get("reset_every",max(1,round(3000*factor)))),"verbose":False}
        cap_policy = "gradient_budget"
        if recipe is not None:
            strategy_config,cap_policy = recipe["strategy"],recipe["cap_policy"]
    if not resume and recipe is not None:
        scene_scale=position_spatial_factor(cameras,train_ids,recipe.get("position_lr_spatial","public_all_median"))
    optimized = cfg.get("training_backend","legacy") == "optimized"
    if optimized:
        from .training_performance import ImageTensorCache
        images = ImageTensorCache(reconstruction["image_path"],cameras,
            host_cache_bytes=int(cfg.get("image_host_cache_mb",1024))*1048576,
            gpu_cache_bytes=int(cfg.get("image_gpu_cache_mb",256))*1048576)
    else:
        images = Images(reconstruction["image_path"],cameras)
    optimizers = optimizers_for(params,scene_scale,cfg.get("adam_backend","auto") if optimized else "legacy",recipe)
    from .projection_policy import position_scheduler
    schedulers = {"means":position_scheduler(optimizers["means"],
        recipe.get("position_lr_horizon",cfg["steps"]) if recipe else cfg["steps"],
        recipe.get("position_lr_phase","post_step_legacy") if recipe else "post_step_legacy")}
    is_v2 = recipe is not None and recipe["name"] in ("quality_v2","observed_v3")
    is_observed = recipe is not None and recipe["name"] in ("observed_v4","observed_v5","observed_v6")
    if is_observed:
        from .splat3_strategy import ObservedStrategy
        strategy=ObservedStrategy(recipe,cfg["max_splats"])
    else:
        strategy = make_v2_strategy(recipe,cfg["max_splats"]) if is_v2 else make_strategy(strategy_config,cfg["max_splats"],cap_policy)
    strategy.check_sanity(params,optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    if resume:
        for k,o in optimizers.items():
            o.load_state_dict(state["optimizers"][k])
        for k,s in schedulers.items():
            s.load_state_dict(state["schedulers"][k])
        strategy_state = state["strategy_state"]
        sampler.load_state_dict(state["sampler"])
        random.setstate(state["rng_python"]); np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["rng_cuda"]])
        # Refinement replaces parameter/moment tensors. Keeping the loaded
        # bundle alive would retain the obsolete pre-resume tensors on CUDA.
        del state
    p["status"] = "training"
    p.pop("last_error",None)
    camera_cache=None
    if recipe and recipe.get("camera_tensor_cache",False):
        camera_cache=[{"w2c":torch.tensor(c["w2c"],dtype=torch.float32,device="cuda")[None],
                       "K":torch.tensor(c["K"],dtype=torch.float32,device="cuda")[None]} for c in cameras]
        if recipe.get("camera_center_cache",False):
            # Same CUDA inverse and shape as the former per-view forward.
            # Cache the immutable result instead of synchronizing every step.
            for camera_tensors in camera_cache:
                camera_tensors["center"]=torch.inverse(camera_tensors["w2c"])[...,:3,3]
    p["settings"]["steps"],p["settings"]["max_splats"] = target_steps,cfg["max_splats"]
    project.save(p)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    log_path = root/"logs"/"metrics.jsonl"
    final_model = root/"models"/"trained.ply"
    step = start
    algorithm = (f"independent {recipe['name']} recovered policy with public projection" if is_observed else
                 "independent " + recipe["name"] + " ADC" if is_v2 else "gsplat DefaultStrategy ADC")
    def save_checkpoint():
        atomic_checkpoint(checkpoint,checkpoint_state(params,optimizers,schedulers,strategy_state,
                                                     strategy_config,sampler,step,cfg,dataset_key,scene_scale,cap_policy,recipe,
                                                     auxiliary=auxiliary,experiment_identity=p.get("studio_training")))
        p["training"] = {"step":step,"target_steps":target_steps,"checkpoint":str(checkpoint),
                         "count":len(params["means"]),"algorithm":algorithm,"recipe":recipe,
                         "train_indices":train_ids,"test_indices":test_ids,"dataset_key":dataset_key,
                         "seconds_this_run":time.perf_counter()-started,
                         "peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1048576,
                         "peak_reserved_memory_mb":torch.cuda.max_memory_reserved()/1048576,
                         "memory_metric":"PyTorch allocated/reserved peaks, not complete process VRAM"}
        project.save(p)
        if checkpoint_observer is not None:
            checkpoint_observer(step)
    def boundary_context():
        # Lazy snapshot is used by the explicit restoration audit only.
        return {"step":step,"target":target_steps,"project":p,
                "snapshot":lambda:checkpoint_state(params,optimizers,schedulers,strategy_state,
                    strategy_config,sampler,step,cfg,dataset_key,scene_scale,cap_policy,recipe,
                    auxiliary=auxiliary,experiment_identity=p.get("studio_training"))}
    def preview():
        nonlocal final_model
        if studio:
            # Pause releases the project for independent editing. A later
            # training continuation must not replace an editor's source cloud.
            final_model = root/"models"/f"trained-step-{step:08d}.ply"
        write_ply(final_model,cloud_from_parameters(params))
        emit("preview",{"path":str(final_model),"step":step,"count":len(params["means"])})
    file_controller = None
    if observer is None:
        from .studio_control import TrainingControl
        file_controller = TrainingControl(root,project.identity({"project":p["id"],"dataset":dataset_key}))
        observer = lambda _params,context,save:file_controller.observe(context["step"],save,target=target_steps)
    # Establish a valid initial checkpoint before any GPU optimization.
    if not resume:
        save_checkpoint()
    emit("log",{"message":f"Independent ADC training: {len(train_ids)} train / {len(test_ids)} held-out cameras; {len(params['means'])} initial Gaussians"})
    emit("log",{"message":"Resolved training configuration","algorithm":algorithm,"recipe":recipe,
                "training_backend":cfg.get("training_backend","legacy"),"strategy":strategy_config})
    try:
        with log_path.open("a",encoding="utf-8") as metric_log:
            while step<target_steps:
                if observer is not None:
                    action = observer(params,boundary_context(),save_checkpoint)
                    if action in ("pause","stop"):
                        save_checkpoint()
                        preview()
                        p["status"] = "paused" if action == "pause" else "stopped"
                        break
                if control.is_set():
                    save_checkpoint()
                    preview()
                    p["status"] = "paused" if pause_requested.is_set() else "cancelled"
                    break
                if is_observed:strategy.prepare_step(params,optimizers,strategy_state,step,cameras,train_ids)
                index = sampler.next()
                camera = cameras[index]
                target = images.get_tensor(index,device="cuda") if optimized else torch.from_numpy(images.get(index)).to("cuda")[None]
                degree = min(cfg["sh_degree"],step//cfg["sh_degree_interval"])
                prediction,alpha,info = render_parameters(params,camera,degree,absgrad=bool(strategy.absgrad),
                    **({"with_depth":True} if auxiliary is not None and auxiliary.need_depth else {}),
                    **({"projection_policy":recipe["projection"],"training":True} if recipe and recipe.get("projection") else {}),
                    **({"camera_tensors":camera_cache[index]} if camera_cache is not None else {}))
                depth,keep = None,None
                if auxiliary is not None:
                    if auxiliary.need_depth:
                        depth,prediction = prediction[...,3],prediction[...,:3]
                    prediction = auxiliary.compensate(prediction,index)
                    keep = auxiliary.mask(index,prediction.device)
                strategy.step_pre_backward(params,optimizers,strategy_state,step,info)
                if keep is not None:
                    loss_parts = auxiliary.loss(prediction,target,keep,params,recipe,cfg,is_observed)
                    loss = loss_parts["loss"]
                elif is_observed:
                    from .splat3_strategy import observed_loss_and_error
                    loss_parts=observed_loss_and_error(prediction,target,params,recipe)
                    loss=loss_parts["loss"]
                elif recipe is not None:
                    loss_parts = training_loss(prediction,target,params,recipe)
                    loss = loss_parts["loss"]
                else:
                    l1 = (prediction-target).abs().mean()
                    weight = float(cfg.get("ssim_lambda",.2))
                    if optimized:
                        from .training_performance import uniform_ssim
                        # Preserve the legacy pool reduction order, especially in
                        # low-texture regions where variance cancellation matters.
                        score = uniform_ssim(prediction,target,method="direct")
                    else:
                        score = ssim(prediction,target)
                    loss = (1-weight)*l1+weight*(1-score)
                if auxiliary is not None and (auxiliary.photometric is not None or auxiliary.need_depth):
                    loss = loss+auxiliary.extra_loss(depth,index,prediction)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; last valid checkpoint is preserved")
                if is_observed:strategy.collect_statistics(strategy_state,info,loss_parts["densify_error_map"])
                loss.backward()
                if is_observed:strategy.add_regularization(params,strategy_state,info)
                if is_observed:
                    # Native mode3 consumes statistics and clears selected
                    # parent/child gradients and moments BEFORE Adam updates.
                    strategy.step_post_backward(params,optimizers,strategy_state,step+1,info,packed=False)
                # Optimizers consume the current gradients before strategy changes tensors.
                for optimizer in optimizers.values():
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                if auxiliary is not None:
                    auxiliary.step()
                for scheduler in schedulers.values():
                    scheduler.step()
                if not is_observed:strategy.step_post_backward(params,optimizers,strategy_state,step,info,packed=False)
                guard_every = int(recipe.get("global_screen_guard_every",0)) if recipe else 0
                if guard_every and step>0 and step%guard_every==0 and step<strategy.refine_stop_iter:
                    from .training_recipe import training_camera_guard
                    mask,_ = training_camera_guard(params,cameras,train_ids,recipe["global_screen_guard_threshold"])
                    removed = int(mask.sum())
                    if removed == len(params["means"]):
                        raise RuntimeError("Global training-camera size guard would remove all Gaussians")
                    if removed:
                        remove(params=params,optimizers=optimizers,state=strategy_state,mask=mask)
                    emit("log",{"message":"Training-camera screen guard","step":step,"removed":removed,
                                "training_cameras":len(train_ids),"heldout_RGB_used":False})
                if len(params["means"])>cfg["max_splats"] and cap_policy=="opacity_topk_legacy":
                    n = len(params["means"])
                    keep = params["opacities"].detach().topk(cfg["max_splats"]).indices
                    mask = torch.ones(n,dtype=torch.bool,device="cuda"); mask[keep]=False
                    remove(params=params,optimizers=optimizers,state=strategy_state,mask=mask)
                if len(params["means"])<1:
                    raise RuntimeError("Densification pruned all Gaussians; restart with different reconstruction")
                step+=1
                if step==1 or step%10==0 or step==target_steps:
                    mse = ((prediction.detach()-target)**2).mean().clamp_min(1e-12)
                    if keep is not None:
                        mse = torch.where(keep[...,None],(prediction.detach()-target).square(),0).sum()/(3*keep.sum())
                        mse = mse.clamp_min(1e-12)
                    metric = {"step":step,"loss":float(loss.detach()),"psnr":float(-10*torch.log10(mse)),
                              "splats":len(params["means"]),"elapsed_seconds":time.perf_counter()-started,
                              "gpu_memory_mb":torch.cuda.max_memory_allocated()/1048576,
                              "gpu_reserved_memory_mb":torch.cuda.max_memory_reserved()/1048576,
                              "sh_degree":degree,"camera":camera["name"]}
                    if recipe is not None:
                        metric["train_ssim_gaussian"] = float(loss_parts["ssim"].detach())
                        metric["refinement"] = strategy_state.get("last_refinement")
                    emit("metric",metric)
                    metric_log.write(json.dumps(metric)+"\n"); metric_log.flush()
                    emit("progress",{"stage":"training","progress":step/target_steps,"message":f"Iteration {step}/{target_steps}"})
                dedup_final=bool(recipe and recipe.get("deduplicate_final_artifacts",False) and step==target_steps)
                if step%int(cfg["checkpoint_every"])==0 and not dedup_final:
                    save_checkpoint()
                if step%int(cfg["preview_every"])==0 and not dedup_final:
                    preview()
            else:
                action = observer(params,boundary_context(),save_checkpoint) if observer is not None else None
                p["status"] = "paused" if action == "pause" else "stopped" if action == "stop" else "trained"
                save_checkpoint()
                preview()
    except torch.cuda.OutOfMemoryError as exc:
        p["status"] = "error"
        p["last_error"] = "GPU out of memory; reduce image size/max_splats. Last complete checkpoint preserved."
        project.save(p)
        if file_controller is not None:
            file_controller.complete("failed",step)
        torch.cuda.empty_cache()
        raise RuntimeError(p["last_error"]) from exc
    except BaseException as exc:
        p["status"],p["last_error"] = "error",str(exc)
        project.save(p)
        if file_controller is not None:
            file_controller.complete("failed",step)
        raise
    existing = next((m for m in p["models"] if m.get("role")=="training"),None)
    preserve_snapshot = bool(studio and existing and existing.get("editor") and Path(existing["path"])!=final_model)
    model = project.model_record(final_model,len(params["means"]),"Trained Gaussian model",existing["id"] if existing and not preserve_snapshot else None)
    model["role"] = "training"
    if preserve_snapshot:
        existing["role"],existing["visible"],existing["superseded_by"] = "training_snapshot",False,model["id"]
        p["models"].append(model)
    elif existing:
        model.update({k:existing[k] for k in ("transform","crop","visible")})
        p["models"] = [model if m["id"]==existing["id"] else m for m in p["models"]]
    else:
        p["models"].append(model)
    p["training"] = {"step":step,"target_steps":target_steps,"checkpoint":str(checkpoint),
                     "count":len(params["means"]),"algorithm":algorithm,"recipe":recipe,
                     "train_indices":train_ids,"test_indices":test_ids,"dataset_key":dataset_key,
                     "seconds_this_run":time.perf_counter()-started,
                     "peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/1048576,
                     "peak_reserved_memory_mb":torch.cuda.max_memory_reserved()/1048576,
                     "memory_metric":"PyTorch allocated/reserved peaks, not complete process VRAM",
                     "strategy_config":strategy_config,"cap_policy":cap_policy,"opacity_reset_fix":True}
    p["pipeline"]["training"] = {"key":project.identity({"dataset":dataset_key,"config":cfg}),
                                "status":p["status"],"step":step}
    project.save(p)
    if file_controller is not None:
        file_controller.complete("succeeded" if p["status"]=="trained" else "paused" if p["status"]=="paused" else "stopped",step)
    del params,optimizers,schedulers,strategy_state
    torch.cuda.empty_cache()
    return p


def tensors_from_cloud(cloud):
    import torch
    return {"means":torch.as_tensor(cloud["means"],device="cuda",dtype=torch.float32),
            "scales":torch.as_tensor(cloud["scales"],device="cuda",dtype=torch.float32),
            "quats":torch.as_tensor(cloud["quats"],device="cuda",dtype=torch.float32),
            "opacities":torch.as_tensor(cloud["opacities"],device="cuda",dtype=torch.float32),
            "sh0":torch.as_tensor(cloud["sh"][:,:1],device="cuda",dtype=torch.float32),
            "shN":torch.as_tensor(cloud["sh"][:,1:],device="cuda",dtype=torch.float32)}


def scene_parameters(p):
    import torch
    models = [m for m in p["models"] if m.get("visible",True)]
    if not models:
        raise ValueError("No visible Gaussian models to render")
    from .studio_edit import model_cloud
    clouds = [model_cloud(p,m) for m in models]
    clouds = [c for c in clouds if len(c["means"])]
    if not clouds:
        raise ValueError("Crop removed every visible Gaussian")
    k = max(c["sh"].shape[1] for c in clouds)
    for c in clouds:
        if c["sh"].shape[1]<k:
            c["sh"] = np.pad(c["sh"],((0,0),(0,k-c["sh"].shape[1]),(0,0)))
    return tensors_from_cloud({key:np.concatenate([c[key] for c in clouds]) for key in clouds[0]}),round(math.sqrt(k)-1)


@operation_locked("render")
def render(path, camera=None, output=None):
    import torch
    p = project.load(path)
    if camera is None:
        if not p.get("reconstruction",{}).get("cameras"):
            raise ValueError("Provide an RDF projection camera with K,w2c,width,height")
        camera = p["reconstruction"]["cameras"][0]
    params,degree = scene_parameters(p)
    with torch.no_grad():
        image,alpha,_ = render_parameters(params,camera,degree,projection_policy=(p.get("training",{}).get("recipe") or {}).get("projection"))
    file = Path(output) if output else Path(p["path"])/"renders"/"render.png"
    file.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray((image[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(file)
    return {"path":str(file.resolve()),"width":camera["width"],"height":camera["height"]}


@operation_locked("evaluate")
def evaluate(path, emit, control):
    import torch
    p = project.load(path)
    if p.get("studio_dataset"):
        from .studio_training import evaluate_project
        return evaluate_project(p,emit,control)
    if "reconstruction" not in p:
        raise ValueError("Evaluation requires reconstructed camera poses")
    cams = p["reconstruction"]["cameras"]
    _,indices = split_indices(cams,p["settings"]["test_every"])
    if not indices:
        raise ValueError("Evaluation requires at least four registered cameras for a held-out split")
    params,degree = scene_parameters(p)
    images = Images(p["reconstruction"]["image_path"],cams)
    rows = []
    for i,index in enumerate(indices):
        if control.is_set():
            raise Cancelled("Evaluation cancelled")
        with torch.no_grad():
            image,alpha,_ = render_parameters(params,cams[index],degree,projection_policy=(p.get("training",{}).get("recipe") or {}).get("projection"))
            truth = torch.from_numpy(images.get(index)).to("cuda")[None]
            mse = ((image-truth)**2).mean().clamp_min(1e-12)
            row = {"name":cams[index]["name"],"index":index,
                   "psnr":float(-10*torch.log10(mse)),"ssim":float(ssim(image,truth))}
        file = Path(p["path"])/"renders"/f"eval_{index:04d}.png"
        Image.fromarray((image[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(file)
        row["render"] = str(file)
        rows.append(row)
        emit("progress",{"stage":"evaluation","progress":(i+1)/len(indices),"message":row["name"]})
    result = {"views":rows,"psnr":float(np.mean([r["psnr"] for r in rows])),
              "ssim":float(np.mean([r["ssim"] for r in rows])),"count":len(params["means"]),
              "metric":"RGB sRGB PSNR; 11x11 uniform-window SSIM; black background",
              "split":"Every test_every-th sorted reconstructed camera; excluded from training"}
    project.atomic_json(Path(p["path"])/"evaluation.json",result)
    return result
