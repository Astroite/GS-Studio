"""Independent implementation of statically verified mode-3 policy components.

The current-cache host/PTX evidence is retained under evidence/splat3/round2-*.
This module never imports or executes original code. Projection/visibility and
contribution kernels remain explicitly identified public-rasterizer substitutes.
Random variates use checkpointed Torch RNG, not the original PCG stream.
"""
from __future__ import annotations

import math


def growth_budget(active: int, maximum: int) -> int:
    """Recovered host budget; .01f is widened before adding1 in float64.

    The budget bounds new children, not the number of eligible parents. The
    representation here compacts inactive rows instead of retaining hole slots.
    """
    if isinstance(active, bool) or isinstance(maximum, bool) or int(active)!=active or int(maximum)!=maximum or active<0 or maximum<1:
        raise ValueError("Integer active>=0 and maximum>=1 required")
    active,maximum=int(active),int(maximum)
    x=max(1.-active/maximum,0.)
    q=min(1.0099999997764826-math.sqrt(x),1.)
    increment=int((1.-(1.-q)**2)*maximum)-int((1.-x)*maximum)
    return min(max(increment,maximum//200),max(maximum-active,0))


def select_parents(sum_error, view_age, opacity_logits, anchors, maximum: int,
                   min_error: float=.1, *, uniforms=None):
    """Mature error candidates and weighted CDF draws, then unique ID order.

    The original oversamples Q=trunc(B*(1+B/C)) with replacement. Candidate
    weights are alpha+max(mean_error-min_error,0). Torch cumsum/searchsorted
    match this distribution, not the original parallel scan rounding or PCG.
    """
    import torch
    n=len(sum_error)
    if any(v.shape!=(n,) for v in (sum_error,view_age,opacity_logits,anchors)):
        raise ValueError("One statistic/opacity/anchor per Gaussian required")
    budget=growth_budget(n,maximum)
    if budget==0:
        return torch.empty(0,dtype=torch.long,device=sum_error.device),{"eligible":0,"budget":0,"draws":0}
    mean_error=torch.where(view_age>=10,sum_error/view_age.clamp_min(1),0.)
    # i_197 replaces mature sum1 values with their mean even if there are no
    # eligible parents. No-candidate passes do not clear this retained state.
    with torch.no_grad():sum_error.copy_(torch.where(view_age>=10,mean_error,sum_error))
    eligible=(view_age>=10)&(anchors>=0)&(mean_error>min_error)
    ids=torch.where(eligible)[0]
    if len(ids)<=budget:
        return ids,{"eligible":len(ids),"budget":budget,"draws":0}
    draws=int(budget*(1.+budget/len(ids)))
    weights=opacity_logits[ids].sigmoid()+(mean_error[ids]-min_error).clamp_min(0)
    cdf=weights.cumsum(0)
    if uniforms is None:
        uniforms=torch.rand(draws,dtype=cdf.dtype,device=cdf.device)
    elif uniforms.shape!=(draws,) or uniforms.device!=cdf.device:
        raise ValueError("Explicit variates must match the frozen draw count/device")
    targets=(uniforms*cdf[-1]).clamp_min(torch.finfo(cdf.dtype).tiny)
    selected=ids[torch.searchsorted(cdf,targets).clamp_max(len(ids)-1)].unique(sorted=True)
    # Original host truncates the compacted selected-ID span to its budget.
    return selected[:budget],{"eligible":len(ids),"budget":budget,"draws":draws,
                              "unique_before_budget":len(selected)}


def split_selected(params, optimizers, state, selected, *, noise=None):
    """One retained-position child plus one perturbed parent; reset both Adam.

    Both scales *=5/8 and both alpha probabilities *=4/5. The independent
    parameter convention is wxyz; the recovered model view stores xyzw.
    All untouched rows and optimizer counters/groups are preserved.
    """
    import torch
    import torch.nn.functional as F
    from gsplat.utils import normalized_quat_to_rotmat
    from gsplat.strategy.ops import _update_param_with_optimizer
    n=len(params["means"])
    selected=selected.to(device=params["means"].device,dtype=torch.long)
    if selected.ndim!=1 or (len(selected) and (int(selected.min())<0 or int(selected.max())>=n)):
        raise ValueError("Valid unique parent indices required")
    if len(selected)!=len(selected.unique()):
        raise ValueError("A parent can split only once in one refinement")
    count=len(selected)
    if not count:return
    with torch.no_grad():
        scales=params["scales"][selected].exp()
        rotations=normalized_quat_to_rotmat(F.normalize(params["quats"][selected],dim=-1))
        if noise is None:noise=torch.randn((count,3),dtype=scales.dtype,device=scales.device)
        if noise.shape!=(count,3) or noise.device!=scales.device:
            raise ValueError("One3D normal variate per selected parent required")
        displacement=torch.einsum("nij,nj->ni",rotations,scales*noise)
        old_means=params["means"][selected].clone()
        new_scales=params["scales"][selected]+math.log(.625)
        new_opacity=torch.logit(params["opacities"][selected].sigmoid()*.8)

        def parameter(name,value):
            kept=value.detach().clone()
            child=value[selected].detach().clone()
            if name=="means":kept[selected]=old_means+displacement;child=old_means
            elif name=="scales":kept[selected]=new_scales;child=new_scales
            elif name=="opacities":kept[selected]=new_opacity;child=new_opacity
            result=torch.nn.Parameter(torch.cat((kept,child)),requires_grad=value.requires_grad)
            if value.grad is not None:
                gradient=value.grad.detach().clone();gradient[selected]=0
                result.grad=torch.cat((gradient,torch.zeros_like(value.grad[selected])))
            return result

        def optimizer_field(name,value):
            kept=value.clone();kept[selected]=0
            return torch.cat((kept,torch.zeros_like(value[selected])))

        _update_param_with_optimizer(parameter,optimizer_field,params,optimizers)
        for key,value in list(state.items()):
            if isinstance(value,torch.Tensor) and value.ndim and len(value)==n:
                kept=value.clone();child=value[selected].clone()
                if key=="sum2":kept[selected]=-20.;child.fill_(-20.)
                elif key=="max_weight":kept[selected]=0.;child.zero_()
                elif key=="sum1":child.zero_()
                state[key]=torch.cat((kept,child))


def clear_completed_windows(state, *, reset=False):
    """Recovered i_YH7: preserve an age strictly between0 and10 only."""
    import torch
    age=state["sum2"]
    keep=(age>0)&(age<10) if not reset else torch.zeros_like(age,dtype=torch.bool)
    state["sum1"].masked_fill_(~keep,0.)
    state["sum2"].copy_(torch.where(keep,age,age.clamp_max(0)))


def compact_dead(params,optimizers,state,mask):
    """Physically remove logical inactive rows while retaining current gradients.

    Native mode3 runs before Adam. Public gsplat's post-step remove deliberately
    drops gradients; that operation cannot implement this ordering unchanged.
    """
    import torch
    from gsplat.strategy.ops import _update_param_with_optimizer
    with torch.no_grad():
        keep=torch.where(~mask)[0]
        n=len(mask)
        def parameter(name,value):
            result=torch.nn.Parameter(value[keep],requires_grad=value.requires_grad)
            if value.grad is not None:result.grad=value.grad[keep]
            return result
        _update_param_with_optimizer(parameter,lambda key,value:value[keep],params,optimizers)
        for key,value in list(state.items()):
            if isinstance(value,torch.Tensor) and value.ndim and len(value)==n:
                state[key]=value[keep]


def reset_opacity_before_adam(params,optimizers,cap):
    import torch
    with torch.no_grad():
        opacity=params["opacities"]
        opacity.clamp_(max=math.log(cap/(1-cap)))
        if opacity.grad is not None:opacity.grad.zero_()
        for name,value in optimizers["opacities"].state.get(opacity,{}).items():
            if name!="step" and isinstance(value,torch.Tensor):value.zero_()


def observed_loss_and_error(prediction,target,params,recipe):
    """Reuse one fused Gaussian map for photometric gradient and error stats.

    The recovered diagnostic pixel map has globally normalized RGB L1 and an
    unnormalized channel-mean DSSIM term. It is distinct from the scalar training
    objective; its producer/consumer binding is recorded in the native evidence.
    """
    import torch
    from fused_ssim import FusedSSIMMap
    if recipe["ssim_backend"]!="fused" or recipe["ssim_padding"]!="same":
        raise RuntimeError("Observed v4 requires its frozen fused/same Gaussian map backend")
    x,y=prediction.permute(0,3,1,2).contiguous(),target.permute(0,3,1,2).contiguous()
    score_map=FusedSSIMMap.apply(.01**2,.03**2,x,y,"same",True,2)
    absolute=(prediction-target).abs()
    score,l1=score_map.mean(),absolute.mean()
    loss=.8*l1+.2*(1-score)
    if recipe["opacity_reg"]:loss=loss+params["opacities"].sigmoid().mean()*recipe["opacity_reg"]
    if recipe["scale_reg"]:loss=loss+params["scales"].exp().mean()*recipe["scale_reg"]
    with torch.no_grad():
        height,width=prediction.shape[1:3]
        error=.8*absolute.detach().sum(-1)[0]/(3*width*height)+.2*(1-score_map.detach().mean(1)[0])
    return {"loss":loss,"l1":l1,"ssim":score,"densify_error_map":error.contiguous()}


class ObservedStrategy:
    """Recovered mode-3 policy over explicitly substituted public projections."""
    absgrad=False

    def __init__(self,recipe,max_splats):
        self.recipe,self.max_splats=recipe,int(max_splats)
        cfg=recipe["strategy"]
        self.refine_start_iter=cfg["refine_start_iter"]
        self.refine_stop_iter=2**31-1  # Native densify_end_step=0: no end gate.
        self.refine_every=cfg["refine_every"]
        self.reset_every=cfg["reset_every"]
        self.prune_opa=cfg["prune_opa"]
        self.n_images=recipe["train_image_count"]

    def check_sanity(self,params,optimizers):
        from gsplat import DefaultStrategy
        DefaultStrategy().check_sanity(params,optimizers)
        # Fail before creating a checkpoint with an unavailable required backend.
        from .splat3_stats import contribution_identity
        contribution_identity(self.recipe)

    def initialize_state(self,scene_scale):
        return {"scene_scale":scene_scale,"sum1":None,"sum2":None,"max_weight":None,
                "min_scale":None,"last_anchor_step":-1,"anchor_refreshes":0}

    def prepare_step(self,params,optimizers,state,step,cameras,train_ids):
        """Refresh anchor on the observed N-image epoch, never on heldout RGB."""
        import time
        import torch
        from .splat3_regularization import compute_min_scale_anchors
        if state["sum1"] is None:
            for name in ("sum1","sum2","max_weight"):
                state[name]=torch.zeros(len(params["means"]),device=params["means"].device)
        if step%self.n_images or state["last_anchor_step"]==step:return
        started=time.perf_counter()
        state["min_scale"]=compute_min_scale_anchors(params,cameras,train_ids,self.recipe["min_scale_px"])
        invalid=state["min_scale"]<0
        count=int(invalid.sum())
        if count==len(invalid):raise RuntimeError("No Gaussian is supported by the training cameras")
        if count:compact_dead(params,optimizers,state,invalid)
        state["last_anchor_step"]=int(step)
        state["anchor_refreshes"]+=1
        state["last_anchor_refresh"]={"step":int(step),"pruned_invalid":count,
                                       "wall_seconds":time.perf_counter()-started}
        state["anchor_wall_seconds_total"]=state.get("anchor_wall_seconds_total",0.)+state["last_anchor_refresh"]["wall_seconds"]

    def step_pre_backward(self,params,optimizers,state,step,info):
        pass  # This policy consumes photometric contributions, not screen gradients.

    def collect_statistics(self,state,info,error_map):
        import torch
        from .splat3_stats import contribution_stats
        native_renderer=self.recipe["contribution_backend"]=="native_renderer_v1"
        if native_renderer:
            from .splat3_renderer import contribution_stats
        with torch.no_grad():
            stats=contribution_stats(info["means2d"].detach().contiguous(),
                info["conics"].detach().contiguous(),info["opacities"].detach().contiguous(),
                info["isect_offsets"],info["flatten_ids"],error_map,
                info["width"],info["height"],info["tile_size"],
                **({} if native_renderer else {"backend":self.recipe["contribution_backend"]}))
            visible=(info["radii"]>0).all(-1)[0]
            # Projection backward discards the error accumulated during a
            # negative age. After20 supported views, begin a fresh10-view window.
            state["sum1"].add_(stats.weighted_error_sum)
            state["sum1"].masked_fill_(visible&(state["sum2"]<0),0.)
            state["sum2"].add_(visible.to(state["sum2"].dtype))
            torch.maximum(state["max_weight"],stats.max_weight,out=state["max_weight"])

    def add_regularization(self,params,state,info):
        if self.recipe["scale_anchor_regularization"]:
            from .splat3_regularization import fast_add_regularization_gradient_
            fast_add_regularization_gradient_(params["scales"],state["min_scale"],
                (info["radii"]>0).all(-1)[0],upper=self.recipe["scale_rel_l2reg"],
                lower=self.recipe["scale_lower_l2reg"])

    def step_post_backward(self,params,optimizers,state,step,info,packed=False):
        import torch
        from .training_recipe import trim_cuda_cache_if_needed
        with torch.no_grad():
            phase=step%self.reset_every
            allowed=step<=self.reset_every or phase==0 or phase>=self.n_images
            active_phase=step>=max(self.refine_start_iter,2*self.n_images)
            opacity_due=active_phase and step%self.refine_every==0 and allowed
            weight_due=active_phase and step%self.n_images==0 and allowed
            removed=0
            if opacity_due or weight_due:
                dead=state["min_scale"]<0
                if opacity_due:dead|=params["opacities"].sigmoid()<self.prune_opa
                if weight_due:
                    mature=state["sum2"]>=10
                    dead|=mature&(state["max_weight"]<self.recipe["native_prune_min_weight"])
                    state["max_weight"].masked_fill_(mature,0.)
                removed=int(dead.sum())
                if removed==len(dead):raise RuntimeError("Observed pruning removed all Gaussians")
                if removed:compact_dead(params,optimizers,state,dead)
            if step>0 and step>=self.refine_start_iter and step<self.refine_stop_iter and step%self.refine_every==0:
                if not allowed:
                    clear_completed_windows(state,reset=True)
                else:
                    selected,details=select_parents(state["sum1"],state["sum2"],params["opacities"],
                        state["min_scale"],self.max_splats,self.recipe["native_min_error"])
                    if len(selected):
                        clear_completed_windows(state)
                        split_selected(params,optimizers,state,selected)
                    state["last_refinement"]={"step":int(step),"pruned":removed,"cloned":0,
                        "split":len(selected),"count":len(params["means"]),**details}
                    trim=trim_cuda_cache_if_needed(self.recipe,params["means"].device)
                    if trim is not None:
                        state["cache_trim_count"]=state.get("cache_trim_count",0)+1
                        state["last_cache_trim"]={"step":int(step),**trim}
            tail_ok=step+self.recipe["reset_stop_margin"]<=self.recipe["schedule_steps"]
            if step>0 and phase==0 and tail_ok:
                reset_opacity_before_adam(params,optimizers,self.prune_opa*self.recipe["opacity_reset_multiplier"])
