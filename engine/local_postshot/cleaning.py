"""Explicit model editing: prune oversized training-camera projections.

This is a transparent public gsplat screen-size rule, not a recovered Postshot
feature. It never reads held-out RGBs and never modifies a raw checkpoint.
"""
from pathlib import Path
import time
import uuid
import numpy as np
from . import project
from .formats import write_ply
from .locks import operation_locked
from .reconstruction import Cancelled
from .training import split_indices


def radius_exceeds(radii, width, height, threshold):
    """Same per-camera normalized-radius definition as DefaultStrategy."""
    return np.max(np.asarray(radii),axis=-1)/max(width,height)>threshold


@operation_locked("model.clean")
def clean(path, model_id, emit, control, max_screen_radius=.15):
    import torch
    from gsplat.cuda._wrapper import fully_fused_projection
    threshold=float(max_screen_radius)
    if not 0<threshold<=1:
        raise ValueError("Normalized screen radius threshold must be greater than zero and at most one")
    p=project.load(path)
    if not p.get("reconstruction",{}).get("cameras"):
        raise ValueError("Screen-size cleaning requires reconstructed camera poses")
    source=next((m for m in p["models"] if m["id"]==model_id),None)
    if source is None:raise ValueError("Unknown model ID")
    cameras=p["reconstruction"]["cameras"]
    train_ids=p.get("training",{}).get("train_indices")
    if train_ids is None:train_ids,_=split_indices(cameras,p["settings"]["test_every"])
    if not train_ids:raise ValueError("No training cameras available")
    from .studio_edit import model_cloud,cleaned_history_record
    # All geometry comes from the same committed deletion/crop/transform state
    # used by export and native rendering. Original indices remain the history
    # identity; a cleaned derivative extends that chain without changing source.
    world=model_cloud(p,source,with_indices=True)
    original_indices=world.pop('_indices')
    cloud=world
    means=torch.as_tensor(world["means"],device="cuda",dtype=torch.float32)
    quats=torch.as_tensor(world["quats"],device="cuda",dtype=torch.float32)
    scales=torch.as_tensor(world["scales"],device="cuda",dtype=torch.float32).exp()
    opacity=torch.as_tensor(world["opacities"],device="cuda",dtype=torch.float32).sigmoid()
    maximum=torch.zeros(len(means),device="cuda")
    seen=torch.zeros(len(means),device="cuda",dtype=torch.int32)
    started=time.perf_counter()
    for n,index in enumerate(train_ids):
        if control.is_set():raise Cancelled("Model cleaning cancelled; source and checkpoint are unchanged")
        camera=cameras[index]
        with torch.no_grad():
            projection=fully_fused_projection(means=means,covars=None,quats=quats,scales=scales,
                        viewmats=torch.tensor(camera["w2c"],device="cuda",dtype=torch.float32)[None],
                        Ks=torch.tensor(camera["K"],device="cuda",dtype=torch.float32)[None],
                        width=int(camera["width"]),height=int(camera["height"]),
                        near_plane=.01,far_plane=1e8,packed=False,opacities=opacity)
            radii=projection[0][0]
            visible=(radii>0).all(-1)
            normalized=radii.max(-1).values.float()/max(camera["width"],camera["height"])
            maximum=torch.maximum(maximum,torch.where(visible,normalized,0))
            seen+=visible.to(torch.int32)
        emit("progress",{"stage":"model cleaning","progress":(n+1)/len(train_ids),
                         "message":f"Projecting training camera {n+1}/{len(train_ids)}"})
    remove=(maximum>threshold).cpu().numpy()
    kept=int((~remove).sum())
    if not kept:raise ValueError("This threshold removes every Gaussian; increase the radius limit")
    filtered={key:value[~remove] for key,value in cloud.items()}
    uid=str(uuid.uuid4())
    root=Path(p["path"])
    output=root/"models"/(uid+"-cleaned.ply")
    mask_file=root/"models"/(uid+"-clean-mask.npz")
    report_file=root/"models"/(uid+"-clean-report.json")
    write_ply(output,filtered)
    temp=mask_file.with_name(mask_file.name+".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f,remove=remove,max_normalized_radius=maximum.cpu().numpy(),
                            visible_training_camera_count=seen.cpu().numpy())
    temp.replace(mask_file)
    report={"method":"Public gsplat projected screen-size pruning; independent model edit",
            "rule":"Remove Gaussian if max projected axis radius / max(image width,height) exceeds threshold in any training camera",
            "max_screen_radius":threshold,"reference_default":"gsplat1.5.3 DefaultStrategy.prune_scale2d=.15",
            "training_camera_indices":train_ids,"training_camera_names":[cameras[i].get("source_name",cameras[i]["name"]) for i in train_ids],
            "heldout_rgb_used":False,"camera_source":"Reconstructed cameras; training split only",
            "source_id":source["id"],"source_path":source["path"],"source_sha256":project.digest(source["path"]),
            "original_count":len(remove),"removed_count":int(remove.sum()),"kept_count":kept,
            "mask_path":str(mask_file),"seconds":time.perf_counter()-started,
            "transform":source.get("transform"),"crop_behavior":"Committed crop/deletion support; original-index undo history retained",
            "source_editor_revision":source.get("editor",{}).get("revision",0),
            "preview_path":str(output)}
    project.atomic_json(report_file,report)
    record=cleaned_history_record(p,source,original_indices[remove],uid,source["name"]+" · screen cleaned")
    record["provenance"]={"source_id":source["id"],"method":"screen_radius","max_screen_radius":threshold,
                          "removed_count":int(remove.sum()),"mask":str(mask_file),"report":str(report_file)}
    source["visible"]=False
    p["models"].append(record)
    project.save(p)
    emit("preview",{"path":str(output),"count":kept,"step":p.get("training",{}).get("step",0)})
    return p
