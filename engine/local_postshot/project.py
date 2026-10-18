from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import uuid
from .runtime import config
from .locks import operation_locked

LOCK = threading.RLock()
DEFAULTS = {"steps":10000,"max_splats":1000000,"max_image_size":1600,
            "video_fps":1.0,"max_frames":128,"sh_degree":3,"sh_degree_interval":333,
            "checkpoint_every":1000,"preview_every":1000,"test_every":8,"seed":42,
            "training_recipe":"quality_v2","training_backend":"optimized","adam_backend":"auto",
            "image_host_cache_mb":1024,"image_gpu_cache_mb":256}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    import os
    import time

    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    with temp.open("w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    # Retry only the commit: never rewrite/truncate the durable destination or
    # discard the fsynced .tmp evidence if Windows keeps refusing replacement.
    deadline = time.monotonic() + .5
    delay = .01
    while True:
        try:
            temp.replace(path)
            return
        except OSError as error:
            if os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, .05)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def identity(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def project_file(path):
    path = Path(path).resolve()
    return path if path.name=="project.json" else path/"project.json"


def load(path):
    file = project_file(path)
    with LOCK:
        p = json.loads(file.read_text(encoding="utf-8"))
    p["path"] = str(file.parent)
    for model in p.get("models",[]):
        if not Path(model["path"]).is_absolute():
            model["path"] = str(file.parent/model["path"])
        for key in ("mask","report"):
            value=model.get("provenance",{}).get(key)
            if value and not Path(value).is_absolute():
                model["provenance"][key]=str(file.parent/value)
    for image in p.get("assets",{}).get("images",[]):
        if not Path(image["path"]).is_absolute():
            image["path"] = str(file.parent/image["path"])
    for section, keys in (("reconstruction",("path","image_path","sparse_path")),
                          ("training",("checkpoint",))):
        for key in keys:
            value = p.get(section,{}).get(key)
            if value and not Path(value).is_absolute():
                p[section][key] = str(file.parent/value)
    return p


def save(p):
    import copy
    with LOCK:
        p["updated_at"] = now()
        stored = copy.deepcopy(p)
        root = Path(p["path"])
        for model in stored.get("models",[]):
            try:
                model["path"] = str(Path(model["path"]).relative_to(root))
            except ValueError:
                pass
            for key in ("mask","report"):
                value=model.get("provenance",{}).get(key)
                if value:
                    try:model["provenance"][key]=str(Path(value).relative_to(root))
                    except ValueError:pass
        for image in stored.get("assets",{}).get("images",[]):
            try:
                image["path"] = str(Path(image["path"]).relative_to(root))
            except ValueError:
                pass
        for section, keys in (("reconstruction",("path","image_path","sparse_path")),
                              ("training",("checkpoint",))):
            for key in keys:
                value = stored.get(section,{}).get(key)
                if value:
                    try:
                        stored[section][key] = str(Path(value).relative_to(root))
                    except ValueError:
                        pass
        atomic_json(root/"project.json",stored)
    return p


def create(name, path=None):
    uid = str(uuid.uuid4())
    safe = re.sub(r"[^\w -]","",name).strip()[:60] or "Project"
    root = Path(path) if path else Path(config()["data_root"])/".models"/(safe+"-"+uid[:8])
    root = root.resolve()
    if (root/"project.json").exists():
        raise ValueError("A project already exists at that location")
    root.mkdir(parents=True,exist_ok=True)
    for directory in ("images","logs","models","checkpoints","renders"):
        (root/directory).mkdir(exist_ok=True)
    return save({"id":uid,"name":name,"path":str(root),"created_at":now(),"updated_at":now(),
                 "status":"empty","settings":DEFAULTS.copy(),"models":[],
                 "assets":{"images":[],"videos":[]},"pipeline":{}})


def list_projects():
    root = Path(config()["data_root"])/".models"
    result = []
    for f in root.glob("*/project.json"):
        try:
            result.append(load(f))
        except (OSError,ValueError):
            pass
    return sorted(result,key=lambda p:p["updated_at"],reverse=True)


@operation_locked("project.update")
def update(path, settings=None, models=None):
    p = load(path)
    if settings:
        allowed = set(DEFAULTS)|{"ssim_lambda","refine_start","refine_every","refine_stop","reset_every",
                                 "training_recipe","training_recipe_overrides","training_backend","adam_backend",
                                 "image_host_cache_mb","image_gpu_cache_mb"}
        unknown = set(settings)-allowed
        if unknown:
            raise ValueError("Unknown settings: "+", ".join(sorted(unknown)))
        merged = {**p["settings"],**settings}
        if any(int(merged[k])<=0 for k in ("steps","max_splats","max_image_size","max_frames","sh_degree_interval","checkpoint_every","preview_every")):
            raise ValueError("Steps, limits and intervals must be positive")
        if not 0<=int(merged["sh_degree"])<=3 or float(merged["video_fps"])<=0:
            raise ValueError("Invalid SH degree or video FPS")
        if not math.isfinite(float(merged["video_fps"])) or int(merged["max_splats"])<4:
            raise ValueError("FPS must be finite and Gaussian limit at least four")
        for key in ("refine_every","reset_every"):
            if key in merged and int(merged[key])<=0:
                raise ValueError(f"{key} must be positive")
        for key in ("refine_start","refine_stop"):
            if key in merged and int(merged[key])<0:
                raise ValueError(f"{key} must not be negative")
        if int(merged["test_every"])<2 or not 0<=int(merged["seed"])<2**32:
            raise ValueError("Test interval must be at least two; seed must fit uint32")
        if not 0<=float(merged.get("ssim_lambda",.2))<=1:
            raise ValueError("SSIM loss weight must be between zero and one")
        if merged.get("training_recipe","legacy_adc") not in ("legacy_adc","quality_v2","observed_v3","observed_v4","observed_v5","observed_v6"):
            raise ValueError("Unknown training recipe")
        if merged.get("training_backend","legacy") not in ("legacy","optimized"):
            raise ValueError("Unknown training performance backend")
        if merged.get("adam_backend","auto") not in ("auto","fused","foreach","single"):
            raise ValueError("Unknown Adam backend")
        for key in ("image_host_cache_mb","image_gpu_cache_mb"):
            value=merged.get(key,0)
            if isinstance(value,bool) or int(value)!=value or not 0<=value<=32768:
                raise ValueError(f"{key} must be an integer MiB budget between0 and32768")
        if "training_recipe_overrides" in merged:
            if not isinstance(merged["training_recipe_overrides"],dict):
                raise ValueError("Recipe overrides must be an object")
            from .training_recipe import recipe_config
            recipe_config(merged.get("training_recipe","legacy_adc"),int(merged["steps"]),merged["training_recipe_overrides"])
        training_keys = set(settings)-{"max_image_size","video_fps","max_frames","steps","checkpoint_every","preview_every"}
        p["settings"] = merged
        if training_keys:
            p["pipeline"].pop("training",None)
    if models is not None:
        current = {m["id"]:m for m in p["models"]}
        result = []
        for edit in models:
            if edit.get("id") not in current:
                raise ValueError("Cannot add unknown model via project.update; use model.import")
            m = current[edit["id"]]
            if m.get("editor") and any(k in edit for k in ("transform","crop")):
                from .studio_edit import update_model_edits
                update_model_edits(p,m,{k:edit[k] for k in ("transform","crop") if k in edit})
            for key in ("name","visible","transform","crop"):
                if key in edit:
                    m[key] = edit[key]
            # Validate transforms even before an expensive model export.
            import numpy as np
            from .geometry import apply_edits
            apply_edits({"means":np.zeros((1,3)),"scales":np.zeros((1,3)),
                         "quats":np.array([[1.,0,0,0]]),"sh":np.zeros((1,1,3)),"opacities":np.zeros(1)},m["transform"],m["crop"])
            result.append(m)
        p["models"] = result
    return save(p)


def model_record(file, count, name=None, mid=None):
    return {"id":mid or str(uuid.uuid4()),"name":name or Path(file).stem,"path":str(Path(file).resolve()),
            "count":int(count),"visible":True,"coordinate_system":"RUB",
            "transform":{"position":[0,0,0],"rotation":[0,0,0],"scale":1},
            "crop":{"enabled":False,"min":[-1,-1,-1],"max":[1,1,1],"invert":False}}
