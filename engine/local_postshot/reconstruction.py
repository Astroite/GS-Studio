import json
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
import uuid
import numpy as np
from PIL import Image, ImageOps
from . import project
from .geometry import CV_TO_WORLD, quat_matrix
from .runtime import executable
from .locks import operation_locked

IMAGE_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"}


class Cancelled(RuntimeError):
    pass


def process(command, emit, control, log_path=None):
    emit("log",{"message":"Running "+str(command[0])+" "+" ".join(map(str,command[1:]))})
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess,"CREATE_NO_WINDOW") else 0
    p = subprocess.Popen([str(x) for x in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                         text=True,encoding="utf-8",errors="replace",creationflags=flags)
    lines = queue.Queue()
    def reader():
        for line in p.stdout:
            lines.put(line.rstrip())
        lines.put(None)
    threading.Thread(target=reader,daemon=True).start()
    tail = []
    log = Path(log_path).open("a",encoding="utf-8") if log_path else None
    try:
        while True:
            if control and control.is_set():
                if hasattr(subprocess,"CREATE_NO_WINDOW"):
                    subprocess.run(["taskkill","/PID",str(p.pid),"/T","/F"],capture_output=True,creationflags=flags)
                else:
                    p.terminate()
                p.wait(timeout=20)
                raise Cancelled("Task cancelled; completed artifacts are preserved")
            try:
                line = lines.get(timeout=.2)
            except queue.Empty:
                continue
            if line is None:
                break
            tail.append(line)
            tail = tail[-30:]
            emit("log",{"message":line})
            if log:
                log.write(line+"\n")
                log.flush()
        p.wait()
        if p.returncode:
            raise RuntimeError(f"{Path(command[0]).name} exited {p.returncode}: "+"\n".join(tail))
    finally:
        if log:
            log.close()
        if p.poll() is None:
            p.kill()


@operation_locked("project.import")
def import_media(path, files, kind, emit, control, **options):
    p = project.load(path)
    settings = {k:options[k] for k in ("max_image_size","video_fps","max_frames") if options.get(k) is not None}
    if settings:
        p = project.update(path,settings=settings)
    if not files:
        raise ValueError("Choose at least one input file")
    root = Path(p["path"])
    sources = []
    origins = {}
    videos = list(p["assets"]["videos"])
    replacing_video_hashes = set()
    if kind == "video":
        ffmpeg = executable("ffmpeg")
        for file in files:
            source = Path(file).resolve()
            sha = project.digest(source)
            limit = int(p["settings"]["max_frames"])-len(sources)
            if limit<=0:
                break
            size = int(p["settings"]["max_image_size"])
            video_key = project.identity({"sha256":sha,"fps":p["settings"]["video_fps"],
                                          "max_frames":limit,"max_image_size":size})
            # Every extraction gets a fresh directory. A failed or shorter
            # extraction can never discover stale frames from an earlier run.
            dest = root/"video_frames"/(video_key[:16]+"-"+uuid.uuid4().hex[:8])
            dest.mkdir(parents=True,exist_ok=True)
            vf = (f"fps={float(p['settings']['video_fps'])},"
                  f"scale=w='min({size},iw)':h='min({size},ih)':"
                  "force_original_aspect_ratio=decrease:force_divisible_by=2")
            process([ffmpeg,"-hide_banner","-y","-i",source,"-vf",vf,"-frames:v",limit,
                     dest/"frame_%06d.jpg"],emit,control,root/"logs"/"import.log")
            frames = sorted(dest.glob("frame_*.jpg"))[:limit]
            sources.extend(frames)
            origins.update({str(f):{"video_source_sha256":sha,"video_key":video_key} for f in frames})
            replacing_video_hashes.add(sha)
            videos = [v for v in videos if v["sha256"]!=sha]
            videos.append({"path":str(source),"sha256":sha,"frames":len(frames),"key":video_key,
                           "fps":p["settings"]["video_fps"],"max_image_size":size})
    elif kind == "images":
        for file in files:
            f = Path(file).resolve()
            sources.extend(sorted(x for x in f.iterdir() if x.suffix.lower() in IMAGE_EXTS) if f.is_dir() else [f])
    else:
        raise ValueError("Import kind must be images or video")
    if not sources:
        raise ValueError("Input produced no images")
    # Validate all sources before publishing converted files into the project.
    for source in sources:
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image: {source.name}")
        with Image.open(source) as im:
            im.verify()
    before = [(i["source_sha256"],i["sha256"]) for i in p["assets"]["images"]]
    images = {item["source_sha256"]:item for item in p["assets"]["images"]
              if item.get("video_source_sha256") not in replacing_video_hashes}
    staging = root/"imports"/("staging-"+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    pending = []
    converted = set()
    try:
        for i, source in enumerate(sources):
            if control.is_set():
                raise Cancelled("Import cancelled")
            sha = project.digest(source)
            size = int(p["settings"]["max_image_size"])
            if (sha,size) in converted:
                continue
            converted.add((sha,size))
            old = images.get(sha)
            if (old and old.get("import_max_image_size")==size and old.get("import_conversion_version")==2
                and Path(old["path"]).is_file()):
                continue
            with Image.open(source) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((size,size),Image.Resampling.LANCZOS)
                dest = root/"images"/(sha[:16]+"_"+source.stem+".png")
                staged = staging/dest.name
                exif=im.getexif()
                if exif:
                    # Keep camera/focal priors for COLMAP. exif_transpose has
                    # already consumed orientation so pixels and metadata agree.
                    im.save(staged,exif=exif.tobytes())
                else:
                    im.save(staged)
                images[sha] = {"path":str(dest),"source":str(source),"source_sha256":sha,
                               "sha256":project.digest(staged),"width":im.width,"height":im.height,
                               "import_max_image_size":size,"import_conversion_version":2,
                               **origins.get(str(source),{})}
                pending.append((staged,dest))
            emit("progress",{"stage":"import","progress":(i+1)/len(sources),"message":source.name})
        if control.is_set():
            raise Cancelled("Import cancelled")
        for staged,dest in pending:
            staged.replace(dest)
    finally:
        # This directory is created and owned by this invocation. No recursive
        # deletion or traversal of user-controlled input directories is needed.
        for staged in staging.iterdir():
            if staged.is_file():
                staged.unlink()
        staging.rmdir()
    p["assets"]["images"] = list(images.values())
    p["assets"]["videos"] = videos
    p.pop("last_error",None)
    after = [(i["source_sha256"],i["sha256"]) for i in p["assets"]["images"]]
    if before!=after:
        p["pipeline"].pop("reconstruction",None)
        p["pipeline"].pop("training",None)
        p["status"] = "imported"
    return project.save(p)


def _read(f, fmt):
    s = struct.Struct("<"+fmt)
    data = f.read(s.size)
    if len(data)!=s.size:
        raise ValueError("Truncated COLMAP binary model")
    return s.unpack(data)


def read_sparse(directory):
    directory = Path(directory)
    cameras = {}
    param_counts = {0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12,11:16}
    with (directory/"cameras.bin").open("rb") as f:
        for _ in range(_read(f,"Q")[0]):
            cid, model, w, h = _read(f,"iiQQ")
            if model not in param_counts:
                raise ValueError(f"Unsupported COLMAP camera model {model}")
            params = _read(f,"d"*param_counts[model])
            if model in (0,2,3,8,9):
                fx, cx, cy = params[:3]
                fy = fx
            else:
                fx, fy, cx, cy = params[:4]
            cameras[cid] = {"width":int(w),"height":int(h),"model":model,
                            "K":[[fx,0,cx],[0,fy,cy],[0,0,1]]}
    images = []
    with (directory/"images.bin").open("rb") as f:
        for _ in range(_read(f,"Q")[0]):
            iid,*values = _read(f,"idddddddi")
            q = np.array(values[:4])
            t = values[4:7]
            cid = values[7]
            chars = bytearray()
            while True:
                char = f.read(1)
                if not char:
                    raise ValueError("Truncated image name")
                if char == b"\0":
                    break
                chars.extend(char)
            name = chars.decode("utf-8")
            n = _read(f,"Q")[0]
            f.seek(n*24,1)
            w2c = np.eye(4)
            w2c[:3,:3] = quat_matrix(q)
            w2c[:3,3] = t
            images.append({"id":iid,"name":name,**cameras[cid],
                           "w2c":(w2c@CV_TO_WORLD).tolist()})
    xyz, rgb = [], []
    with (directory/"points3D.bin").open("rb") as f:
        for _ in range(_read(f,"Q")[0]):
            values = _read(f,"QdddBBBd")
            xyz.append(values[1:4]); rgb.append(values[4:7])
            tracklen = _read(f,"Q")[0]
            f.seek(tracklen*8,1)
    points = np.asarray(xyz,dtype=np.float32).reshape(-1,3)@CV_TO_WORLD[:3,:3].T
    colors = np.asarray(rgb,dtype=np.float32).reshape(-1,3)/255
    return sorted(images,key=lambda im:im["name"]),points,colors


def reconstruction_key(p):
    return project.identity({"images":[(Path(i["path"]).name,project.digest(i["path"])) for i in p["assets"]["images"]],
                             "camera":"SIMPLE_RADIAL","colmap":"4.2.1","pipeline":1})


def artifact_manifest(work, dense, cameras):
    work,dense=Path(work),Path(dense)
    files=[work/"sparse_points.npz",work/"cameras.json"]+[dense/"sparse"/name for name in ("cameras.bin","images.bin","points3D.bin")]
    return {"files":[{"path":str(f.relative_to(work)),"size":f.stat().st_size,"sha256":project.digest(f)} for f in files],
            "images":[{"name":c["name"],"size":(dense/"images"/c["name"]).stat().st_size} for c in cameras]}


def reconstruction_artifacts_valid(p, work, dense):
    work,dense=Path(work),Path(dense)
    cameras=p.get("reconstruction",{}).get("cameras",[])
    if not cameras:
        return False
    required=[work/"sparse_points.npz",work/"cameras.json"]+[dense/"sparse"/name for name in ("cameras.bin","images.bin","points3D.bin")]
    required += [dense/"images"/c["name"] for c in cameras]
    if any(not f.is_file() or f.stat().st_size==0 for f in required):
        return False
    manifest=p.get("reconstruction",{}).get("artifacts")
    if manifest:
        for item in manifest["files"]:
            f=work/item["path"]
            if not f.is_file() or f.stat().st_size!=item["size"] or project.digest(f)!=item["sha256"]:
                return False
        for item in manifest["images"]:
            f=dense/"images"/item["name"]
            if not f.is_file() or f.stat().st_size!=item["size"]:
                return False
    return True


@operation_locked("reconstruct")
def reconstruct(path, emit, control):
    p = project.load(path)
    if len(p["assets"]["images"])<3:
        raise ValueError("Reconstruction needs at least three overlapping photos")
    key = reconstruction_key(p)
    root = Path(p["path"])
    work = root/"reconstruction"/key[:16]
    dense = work/"undistorted"
    if (p.get("pipeline",{}).get("reconstruction",{}).get("key")==key and
        reconstruction_artifacts_valid(p,work,dense)):
        emit("progress",{"stage":"reconstruction","progress":1,"message":"Reusing matching reconstruction"})
        return p
    colmap = executable("colmap")
    work.mkdir(parents=True,exist_ok=True)
    image_list = work/"input-images.txt"
    image_list.write_text("\n".join(Path(i["path"]).name for i in p["assets"]["images"])+"\n",encoding="utf-8")
    sparse = work/"sparse"
    sparse.mkdir(exist_ok=True)
    database = work/"database.db"
    log = root/"logs"/"reconstruction.log"
    p["status"] = "reconstructing"
    p.pop("last_error",None)
    project.save(p)
    stages = [
        ("features",[colmap,"feature_extractor","--database_path",database,"--image_path",root/"images",
                     "--image_list_path",image_list,
                     "--ImageReader.camera_model","SIMPLE_RADIAL","--FeatureExtraction.use_gpu","1"]),
        ("matching",[colmap,"exhaustive_matcher","--database_path",database,"--FeatureMatching.use_gpu","1"]),
        ("mapping",[colmap,"mapper","--database_path",database,"--image_path",root/"images","--output_path",sparse])]
    for i,(name,command) in enumerate(stages):
        emit("progress",{"stage":name,"progress":i/4,"message":"COLMAP "+name})
        process(command,emit,control,log)
    models = []
    for f in sparse.glob("*/images.bin"):
        with f.open("rb") as stream:
            models.append((_read(stream,"Q")[0],f.parent))
    if not models:
        raise RuntimeError("COLMAP could not reconstruct any camera model; use overlapping sharp photos")
    count, selected = max(models,key=lambda x:x[0])
    emit("progress",{"stage":"undistort","progress":.75,"message":f"Undistorting {count} registered cameras"})
    process([colmap,"image_undistorter","--image_path",root/"images","--input_path",selected,
             "--output_path",dense,"--output_type","COLMAP"],emit,control,log)
    cams,points,colors = read_sparse(dense/"sparse")
    if len(cams)<3 or len(points)<4:
        raise RuntimeError("Reconstruction has insufficient registered cameras or sparse points")
    for c in cams:
        if c["model"] not in (0,1):
            raise RuntimeError("Undistorted camera must be SIMPLE_PINHOLE or PINHOLE")
    by_name = {Path(item["path"]).name:item for item in p["assets"]["images"]}
    for c in cams:
        asset = by_name.get(c["name"],{})
        c["source_name"] = Path(asset.get("source",c["name"])).name
        c["source_sha256"] = asset.get("source_sha256")
    cams.sort(key=lambda c:(c["source_name"],c["name"]))
    np.savez(work/"sparse_points.npz",points=points,colors=colors)
    project.atomic_json(work/"cameras.json",cams)
    info = {"key":key,"path":str(work),"image_path":str(dense/"images"),"sparse_path":str(dense/"sparse"),
            "camera_count":len(cams),"point_count":len(points),"cameras":cams,
            "coordinate_system":"RUB world; RDF world-to-camera projection"}
    info["artifacts"]=artifact_manifest(work,dense,cams)
    p["reconstruction"] = info
    p["pipeline"]["reconstruction"] = {"key":key,"status":"complete"}
    p["pipeline"].pop("training",None)
    p["status"] = "reconstructed"
    project.save(p)
    emit("progress",{"stage":"reconstruction","progress":1,"message":f"{len(cams)} cameras, {len(points)} sparse points"})
    return p
