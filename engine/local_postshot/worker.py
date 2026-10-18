"""JSONL RPC worker. The stdin control reader stays responsive during CUDA work."""
import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import traceback
import uuid
import tempfile
from . import project
from .runtime import config, health
from .locks import ProjectBusy,operation_lock,is_active


class Worker:
    def __init__(self, output=None):
        self.output = output or sys.stdout
        self.output_lock = threading.Lock()
        self.lock = threading.RLock()
        self.thread = None
        self.active = None
        self.control = threading.Event()
        self.pause_requested = threading.Event()
        self.shutting_down = False
        self.failed = False
        self.read_threads = set()
        self.stop_directory = None
        self.stop_file = None

    def send(self, data):
        with self.output_lock:
            self.output.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+"\n")
            self.output.flush()

    def error(self, rid, code, message, details=None):
        self.failed = True
        self.send({"type":"error","id":rid,"error":{"code":code,"message":message,"details":details or {}}})

    def result(self, rid, result):
        self.send({"type":"result","id":rid,"result":result})

    def emit(self, rid, event, data):
        self.send({"type":"event","id":rid,"event":event,"data":data})

    def _dispatch(self, method, params, emit):
        if method in ('studio.catalog','studio.operation'):
            from .studio_bridge import call
            return call(method,params,request_id=self.active['id'],emit=emit,cancel=self.control)
        if method == 'editor.operation':
            from .studio_edit import dispatch
            return dispatch(params,emit=emit,control=self.control)
        if method == "project.create":
            return project.create(params.get("name","Untitled project"),params.get("path"))
        if method == "project.update":
            return project.update(params["path"],params.get("settings"),params.get("models"))
        if method == "project.import":
            from .reconstruction import import_media
            return import_media(params["path"],params["files"],params["kind"],emit,self.control,
                                **{k:params[k] for k in ("max_image_size","video_fps","max_frames") if k in params})
        if method == "reconstruct":
            from .reconstruction import reconstruct
            return reconstruct(params["path"],emit,self.control)
        if method in ("train","resume"):
            from .training import train
            return train(params["path"],emit,self.control,self.pause_requested,
                         steps=params.get("steps"),max_splats=params.get("max_splats"),
                         resume=method=="resume" or params.get("resume",False))
        if method == "model.import":
            from .formats import load_cloud, write_ply
            cloud = load_cloud(params["file"])
            p = project.load(params["path"])
            import uuid
            dest = Path(p["path"])/"models"/(str(uuid.uuid4())+".ply")
            write_ply(dest,cloud)
            p["models"].append(project.model_record(dest,len(cloud["means"]),Path(params["file"]).stem))
            if p["status"] == "empty":
                p["status"] = "model imported"
            return project.save(p)
        if method == "model.clean":
            from .cleaning import clean
            return clean(params["path"],params["model_id"],emit,self.control,params.get("max_screen_radius",.15))
        if method == "model.export":
            from .formats import load_cloud, write_ply, write_spz
            from .geometry import apply_edits
            p = project.load(params["path"])
            model = next((m for m in p["models"] if m["id"]==params["model_id"]),None)
            if model is None:
                raise ValueError("Unknown model ID")
            destination=Path(params['file']).resolve()
            root=Path(p['path']).resolve()
            protected={Path(m['path']).resolve() for m in p.get('models',[])}
            protected.update(root/name for name in ('project.json','config.json','training.json',
                                                    'dispatch.json','model.ply'))
            for group in p.get('assets',{}).values():
                if isinstance(group,list):
                    for item in group:
                        for key in ('path','source'):
                            if item.get(key): protected.add(Path(item[key]).resolve())
            boundaries=[root/'checkpoints',root/'prepared',root/'images']
            if p.get('studio_dataset',{}).get('package'):
                boundaries.append(Path(p['studio_dataset']['package']).resolve())
            if destination in protected or any(destination.is_relative_to(boundary) for boundary in boundaries):
                raise ValueError('Export destination would overwrite an immutable model, source, dataset or checkpoint; choose a new export file')
            cloud = load_cloud(model["path"])
            if params.get("bake_edits",True):
                from .studio_edit import edited_cloud
                cloud = edited_cloud(p,model)
            fmt = params.get("format",Path(params["file"]).suffix.lstrip(".")).lower()
            if fmt not in ("ply","spz"):
                raise ValueError("Export format must be ply or spz")
            (write_ply if fmt=="ply" else write_spz)(params["file"],cloud)
            return {"path":str(Path(params["file"]).resolve()),"count":len(cloud["means"]),"coordinate_system":"RUB"}
        if method == "render":
            from .training import render
            return render(params["path"],params.get("camera"),params.get("output"))
        if method == "evaluate":
            from .training import evaluate
            return evaluate(params["path"],emit,self.control)
        raise ValueError(f"Unknown method {method}")

    def run_read(self, rid, method, params):
        try:
            from .studio_bridge import call
            value = call(method,params,request_id=rid,emit=lambda e,d:self.emit(rid,e,d))
            self.result(rid,value)
        except Exception as exc:
            self.error(rid,getattr(exc,'code','PIPELINE_ERROR'),str(exc),getattr(exc,'details',{}))
        finally:
            with self.lock:
                self.read_threads.discard(threading.current_thread())

    def run_task(self, rid, method, params):
        terminal = None
        try:
            # Third-party native modules must never corrupt stdout protocol framing.
            with contextlib.redirect_stdout(sys.stderr):
                with contextlib.ExitStack() as stack:
                    # Preprocessing and GS training children own their lock.
                    # Never lock around a subprocess that acquires this byte lock.
                    if method in {'train','resume'}:
                        from .studio_bridge import training_control_scope
                        stack.enter_context(training_control_scope(self.control))
                    gpu_owned_here=method in {'reconstruct','model.clean','render','evaluate'}
                    if method in {'train','resume'}:
                        gpu_owned_here=not bool(project.load(params['path']).get('studio_dataset'))
                    if gpu_owned_here:
                        from .studio_bridge import gpu_session
                        stack.enter_context(gpu_session(cancel=self.control))
                    if params.get("path") and method!="project.create":
                        stack.enter_context(operation_lock(params["path"],method))
                    value = self._dispatch(method,params,lambda e,d:self.emit(rid,e,d))
            terminal = {"type":"result","id":rid,"result":value}
        except Exception as exc:
            self.failed = True
            from .reconstruction import Cancelled
            if hasattr(exc,'code'):
                code = exc.code
            elif isinstance(exc,ProjectBusy):
                code = "PROJECT_BUSY"
            elif isinstance(exc,Cancelled):
                code = "CANCELLED"
            elif isinstance(exc,(ValueError,KeyError,FileNotFoundError)):
                code = "INVALID_INPUT"
            elif "out of memory" in str(exc).lower():
                code = "GPU_OUT_OF_MEMORY"
            else:
                code = "ENGINE_ERROR"
            if code!="PROJECT_BUSY" and params.get("path") and method in ("train","resume","reconstruct","project.import"):
                try:
                    p = project.load(params["path"])
                    p["status"] = "cancelled" if code=="CANCELLED" else "error"
                    p["last_error"] = str(exc)
                    project.save(p)
                except Exception:
                    pass
            traceback.print_exc(file=sys.stderr)
            terminal = {"type":"error","id":rid,"error":{"code":code,"message":str(exc),"details":{"method":method}}}
        finally:
            with self.lock:
                self.active = None
                if self.stop_file:
                    self.stop_file.unlink(missing_ok=True)
                    try:
                        self.stop_directory.rmdir()
                    except OSError:
                        pass
                    self.stop_file = self.stop_directory = None
            if terminal:
                self.send(terminal)

    def receive(self, request):
        if not isinstance(request,dict):
            self.error("unknown","INVALID_REQUEST","Expected a JSON request object")
            return
        rid = request.get("id")
        method = request.get("method")
        params = request.get("params")
        if params is None:
            params = {}
        if not isinstance(rid,str) or not isinstance(method,str) or not isinstance(params,dict):
            self.error(str(rid),"INVALID_REQUEST","Expected string id/method and object params")
            return
        try:
            if method == "health":
                self.result(rid,health())
                return
            if method == "project.list":
                self.result(rid,project.list_projects())
                return
            if method == "project.open":
                p = project.load(params["path"])
                with self.lock:
                    active_here = self.active and self.active.get("path") and Path(self.active["path"]).resolve()==Path(p["path"]).resolve()
                if not active_here and not is_active(p["path"]) and p["status"] in ("training","reconstructing"):
                    try:
                        with operation_lock(p["path"],"recover"):
                            p=project.load(p["path"])
                            if p["status"] in ("training","reconstructing"):
                                p["status"] = "interrupted"
                                p["last_error"] = "Previous worker stopped. Completed artifacts and checkpoints are preserved."
                                project.save(p)
                    except ProjectBusy:
                        pass
                self.result(rid,p)
                return
            if method == "status":
                self.result(rid,{"active":self.active,"pause_requested":self.pause_requested.is_set(),
                                 "cancel_requested":self.control.is_set()})
                return
            if method in ('studio.catalog','studio.operation'):
                from .studio_bridge import config
                params = {**params,'root':params.get('root',config()['data_root'])}
                if method=='studio.operation' and params.get('action')=='control_run':
                    with self.lock:
                        if (not self.active or self.stop_file is None or
                                params.get('job_id')!=self.active['id'] or
                                params.get('generation_id')!=self.active['generation_id'] or
                                params.get('control_action')!='stop_after_stage'):
                            raise ValueError('Run control must bind an active Run job/generation and stop_after_stage')
                        self.stop_file.write_text(json.dumps({'job_id':self.active['id'],
                                                  'generation_id':self.active['generation_id']}),encoding='utf-8')
                        self.result(rid,{'job_id':self.active['id'],'generation_id':self.active['generation_id'],
                                         'control_action':'stop_after_stage','status':'stop_after_stage requested'})
                    return
                read = method=='studio.catalog' or params.get('action') in {
                    'capture_config','source_hashes','run_config','masks.review','masks.read','qa_report',
                    'segment_choices','experiments','training_state','preview.read','verified_models',
                    'model_open','list_external_imports','inspect_external_import','control_training','preview.request'}
                read = read or params.get('action') in {'sources.list','experiment.read','run_status'}
                if read:
                    with self.lock:
                        if len(self.read_threads)>=8:
                            self.error(rid,'BUSY','Too many pending read requests; retry shortly')
                            return
                        thread=threading.Thread(target=self.run_read,args=(rid,method,params),daemon=False)
                        self.read_threads.add(thread)
                        thread.start()
                    return
            if method in ("pause","cancel","suspend","unpause","checkpoint","stop"):
                with self.lock:
                    if not self.active:
                        self.result(rid,{"active":None,"status":"idle"})
                        return
                    if params.get('job_id') != self.active['id'] or params.get('generation_id') != self.active['generation_id']:
                        raise ValueError('Control must bind active job_id and generation_id from status')
                    if self.active['method']=='studio.operation':
                        if method!='cancel':
                            raise ValueError('Use studio.operation control_training for identity-bound GS experiments')
                        self.control.set()
                        self.result(rid,{'status':'cancel requested','active':dict(self.active)})
                        return
                    if params.get("path") and Path(params["path"]).resolve()!=Path(self.active.get("path","")).resolve():
                        raise ValueError("Control request path differs from active project")
                    if method!="cancel" and self.active["method"] not in ("train","resume"):
                        raise ValueError("Pause is available for training; use cancel for other operations")
                    if self.active['method'] in ('train','resume') and method in ('pause','suspend','unpause','checkpoint','stop'):
                        from .studio_control import submit_control,read_control_state
                        path=self.active.get('path')
                        state=read_control_state(path)
                        if not state:
                            if method in ('pause','stop'):
                                if method=='pause': self.pause_requested.set()
                                self.control.set()
                                self.result(rid,{'status':method+' requested','active':dict(self.active)})
                                return
                            raise ValueError('Training has not published control state yet')
                        request=submit_control(path,state['package_sha256'],method)
                        self.result(rid,request)
                        return
                    if method=="pause":
                        self.pause_requested.set()
                    self.control.set()
                    self.result(rid,{"status":method+" requested","active":self.active,
                                     "message":"Training stops after the current iteration and writes a complete checkpoint"})
                return
            with self.lock:
                if self.active:
                    self.error(rid,"BUSY","Another engine task is active",self.active)
                    return
                self.control.clear(); self.pause_requested.clear()
                self.active = {"id":rid,"method":method,"path":params.get("path"),
                               "generation_id":uuid.uuid4().hex,'action':params.get('action')}
                if method=='studio.operation' and params.get('action') in {'create_run','resume_run','run_stage'}:
                    self.stop_directory=Path(tempfile.mkdtemp(prefix='gsstudio-run-control-'))
                    self.stop_file=self.stop_directory/'stop.json'
                    params={**params,'stop_file':str(self.stop_file)}
                self.thread = threading.Thread(target=self.run_task,args=(rid,method,params),daemon=False)
                self.thread.start()
        except Exception as exc:
            self.error(rid,"INVALID_REQUEST",str(exc))


def setup_environment():
    cfg = config()
    os.environ.setdefault("PYTHONUNBUFFERED","1")
    os.environ.setdefault('GSSTUDIO_GPU_LOCK_PATH',cfg['gpu_lock_path'])
    os.environ.setdefault('GSSTUDIO_DATA_ROOT',cfg['data_root'])
    os.environ.setdefault('GSSTUDIO_CACHE_ROOT',cfg['cache_root'])
    os.environ.setdefault('HF_HOME',str(Path(cfg['cache_root'])/'huggingface'))
    if cfg.get('torch_home'):
        os.environ.setdefault('TORCH_HOME', cfg['torch_home'])
    if cfg.get("torch_extensions_dir"):
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", cfg["torch_extensions_dir"])
    if cfg.get("cuda_path"):
        os.environ.setdefault("CUDA_PATH", cfg["cuda_path"])
        os.environ.setdefault("CUDA_HOME", cfg["cuda_path"])


def initialize_native_runtime():
    """Initialize Windows native extensions before the blocking stdin reader.

    Loading scipy's C extensions for the first time on a task thread while the
    main thread blocks in CRT stdin can deadlock on Windows. Import on the main
    thread, retain missing-dependency diagnostics, and only then accept requests.
    """
    for module in ("numpy","scipy.spatial","scipy.spatial.transform","scipy.special",
                   "PIL.Image","plyfile","torch","torch.optim","torch._dynamo",
                   "gsplat.csrc","gsplat","spz"):
        try:
            importlib.import_module(module)
        except (ImportError,OSError) as exc:
            print(f"Native preflight {module}: {exc}",file=sys.stderr,flush=True)
    # Optional fused SSIM is also a Windows native extension. Load it before
    # blocking CRT stdin; a saved fused checkpoint must retain this backend.
    if importlib.util.find_spec("fused_ssim") is not None:
        try:
            importlib.import_module("fused_ssim")
        except (ImportError,OSError) as exc:
            print(f"Native preflight fused_ssim: {exc}",file=sys.stderr,flush=True)
    for module in ("local_splat3_stats", "local_splat3_render", "local_splat3_tiles", "local_splat3_projection"):
        if importlib.util.find_spec(module) is not None:
            try:
                importlib.import_module(module)
            except (ImportError,OSError) as exc:
                print(f"Native preflight {module}: {exc}",file=sys.stderr,flush=True)
    # CUDA is initialized only within an explicitly requested, locked operation.


def main():
    for stream in (sys.stdin,sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="replace")
    setup_environment()
    parser = argparse.ArgumentParser(description="Local Postshot independent JSONL engine")
    parser.add_argument("--method",help="Run one JSON RPC method and exit")
    parser.add_argument("--params",default="{}",help="JSON object for single-method mode")
    args = parser.parse_args()
    # Keep a private protocol descriptor, then redirect fd1 itself. Python's
    # redirect_stdout cannot intercept printf/std::cout from native SPZ code.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()),"w",encoding="utf-8",buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    initialize_native_runtime()
    worker = Worker(protocol)
    if args.method:
        worker.receive({"id":"cli","method":args.method,"params":json.loads(args.params)})
        if worker.thread:
            worker.thread.join()
        for thread in list(worker.read_threads):
            thread.join()
        raise SystemExit(1 if worker.failed else 0)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            worker.receive(request)
        except (ValueError,TypeError) as exc:
            worker.error("unknown","INVALID_JSON",str(exc))
    # EOF means the desktop closed. Preserve a complete checkpoint on the next
    # iteration, and prevent a silent orphan process from continuing training.
    if worker.active:
        worker.pause_requested.set(); worker.control.set()
        if worker.active['method'] in ('train','resume') and worker.active.get('path'):
            # Stop supersedes pending checkpoint/suspend and wakes hot suspend.
            try:
                from .studio_control import submit_control,read_control_state
                state=read_control_state(worker.active['path'])
                if state:
                    submit_control(worker.active['path'],state['package_sha256'],'stop')
            except (OSError,ValueError,RuntimeError):
                pass  # Event scope handles startup before control-state exists.
    if worker.thread:
        worker.thread.join()
    for thread in list(worker.read_threads):
        thread.join()


if __name__=="__main__":
    main()
