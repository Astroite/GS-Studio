"""Source-only legacy editor runtime retained for current validation helpers.

This is not the release desktop/worker runtime. Current production entry points
use the independent engine; no old Run/checkpoint compatibility is introduced.

All project/progress JSON is UTF-8 on disk (`json_write` / explicit encoding);
open() never relies on the Windows ANSI code page.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.editor.data import Dataset
from gsstudio.pipeline.editor.edit import Editor
from gsstudio.pipeline.training.data import json_write


class Runtime:
    def __init__(self):
        self.events=queue.Queue()
        self.tasks=queue.Queue()
        self.pause=threading.Event()
        self.stop=threading.Event()
        self.save_checkpoint=threading.Event()
        self.quit=threading.Event()
        self.active_task=threading.Event()
        self.dataset=None
        self.editor=None
        self.model_identity=None
        self.project=None
        self.experiment=None
        self.settings=None
        self.pending_view=None
        self.state='empty'
        self.camera=None
        self.preview_interval=0.5
        self.preview_revision=0
        self.last_preview=-1
        self.last_preview_time=0
        self.params=None
        self.params_revision=None
        self.preview_matrix=None
        self.preview_keep=None
        self.edit_preview_revision=0
        self.thread=threading.Thread(target=self._loop,daemon=True,name='studio-worker')
        self.thread.start()

    def emit(self,kind,**payload):
        self.events.put((kind,payload))

    def status(self,state,**payload):
        self.state=state
        self.emit('status',state=state,**payload)

    def submit(self,fn,*args,**kwargs):
        self.tasks.put((fn,args,kwargs))

    def set_camera(self,row):
        self.camera=row
        self.preview_revision+=1

    def _loop(self):
        while not self.quit.is_set():
            try:
                fn,args,kwargs=self.tasks.get(timeout=0.05)
            except queue.Empty:
                if self.editor and self.camera and self.preview_revision!=self.last_preview:
                    try:
                        with gpu_session(cancel=self.quit,timeout=0.2):
                            identity=(id(self.editor),self.editor.revision,self.edit_preview_revision)
                            t=None
                            if self.params_revision!=identity:
                                import torch
                                from gsstudio.pipeline.training.native import configure_windows_cuda
                                configure_windows_cuda()
                                matrix=(self.preview_matrix@self.editor.matrix
                                        if self.preview_matrix is not None else None)
                                data=self.editor.materialize(matrix_override=matrix,
                                                             keep_override=self.preview_keep)
                                t=torch.tensor(data,device='cuda')
                                self.params=dict(means=t[:,:3],sh0=t[:,6:9,None].transpose(1,2),
                                    shN=t[:,9:54].reshape(-1,3,15).transpose(1,2).contiguous(),
                                    opacities=t[:,54],scales=t[:,55:58],quats=t[:,58:62])
                                self.params_revision=identity
                            self._preview(self.params)
                            # Idle viewing must release its GPU allocations before
                            # another pipeline stage acquires the workspace lock.
                            self.params=None
                            self.params_revision=None
                            if t is not None:
                                del t
                            torch.cuda.empty_cache()
                    except (TimeoutError,InterruptedError):
                        self.last_preview=self.preview_revision
                        self.emit('notice',message='GPU 被其他工作区任务使用；移动视角可重新请求预览。')
                    except Exception as e:
                        self.params=None
                        self.params_revision=None
                        self.last_preview=self.preview_revision
                        self.emit('error',message=f'模型渲染失败: {e}')
                continue
            try:
                self.active_task.set()
                fn(*args,**kwargs)
            except Exception as e:
                self.emit('error',message=str(e))
                self.emit('status',state=self.state)
            finally:
                self.active_task.clear()

    def import_data(self,**kwargs):
        self.status('preparing')
        try:
            dataset=Dataset(**kwargs)
            self.dataset=dataset
            self.editor=None
            self.model_identity=None
            self.params=None
            self.params_revision=None
            self.project=None
            self.experiment=None
            self.emit('dataset',dataset=dataset)
            self.status('ready' if not dataset.errors else 'invalid')
        except Exception:
            self.status('invalid')
            raise

    def _preview(self,params,force=False):
        if self.camera is None or (not force and time.monotonic()-self.last_preview_time<self.preview_interval):
            return
        import torch
        from gsstudio.pipeline.training.native import render
        row=self.camera
        revision=self.preview_revision
        with torch.no_grad():
            rgb,_,_=render(params,row)
            pixels=(rgb[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        self.emit('preview',pixels=pixels,camera=row,revision=revision,
                  model_identity=self.model_identity)
        self.last_preview=revision
        self.last_preview_time=time.monotonic()

    def _observer(self,params,progress,save):
        self.params=params
        now=time.monotonic()
        if now-self.last_progress>=0.25 or progress['step']==0:
            self.emit('progress',**progress)
            if self.experiment:
                # utf-8 + ensure_ascii=False so Chinese status/progress text survives on disk.
                with (self.experiment/'studio-progress.jsonl').open('a',encoding='utf-8',newline='\n') as f:
                    f.write(json.dumps(progress,ensure_ascii=False)+'\n')
            self.last_progress=now
        if self.save_checkpoint.is_set():
            save()
            self.save_checkpoint.clear()
            if self.project:
                self.save_project()
            self.emit('notice',message='检查点已保存')
        try:
            self._preview(params)
        except Exception as e:
            # A bad preview does not invalidate the optimization state.
            self.emit('notice',message=f'预览暂不可用: {e}')
            self.last_preview_time=now
        if self.pause.is_set():
            save()
            self.status('paused')
            while self.pause.is_set() and not self.stop.is_set():
                if self.save_checkpoint.is_set():
                    save()
                    self.save_checkpoint.clear()
                if self.preview_revision!=self.last_preview:
                    try:
                        self._preview(params)
                    except Exception as e:
                        self.emit('notice',message=f'预览暂不可用: {e}')
                time.sleep(0.05)
            if not self.stop.is_set():
                self.status('training')
        return 'stop' if self.stop.is_set() else None

    def start(self,output,settings,resume=False):
        if self.dataset is None:
            raise ValueError('请先导入数据集')
        if settings['steps']<1 or settings['max_size']<0 or settings['checkpoint_every']<1 or settings['sh_degree'] not in (0,1,2,3):
            raise ValueError('无效的训练设置')
        if resume and (self.experiment is None or self.settings!=settings):
            raise ValueError('恢复时必须保留原配置；改配置请新建实验')
        self.status('preparing')
        self.stop.clear()
        self.pause.clear()
        self.params=None
        self.params_revision=None
        try:
            # Short tasks release allocations before they release the GPU lock.
            import torch
            from gsstudio.pipeline.training.native import configure_windows_cuda,train
            if not torch.cuda.is_available():
                raise ValueError('本机 NVIDIA CUDA 不可用')
            if resume:
                package=self.project/'dataset'
                # Reopening the package also checks the image/mask inventory.
                self.dataset.verify()
            else:
                root=Path(output).resolve()
                root.mkdir(parents=True,exist_ok=True)
                self.project=root/f'studio-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:8]}'
                self.project.mkdir()
                self.experiment=self.project/'training'
                self.settings=dict(settings)
                package=self.dataset.prepare(self.project/'dataset',settings['max_size'])
                self.dataset=Dataset(package)
                self.editor=None
                self.emit('dataset',dataset=self.dataset)
            self.save_project()
            self.status('waiting',message='等待工作区 GPU 锁')
            with gpu_session(cancel=self.stop):
                configure_windows_cuda()
                free,total=torch.cuda.mem_get_info()
                self.emit('device',name=torch.cuda.get_device_name(),free=free,total=total)
                if self.stop.is_set():
                    self.status('stopped')
                    return
                self.status('training')
                self.last_progress=0
                result=train(package,self.experiment,steps=settings['steps'],photo_comp=settings['photo_comp'],
                    resume=resume,checkpoint_every=settings['checkpoint_every'],observer=self._observer,
                    run_evaluation=False,sh_degree=settings['sh_degree'],
                    use_bilateral_grid=settings.get('use_bilateral_grid',True),
                    use_sparse_depth=settings.get('use_sparse_depth',True))
            self.params=None
            import shutil
            snapshot=self.project/f'snapshot-{uuid.uuid4().hex}.ply'
            shutil.copy2(self.experiment/'model.ply',snapshot)
            self.editor=Editor(snapshot)
            self.emit('editor',editor=self.editor)
            self.preview_revision+=1
            self.status('completed' if result['status']=='succeeded' else 'stopped')
            self.save_project()
        except InterruptedError:
            self.status('stopped',message='已取消 GPU 等待，尚未训练')
            self.save_project()
        except Exception as e:
            self.params=None
            self.status('failed')
            if self.project:
                json_write(self.project/'studio-failure.json',dict(error=str(e),
                    checkpoint=str(self.experiment/'checkpoint.pt') if self.experiment else None))
                self.save_project()
            raise

    def load_model(self,path,state=None):
        editor=Editor(path)
        if state is not None:
            editor.load_state(state)
        self.editor=editor
        self.model_identity=str(Path(path).resolve())
        self.dataset=None
        self.project=None
        self.experiment=None
        self.settings=None
        self.pending_view=None
        self.emit('clear_dataset')
        self.params=None
        self.params_revision=None
        self.preview_matrix=None
        self.preview_keep=None
        self.emit('editor',editor=self.editor,model_identity=self.model_identity)
        self.preview_revision+=1
        self.status('viewing')

    def edit(self,method,*args,**kwargs):
        if self.editor is None:
            raise ValueError('请先打开模型或完成训练')
        if method=='select' and not args[2]:
            from gsstudio.pipeline.editor.visibility import visible_centers
            from gsstudio.infrastructure.adapters.gsplat import configure_windows_cuda
            with gpu_session(cancel=self.quit,timeout=0.2):
                configure_windows_cuda()
                visible=visible_centers(self.editor,args[0],args[1])
            args=(*args,visible)
        self.preview_matrix=None
        self.preview_keep=None
        self.edit_preview_revision+=1
        self.preview_revision+=1
        result=getattr(self.editor,method)(*args,**kwargs)
        self.emit('editor',editor=self.editor,fit=False,model_identity=self.model_identity)
        return result

    def set_edit_preview(self, *, transform=None, crop=None):
        if self.editor is None:
            return
        self.preview_matrix=None
        self.preview_keep=None
        self.edit_preview_revision+=1
        self.preview_revision+=1
        if transform is not None:
            self.preview_matrix=self.editor.transform_matrix(*transform)
        if crop is not None:
            low,high,keep_inside=crop
            low,high=np.asarray(low,float),np.asarray(high,float)
            if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all() or np.any(low>=high):
                raise ValueError('裁剪预览范围无效')
            xyz=self.editor.xyz()
            inside=((xyz>=low)&(xyz<=high)).all(axis=1)
            self.preview_keep=self.editor.keep & (inside if keep_inside else ~inside)
            if not self.preview_keep.any():
                raise ValueError('裁剪将删除全部高斯')

    def clear_edit_preview(self):
        self.set_edit_preview()

    def save_project(self,view=None):
        if self.project is None:
            raise ValueError('尚未创建项目；请开始训练或另存项目')
        if view is not None:
            self.pending_view=view
        view=self.pending_view
        previous={}
        file=self.project/'project.json'
        if file.exists():
            previous=json.loads(file.read_text(encoding='utf-8'))
        model=None
        if self.editor:
            # Snapshot source is immutable; edits and history are stored separately.
            source=self.editor.source
            if not source.is_relative_to(self.project):
                import shutil
                target=self.project/f'snapshot-{uuid.uuid4().hex}.ply'
                shutil.copy2(source,target)
                self.editor.source=target
                source=target
            state=f'edit-{uuid.uuid4().hex}.npz'
            self.editor.save_state(self.project/state)
            model=dict(path=source.relative_to(self.project).as_posix(),sha256=sha256_file(source),state=state)
        meta=dict(schema_version=1,settings=self.settings,state=self.state,
            experiment=self.experiment.relative_to(self.project).as_posix() if self.experiment else None,
            dataset='dataset' if (self.project/'dataset').exists() else None,
            model=model,view=view if view is not None else previous.get('view'),
            assessment='Functional result only; visual quality has not been approved.')
        json_write(file,meta)
        self.emit('notice',message=f'项目已保存: {file}')

    def open_project(self,path):
        from gsstudio.pipeline.editor.data import inside
        path=Path(path).resolve()
        meta=json.loads(path.read_text(encoding='utf-8'))
        if meta.get('schema_version')!=1:
            raise ValueError('不支持的项目版本')
        root=path.parent
        dataset=Dataset(inside(root,meta['dataset'])) if meta.get('dataset') else None
        editor=None
        if meta.get('model'):
            source=inside(root,meta['model']['path'])
            if sha256_file(source)!=meta['model']['sha256']:
                raise ValueError('项目源模型已改变')
            editor=Editor(source)
            editor.load_state(inside(root,meta['model']['state']))
        self.project=root
        self.dataset=dataset
        self.editor=editor
        self.settings=meta['settings']
        self.pending_view=meta.get('view')
        self.experiment=inside(root,meta['experiment']) if meta.get('experiment') else None
        self.params=None
        self.params_revision=None
        if dataset:
            self.emit('dataset',dataset=dataset)
        if editor:
            self.emit('editor',editor=editor)
        self.emit('project',settings=self.settings,view=meta.get('view'))
        self.status('stopped' if self.experiment else 'viewing')
        self.preview_revision+=1

    def save_as(self,root,view):
        if self.project:
            self.save_project(view)
            import shutil
            target=Path(root)/f'studio-copy-{uuid.uuid4().hex[:8]}'
            shutil.copytree(self.project,target)
            self.open_project(target/'project.json')
        else:
            self.project=Path(root)/f'studio-project-{uuid.uuid4().hex[:8]}'
            self.project.mkdir(parents=True)
            if self.dataset:
                self.dataset.prepare(self.project/'dataset')
                self.dataset=Dataset(self.project/'dataset')
            self.save_project(view)
