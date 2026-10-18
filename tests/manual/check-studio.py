"""Studio integration checks in an explicitly new output folder; no production runs."""
import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

from gsstudio.pipeline.editor.data import write_image
from gsstudio.pipeline.training.data import json_write
from gsstudio.infrastructure.adapters.media import sha256_file


def make_package(root, gpu=False):
    package=root/'synthetic-data'
    package.mkdir()
    rng=np.random.default_rng(7)
    xyz=rng.uniform([-1,-1,2],[1,1,4],(100,3)).astype(np.float32)
    np.savez(package/'points.npz',xyz=xyz,rgb=rng.integers(30,220,(100,3),dtype=np.uint8))
    params=None
    if gpu:
        from gsstudio.pipeline.training import native
        native.configure_windows_cuda()
        params=native.initialize(package)
    rows=[]
    for i in range(12):
        matrix=np.eye(4)
        matrix[0,3]=(i-6)*.03
        row=dict(image=f'images/{i}.png',mask=f'masks/{i}.png',width=128,height=96,
            K=[[90.,0,64],[0,90.,48],[0,0,1]],world_to_camera=matrix.tolist(),frame=i,
            timestamp_seconds=i*.5,split='validation' if i%8==7 else 'train')
        if gpu:
            import torch
            with torch.no_grad():
                pixels,_,_=native.render(params,row)
                data=(pixels[0].cpu().numpy()[...,::-1].clip(0,1)*255).astype(np.uint8)
        else:
            data=np.zeros((96,128,3),np.uint8)
            data[15:80,25:100]=[70,150,200]
        write_image(package/row['image'],data)
        write_image(package/row['mask'],np.full((96,128),255,np.uint8))
        rows.append(row)
    json_write(package/'dataset.json',dict(schema_version=1,validation='passed',synthetic=True,images=rows,
        files={p.relative_to(package).as_posix():sha256_file(p) for p in package.rglob('*') if p.is_file()}))
    return package,rows


def gpu_check(root):
    import queue
    import torch
    from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
    from gsstudio.pipeline.editor.runtime import Runtime
    with gpu_session(timeout=30):
        package,rows=make_package(root,True)
    runtime=Runtime()
    runtime.import_data(root=package)
    runtime.set_camera(rows[0])
    original=runtime._observer
    pause_once=[False]
    stop_once=[False]
    def observer(params,progress,save):
        if progress['step']==2 and not pause_once[0]:
            pause_once[0]=True
            runtime.pause.set()
        if progress['step']==8 and not stop_once[0]:
            stop_once[0]=True
            runtime.stop.set()
        return original(params,progress,save)
    runtime._observer=observer
    settings=dict(steps=12,max_size=0,sh_degree=3,photo_comp=True,checkpoint_every=2,use_bilateral_grid=True,use_sparse_depth=True)
    runtime.submit(runtime.start,root/'projects',settings)
    states=[]
    preview_count=0
    resumed=False
    deadline=time.monotonic()+180
    try:
        while time.monotonic()<deadline:
            try:
                kind,p=runtime.events.get(timeout=1)
            except queue.Empty:
                continue
            if kind=='error':
                raise RuntimeError(p['message'])
            if kind=='preview':
                assert p['pixels'].shape==(96,128,3)
                preview_count+=1
            if kind=='status':
                states.append(p['state'])
                if p['state']=='paused':
                    saved=torch.load(runtime.experiment/'checkpoint.pt',map_location='cpu')
                    assert saved['step']==2
                    runtime.pause.clear()
                elif p['state']=='stopped':
                    assert not resumed
                    saved=torch.load(runtime.experiment/'checkpoint.pt',map_location='cpu')
                    assert saved['step']==8
                    resumed=True
                    runtime.submit(runtime.start,root/'projects',settings,True)
                elif p['state']=='completed':
                    break
        else:
            raise TimeoutError('Studio integration timed out')
        assert preview_count and resumed and 'paused' in states
        model=runtime.experiment/'model.ply'
        assert model.is_file()
        summary=json.loads((runtime.experiment/'training.json').read_text())
        assert summary['status']=='succeeded' and summary['steps']==12
        # Stop worker before editing CPU state in this test.
        runtime.quit.set()
        runtime.thread.join(10)
        runtime.editor.transform([1,0,0],[0,15,0],1.1)
        runtime.editor.export(root/'edited.ply')
        runtime.save_project()
        runtime.open_project(runtime.project/'project.json')
        assert len(runtime.editor.history)==1
        json_write(root/'gpu-result.json',dict(status='passed',states=states,previews=preview_count,
            paused_step=2,stopped_step=8,resumed_step=12,export_roundtrip=summary['export_roundtrip_max_error'],
            synthetic_only=True,quality_approved=False))
    finally:
        runtime.stop.set()
        runtime.pause.clear()
        runtime.quit.set()
        runtime.thread.join(10)


def gui_check(root):
    import tkinter as tk
    from PIL import Image
    from prototypes.tk_studio.studio import Studio
    from gsstudio.pipeline.editor.data import Dataset
    package,rows=make_package(root)
    window=tk.Tk()
    app=Studio(window)
    try:
        app.runtime.emit('dataset',dataset=Dataset(package))
        app.runtime.status('ready')
        window.update()
        app.poll()
        app.image_list.selection_set(0)
        app.select_image()
        for _ in range(5):
            window.update()
            time.sleep(.05)
        assert app.dataset and len(app.dataset.rows)==12
        # PrintWindow captures this test application's client area even when
        # the interactive desktop is locked; never capture other applications.
        import ctypes
        from ctypes import wintypes
        user,gdi=ctypes.windll.user32,ctypes.windll.gdi32
        hwnd=window.winfo_id()
        user.GetDC.argtypes=[wintypes.HWND]
        user.GetDC.restype=wintypes.HDC
        gdi.CreateCompatibleDC.argtypes=[wintypes.HDC]
        gdi.CreateCompatibleDC.restype=wintypes.HDC
        gdi.CreateCompatibleBitmap.argtypes=[wintypes.HDC,ctypes.c_int,ctypes.c_int]
        gdi.CreateCompatibleBitmap.restype=wintypes.HBITMAP
        gdi.SelectObject.argtypes=[wintypes.HDC,wintypes.HGDIOBJ]
        gdi.SelectObject.restype=wintypes.HGDIOBJ
        user.PrintWindow.argtypes=[wintypes.HWND,wintypes.HDC,wintypes.UINT]
        gdi.GetDIBits.argtypes=[wintypes.HDC,wintypes.HBITMAP,wintypes.UINT,wintypes.UINT,ctypes.c_void_p,ctypes.c_void_p,wintypes.UINT]
        gdi.DeleteObject.argtypes=[wintypes.HGDIOBJ]
        gdi.DeleteDC.argtypes=[wintypes.HDC]
        user.ReleaseDC.argtypes=[wintypes.HWND,wintypes.HDC]
        w,h=window.winfo_width(),window.winfo_height()
        dc=user.GetDC(hwnd)
        memory=gdi.CreateCompatibleDC(dc)
        bitmap=gdi.CreateCompatibleBitmap(dc,w,h)
        old=gdi.SelectObject(memory,bitmap)
        try:
            if not user.PrintWindow(hwnd,memory,3):
                raise RuntimeError('PrintWindow failed; visual QA unavailable')
            header=struct.pack('<IiiHHIIiiII',40,w,-h,1,32,0,w*h*4,0,0,0,0)
            buffer=ctypes.create_string_buffer(w*h*4)
            if not gdi.GetDIBits(memory,bitmap,0,h,buffer,header,0):
                raise RuntimeError('Window image readback failed')
            im=Image.frombuffer('RGBA',(w,h),buffer.raw,'raw','BGRA',0,1).convert('RGB')
            if np.asarray(im).std()<3:
                raise RuntimeError('Window capture is blank; visual QA unavailable')
            im.save(root/'studio-window.png')
        finally:
            gdi.SelectObject(memory,old)
            gdi.DeleteObject(bitmap)
            gdi.DeleteDC(memory)
            user.ReleaseDC(hwnd,dc)
        json_write(root/'gui-result.json',dict(status='passed',width=window.winfo_width(),height=window.winfo_height(),synthetic_only=True))
    finally:
        app.runtime.quit.set()
        window.destroy()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.gpu:
        gpu_check(args.output)
    else:
        gui_check(args.output)
    print(json.dumps({'output':str(args.output),'status':'passed'}))
