"""Bounded atomic GS source/render preview; identical camera, no heldout fitting."""
import json
from pathlib import Path
import re
import time
import numpy as np
from PIL import Image
from .project import atomic_json, digest

REQUEST_ID = re.compile(r'^[0-9a-f]{32}$')


class Preview:
    def __init__(self,p,generation_id,emit=None):
        self.p,self.output,self.generation_id = p,Path(p['path']),generation_id
        self.emit = emit or (lambda *a:None)
        self.last_request,self.last_step = None,None
        self._last_poll = float('-inf')
        self._pending_due = False
        # A stale request is never applied to a new generation.
        try:
            self.last_request = json.loads((self.output/'preview-request.json').read_text())['request_id']
        except (OSError,ValueError,KeyError):
            pass

    def tick(self,params,step,target,force=False):
        every = self.p['studio_training'].get('preview_every',0)
        self._pending_due |= every>0 and step%every==0 and step!=self.last_step
        now = time.monotonic()
        if not force and now-self._last_poll < .15:
            return
        self._last_poll = now
        request,row = 'default',self.p['reconstruction']['cameras'][self.p['studio_dataset']['train_indices'][0]]
        try:
            value = json.loads((self.output/'preview-request.json').read_text(encoding='utf-8'))
        except FileNotFoundError:
            value = None
        except (OSError,ValueError) as error:
            self.emit('log',{'message':f'Preview request rejected: {error}'})
            value = None
        if value is not None:
            if (not isinstance(value,dict) or value.get('schema_version')!=1 or
                    value.get('package_sha256')!=self.p['studio_dataset']['package_sha256'] or
                    not REQUEST_ID.fullmatch(str(value.get('request_id',''))) or
                    value.get('generation_id',self.generation_id)!=self.generation_id):
                self.emit('log',{'message':'Preview request identity rejected'})
                value = None
            elif value['request_id'] == self.last_request:
                value = None
            else:
                request = value['request_id']
                matches = [r for r in self.p['reconstruction']['cameras'] if r['image']==value.get('image')]
                if not matches:
                    raise ValueError('Preview camera is outside verified package')
                row = matches[0]
        due = self._pending_due
        if not (force or due or value):
            return
        import torch
        from .training import render_parameters
        factor = min(1.,640/max(row['width'],row['height']))
        w,h = max(1,round(row['width']*factor)),max(1,round(row['height']*factor))
        K = np.array(row['K'],copy=True,dtype=float)
        K[0]*=w/row['width']; K[1]*=h/row['height']
        camera = {**row,'width':w,'height':h,'K':K.tolist()}
        with torch.no_grad():
            degree = min(self.p['settings']['sh_degree'],step//self.p['settings']['sh_degree_interval'])
            image,_,_ = render_parameters(params,camera,degree,projection_policy=(self.p.get('training',{}).get('recipe') or {}).get('projection'))
            pixels = (image[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        with Image.open(Path(self.p['studio_dataset']['package'])/row['image']) as im:
            source = im.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
        prior = None
        try:
            prior = json.loads((self.output/'preview.json').read_text()).get('slot')
        except (OSError,ValueError):
            pass
        slot = 1-prior if prior in (0,1) else 0
        names = [f'preview-slot-{slot}-{kind}.png' for kind in ('source','render')]
        for name,im in zip(names,[source,Image.fromarray(pixels)]):
            temp = self.output/(name+'.tmp')
            im.save(temp,format='PNG')
            temp.replace(self.output/name)
        payload = {'schema_version':1,'slot':slot,'package_sha256':self.p['studio_dataset']['package_sha256'],
                   'generation_id':self.generation_id,'request_id':request,'image':row['image'],'split':row['split'],
                   'step':step,'target':target,'source_file':names[0],'render_file':names[1],
                   'source_sha256':digest(self.output/names[0]),'render_sha256':digest(self.output/names[1]),
                   'camera':camera}
        atomic_json(self.output/'preview.json',payload)
        self.emit('preview',payload)
        self.last_request,self.last_step = request,step
        self._pending_due = False
