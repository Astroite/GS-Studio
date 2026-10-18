"""Non-destructive snapshot edits, with full quaternion and SH rotation."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from gsstudio.pipeline.editor.ply import (FLOAT_PROPERTIES, read_ply_header, _rotate_quaternions_wxyz,
                  _sh_rotation_matrix)


def project_points(xyz, camera):
    matrix, K = np.asarray(camera['world_to_camera']), np.asarray(camera['K'])
    local = xyz @ matrix[:3,:3].T + matrix[:3,3]
    pixel = local @ K.T
    xy = pixel[:,:2]/np.maximum(pixel[:,2:3],1e-9)
    return xy,local[:,2]


class Editor:
    def __init__(self, source):
        self.source = Path(source).resolve()
        self.header,n,props = read_ply_header(self.source)
        if tuple(props)!=tuple(FLOAT_PROPERTIES) or self.source.stat().st_size!=len(self.header)+n*len(props)*4:
            raise ValueError('需要完整的标准 SH3 float Gaussian PLY')
        self.data = np.fromfile(self.source,dtype='<f4',offset=len(self.header)).reshape(n,len(props))
        if not n or not np.isfinite(self.data).all():
            raise ValueError('PLY 为空或包含非有限值')
        self.keep = np.ones(n,bool)
        self.selected = np.zeros(n,bool)
        self.matrix = np.eye(4)
        self.history, self.future = [], []
        self.revision = 0

    def _state(self):
        return np.packbits(self.keep), self.matrix.copy()

    def _restore(self, state):
        self.keep = np.unpackbits(state[0],count=len(self.data)).astype(bool)
        self.matrix = state[1].copy()
        self.selected[:] = False
        self.revision += 1

    def _remember(self):
        self.history.append(self._state())
        self.future.clear()
        self.revision += 1

    def undo(self):
        if self.history:
            self.future.append(self._state())
            self._restore(self.history.pop())

    def redo(self):
        if self.future:
            self.history.append(self._state())
            self._restore(self.future.pop())

    def xyz(self):
        return self.data[:,:3]@self.matrix[:3,:3].T+self.matrix[:3,3]

    def select(self, camera, rectangle, through=False, visible=None):
        xy,z = project_points(self.xyz(),camera)
        x0,y0,x1,y1 = rectangle
        candidate = self.keep & (z>0) & (xy[:,0]>=min(x0,x1)) & (xy[:,0]<=max(x0,x1)) & (xy[:,1]>=min(y0,y1)) & (xy[:,1]<=max(y0,y1))
        candidate &= (xy[:,0]>=0)&(xy[:,0]<camera['width'])&(xy[:,1]>=0)&(xy[:,1]<camera['height'])
        if not through:
            if visible is None or np.shape(visible) != (len(self.data),):
                raise ValueError('非穿透选择需要当前渲染的可见贡献结果')
            candidate &= visible
        self.selected = candidate

    def invert(self):
        self.selected = self.keep & ~self.selected

    def delete(self):
        if not self.selected.any():
            return
        if not (self.keep & ~self.selected).any():
            raise ValueError('不能删除全部高斯')
        self._remember()
        self.keep &= ~self.selected
        self.selected[:] = False

    def crop(self, low, high, keep_inside=True, commit=False):
        low,high = np.asarray(low,float),np.asarray(high,float)
        if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all() or np.any(low>=high):
            raise ValueError('裁剪框最小值必须小于最大值')
        xyz = self.xyz()
        inside = ((xyz>=low)&(xyz<=high)).all(axis=1)
        retained = self.keep & (inside if keep_inside else ~inside)
        if commit:
            if not retained.any():
                raise ValueError('裁剪将删除全部高斯')
            self._remember()
            self.keep = retained
            self.selected[:] = False
        else:
            self.selected = self.keep & ~retained
        return int(retained.sum())

    def transform(self, translation=(0,0,0), rotation=(0,0,0), scale=1, pivot=(0,0,0)):
        m=self.transform_matrix(translation,rotation,scale,pivot)
        self._remember()
        self.matrix=m@self.matrix

    @staticmethod
    def transform_matrix(translation=(0,0,0), rotation=(0,0,0), scale=1, pivot=(0,0,0)):
        if not np.isfinite([*translation,*rotation,scale,*pivot]).all() or scale<=0:
            raise ValueError('变换必须有限，缩放必须大于零')
        x,y,z = np.radians(rotation)
        rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        m=np.eye(4)
        m[:3,:3]=scale*(rz@ry@rx)
        pivot=np.asarray(pivot,float)
        m[:3,3]=np.asarray(translation,float)+pivot-m[:3,:3]@pivot
        return m

    def materialize(self, matrix_override=None, keep_override=None):
        matrix=self.matrix if matrix_override is None else np.asarray(matrix_override)
        keep=self.keep if keep_override is None else np.asarray(keep_override, dtype=bool)
        data=self.data[keep].copy()
        scale=np.cbrt(np.linalg.det(matrix[:3,:3]))
        rotation=matrix[:3,:3]/scale
        data[:,:3]=data[:,:3]@matrix[:3,:3].T+matrix[:3,3]
        data[:,3:6]=data[:,3:6]@rotation.T
        offset=FLOAT_PROPERTIES.index('scale_0')
        data[:,offset:offset+3]+=np.log(scale)
        offset=FLOAT_PROPERTIES.index('rot_0')
        _rotate_quaternions_wxyz(data[:,offset:offset+4],rotation)
        transform=_sh_rotation_matrix(rotation)
        for channel in range(3):
            offset=9+15*channel
            data[:,offset:offset+15]=data[:,offset:offset+15]@transform.T
        if not np.isfinite(data).all():
            raise ValueError('变换产生非有限参数')
        return data

    def export(self, destination):
        destination=Path(destination)
        if destination.exists():
            raise FileExistsError('请选择新文件，原始成果不会被覆盖')
        data=self.materialize()
        header=re.sub(rb'element vertex \d+',f'element vertex {len(data)}'.encode(),self.header)
        temporary=destination.with_suffix(destination.suffix+'.tmp')
        with temporary.open('xb') as f:
            f.write(header)
            f.write(data.astype('<f4').tobytes())
        temporary.rename(destination)

    def save_state(self, destination):
        states=self.history+[self._state()]
        np.savez_compressed(destination, keep=np.stack([s[0] for s in states]),
            matrices=np.stack([s[1] for s in states]),
            future_keep=np.stack([s[0] for s in self.future]) if self.future else np.empty((0,len(states[0][0])),np.uint8),
            future_matrices=np.stack([s[1] for s in self.future]) if self.future else np.empty((0,4,4)))

    def load_state(self, source):
        with np.load(source,allow_pickle=False) as s:
            states=list(zip(s['keep'],s['matrices']))
            if not states or any(m.shape!=(4,4) or not np.isfinite(m).all() or len(k)!=(len(self.data)+7)//8 for k,m in states):
                raise ValueError('编辑状态与模型不匹配')
            self.history=states[:-1]
            self.future=list(zip(s['future_keep'],s['future_matrices']))
            self._restore(states[-1])
