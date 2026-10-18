"""Read-only COLMAP inspection and immutable Studio training packages.

No SfM is performed here. PINHOLE and SIMPLE_PINHOLE are deliberately the
supported camera set: distorted images must first be exported undistorted.

Text/JSON I/O in this package uses explicit UTF-8 (COLMAP txt may be utf-8-sig);
image/model binaries stay binary. Chinese paths are supported end-to-end.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.training.data import json_write, validate_package


def read_image(path, grayscale=False):
    data = np.fromfile(Path(path), dtype=np.uint8)
    result = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f'无法读取图片: {path}')
    return result


def write_image(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, data)
    if not ok:
        raise ValueError(f'无法编码图片: {path}')
    encoded.tofile(path)


def safe_relative(name):
    name = name.replace('\\', '/').removeprefix('./')
    path = Path(name)
    if path.is_absolute() or ':' in name or '..' in path.parts or not name:
        raise ValueError(f'非法相对路径: {name}')
    return name


def inside(root, name):
    path = (root / safe_relative(name)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'文件链接指向数据集外部: {name}')
    return path


def discover_models(root):
    root = Path(root)
    candidates = [root, root/'colmap', root/'sparse']
    if root.is_dir():
        candidates += sorted(p for p in root.iterdir() if p.is_dir())
    if (root/'sparse').is_dir():
        candidates += sorted(p for p in (root/'sparse').iterdir() if p.is_dir())
    if (root/'colmap').is_dir():
        candidates += sorted(p for p in (root/'colmap').iterdir() if p.is_dir())
    return list(dict.fromkeys(str(p.resolve()) for p in candidates
        if any(all((p/f'{n}.{ext}').is_file() for n in ('cameras', 'images', 'points3D'))
               for ext in ('bin', 'txt'))))


def _unpack(stream, fmt):
    size = struct.calcsize('<'+fmt)
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError('COLMAP 文件被截断')
    return struct.unpack('<'+fmt, raw)


def _unique(items, key, value):
    if key in items or key < 0:
        raise ValueError(f'重复或非法 COLMAP ID: {key}')
    items[key] = value


def read_model(folder):
    folder = Path(folder)
    cameras, images, points = {}, {}, {}
    # Parameter lengths allow unsupported models to be reported without
    # guessing how to render their distortion.
    models = {0: ('SIMPLE_PINHOLE', 3), 1: ('PINHOLE', 4), 2: ('SIMPLE_RADIAL', 4),
              3: ('RADIAL', 5), 4: ('OPENCV', 8), 5: ('OPENCV_FISHEYE', 8),
              6: ('FULL_OPENCV', 12), 7: ('FOV', 5), 8: ('SIMPLE_RADIAL_FISHEYE', 4),
              9: ('RADIAL_FISHEYE', 5), 10: ('THIN_PRISM_FISHEYE', 12)}
    if all((folder/f'{n}.bin').is_file() for n in ('cameras', 'images', 'points3D')):
        with (folder/'cameras.bin').open('rb') as f:
            for _ in range(_unpack(f, 'Q')[0]):
                cid, model, w, h = _unpack(f, 'iiQQ')
                if model not in models:
                    raise ValueError(f'不支持的 COLMAP 相机类型 ID: {model}')
                name, n = models[model]
                _unique(cameras, cid, (name, w, h, _unpack(f, 'd'*n)))
        with (folder/'images.bin').open('rb') as f:
            for _ in range(_unpack(f, 'Q')[0]):
                vals = _unpack(f, 'i7di')
                name = bytearray()
                while True:
                    ch = f.read(1)
                    if not ch:
                        raise ValueError('COLMAP 图片名被截断')
                    if ch == b'\0':
                        break
                    name.extend(ch)
                n = _unpack(f, 'Q')[0]
                # Keep observations for training-only initialization colors.
                obs = [_unpack(f, 'ddq') for _ in range(n)]
                _unique(images, vals[0], dict(q=vals[1:5], t=vals[5:8], camera=vals[8],
                    name=safe_relative(name.decode('utf-8')), obs=obs))
        with (folder/'points3D.bin').open('rb') as f:
            for _ in range(_unpack(f, 'Q')[0]):
                v = _unpack(f, 'Q3d3BdQ')
                _unique(points, v[0], (v[1:4], v[4:7]))
                length = v[-1]*8
                if f.tell()+length > (folder/'points3D.bin').stat().st_size:
                    raise ValueError('COLMAP 点轨迹被截断')
                f.seek(length, 1)
    else:
        def lines(name):
            return (line.strip() for line in (folder/name).read_text(encoding='utf-8-sig').splitlines()
                    if not line.lstrip().startswith('#'))
        for line in lines('cameras.txt'):
            if line:
                v = line.split()
                _unique(cameras, int(v[0]), (v[1], int(v[2]), int(v[3]), tuple(map(float, v[4:]))))
        it = iter(lines('images.txt'))
        for line in it:
            if not line:
                continue
            v = line.split(maxsplit=9)
            if len(v) != 10:
                raise ValueError('无效的 COLMAP 图片记录')
            obs = next(it, None)
            if obs is None:
                raise ValueError('COLMAP 图片缺少观测行')
            fields = obs.split()
            if len(fields) % 3:
                raise ValueError('无效的 COLMAP 观测记录')
            _unique(images, int(v[0]), dict(q=tuple(map(float,v[1:5])), t=tuple(map(float,v[5:8])),
                camera=int(v[8]), name=safe_relative(v[9]),
                obs=[(float(fields[j]),float(fields[j+1]),int(fields[j+2])) for j in range(0,len(fields),3)]))
        for line in lines('points3D.txt'):
            if line:
                v = line.split()
                _unique(points,int(v[0]),(tuple(map(float,v[1:4])),tuple(map(int,v[4:7]))))
    return cameras, images, points


def pose(q, t):
    q, t = np.asarray(q, float), np.asarray(t, float)
    if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError('非法相机位姿')
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
        raise ValueError('相机四元数未归一化')
    w,x,y,z = q/np.linalg.norm(q)
    m = np.eye(4)
    m[:3,:3] = [[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]]
    m[:3,3] = t
    return m.tolist()


def _write_training_colmap(output, records, source_points, colors, counts, ids):
    """Publish a PINHOLE model matching the normalized training image names."""
    folder = output/'colmap'
    folder.mkdir()
    tracks = {pid: [] for pid in ids}
    with (folder/'cameras.bin').open('wb') as cameras, (folder/'images.bin').open('wb') as images:
        cameras.write(struct.pack('<Q',len(records)))
        images.write(struct.pack('<Q',len(records)))
        for record in records:
            iid, row, source, observations = record
            k = row['K']
            cameras.write(struct.pack('<iiQQ4d',iid,1,row['width'],row['height'],
                k[0][0],k[1][1],k[0][2],k[1][2]))
            images.write(struct.pack('<i7di',iid,*source['q'],*source['t'],iid))
            images.write(Path(row['image']).name.encode('utf-8')+b'\0')
            images.write(struct.pack('<Q',len(observations)))
            for index,(x,y,pid) in enumerate(observations):
                accepted = pid if pid in tracks else -1
                images.write(struct.pack('<ddq',x,y,accepted))
                if accepted != -1:
                    tracks[accepted].append((iid,index))
    with (folder/'points3D.bin').open('wb') as points:
        points.write(struct.pack('<Q',len(ids)))
        for pid in ids:
            xyz = source_points[pid][0]
            rgb = np.rint(colors[pid]/counts[pid]).astype(np.uint8)
            track = tracks[pid]
            points.write(struct.pack('<Q3d3BdQ',pid,*xyz,*rgb,0.,len(track)))
            for image_id,index in track:
                points.write(struct.pack('<ii',image_id,index))


class Dataset:
    def __init__(self, root, images=None, model=None, masks=None, white_ignore=False,
                 groups=None, independent_images=False):
        self.root = Path(root).resolve()
        self.image_root = Path(images).resolve() if images else self.root/'images'
        self.mask_root = Path(masks).resolve() if masks else None
        self.white_ignore = white_ignore
        self.group_source = 'shared_package' if (self.root/'dataset.json').is_file() else 'required'
        self.group_labels = {}
        self.errors, self.warnings, self.rows, self.files = [], [], [], {}
        self.model = None
        self.package = (self.root/'dataset.json').is_file()
        if self.package:
            meta = validate_package(self.root)
            self.rows = [dict(r, source_image=r.get('source_image',r['image']),
                              source_path=str(self.root/r['image']), mask_path=str(self.root/r['mask']))
                         for r in meta['images']]
            with np.load(self.root/'points.npz') as p:
                self.xyz, self.rgb = p['xyz'].copy(), p['rgb'].copy()
            self.files = {str(self.root/n): h for n,h in meta['files'].items()}
            self.files[str(self.root/'dataset.json')] = sha256_file(self.root/'dataset.json')
            self.white_ignore = False
            self.inventory = len(self.rows)
            self.group_labels = {r['source_image']: str(r['frame']) for r in self.rows}
            self.warnings.append('已保留共享包的训练/验证分组、遮罩与初始化点；忽略外部遮罩选项。')
            for row in self.rows:
                pixels=read_image(row['source_path'])
                mask=read_image(row['mask_path'],True)
                R=np.asarray(row['world_to_camera'],float)
                K=np.asarray(row['K'],float)
                if pixels.shape[:2]!=(row['height'],row['width']) or mask.shape!=pixels.shape[:2]:
                    self.errors.append(f'图片或遮罩尺寸错误: {row["source_image"]}')
                if not np.isin(mask,[0,255]).all() or not (mask==255).any():
                    self.errors.append(f'无效的二值遮罩: {row["source_image"]}')
                if not np.allclose(R[3],[0,0,0,1]) or not np.allclose(R[:3,:3].T@R[:3,:3],np.eye(3),atol=1e-4) or np.linalg.det(R[:3,:3])<.99:
                    self.errors.append(f'非法位姿: {row["source_image"]}')
                if min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]):
                    self.errors.append(f'非法内参: {row["source_image"]}')
        else:
            if any((self.root/n).exists() for n in ('segments.json','mask-filter.json','mask-final.json','manifest.yaml')):
                raise ValueError('工作流数据请先通过分段 QA 并导入 training-data 中的共享包，不能绕过现有门禁。')
            models = discover_models(self.root)
            if not model and len(models) != 1:
                raise ValueError('请选择一个 COLMAP 模型: '+', '.join(models))
            self.model = Path(model or models[0]).resolve()
            cameras, images_by_id, self.points = read_model(self.model)
            self.cameras = cameras
            self.image_records = images_by_id
            inventory = sorted(p for p in self.image_root.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png'))
            self.inventory = len(inventory)
            used, names = set(), set()
            names_in_model = {im['name'] for im in images_by_id.values()}
            if groups is not None and independent_images:
                self.errors.append('显式分组与每图独立确认不能同时使用。')
            if groups is not None:
                if not isinstance(groups, dict):
                    self.errors.append('分组表须为图片名到分组 ID 的映射。')
                else:
                    missing = names_in_model - set(groups)
                    extra_groups = set(groups) - names_in_model
                    if missing or extra_groups or any(not str(value).strip() for value in groups.values()):
                        self.errors.append('分组表必须精确覆盖 COLMAP 图片：缺少 '
                            + ', '.join(sorted(missing)) + '；多余 ' + ', '.join(sorted(extra_groups)))
                    else:
                        self.group_source = 'explicit_mapping'
                        self.group_labels = {name: str(groups[name]).strip() for name in names_in_model}
            elif independent_images:
                self.group_source = 'independent_images_confirmed'
                self.group_labels = {im['name']: f'{iid:012d}' for iid, im in images_by_id.items()}
            else:
                self.errors.append('普通 COLMAP 缺少可信分组；请提供完整分组表，或明确确认每图独立。')
            labels = set(self.group_labels.values())
            ordered_labels = sorted(labels, key=int) if labels and all(label.isdecimal() for label in labels) else sorted(labels)
            group_ids = {label: index + 1 for index, label in enumerate(ordered_labels)}
            group_order = {label: index for index, label in enumerate(ordered_labels)}
            splits = {label: ('validation' if (index % 8 == 7 or
                (len(group_ids) < 8 and index == len(group_ids) - 1)) else 'train')
                for label, index in group_order.items()}
            if self.group_source == 'independent_images_confirmed':
                self.warnings.append('已明确确认每图独立；时间仅为图片序号，不能据此推断全景采样组。')
            elif self.group_source == 'explicit_mapping':
                self.warnings.append('使用用户提供的完整分组表；时间仅为分组序号。')
            for iid, im in sorted(images_by_id.items()):
                try:
                    name = im['name']
                    if name.casefold() in names:
                        raise ValueError(f'图片命名歧义: {name}')
                    names.add(name.casefold())
                    source = inside(self.image_root, name)
                    pixels = read_image(source)
                    mode,w,h,p = cameras[im['camera']]
                    if mode == 'SIMPLE_PINHOLE' and len(p)==3:
                        fx,cx,cy = p
                        fy = fx
                    elif mode == 'PINHOLE' and len(p)==4:
                        fx,fy,cx,cy = p
                    else:
                        raise ValueError(f'{name}: 不支持 {mode}，请从 RealityScan 导出去畸变图片与 PINHOLE 相机')
                    if w<=0 or h<=0 or not np.isfinite([fx,fy,cx,cy]).all() or min(fx,fy)<=0:
                        raise ValueError(f'{name}: 非法内参或尺寸')
                    if pixels.shape[:2] != (h,w):
                        raise ValueError(f'{name}: 图片尺寸与相机不一致')
                    mask_path = None
                    if self.mask_root:
                        rel = Path(name)
                        candidates = {inside(self.mask_root,n) for n in (name, name+'.png',
                            rel.with_suffix('.png').as_posix(), rel.with_suffix('.mask.png').as_posix(), name+'.mask.png')}
                        matches = [p for p in candidates if p.is_file()]
                        if len(matches)!=1:
                            raise ValueError(f'{name}: 遮罩缺失或对应不唯一')
                        mask_path = matches[0]
                        mask = read_image(mask_path,True)
                        if mask.shape!=(h,w):
                            raise ValueError(f'{name}: 遮罩尺寸错误')
                        if not np.isin(mask,[0,255]).all():
                            raise ValueError(f'{name}: 遮罩必须为 0/255 二值图')
                        if not (mask == (0 if white_ignore else 255)).any():
                            raise ValueError(f'{name}: 整张图片被遮罩排除')
                        self.files[str(mask_path)] = sha256_file(mask_path)
                    label = self.group_labels.get(name)
                    group = group_ids.get(label, iid)
                    self.rows.append(dict(source_image=name,source_path=str(source),mask_path=str(mask_path) if mask_path else None,
                        image_id=iid,frame=group,timestamp_seconds=float(group_order.get(label, iid)),
                        split=splits.get(label, 'unassigned'),
                        width=w,height=h,K=[[fx,0,cx],[0,fy,cy],[0,0,1]],world_to_camera=pose(im['q'],im['t'])))
                    self.files[str(source)] = sha256_file(source)
                    used.add(source)
                except (ValueError,KeyError,OSError) as e:
                    self.errors.append(str(e))
            extra = [str(p.relative_to(self.image_root)) for p in inventory if p.resolve() not in used]
            if extra:
                self.warnings.append('未参与训练的图片: '+', '.join(extra))
            for p in self.model.iterdir():
                if p.name in [f'{n}.{ext}' for n in ('cameras','images','points3D') for ext in ('bin','txt')]:
                    self.files[str(p)] = sha256_file(p)
            self.xyz = np.asarray([p[0] for p in self.points.values()],np.float32).reshape(-1,3)
            self.rgb = np.asarray([p[1] for p in self.points.values()],np.uint8).reshape(-1,3)
        if len(self.xyz)<4 or not np.isfinite(self.xyz).all():
            self.errors.append('初始化点不足或包含非有限值')
        if not any(r['split']=='train' for r in self.rows) or not any(r['split']=='validation' for r in self.rows):
            self.errors.append('需要至少两个独立采样组，以保留训练与验证分组')
        self.warnings.append('请确认相机畸变参数与图片匹配；无法自动识别所有错误配对。')

    def keep(self, row):
        if not row['mask_path']:
            return np.ones((row['height'],row['width']),bool)
        return read_image(row['mask_path'],True)==(0 if self.white_ignore else 255)

    def verify(self):
        if self.errors:
            raise ValueError('\n'.join(self.errors))
        for name,digest in self.files.items():
            if sha256_file(Path(name))!=digest:
                raise ValueError(f'输入已经改变，请重新导入并新建实验: {name}')

    def prepare(self, output, max_size=0):
        self.verify()
        if self.package:
            from gsstudio.pipeline.training.resize import resize_package
            if max_size:
                resize_package(self.root, Path(output), max_size)
            else:
                import shutil
                shutil.copytree(self.root, output)
                validate_package(Path(output))
            return Path(output)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        colors, counts = {}, {}
        track_images = {}
        colmap_records = []
        for i,row in enumerate(self.rows):
            pixels, keep = read_image(row['source_path']), self.keep(row)
            source_keep = keep.copy()
            factor = min(1,max_size/max(row['width'],row['height'])) if max_size else 1
            w,h = max(1,round(row['width']*factor)),max(1,round(row['height']*factor))
            if factor!=1:
                pixels = cv2.resize(pixels,(w,h),interpolation=cv2.INTER_AREA)
                keep = cv2.resize(keep.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
            K = np.asarray(row['K']).copy()
            K[0] *= w/row['width']
            K[1] *= h/row['height']
            image,mask = f'images/{i:08d}.png',f'masks/{i:08d}.png'
            write_image(output/image,pixels)
            write_image(output/mask,keep.astype(np.uint8)*255)
            rows.append({k:v for k,v in dict(row,image=image,mask=mask,width=w,height=h,K=K.tolist()).items()
                         if k not in ('source_path','mask_path')})
            if not self.package and row['split']=='train':
                scaled = []
                for x,y,pid in self.image_records[row['image_id']]['obs']:
                    if not np.isfinite([x,y]).all():
                        scaled.append((0.,0.,-1))
                        continue
                    px0,py0 = int(round(x)),int(round(y))
                    x,y = x*w/row['width'],y*h/row['height']
                    px,py = int(round(x)),int(round(y))
                    valid = (pid in self.points and 0<=px0<row['width'] and 0<=py0<row['height']
                             and source_keep[py0,px0] and 0<=px<w and 0<=py<h and keep[py,px])
                    scaled.append((float(x),float(y),pid if valid else -1))
                    if valid:
                        colors[pid] = colors.get(pid,np.zeros(3))+pixels[py,px,::-1]
                        counts[pid] = counts.get(pid,0)+1
                        track_images.setdefault(pid,set()).add(row['image_id'])
                colmap_records.append((row['image_id'],rows[-1],
                                       self.image_records[row['image_id']],scaled))
        if self.package:
            xyz,rgb = self.xyz,self.rgb
        else:
            ids = [p for p,n in counts.items() if n>=2 and len(track_images[p])>=2]
            if len(ids)<4:
                raise ValueError('独立引擎至少需要 4 个有两次未遮罩训练观测的初始化点')
            xyz = np.asarray([self.points[p][0] for p in ids],np.float32)
            rgb = np.asarray([np.rint(colors[p]/counts[p]) for p in ids],np.uint8)
            _write_training_colmap(output,colmap_records,self.points,colors,counts,ids)
        np.savez(output/'points.npz',xyz=xyz,rgb=rgb)
        json_write(output/'source.json',dict(root=str(self.root),files=self.files,white_ignore=self.white_ignore,
            max_size=max_size,group_source=self.group_source,group_labels=self.group_labels))
        json_write(output/'dataset.json',dict(schema_version=1,validation='passed',images=rows,
            coordinates='COLMAP',mask_polarity='white_keep',initial_points=len(xyz),
            group_source=self.group_source,source_kind='colmap',
            initialization_colors='unmasked_training_observations_only',
            files={p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob('*') if p.is_file()}))
        validate_package(output)
        return output
