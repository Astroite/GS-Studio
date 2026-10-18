"""Optional GS observations/objectives; ordinary-photo recipes are untouched."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class InvalidMask(ValueError):
    pass


def masked_statistics(prediction, target, keep, *, kind='gaussian11', padding='valid'):
    """Differentiable NHWC metrics; masked samples never enter local moments.

    Only windows whose entire footprint is kept enter SSIM. Masked RGB/NaNs
    never enter moments or L1. Window kind/padding follows the frozen recipe.
    """
    if prediction.ndim == 3:
        prediction, target = prediction[None], target[None]
    if keep.ndim == 2:
        keep = keep[None]
    if (prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[-1] != 3 or
            keep.shape != prediction.shape[:-1] or keep.dtype != torch.bool or
            prediction.device != target.device or keep.device != prediction.device):
        raise ValueError('Expected matching NHWC RGB and boolean NHW support')
    if not bool(keep.any()):
        raise InvalidMask('All pixels excluded by mask')
    if padding not in ('valid','same') or kind not in ('gaussian11','uniform11'):
        raise ValueError('Unknown masked SSIM window')
    x, y = prediction.permute(0,3,1,2), target.permute(0,3,1,2)
    w = keep[:,None].to(x.dtype)
    # where (rather than multiplication alone) also excludes invalid NaN RGB.
    x, y = torch.where(w.bool(), x, 0), torch.where(w.bool(), y, 0)
    count = w.sum()*3
    l1 = (x-y).abs().sum()/count
    mse = (x-y).square().sum()/count
    if kind == 'gaussian11' and min(x.shape[-2:]) < 11:
        raise InvalidMask('Gaussian11 SSIM requires images at least11 pixels on each axis')
    size = min(11, x.shape[-2], x.shape[-1])
    size -= size % 2 == 0
    coordinate = torch.arange(size, device=x.device, dtype=x.dtype)-(size-1)/2
    kernel = torch.exp(-.5*(coordinate/1.5).square()) if kind == 'gaussian11' else torch.ones_like(coordinate)
    kernel = kernel/kernel.sum()
    pad = size//2 if padding == 'same' else 0
    def blur(t):
        c = t.shape[1]
        t = F.conv2d(t, kernel[None,None,None,:].expand(c,1,1,size), groups=c, padding=(0,pad))
        return F.conv2d(t, kernel[None,None,:,None].expand(c,1,size,1), groups=c, padding=(pad,0))
    # Integer footprint counts, not Gaussian mass: even the smallest nonzero
    # Gaussian tap touching an ignored pixel excludes the whole window.
    support = F.conv2d(w,torch.ones((1,1,size,size),device=x.device,dtype=x.dtype),padding=pad)
    valid = support == size*size
    if not bool(valid.any()):
        raise InvalidMask('No full SSIM window remains after masking')
    values = torch.cat([x,y,x*x,y*y,x*y],1)
    moments = (F.avg_pool2d(values,size,1,pad) if kind == 'uniform11' else blur(values))
    a,b,xx,yy,xy = moments.chunk(5,1)
    score_map = ((2*a*b+.01**2)*(2*(xy-a*b)+.03**2))/((a*a+b*b+.01**2)*(xx-a*a+yy-b*b+.03**2))
    score = torch.where(valid, score_map, 0).sum()/(3*valid.sum())
    # Full-size map for observed recipe statistics; excluded centers are zero.
    error = torch.where(valid, (1-score_map).mean(1,keepdim=True), 0)
    if not pad and size > 1:
        error = F.pad(error, [size//2]*4)
    return {'l1':l1, 'mse':mse, 'ssim':score, 'dssim_map':error[:,0]}


class BilateralGrid(torch.nn.Module):
    def __init__(self, num_groups, spatial=8, bins=8):
        super().__init__()
        if min(num_groups,spatial,bins) < 1:
            raise ValueError('Positive bilateral grid dimensions required')
        self.spatial, self.bins = spatial,bins
        grid = torch.zeros(num_groups,2,3,bins,spatial,spatial)
        grid[:,0] = 1
        self.grid = torch.nn.Parameter(grid)

    def forward(self, rgb, group):
        h,w,_ = rgb.shape
        ys,xs = torch.meshgrid(torch.linspace(-1,1,h,device=rgb.device,dtype=rgb.dtype),
                              torch.linspace(-1,1,w,device=rgb.device,dtype=rgb.dtype), indexing='ij')
        coords = torch.stack([xs,ys,rgb.detach().mean(-1)*2-1],-1)[None,None]
        volume = self.grid[group].reshape(1,6,self.bins,self.spatial,self.spatial)
        affine = F.grid_sample(volume, coords, mode='bilinear', padding_mode='border', align_corners=True)[0,:,0].permute(1,2,0)
        return affine[...,:3]*rgb+affine[...,3:]

    def regularization(self):
        g = self.grid
        loss = (g[:,0]-1).square().mean()+g[:,1].square().mean()
        for axis in (-1,-2,-3):
            if g.shape[axis] > 1:
                loss = loss+.1*torch.diff(g,dim=axis).square().mean()
        return loss


class PanoramaExposure(torch.nn.Module):
    def __init__(self, timestamps):
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(len(timestamps),6))
        self.register_buffer('anchor',torch.tensor([0.]+[1.]*(len(timestamps)-1))[:,None])
        order = np.argsort(timestamps,kind='stable')
        self.register_buffer('order',torch.tensor(order,dtype=torch.long))
        self.register_buffer('dt',torch.tensor(np.diff(np.asarray(timestamps)[order]),dtype=torch.float32).clamp_min(.1))

    def forward(self, rgb, group):
        v = (self.values*self.anchor)[group]
        return rgb*v[:3].exp()+v[3:]

    def regularization(self):
        v = self.values*self.anchor
        ordered = v[self.order]
        smooth = (torch.diff(ordered,dim=0).square()/self.dt[:,None]).mean() if len(v)>1 else v.sum()*0
        return 1e-3*(v.square().mean()+smooth)


def select_photometric(photo_comp, use_bilateral_grid, group_times):
    if not photo_comp:
        return None
    return BilateralGrid(len(group_times)) if use_bilateral_grid else PanoramaExposure(group_times)


class StudioAuxiliary:
    def __init__(self, p, device='cuda'):
        self.p = p
        self.root = Path(p['studio_dataset']['package'])
        self.cameras = p['reconstruction']['cameras']
        self.config = p['studio_training']
        self.groups = {}
        times = []
        for i in p['studio_dataset']['train_indices']:
            row = self.cameras[i]
            if row['frame'] not in self.groups:
                self.groups[row['frame']] = len(times)
                times.append(row['timestamp_seconds'])
        self.photometric = select_photometric(self.config['photo_comp'],self.config['use_bilateral_grid'],times)
        if self.photometric is not None:
            self.photometric.to(device)
        self.optimizer = torch.optim.Adam(self.photometric.parameters(), lr=1e-3) if self.photometric is not None else None
        self.anchors = None
        file = Path(p['reconstruction']['path'])/'sparse_depth.npz'
        if self.config['use_sparse_depth'] and 'sparse_depth.npz' in p['reconstruction']['artifacts'] and file.is_file():
            with np.load(file, allow_pickle=False) as data:
                self.anchors = {k:data[k] for k in data.files}
        self.masks = OrderedDict()
        self.anchor_rows = {}
        if self.anchors is not None:
            for index in np.unique(self.anchors['row_index']):
                selected = self.anchors['row_index']==index
                self.anchor_rows[int(index)] = {
                    'u':torch.as_tensor(self.anchors['u'][selected],device=device,dtype=torch.long),
                    'v':torch.as_tensor(self.anchors['v'][selected],device=device,dtype=torch.long),
                    'depth':torch.as_tensor(self.anchors['depth'][selected],device=device,dtype=torch.float32)}
        if self.config['use_sparse_depth'] and not self.need_depth:
            raise ValueError('Sparse depth enabled without valid training anchors; resolve package availability first')

    @property
    def need_depth(self):
        return self.anchors is not None and bool(len(self.anchors['depth']))

    def mask(self, index, device):
        if not self.cameras[index]['has_mask']:
            return None
        if index not in self.masks:
            with Image.open(self.root/self.cameras[index]['mask']) as im:
                self.masks[index] = np.asarray(im).copy() == 255
            while len(self.masks) > 8:
                self.masks.popitem(last=False)
        self.masks.move_to_end(index)
        return torch.as_tensor(self.masks[index],device=device)[None]

    def compensate(self, image, index):
        if self.photometric is None:
            return image
        row = self.cameras[index]
        if row['split'] != 'train':
            raise ValueError('No heldout photometric fitting')
        return self.photometric(image[0],self.groups[row['frame']])[None]

    def loss(self, prediction, target, keep, params, recipe, cfg, observed=False):
        parts = masked_statistics(prediction,target,keep,kind=recipe['ssim_kind'] if recipe else 'uniform11',
                                  padding=recipe['ssim_padding'] if recipe else 'same')
        weight = recipe['ssim_lambda'] if recipe else cfg.get('ssim_lambda',.2)
        loss = (1-weight)*parts['l1']+weight*(1-parts['ssim'])
        if recipe:
            if recipe['opacity_reg']:
                loss = loss+recipe['opacity_reg']*params['opacities'].sigmoid().mean()
            if recipe['scale_reg']:
                loss = loss+recipe['scale_reg']*params['scales'].exp().mean()
        if observed:
            absolute = (torch.where(keep[...,None],prediction,0)-torch.where(keep[...,None],target,0)).abs()
            parts['densify_error_map'] = ((1-weight)*absolute.sum(-1)[0]/(3*keep.sum())+
                                          weight*parts['dssim_map'][0]).detach().contiguous()
        return {**parts,'loss':loss}

    def extra_loss(self, depth, index, prediction):
        loss = prediction.sum()*0
        if self.photometric is not None:
            loss = loss+self.photometric.regularization()
        if depth is not None and self.need_depth:
            anchors = self.anchor_rows.get(index)
            if anchors is not None:
                loss = loss+self.config.get('sparse_depth_weight',.1)*(depth[0,anchors['v'],anchors['u']]-anchors['depth']).abs().mean()
        return loss

    def step(self):
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return {'schema_version':1, 'identity':self.config,
                'photometric':self.photometric.state_dict() if self.photometric is not None else None,
                'optimizer':self.optimizer.state_dict() if self.optimizer else None}

    def load_state_dict(self, state):
        if not state or state.get('schema_version') != 1 or state.get('identity') != self.config:
            raise ValueError('Checkpoint auxiliary identity mismatch')
        if (state['photometric'] is None) != (self.photometric is None) or (state['optimizer'] is None) != (self.optimizer is None):
            raise ValueError('Checkpoint auxiliary state is incomplete')
        if self.photometric is not None:
            self.photometric.load_state_dict(state['photometric'],strict=True)
            self.optimizer.load_state_dict(state['optimizer'])
