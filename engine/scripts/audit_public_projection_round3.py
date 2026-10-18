"""CPU-only public gsplat/R8 audit and independent mathematical diagnostics.

No production source modification, package upgrade, compilation or CUDA tensor
allocation occurs. The conic compensation function is an audit prototype, not
an automatically installed rendering change.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[1]
SOURCE=Path('E:/AI/LocalPostshot/sources/gsplat')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compensation2d(covariance,eps2d=.3,policy="gsplat",det_floor=1e-6):
    import torch
    assert covariance.shape[-2:]==(2,2)
    det_raw=covariance[...,0,0]*covariance[...,1,1]-covariance[...,0,1]*covariance[...,1,0]
    det_filtered=(covariance[...,0,0]+eps2d)*(covariance[...,1,1]+eps2d)-covariance[...,0,1]*covariance[...,1,0]
    if policy=="gsplat":
        return (det_raw/det_filtered).clamp_min(0).sqrt()
    if policy=="observed_floor":
        return (det_raw.clamp_min(det_floor)/det_filtered.clamp_min(det_floor)).sqrt()
    raise ValueError("Unknown independently specified compensation policy")


def floor_compensation_from_conics(conic,eps2d=.3,det_floor=1e-6):
    """Audit-only algebra: conic=inverse(raw covariance+epsI), C=1 or batched.

    Reconstructing a tiny raw determinant from float32 conics loses precision;
    exact original forward/backward equivalence needs independent CUDA gates.
    """
    a,b,c=conic.unbind(-1)
    inverse_det=a*c-b*b
    filtered_det=1/inverse_det
    raw_det=(1-eps2d*(a+c))/inverse_det+eps2d**2
    return (raw_det.clamp_min(det_floor)/filtered_det.clamp_min(det_floor)).sqrt()


def observed_near_double_reference(camera):
    """Audit scalar formula reported by parallel host/PTX audit.

    Both fx and fy use image width in the denominator. This float64 expression
    is not a claim of the host's exact float32 operation ordering.
    """
    width=float(camera['width'])
    fx,fy=float(camera['K'][0][0]),float(camera['K'][1][1])
    return min(.3,max(.1,4*.036*max(abs(fx/width),abs(fy/width))))


def cpu_diagnostics():
    import torch
    cuda_before=torch.cuda.is_initialized()
    torch.set_num_threads(4)
    covariance=torch.diag_embed(torch.tensor([[1.,1.],[.01,.01],[1e-5,1e-5],[1e-10,1e-10]],dtype=torch.float64))
    p=compensation2d(covariance)
    n=compensation2d(covariance,policy="observed_floor")
    filtered=covariance+torch.eye(2,dtype=torch.float64)*.3
    inverse=torch.linalg.inv(filtered)
    conic=torch.stack([inverse[...,0,0],inverse[...,0,1],inverse[...,1,1]],-1)
    reconstructed=floor_compensation_from_conics(conic)
    # Ordinary dense Adam keeps decaying moments on rows whose current image
    # contributes zero gradient. It differs from freezing hidden rows.
    parameter=torch.nn.Parameter(torch.tensor([1.,1.],dtype=torch.float64))
    optimizer=torch.optim.Adam([parameter],lr=.01,betas=(.9,.997),eps=1e-15)
    parameter.grad=torch.tensor([.3,.3],dtype=torch.float64);optimizer.step()
    before_parameter=parameter.detach().clone()
    before_moment=optimizer.state[parameter]['exp_avg'].clone()
    parameter.grad=torch.tensor([.3,0.],dtype=torch.float64);optimizer.step()
    after_parameter=parameter.detach().clone()
    after_moment=optimizer.state[parameter]['exp_avg'].clone()
    # Public SelectiveAdam's CUDA kernel lacks Adam bias correction even with
    # an all-true visibility mask; this is another semantic difference.
    b1,b2,gradient,lr=.9,.997,.3,.01
    plain_step=lr*((1-b1)*gradient)/(math.sqrt((1-b2)*gradient**2)+1e-15)
    standard_first_step=.01
    opacity=.005; variance=.3; center_x=-.99
    radius=math.ceil(min(3.33,math.sqrt(2*math.log(opacity/(1/255))))*math.sqrt(variance))
    first_pixel_alpha=opacity*math.exp(-.5*(center_x-.5)**2/variance)
    assert torch.cuda.is_initialized()==cuda_before
    return {'cuda_initialized':torch.cuda.is_initialized(),
        'cuda_initialized_on_entry':cuda_before,'cuda_state_unchanged':True,
        'compensation_cases':[{'raw_diagonal':row.diagonal().tolist(),'public_compensation':float(a),
                               'observed_floor_compensation':float(b),'conic_reconstructed_floor':float(c)}
                              for row,a,b,c in zip(covariance,p,n,reconstructed)],
        'dense_adam_zero_gradient':{'before_parameter':before_parameter.tolist(),'after_parameter':after_parameter.tolist(),
                                    'before_first_moment':before_moment.tolist(),'after_first_moment':after_moment.tolist(),
                                    'hidden_parameter_delta':float(after_parameter[1]-before_parameter[1]),
                                    'hidden_moment_decay_ratio':float(after_moment[1]/before_moment[1])},
        'selective_adam_all_visible_first_step':{'public_no_bias_correction_step':plain_step,
                                                'standard_bias_corrected_step':standard_first_step,
                                                'ratio':plain_step/standard_first_step},
        'support_not_contribution_example':{'center_x':center_x,'variance':variance,'opacity':opacity,'radius':radius,
                                            'bbox_overlaps_image':center_x+radius>0,'first_pixel_alpha':first_pixel_alpha,
                                            'alpha_threshold':1/255,'pixel_contribution_zero':first_pixel_alpha<1/255}}


def line_refs(path,needles):
    lines=path.read_text(encoding='utf-8-sig').splitlines()
    return [{'line':i+1,'text':line.strip()} for i,line in enumerate(lines) if any(n in line for n in needles)]


def inventory():
    root=Path(importlib.metadata.distribution('gsplat').locate_file('gsplat'))
    files={'rendering.py':['eps2d:','rasterize_mode:','near_plane:','far_plane:','radius_clip:','if compensations is not None','opacities = opacities * compensations'],
        'cuda/_wrapper.py':['def fully_fused_projection(','def rasterize_to_pixels(','sparse_grad is only supported'],
        'cuda/_torch_impl.py':['det = det.clamp','valid = (det > 0)','radius_x = torch.ceil'],
        'cuda/include/Utils.cuh':['add_blur(','compensation = sqrt','v_sqr_comp =','lim_x_pos =','lim_y_pos ='],
        'cuda/csrc/ProjectionEWA3DGSFused.cu':['mean_c.z < near_plane','float det = add_blur','opacity *= compensation','opacity < ALPHA_THRESHOLD','extend = min','radius_x = ceilf','radius_x <= radius_clip','radii[idx * 2] <= 0'],
        'cuda/csrc/RasterizeToPixels3DGSFwd.cu':['float alpha = min','sigma < 0.f','next_T <=','const float vis'],
        'optimizers/selective_adam.py':['def step(','state["step"]','adam('],
        'cuda/csrc/AdamCUDA.cu':['if (valid != nullptr','float step =','register_exp_avg =','register_exp_avg_sq =']}
    return {'version':importlib.metadata.version('gsplat'),
        'source_commit':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        'source_status':subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip(),
        'files':[{'relative_path':name,'source_path':str(SOURCE/'gsplat'/name),'source_sha256':sha(SOURCE/'gsplat'/name),
                  'installed_path':str(root/name),'installed_sha256':sha(root/name),
                  'match':sha(SOURCE/'gsplat'/name)==sha(root/name),'lines':line_refs(SOURCE/'gsplat'/name,needles)}
                 for name,needles in files.items()]}


def checkpoint_audit(path):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert not torch.cuda.is_initialized()
    p=json.loads((path.parent.parent/'project.json').read_text(encoding='utf-8-sig'))
    cameras=p['reconstruction']['cameras']
    near_values=[{'camera':c['name'],'width':c['width'],'height':c['height'],
                  'fx':c['K'][0][0],'fy':c['K'][1][1],'near_double_reference':observed_near_double_reference(c)}
                 for c in cameras]
    return {'path':str(path),'step':state['step'],'points':len(state['parameters']['means']),
        'recipe':state['training_recipe'],
        'optimizer_groups':{key:{field:value['param_groups'][0].get(field) for field in ['fused','foreach','betas','eps','lr']}
                            for key,value in state['optimizers'].items()},'cuda_initialized':False,
        'recovered_camera_near_double_reference':{'minimum':min(c['near_double_reference'] for c in near_values),
             'maximum':max(c['near_double_reference'] for c in near_values),'cameras':near_values,
             'note':'Parallel host/PTX formula, pending exact hostfloat32 rounding gate; perspective far is unbounded (n/z),1e6 applies to orthographic only; anchor near remains.1.'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('E:/AI/LocalPostshot/projects/SouthBuilding-Splat3-R8/checkpoints/latest.pt'))
    parser.add_argument('--output',type=Path,default=REPO/'evidence/splat3/round3-public-audit/public-audit.json')
    args=parser.parse_args()
    production={}
    for name,needles in {'training.py':['def render_parameters','near_plane=.01','optimizer.step(); optimizer.zero_grad'],
                         'splat3_regularization.py':['min_scale_px','near_plane=.1','eps2d=.3','depth >= .1'],
                         'splat3_strategy.py':['visible=(info["radii"]','info["opacities"]','state["sum2"].add_'],
                         'splat3_stats.py':['Public alpha cap','native_cuda_v1'],
                         '../native/splat3_stats/stats.cu':['fminf(.999f','alpha < 1.0f/255','next <= 1e-4f']}.items():
        path=REPO/'engine/local_postshot'/name
        production[name]={'path':str(path.resolve()),'sha256':sha(path),'lines':line_refs(path,needles)}
    result={'schema':1,'generated_utc':datetime.now(timezone.utc).isoformat(),'gpu_executed':False,
        'public_inventory':inventory(),'current_production_inventory':production,
        'cpu_diagnostics':cpu_diagnostics(),'r8_checkpoint':checkpoint_audit(args.checkpoint),
        'findings':[
            {'id':'3d_filter_not_proven_missing','conclusion':'gsplat1.5.3 has2D eps blur/AA but no3D mip preprocessing. That absence is not an original-implementation bug: parallel native audit says main i_jy8 only uses min_scale as inactive sentinel, not covariance/opacity. Do not insert min_scale^2 into3D covariance on speculation.',
             'native_evidence_status':'engine_impl message: i_jy8 35088..35089 loads+112 and compares-1; only use. Host compensation flag still pending at audit time.'},
            {'id':'2d_filter_defaults','conclusion':'Current render explicitly near.01/far1e8, leaves eps2d.3,classic,radius_clip0,camera_modelpinhole at public defaults. classic still adds.3I; it disables only the opacity compensation.',
             'fastest_supported_adaptation':'Once original flag binding is known, pass explicit eps2d/rasterize_mode/camera_model/near/far as a frozen projection policy through the current rasterization API. Existing CUDA forward/backward and stats info metadata work directly; no build required.'},
            {'id':'aa_floor_not_exact','conclusion':'Public CUDA compensation=sqrt(max(0,detraw/detblur)); native audit reports sqrt(max(1e-6,detraw)/max(1e-6,detblur)). Public add_blur_vjp also uses1/(compensation+1e-6). Built-in antialiased is an explicit substitute unless those branches/gradients are verified equivalent in the relevant scale range.',
             'minimal_wrapper_design':'For a diagnosed floor mismatch, split rasterization into public fully_fused_projection -> SH -> isect_tiles/offset_encode -> rasterize_to_pixels. Compute a custom differentiable compensation and feed compensated opacity to BOTH pixels and returned stats info. Keep original projection metadata/values from the same forward. A conic inversion prototype is supplied only for CPU algebra checks; float32 cancellation near the floor requires GPU/gradient gates. Do not silently monkeypatch the installed package.'},
            {'id':'visibility_definitions','conclusion':'radii>0 means clipped projected bounding support, not nonzero alpha*T and not lack of occlusion. Native stats uses exactly the renderer means/conics/compensated opacity/tile list. Changing any projection policy must change its metadata producer and anchor policy explicitly, not only visual output.'},
            {'id':'anchor_render_near_split','conclusion':'Current render near.01 differs from recovered main native per-camera near=min(.3,max(.1,4*.036*max(abs(fx/W),abs(fy/W)))). Native perspective Pz/Pw=n/z has no finite far;1e6 is orthographic-only. Original anchor near remains fixed.1; do not unify these two policies. Public near/far kwargs can implement this without compiling. Store an explicit unbounded_perspective far policy string and create float(inf) only at runtime, not in checkpoint JSON. Native CUDA depth boundaries are inclusive; public CPU reference uses strict boundaries and lacks opacity-aware bbox tightening, so it is not a bitwise visibility oracle.',
             'native_evidence_status':'engine_impl message: actual train host camera-derived frustum and i_jy8 normalized depth clipping; final source artifact/float32 rounding proof supplied independently by native agent.'},
            {'id':'dense_adam_is_not_bug','conclusion':'R8 uses standard fused dense Adam and six full parameter tensors. Zero-gradient rows with prior moments still move. Parallel original host audit confirms actual dense kernels and missing-gradientg0 route; do not replace with visible-only Adam as a fix. Bias correction/epsilon/host LR context still require their own original evidence.'},
            {'id':'selective_adam_not_drop_in','conclusion':'Public SelectiveAdam skips invisible rows and its CUDA kernel omits bias correction; wrapper never increments storedstep. It differs from standard Adam even with an all-visible mask. SparseAdam/packed sparse gradients likewise change update semantics.'}],
        'external_public_3d_mip_reference':{'url':'https://github.com/autonomousvision/mip-splatting/blob/main/scene/gaussian_model.py',
            'purpose':'Distinguish actual3D smoothing from current min_scale regularization. Official public source uses sqrt(s^2+f^2) and opacity determinant ratio; no source copied and no3D coefficient imported into Postshot implementation.',
            'original_equivalence_claim':False},
        'production_mutated':False,'packages_upgraded':False,'compiled':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'installed_source_all_match':all(v['match'] for v in result['public_inventory']['files']),
                      'cuda_initialized':result['cpu_diagnostics']['cuda_initialized']}))


if __name__=='__main__':main()
