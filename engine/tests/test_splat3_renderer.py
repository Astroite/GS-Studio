import os
import pytest
import torch

from local_postshot.splat3_renderer import (
    ALPHA_CAP,T_THRESHOLD,contribution_stats,reference_projected,rasterize_projected,rasterize_parameters,
)


def fixture(width=1,height=1,opacity=(.2,.5,.25),*,dtype=torch.float64,device='cpu',offset=.01):
    n=len(opacity);tile=16
    means=torch.full((1,n,2),.5+offset,dtype=dtype,device=device,requires_grad=True)
    conics=torch.tensor([.2,.01,.3],dtype=dtype,device=device).expand(1,n,3).clone().requires_grad_()
    colors=torch.linspace(.1,.9,max(n*3,1),dtype=dtype,device=device)[:n*3].reshape(1,n,3).requires_grad_()
    opacities=torch.tensor(opacity,dtype=dtype,device=device,requires_grad=True)
    tw,th=(width+15)//16,(height+15)//16
    offsets=torch.arange(tw*th,dtype=torch.int32,device=device).reshape(1,th,tw)*n
    ids=torch.arange(n,dtype=torch.int32,device=device).repeat(tw*th)
    bg=torch.tensor([[.2,.3,.1]],dtype=dtype,device=device,requires_grad=True)
    return [means,conics,colors,opacities,offsets,ids,width,height,tile],bg


def test_cpu_inclusive_terminating_contribution_and_cap():
    args,bg=fixture(opacity=(1.,1.,.3),offset=0)
    result=reference_projected(*args,background=bg)
    weights=result['stats'].sum_weight
    assert weights[0]==ALPHA_CAP
    assert weights[1]==ALPHA_CAP/512
    assert weights[2]==0
    assert result['transmittance'].item()==1/(512**2)
    assert result['transmittance'].item()<T_THRESHOLD


def test_cpu_cap_backward_is_straight_through_not_true_min_derivative():
    args,bg=fixture(opacity=(1.,1.,.3),offset=0)
    result=reference_projected(*args,background=bg)
    gradient,=torch.autograd.grad(result['image'].sum(),args[3])
    colors=args[2][0]
    expected=colors[0].sum()-ALPHA_CAP*colors[1].sum()-(1-ALPHA_CAP)*bg.sum()
    torch.testing.assert_close(gradient[0],expected,rtol=1e-12,atol=1e-12)
    assert abs(float(gradient[0]))>.01


def test_cpu_background_gradient_and_unsaturated_gradcheck():
    args,bg=fixture(opacity=(.2,.4),offset=.1)
    def function(means,conics,colors,opacity,background):
        result=reference_projected(means,conics,colors,opacity,*args[4:],background=background)
        return result['image'],result['alpha']
    assert torch.autograd.gradcheck(function,(*args[:4],bg))
    result=reference_projected(*args,background=bg)
    gradient,=torch.autograd.grad(result['image'].sum(),bg)
    torch.testing.assert_close(gradient,result['transmittance'].sum().expand_as(bg),rtol=0,atol=0)


def test_cpu_empty_scene_background_and_no_rng_consumption():
    args,bg=fixture(width=3,height=2,opacity=())
    rng=torch.random.get_rng_state().clone()
    result=reference_projected(*args,background=bg)
    torch.testing.assert_close(result['image'],bg.view(1,1,1,3).expand(1,2,3,3),rtol=0,atol=0)
    assert result['stats'].sum_weight.numel()==0
    assert torch.equal(rng,torch.random.get_rng_state())
    with pytest.raises(ValueError,match='requires CUDA'):
        rasterize_projected(*args,background=bg)


GPU=pytest.mark.skipif(os.environ.get('LOCAL_POSTSHOT_GPU_TESTS')!='1',reason='Opt-in private compositor CUDA audit')


@GPU
@pytest.mark.parametrize('shape',[(1,1),(19,13)])
@pytest.mark.parametrize('saturated',[False,True])
def test_cuda_projected_forward_and_all_vjps_match_cpu_st_oracle(shape,saturated):
    opacity=(1.,1.,.3) if saturated else (.2,.5,.25)
    cpu,bg=fixture(*shape,opacity=opacity)
    gpu=[v.detach().float().cuda().requires_grad_(v.requires_grad) if isinstance(v,torch.Tensor) and v.is_floating_point()
         else v.cuda() if isinstance(v,torch.Tensor) else v for v in cpu]
    gb=bg.detach().float().cuda().requires_grad_()
    actual=rasterize_projected(*gpu,background=gb)
    expected=reference_projected(*cpu,background=bg)
    torch.testing.assert_close(actual[0].cpu().double(),expected['image'],rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(actual[1].cpu().double(),expected['alpha'],rtol=2e-5,atol=2e-6)
    # Fixed nonuniform upstream gradients test RGB and alpha paths independently.
    vg=torch.linspace(-.4,.6,actual[0].numel(),dtype=torch.float64).reshape(expected['image'].shape)
    va=torch.linspace(.1,.3,actual[1].numel(),dtype=torch.float64).reshape(expected['alpha'].shape)
    ng=torch.autograd.grad((actual[0]*vg.float().cuda()).sum()+(actual[1]*va.float().cuda()).sum(),(*gpu[:4],gb))
    rg=torch.autograd.grad((expected['image']*vg).sum()+(expected['alpha']*va).sum(),(*cpu[:4],bg))
    for a,b in zip(ng,rg):
        torch.testing.assert_close(a.cpu().double(),b,rtol=8e-4,atol=3e-5)
    stats=contribution_stats(gpu[0],gpu[1],gpu[3],gpu[4],gpu[5],torch.ones((shape[1],shape[0]),device='cuda'),*gpu[6:])
    for a,b in zip(stats,expected['stats']):
        torch.testing.assert_close(a.cpu().double(),b,rtol=2e-5,atol=2e-6)


@GPU
def test_cuda_projected_uses_current_stream_and_retains_tiny_t():
    args,bg=fixture(opacity=(1.,1.,.3),dtype=torch.float32,device='cuda',offset=0)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        image,alpha=rasterize_projected(*args,background=bg)
        loss=image.sum()+alpha.sum()*.2
        gradients=torch.autograd.grad(loss,(*args[:4],bg))
    stream.synchronize()
    assert float(alpha)>1-1e-4
    assert all(bool(torch.isfinite(g).all()) for g in gradients)
    torch.testing.assert_close(gradients[-1],torch.full_like(bg,1/(512**2)),rtol=1e-6,atol=1e-9)


@GPU
def test_cuda_empty_background_and_alpha_only_vjp():
    args,bg=fixture(width=3,height=2,opacity=(),dtype=torch.float32,device='cuda')
    image,alpha=rasterize_projected(*args,background=bg)
    gradient,=torch.autograd.grad(image.sum(),bg)
    torch.testing.assert_close(gradient,torch.full_like(bg,6),rtol=0,atol=0)
    assert not bool(alpha.any())
    args,bg=fixture(dtype=torch.float32,device='cuda')
    _,alpha=rasterize_projected(*args,background=bg)
    gradients=torch.autograd.grad(alpha.sum(),(*args[:4],bg))
    assert all(bool(torch.isfinite(v).all()) for v in gradients)
    torch.testing.assert_close(gradients[-1],torch.zeros_like(bg),rtol=0,atol=0)


@GPU
def test_cuda_old_t_equality_includes_current_and_stats_crosses_batch():
    # Find representable opacity inputs whose float32 prefix T is exactly the
    # specified float32 threshold. No relaxed tolerance for this branch gate.
    bits=torch.tensor(.95,dtype=torch.float32).view(torch.int32)
    first=(bits+torch.arange(-8192,8193,dtype=torch.int32)).view(torch.float32)
    t=1-first
    second=(1-T_THRESHOLD/t.double()).float()
    matches=(t*(1-second)==T_THRESHOLD)&(second<=ALPHA_CAP)
    index=int(torch.where(matches)[0][0])
    opacity=(float(first[index]),float(second[index]),.5,.3)
    args,bg=fixture(opacity=opacity,dtype=torch.float32,device='cuda',offset=0)
    image,alpha=rasterize_projected(*args,background=bg)
    stats=contribution_stats(args[0],args[1],args[3],args[4],args[5],torch.ones((1,1),device='cuda'),1,1,16)
    assert float(stats.sum_weight[2])==T_THRESHOLD*.5
    assert float(stats.sum_weight[3])==0
    assert bool(torch.isfinite(image).all())
    # Cross the256-Gaussian shared batch boundary with no early termination.
    args,bg=fixture(width=3,height=2,opacity=(.01,)*300,dtype=torch.float32,device='cuda',offset=0)
    args[1]=torch.zeros_like(args[1],requires_grad=True)
    image,alpha=rasterize_projected(*args,background=bg)
    stats=contribution_stats(args[0],args[1],args[3],args[4],args[5],torch.ones((2,3),device='cuda'),3,2,16)
    torch.testing.assert_close(stats.sum_weight.sum(),alpha.sum(),rtol=2e-5,atol=1e-5)
    image.sum().backward()
    assert bool(torch.isfinite(args[3].grad).all())


@GPU
@pytest.mark.parametrize('mode',['classic','antialiased'])
def test_cuda_public_projection_sh_to_private_rgb_end_to_end(mode):
    from local_postshot.projection_policy import validated_projection
    means=torch.tensor([[-.2,0,2.],[.1,.1,2.5],[.2,-.1,3.]],device='cuda')
    params=torch.nn.ParameterDict({
        'means':torch.nn.Parameter(means),'quats':torch.nn.Parameter(torch.tensor([[1.,0,0,0]]*3,device='cuda')),
        'scales':torch.nn.Parameter(torch.full((3,3),-1.5,device='cuda')),
        'opacities':torch.nn.Parameter(torch.zeros(3,device='cuda')),
        'sh0':torch.nn.Parameter(torch.zeros(3,1,3,device='cuda')),
        'shN':torch.nn.Parameter(torch.zeros(3,15,3,device='cuda'))})
    camera={'K':[[20.,0,9.5],[0,20.,6.5],[0,0,1]],'w2c':torch.eye(4).tolist(),'width':19,'height':13}
    image,alpha,info=rasterize_parameters(params,camera,3,policy=validated_projection({'rasterize_mode':mode}),
                                        background=torch.tensor([[.2,.1,.3]],device='cuda'))
    (image.sum()+alpha.sum()*.1).backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values())
    stats=contribution_stats(info['means2d'],info['conics'],info['opacities'],info['isect_offsets'],info['flatten_ids'],
        torch.ones((13,19),device='cuda'),19,13,info['tile_size'])
    torch.testing.assert_close(stats.sum_weight.sum(),alpha.sum(),rtol=2e-5,atol=1e-5)
