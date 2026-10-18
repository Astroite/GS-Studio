import math
import os
import random
import pytest
import torch
from local_postshot.splat3_tiles import (
    ALPHA_CUTOFF, native_circle_radius, native_edge_intersection,
    minimum_quadratic_on_rectangle, reference_tiles, intersect_tiles,
)


def test_native_two_near_edges_equal_convex_four_edge_oracle():
    rng=random.Random(842)
    for _ in range(10000):
        theta=rng.uniform(-math.pi,math.pi)
        s,t=10**rng.uniform(-3,2),10**rng.uniform(-3,2)
        co,si=math.cos(theta),math.sin(theta)
        q=(s*co*co+t*si*si,(s-t)*co*si,s*si*si+t*co*co)
        mean=(rng.uniform(-100,100),rng.uniform(-100,100))
        x,y=rng.randrange(-3,3)*16,rng.randrange(-3,3)*16
        rect=(x,y,x+16,y+16)
        threshold=rng.uniform(.001,12)
        expected=minimum_quadratic_on_rectangle(mean,q,rect)<=threshold
        assert native_edge_intersection(mean,q,rect,threshold)==expected


@pytest.mark.parametrize('mean,threshold,expected', [((-1.,8.),1.,True),((-1.,8.),.999,False),((8.,8.),0.,True),((17.,17.),2.,True),((17.,17.),1.99,False)])
def test_native_closed_edges_tangency(mean,threshold,expected):
    assert native_edge_intersection(mean,(1.,0.,1.),(0.,0.,16.,16.),threshold)==expected


def test_native_circle_includes_discriminant_floor_and_opacity_support():
    radius,tau=native_circle_radius((1.,0.,1.),.5)
    assert radius==pytest.approx(math.sqrt(1.1*tau))
    assert native_circle_radius((1.,0.,1.),ALPHA_CUTOFF/2)[0]==0


def test_cpu_reference_preserves_equal_depth_order_partial_tiles_and_empty_offsets():
    counts,keys,ids,offsets=reference_tiles([(15.9,16.1)]*3,[(10,10)]*3,[2.,2.,1.],[(1.,0.,1.)]*3,[.7]*3,19,33)
    assert counts==[4,4,4]
    assert ids[:3]==[2,0,1]
    assert offsets==[0,3,6,9,12,12]
    assert keys==sorted(keys)
    assert reference_tiles([],[],[],[],[],19,33)==([] ,[],[],[0]*6)


def test_cpu_invalid_conic_keeps_public_rectangle():
    result=reference_tiles([(16.,16.)],[(15,15)],[2.],[(float('nan'),0.,1.)],[.2],32,32)
    assert result[0]==[4]


def test_cpu_conservative_removal_never_loses_any_contributing_pixel():
    rng=random.Random(213)
    width,height=51,39
    for _ in range(100):
        mean=(rng.uniform(-15,65),rng.uniform(-15,50))
        q=(rng.uniform(.02,.6),rng.uniform(-.015,.015),rng.uniform(.02,.6))
        op=10**rng.uniform(-3,0)
        _,keys,_,_=reference_tiles([mean],[(80,80)],[1.],[q],[op],width,height)
        tiles={key>>32 for key in keys};tw=(width+15)//16
        for y in range(height):
            for x in range(width):
                dx,dy=mean[0]-x-.5,mean[1]-y-.5
                power=-.5*(q[0]*dx*dx+2*q[1]*dx*dy+q[2]*dy*dy)
                if op*math.exp(power)>=ALPHA_CUTOFF:
                    assert (y//16)*tw+x//16 in tiles


GPU=pytest.mark.skipif(os.environ.get('LOCAL_POSTSHOT_GPU_TESTS')!='1',reason='Opt-in private tile CUDA audit')


def cuda_scene(n=120,width=67,height=49):
    generator=torch.Generator().manual_seed(523)
    means=torch.rand((1,n,2),generator=generator)*torch.tensor([width+20,height+20])-10
    radii=torch.full((1,n,2),70,dtype=torch.int32)
    depths=(torch.arange(n)%7+1).float()[None]
    conics=torch.zeros((1,n,3));conics[...,0]=torch.rand((1,n),generator=generator)*.4+.01
    conics[...,2]=torch.rand((1,n),generator=generator)*.4+.01
    conics[...,1]=(torch.rand((1,n),generator=generator)-.5)*.005
    opacity=torch.rand((1,n),generator=generator)*.998+.002
    return [v.cuda() for v in (means,radii,depths,conics,opacity)]


@GPU
@pytest.mark.parametrize('mode',['conservative_v1','native_exact_v1'])
def test_cuda_lists_match_independent_cpu_oracle_and_stable_ties(mode):
    values=cuda_scene()
    actual=intersect_tiles(*values,67,49,mode=mode)
    expected=reference_tiles(*(v[0].cpu().tolist() for v in values),67,49,mode=mode)
    for a,e in zip(actual,expected):
        assert a.cpu().reshape(-1).tolist()==e


@GPU
def test_cuda_conservative_forward_gradients_and_stats_match_public_bbox():
    from gsplat.cuda._wrapper import isect_tiles,isect_offset_encode
    from local_postshot.splat3_renderer import rasterize_projected,contribution_stats
    values=cuda_scene()
    means,radii,depth,conic,opacity=values
    public=isect_tiles(means,radii,depth,16,5,4,packed=False)
    old_offsets=isect_offset_encode(public[1],1,5,4)
    counts,keys,ids,offsets=intersect_tiles(*values,67,49)
    assert len(ids)<len(public[2])*.4
    generator=torch.Generator(device='cuda').manual_seed(123)
    colors=torch.rand((1,120,3),device='cuda',generator=generator)
    background=torch.tensor([[.1,.4,.9]],device='cuda')
    error=torch.rand((49,67),device='cuda',generator=generator)
    outputs=[]
    for off,flat in ((old_offsets,public[2]),(offsets,ids)):
        args=[v.detach().clone().requires_grad_() for v in (means,conic,colors,opacity,background)]
        image,alpha=rasterize_projected(*args[:4],off,flat,67,49,background=args[4])
        ((image*error[None,...,None]).sum()+alpha.sum()*.03).backward()
        stats=contribution_stats(means,conic,opacity,off,flat,error,67,49,16)
        outputs.append((image,alpha,[v.grad for v in args],stats))
    torch.testing.assert_close(outputs[0][0],outputs[1][0],rtol=0,atol=0)
    torch.testing.assert_close(outputs[0][1],outputs[1][1],rtol=0,atol=0)
    for a,b in zip(outputs[0][2],outputs[1][2]):
        torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
    for a,b in zip(outputs[0][3],outputs[1][3]):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5)


@GPU
def test_cuda_invalid_conics_and_opacity_keep_public_bbox():
    from gsplat.cuda._wrapper import isect_tiles,isect_offset_encode
    values=cuda_scene(n=7)
    values[3][0]=torch.tensor([[float('nan'),0,1],[-1,0,1],[1,2,1],[1,1,1],[1,0,float('inf')],[1,0,1],[1,0,1]],device='cuda')
    values[4][0,-2:]=torch.tensor([float('nan'),-1],device='cuda')
    public=isect_tiles(values[0],values[1],values[2],16,5,4,packed=False)
    actual=intersect_tiles(*values,67,49)
    for a,b in zip(actual[:3],public):torch.testing.assert_close(a,b,rtol=0,atol=0)


@GPU
def test_cuda_empty_shapes_validation_and_rng_unchanged():
    values=cuda_scene(0)
    rng=torch.cuda.get_rng_state().clone()
    actual=intersect_tiles(*values,19,33)
    assert actual[0].shape==(1,0) and actual[3].shape==(1,3,2)
    assert not actual[3].count_nonzero().item()
    assert torch.equal(rng,torch.cuda.get_rng_state())
    with pytest.raises(ValueError):intersect_tiles(*values,19,33,mode='unknown')
    bad=cuda_scene(1);bad[1]=bad[1].float()
    with pytest.raises(RuntimeError,match='radii'):intersect_tiles(*bad,19,33)


@GPU
def test_cuda_exhaustive_contributors_anisotropy_threshold_boundaries_and_large_coords():
    from local_postshot.splat3_tiles import _extension
    # 1,024 splats x 12,513 pixels, evaluated with the actual float32 compositor
    # expression/__expf. Seeds are fixed; no probabilistic acceptance criterion.
    n=1024;w,h=129,97;tw,th=(w+15)//16,(h+15)//16
    gen=torch.Generator().manual_seed(11965)
    theta=torch.rand(n,generator=gen)*math.pi*2
    eigen1=10**(torch.rand(n,generator=gen)*8-5)
    eigen2=10**(torch.rand(n,generator=gen)*8-5)
    co,si=theta.cos(),theta.sin()
    conic=torch.stack((eigen1*co.square()+eigen2*si.square(),(eigen1-eigen2)*co*si,eigen1*si.square()+eigen2*co.square()),-1)[None]
    means=(torch.rand((1,n,2),generator=gen)-.15)*torch.tensor([w,h])*1.3
    # Center on pixel centers, tile edges, and near the maximum float32 exact
    # integer coordinate range with tiny positive conics.
    means[0,:64]=(means[0,:64]/16).round()*16+.5
    means[0,64:128]=(means[0,64:128]/16).round()*16
    means[0,128:160]=torch.tensor([2**20+.5,2**20+.5])
    conic[0,128:160]=torch.tensor([1e-12,0,1e-12])
    opacity=torch.rand((1,n),generator=gen)*.999
    opacity[0,:128]=torch.tensor(ALPHA_CUTOFF)*(1+(torch.arange(128).float()-64)*2e-7)
    radii=torch.full((1,n,2),2**21,dtype=torch.int32)
    depths=torch.ones((1,n))
    values=[v.cuda() for v in (means,radii,depths,conic,opacity)]
    counts,keys,ids,offsets=intersect_tiles(*values,w,h)
    present=torch.zeros((n,tw*th),device='cuda',dtype=torch.bool)
    present[ids.long(),keys>>32]=True
    missing=_extension().audit_contributors(values[0],values[3],values[4],present,w,h,16)
    assert missing.sum().item()==0, {'missing':missing.sum().item(),'gaussians':missing.nonzero().cpu().tolist()}


@GPU
def test_cuda_native_large_finite_radius_and_float32_underflow_fallback():
    # These radii exceed int32 tile-coordinate conversion range. Clip floats
    # before integer conversion, preserving whole-image support or emptiness.
    for value,mean,expected in [(1e-22,8.,4),(1e-22,1e10,4),(1e-22,-1e10,4),
                                (1e-22,1e20,0),(1e-22,-1e20,0)]:
        means=torch.tensor([[[mean,mean]]],device='cuda')
        radii=torch.tensor([[[2147483647,2147483647]]],device='cuda',dtype=torch.int32)
        depth=torch.ones((1,1),device='cuda')
        conic=torch.tensor([[[value,0.,value]]],device='cuda')
        opacity=torch.ones((1,1),device='cuda')
        actual=intersect_tiles(means,radii,depth,conic,opacity,32,32,mode='native_exact_v1')
        assert actual[0].item()==expected
    # At1e-23 the determinant underflows tozero in float32. This intentionally
    # uses the documented public-bbox invalid-conic fallback, not float64 native
    # geometry extrapolated through an inverse of an unrepresentable determinant.
    conic.fill_(0);conic[...,0]=conic[...,2]=1e-23
    means.fill_(1e10)
    actual=intersect_tiles(means,radii,depth,conic,opacity,32,32,mode='native_exact_v1')
    assert actual[0].item()==0


@GPU
def test_cuda_nondefault_stream_and_nonempty_rng_unchanged():
    values=cuda_scene(47)
    baseline=intersect_tiles(*values,67,49)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    rng=torch.cuda.get_rng_state().clone()
    with torch.cuda.stream(stream):
        copied=[v.clone() for v in values]
        actual=intersect_tiles(*copied,67,49)
    torch.cuda.current_stream().wait_stream(stream)
    for a,b in zip(actual,baseline):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(rng,torch.cuda.get_rng_state())
