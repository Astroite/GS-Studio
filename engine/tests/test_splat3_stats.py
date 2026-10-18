import importlib
import os

import pytest
import torch

from local_postshot.splat3_stats import contribution_stats, reference_statistics


def fixture(width=1,height=1,tile_size=1,opacity=(.2,.5,.25),device="cpu"):
    n=len(opacity)
    means=torch.full((1,n,2),.5,dtype=torch.float32,device=device)
    conics=torch.zeros((1,n,3),dtype=torch.float32,device=device)
    opa=torch.tensor(opacity,dtype=torch.float32,device=device)
    tw=(width+tile_size-1)//tile_size;th=(height+tile_size-1)//tile_size
    offsets=torch.arange(tw*th,dtype=torch.int32,device=device).reshape(1,th,tw)*n
    ids=torch.arange(n,dtype=torch.int32,device=device).repeat(tw*th)
    error=torch.full((height,width),2.,dtype=torch.float32,device=device)
    return [means,conics,opa,offsets,ids,error,width,height,tile_size]


def native_available():
    try:
        return importlib.import_module("local_splat3_stats")
    except (ImportError,OSError):
        pytest.skip("Private local_splat3_stats extension not installed")


def assert_reference(args,backend):
    expected=reference_statistics(*(a.detach().cpu().contiguous() if isinstance(a,torch.Tensor) else a for a in args))
    actual=contribution_stats(*args,backend=backend)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a.cpu().double(),b,rtol=1e-5,atol=1e-6)
    return actual


def test_cpu_double_reference_matches_hand_computed_occlusion():
    result=reference_statistics(*fixture())
    torch.testing.assert_close(result.sum_weight,torch.tensor([.2,.4,.1],dtype=torch.float64),rtol=1e-7,atol=1e-9)
    torch.testing.assert_close(result.max_weight,result.sum_weight,rtol=0,atol=0)
    torch.testing.assert_close(result.weighted_error_sum,2*result.sum_weight,rtol=0,atol=0)


def test_cpu_double_reference_excludes_terminating_contribution():
    result=reference_statistics(*fixture(opacity=(1.,1.,.5)))
    torch.testing.assert_close(result.sum_weight,torch.tensor([.999,0.,0.],dtype=torch.float64),rtol=0,atol=0)


@pytest.mark.parametrize("backend",["reference","native_cpu_v1"])
def test_threshold_negative_sigma_and_zero_error(backend):
    if backend!="reference": native_available()
    args=fixture(opacity=(.003,.1,.5))
    args[0][0,1]=torch.tensor([1.5,.5]);args[1][0,1,0]=-1.
    args[5].zero_()
    result=contribution_stats(*args,backend=backend)
    torch.testing.assert_close(result.sum_weight,torch.tensor([0.,0.,.5],dtype=result.sum_weight.dtype),rtol=0,atol=0)
    assert torch.count_nonzero(result.weighted_error_sum)==0
    assert result.max_weight[2]==.5


@pytest.mark.parametrize("shape",[(1,1,1),(3,2,2),(19,13,16)])
def test_native_cpu_matches_independent_double_reference_and_partial_tiles(shape):
    native_available()
    args=fixture(*shape)
    args[1][0,:,0]=torch.tensor([.1,.2,.3]);args[1][0,:,2]=.15
    args[5]=torch.linspace(0,1,shape[0]*shape[1]).reshape(shape[1],shape[0])
    assert_reference(args,"native_cpu_v1")


@pytest.mark.parametrize("empty_points",[False,True])
def test_empty_lists_and_zero_points(empty_points):
    native_available()
    args=fixture(opacity=() if empty_points else (.4,.7))
    args[4]=torch.empty(0,dtype=torch.int32)
    result=contribution_stats(*args,backend="native_cpu_v1")
    assert all(v.numel()==(0 if empty_points else 2) for v in result)
    assert all(torch.count_nonzero(v)==0 for v in result)


def test_shapes_dtypes_contiguity_backend_and_missing_native_fail_clearly(monkeypatch):
    args=fixture()
    with pytest.raises(ValueError,match="requires CUDA"):
        contribution_stats(*args,backend="native_cuda_v1")
    bad=args.copy();bad[0]=bad[0].double()
    with pytest.raises(ValueError,match="float32"):
        contribution_stats(*bad,backend="reference")
    bad=args.copy();bad[3]=bad[3].long()
    with pytest.raises(ValueError,match="int32"):
        contribution_stats(*bad,backend="reference")
    bad=fixture(width=2,height=3);bad[5]=torch.zeros((2,3)).T
    with pytest.raises(ValueError,match="contiguous"):
        contribution_stats(*bad,backend="reference")
    import local_postshot.splat3_stats as wrapper
    def missing(*a): raise ImportError("injected missing native module")
    monkeypatch.setattr(wrapper.importlib,"import_module",missing)
    with pytest.raises(RuntimeError,match="No CPU fallback"):
        contribution_stats(*args,backend="native_cpu_v1")


def test_native_returns_detached_forward_statistics():
    native_available()
    args=fixture();args[0].requires_grad_();args[2].requires_grad_()
    output=contribution_stats(*args,backend="native_cpu_v1")
    assert not any(t.requires_grad for t in output)


GPU=pytest.mark.skipif(os.environ.get("LOCAL_POSTSHOT_GPU_TESTS")!="1",reason="Opt-in private CUDA extension audit")


@GPU
@pytest.mark.parametrize("shape",[(1,1,1),(3,2,2),(19,13,16)])
def test_cuda_matches_independent_cpu_double_oracle(shape):
    native_available()
    args=fixture(*shape,device="cuda")
    args[1][0,:,0]=torch.tensor([.1,.2,.3],device="cuda");args[1][0,:,2]=.15
    assert_reference(args,"native_cuda_v1")


@GPU
def test_cuda_exclusive_termination_and_empty_inputs():
    native_available()
    assert_reference(fixture(opacity=(1.,1.,.5),device="cuda"),"native_cuda_v1")
    assert_reference(fixture(width=19,height=13,tile_size=16,opacity=(1.,1.,.5),device="cuda"),"native_cuda_v1")
    for opacity in [(),(.3,.7)]:
        args=fixture(opacity=opacity,device="cuda");args[4]=torch.empty(0,dtype=torch.int32,device="cuda")
        assert_reference(args,"native_cuda_v1")


@GPU
def test_cuda_uses_current_nondefault_stream():
    native_available()
    args=fixture(device="cuda")
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(5_000_000)
        args[5].fill_(7.)
        result=contribution_stats(*args,backend="native_cuda_v1")
    stream.synchronize()
    torch.testing.assert_close(result.weighted_error_sum.cpu(),torch.tensor([1.4,2.8,.7]),rtol=1e-6,atol=1e-6)


@GPU
def test_cuda_tiled_skip_threshold_and_negative_sigma():
    native_available()
    args=fixture(width=19,height=13,tile_size=16,opacity=(.003,.1,.5),device="cuda")
    args[0][0,1]=torch.tensor([50.,50.],device="cuda")
    args[1][0,1,0]=-1.
    assert_reference(args,"native_cuda_v1")


@GPU
def test_cuda_tiled_list_crosses_shared_batch_boundary():
    native_available()
    args=fixture(width=19,height=13,tile_size=16,opacity=(.01,)*300,device="cuda")
    assert_reference(args,"native_cuda_v1")


@GPU
def test_real_public_gsplat_projection_metadata_and_alpha_mass():
    native_available()
    from gsplat import rasterization
    means=torch.tensor([[-.2,0,2.],[.1,.1,2.5],[.2,-.1,3.]],device="cuda")
    quats=torch.tensor([[1.,0,0,0]]*3,device="cuda")
    scales=torch.full((3,3),.2,device="cuda")
    opa=torch.tensor([.7,.5,.3],device="cuda")
    K=torch.tensor([[[20.,0,9.5],[0,20.,6.5],[0,0,1]]],device="cuda")
    image,alpha,info=rasterization(means=means,quats=quats,scales=scales,opacities=opa,
        colors=torch.ones(3,3,device="cuda"),viewmats=torch.eye(4,device="cuda")[None],
        Ks=K,width=19,height=13,packed=False,tile_size=16)
    args=[*(info[k].detach().contiguous() for k in ["means2d","conics","opacities","isect_offsets","flatten_ids"]),
          torch.ones((13,19),device="cuda"),19,13,16]
    actual=assert_reference(args,"native_cuda_v1")
    torch.testing.assert_close(actual.sum_weight.sum(),alpha.sum(),rtol=2e-5,atol=1e-5)
    # Separate atomic accumulators can visit pixels in different orders.
    torch.testing.assert_close(actual.weighted_error_sum,actual.sum_weight,rtol=1e-6,atol=1e-5)
