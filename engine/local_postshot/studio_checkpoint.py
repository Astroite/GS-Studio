"""Exact checkpoint-boundary audit, independent of subsequent CUDA variance."""


def assert_restored(saved,current,path='checkpoint'):
    import numpy as np
    import torch
    if isinstance(saved,torch.Tensor):
        assert isinstance(current,torch.Tensor),path
        assert saved.dtype==current.dtype and saved.shape==current.shape,path
        assert torch.equal(saved.cpu(),current.cpu()),path
    elif isinstance(saved,np.ndarray):
        assert isinstance(current,np.ndarray) and saved.dtype==current.dtype,path
        assert np.array_equal(saved,current),path
    elif isinstance(saved,dict):
        assert isinstance(current,dict) and saved.keys()==current.keys(),path
        for key in saved:
            # Wall clock counters are performance evidence, not numerical state.
            if key in ('anchor_wall_seconds_total','wall_seconds'):
                continue
            assert_restored(saved[key],current[key],path+'.'+str(key))
    elif isinstance(saved,(tuple,list)):
        assert type(saved)==type(current) and len(saved)==len(current),path
        for i,(a,b) in enumerate(zip(saved,current)):
            assert_restored(a,b,f'{path}.{i}')
    else:
        assert saved==current,path
