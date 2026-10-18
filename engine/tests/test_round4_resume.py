"""Actual opt-in AA/tile checkpoints and fail-closed binary drift on CUDA."""
import copy
import hashlib
import os
from pathlib import Path
import threading
import pytest
from local_postshot import project
from local_postshot.training import train,load_checkpoint
from test_gpu_resume import fixture_project,assert_exact_tree

pytestmark=pytest.mark.skipif(os.environ.get('LOCAL_POSTSHOT_GPU_TESTS')!='1',reason='Opt-in actual CUDA')


@pytest.mark.parametrize('tile',['conservative_v1','native_exact_v1'])
def test_native_aa_tile_pause_resume_and_binary_drift(tmp_path,monkeypatch,tile):
    import local_postshot.projection_policy as policy_module
    from local_postshot.splat3_strategy import ObservedStrategy
    root=fixture_project(tmp_path/tile,'observed_v6')
    cfg=project.load(root)['settings']['training_recipe_overrides']
    cfg['projection']={'compositor':'native_v1','train_rasterize_mode':'antialiased',
        'eval_rasterize_mode':'antialiased','tile_culling':tile,'aa_compensation':'native_floor_v1',
        'train_background':'random_rgb_v1','eval_near_plane':.1,'far_mode':'unbounded_perspective'}
    project.update(root,{'training_recipe_overrides':cfg})
    stop,pause=threading.Event(),threading.Event()
    def callback(event,data):
        if event=='metric' and data['step']==1:pause.set();stop.set()
    paused=train(root,callback,stop,pause,steps=12,max_splats=100)
    assert paused['status']=='paused'
    file=Path(root)/'checkpoints/latest.pt'
    before=hashlib.sha256(file.read_bytes()).hexdigest()
    saved=load_checkpoint(file)
    assert set(saved['training_recipe']['projection_artifacts'])=={'tiles','aa_projection'}
    original=policy_module.projection_artifact_identity
    with monkeypatch.context() as patch:
        def wrong_identity(policy):
            result=copy.deepcopy(original(policy));result['tiles']['binary_sha256']='0'*64
            return result
        patch.setattr(policy_module,'projection_artifact_identity',wrong_identity)
        with pytest.raises(ValueError,match='tile/AA projection binary differs'):
            train(root,lambda *_:None,threading.Event(),threading.Event(),steps=2,max_splats=100,resume=True)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==before
    checks=[]
    actual=ObservedStrategy.prepare_step
    def inspect(self,params,optimizers,state,step,*args,**kwargs):
        if not checks:
            assert step==1
            assert_exact_tree(dict(params.items()),saved['parameters'],'AA restored model')
            assert_exact_tree(state,saved['strategy_state'],'AA restored strategy')
            for key,optimizer in optimizers.items():
                assert_exact_tree(optimizer.state_dict(),saved['optimizers'][key],'AA restored Adam '+key)
            checks.append(True)
        return actual(self,params,optimizers,state,step,*args,**kwargs)
    monkeypatch.setattr(ObservedStrategy,'prepare_step',inspect)
    resumed=train(root,lambda *_:None,threading.Event(),threading.Event(),steps=2,max_splats=100,resume=True)
    assert checks and resumed['training']['step']==2
    assert resumed['training']['recipe']==saved['training_recipe']
