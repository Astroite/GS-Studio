"""Proof-oriented runner tests. Every trainer process is replaced at its boundary."""
import json
from pathlib import Path
import pytest
from gsstudio.pipeline.training import runner
from gsstudio.pipeline.training.data import json_write
from gsstudio.application import operations as ops
from gsstudio.application.studio import execute_additional


@pytest.fixture
def dispatch_case(tmp_path,monkeypatch,mock_dispatch_package):
    package=tmp_path/'package';package.mkdir()
    (package/'dataset.json').write_text('{}',encoding='utf-8')
    meta={'images':[{'image':'train.png','split':'train'},
                    {'image':'validation.png','split':'validation'}]}
    mock_dispatch_package(meta)
    python=tmp_path/'cu129-python.exe';python.touch()
    (tmp_path/'engine').mkdir(); (tmp_path/'src').mkdir()
    runtime=tmp_path/'runtime.json'
    runtime.write_text(json.dumps({'python':str(python),'engine_root':'engine',
                                  'pipeline_root':'src','data_root':str(tmp_path)}),encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_RUNTIME',str(runtime))
    monkeypatch.delenv('GSSTUDIO_GSPLAT_PYTHON',raising=False)
    return package,tmp_path/'experiment',meta


def test_command_uses_independent_module_and_exact_requested_flags(dispatch_case,monkeypatch):
    package,output,_=dispatch_case
    result=runner.train_package(package,output,steps=9,dry_run=True,photo_comp=False,
                                use_bilateral_grid=False,use_sparse_depth=False,
                                preview_every=500,training_recipe='observed_v5')
    command=result['command']
    assert command[1:5]==['-I','-B','-u','-c']
    assert command[7]=='local_postshot.studio_training'
    assert '--no-photo-comp' in command and '--no-use-bilateral-grid' in command
    assert '--no-use-sparse-depth' in command and command[command.index('--recipe')+1]=='observed_v5'
    assert result['engine']==runner.ENGINE_ID and result['checkpoint_format']=='checkpoints/latest.pt'
    assert result['status']=='prepared' and not (output/'model.ply').exists()


@pytest.mark.parametrize('corrupt',['none','engine','package','checkpoint','model','metrics','budget'])
def test_success_requires_all_durable_identity_proofs(dispatch_case,monkeypatch,independent_evidence,corrupt):
    package,output,_=dispatch_case
    def fake(command,log,**kwargs):
        assert kwargs['env']['PYTHONNOUSERSITE']=='1'
        assert 'site-packages' not in kwargs['env']['PYTHONPATH']
        cfg,summary=independent_evidence(output,package,steps=10)
        if corrupt=='engine': cfg['engine']={'name':'retired'};json_write(output/'config.json',cfg)
        if corrupt=='package': summary['package_sha256']='0'*64
        if corrupt=='checkpoint': (output/'checkpoints'/'latest.pt').write_bytes(b'changed')
        if corrupt=='model': (output/'model.ply').write_bytes(b'changed')
        if corrupt=='metrics': json_write(output/'evaluation'/'metrics.json',{'schema_version':1,'images':[]})
        if corrupt=='budget': summary['completed_steps']=2
        json_write(output/'training.json',summary)
        return {'elapsed_seconds':.1}
    monkeypatch.setattr(runner,'run_logged',fake)
    if corrupt=='none':
        result=runner.train_package(package,output,steps=10)
        assert result['status']=='succeeded' and result['completed_steps']==10
    else:
        with pytest.raises(RuntimeError): runner.train_package(package,output,steps=10)
        assert json.loads((output/'dispatch.json').read_text())['status']=='failed'


def test_pause_is_durable_and_never_success(dispatch_case,monkeypatch,independent_evidence):
    package,output,_=dispatch_case
    def fake(*args,**kwargs):
        independent_evidence(output,package,steps=10,completed_steps=3,status='paused')
        return {}
    monkeypatch.setattr(runner,'run_logged',fake)
    result=runner.train_package(package,output,steps=10)
    assert result['status']=='paused' and result['completed_steps']==3
    assert result['checkpoint'].endswith('latest.pt')


def test_retired_backend_and_checkpoint_fail_actionably(dispatch_case):
    package,output,_=dispatch_case
    with pytest.raises(ValueError,match='retired.*--backend gsplat'):
        runner.train_package(package,output,backend='postshot')
    output.mkdir();json_write(output/'dispatch.json',{'backend':'gsplat','status':'failed'})
    (output/'checkpoint.pt').write_bytes(b'legacy')
    with pytest.raises(RuntimeError,match='Retired experiment/checkpoint'):
        runner.train_package(package,output,resume=True)


def test_training_controls_are_bound_to_package_and_generation(dispatch_case):
    package,output,_=dispatch_case
    output.mkdir()
    digest=runner.sha256_file(package/'dataset.json')
    json_write(output/'dispatch.json',{'backend':'gsplat','status':'prepared','package_sha256':digest})
    json_write(output/'control-state.json',{'schema_version':1,'package_sha256':digest,
               'generation_id':'current','status':'running','request_id':None})
    with pytest.raises(ValueError,match='unpause'):
        ops.control_training(output.parent,output,'resume')
    with pytest.raises(RuntimeError,match='identity changed'):
        ops.control_training(output.parent,output,'stop',generation_id='stale',package_sha256=digest)
    assert not (output/'control-request.json').exists()
    request=ops.control_training(output.parent,output,'suspend',generation_id='current',package_sha256=digest)
    wire=json.loads((output/'control-request.json').read_text())
    assert request['generation_id']=='current' and wire['action']=='suspend'
    assert wire['package_sha256']==digest


def test_preview_rejects_stale_request_and_changed_render(dispatch_case):
    package,output,_=dispatch_case
    output.mkdir();digest=runner.sha256_file(package/'dataset.json')
    json_write(output/'dispatch.json',{'backend':'gsplat','status':'prepared','package_sha256':digest})
    (output/'source.png').write_bytes(b'source');(output/'render.png').write_bytes(b'render')
    json_write(output/'preview.json',{'schema_version':1,'request_id':'a'*32,'image':'train.png',
               'package_sha256':digest,'source_file':'source.png','render_file':'render.png',
               'source_sha256':runner.sha256_file(output/'source.png'),
               'render_sha256':runner.sha256_file(output/'render.png')})
    params={'root':str(output.parent),'output':str(output),'action':'preview.read','request_id':'b'*32}
    assert execute_additional(params)==(True,{'preview':None})
    params['request_id']='a'*32
    assert execute_additional(params)[1]['preview']['source_path']==str(output/'source.png')
    (output/'render.png').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Stale identity'): execute_additional(params)
