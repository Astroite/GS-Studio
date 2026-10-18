"""Explicit model open stays bound to one published Run result and verifies once."""
from types import SimpleNamespace
import pytest
from gsstudio.application import studio,delivery


@pytest.fixture
def model(tmp_path,monkeypatch):
    work=tmp_path/'run';output=work/'experiments/one';output.mkdir(parents=True)
    entry=dict(status='succeeded',backend='gsplat',output=str(output),model_sha256='f'*64)
    run=SimpleNamespace(metrics={'training_experiments':[entry]})
    monkeypatch.setattr(studio,'_run',lambda r:(tmp_path,run,work))
    monkeypatch.setattr(delivery,'verified_models',lambda *a,**kw:pytest.fail('Repeated full model-list verification'))
    return dict(action='model_open',root=str(tmp_path),output=str(output)),run,output,entry


def test_named_model_verified_once_against_ledger_hash(model,monkeypatch):
    request,run,output,entry=model
    calls=[]
    def verify(root,path,expected):
        calls.append((root,path,expected))
        return path,{'model_sha256':expected}
    monkeypatch.setattr(studio,'_verified_output',verify)
    handled,result=studio.execute_additional(request)
    assert handled and result['project_path']==str(output/'project.json')
    assert len(calls)==1 and calls[0][2]==entry['model_sha256']


def test_registered_custom_directory_under_project_root_opens(model,monkeypatch):
    request,run,output,entry=model
    custom=output.parents[2]/'custom-experiments'/'one'
    request['output']=entry['output']=str(custom)
    monkeypatch.setattr(studio,'_verified_output',lambda root,path,digest:(path,{'model_sha256':digest}))
    handled,result=studio.execute_additional(request)
    assert handled and result['path']==str(custom/'model.ply')


@pytest.mark.parametrize('bad',['unregistered','duplicate','failed','retired','other_run','outside_root'])
def test_invalid_registration_never_reaches_expensive_or_gpu_work(model,monkeypatch,bad):
    request,run,output,entry=model
    if bad=='unregistered':run.metrics['training_experiments']=[]
    if bad=='duplicate':run.metrics['training_experiments'].append(dict(entry))
    if bad=='failed':entry['status']='failed'
    if bad=='retired':entry['backend']='old-trainer'
    if bad=='other_run':request['output']=str(output.parents[2]/'other-run/experiment')
    if bad=='outside_root':request['output']=entry['output']=str(output.parents[3]/'outside-project/experiment')
    monkeypatch.setattr(studio,'_verified_output',lambda *a:pytest.fail('Unregistered model verified'))
    with pytest.raises(ValueError):studio.execute_additional(request)


def test_identity_failure_reaches_caller(model,monkeypatch):
    request,run,output,entry=model
    def changed(*a):raise RuntimeError('Checkpoint hash differs from training evidence')
    monkeypatch.setattr(studio,'_verified_output',changed)
    with pytest.raises(RuntimeError,match='Checkpoint hash differs'):
        studio.execute_additional(request)
