"""The actual independent dataset bridge consumes canonical external packages."""
from pathlib import Path
from gsstudio.application.external_imports import prepare_external_import
from gsstudio.infrastructure.adapters.media import sha256_file
from tests.unit.test_external_imports import _source


def test_external_colmap_and_shared_reimport_preserve_core_provenance(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'engine'))
    from local_postshot.studio_dataset import prepare
    images,model=_source(tmp_path)
    first=prepare_external_import(tmp_path/'catalog','raw',images=images,model=model,independent_images=True)
    package=Path(first['package_path'])
    before={p:sha256_file(p) for p in package.rglob('*') if p.is_file()}
    second=prepare_external_import(tmp_path/'catalog','copy',source=package)
    one=prepare(package,tmp_path/'engine-one')
    two=prepare(Path(second['package_path']),tmp_path/'engine-two')
    assert one['studio_dataset']['package_sha256']==first['package_sha256']
    assert two['studio_dataset']['package_sha256']==first['package_sha256']
    assert one['studio_dataset']['train_indices']==two['studio_dataset']['train_indices']
    assert one['studio_dataset']['test_indices']==two['studio_dataset']['test_indices']
    assert one['studio_dataset']['test_indices']
    assert all(sha256_file(path)==digest for path,digest in before.items())


def test_source_camera_handoff_uses_core_world_and_scene_scale(tmp_path,monkeypatch):
    import numpy as np
    from gsstudio.application.studio import execute_additional
    from gsstudio.pipeline.training.data import json_write
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'engine'))
    from local_postshot.studio_dataset import prepare
    images,model=_source(tmp_path)
    prepared=prepare_external_import(tmp_path/'catalog','raw',images=images,model=model,independent_images=True)
    package=Path(prepared['package_path']);output=tmp_path/'experiment'
    p=prepare(package,output,settings={'scene_normalization':{'scale':2.,'center':[1.,2.,3.]}})
    json_write(output/'dispatch.json',{'backend':'gsplat','package_sha256':prepared['package_sha256'],
                                      'command':['python','--dataset',str(package)]})
    _,result=execute_additional({'root':str(tmp_path),'action':'sources.list','output':str(output),
                                'package_sha256':prepared['package_sha256']})
    assert result['coordinate_system']=='RUB world; RDF camera'
    assert len(result['cameras'])==len(p['reconstruction']['cameras'])
    for source,core in zip(result['cameras'],p['reconstruction']['cameras']):
        np.testing.assert_array_equal(source['w2c'],core['w2c'])
        assert source['K']==core['K']
