import pytest


@pytest.fixture(autouse=True)
def isolated_unit_test_gpu_lock(tmp_path,monkeypatch):
    """Unit tests mock GPU commands; keep their real locks off production jobs."""
    from gsstudio.infrastructure.runtime import gpu_lock
    monkeypatch.setenv('GSSTUDIO_GPU_LOCK_PATH',str(tmp_path/'unit-test-gpu.lock'))
    monkeypatch.setattr(gpu_lock,'GPU_LOCK_PATH',tmp_path/'unit-test-gpu.lock')


@pytest.fixture
def independent_evidence():
    """Durable CPU fixture for the new core contract; never starts a trainer."""
    def write(output, package, *, steps=1, completed_steps=None, status='succeeded',
              validation_image='validation.png', photo_comp=True,
              use_bilateral_grid=True,use_sparse_depth=True):
        from gsstudio.pipeline.training.data import json_write
        from gsstudio.infrastructure.adapters.media import sha256_file
        (output/'checkpoints').mkdir(exist_ok=True)
        checkpoint=output/'checkpoints'/'latest.pt'
        checkpoint.write_bytes(b'CPU contract checkpoint fixture')
        (output/'model.ply').write_bytes(b'ply')
        cfg={'package_sha256':sha256_file(package/'dataset.json'),'steps':steps,
             'photo_comp':photo_comp,'use_bilateral_grid':use_bilateral_grid,
             'use_sparse_depth':use_sparse_depth,'checkpoint':str(checkpoint),
             'sparse_depth_requested':use_sparse_depth,'sparse_depth_available':True,
             'sparse_depth_observations':8,'sparse_depth_status':'enabled' if use_sparse_depth else 'disabled',
             'training_recipe':'quality_v2','engine':{
                 'name':'local-postshot-independent-gsstudio','schema_version':1,
                 'sources':{'training.py':'0'*64},'versions':{'torch':'2.8.0+cu129'}}}
        json_write(output/'config.json',cfg)
        summary={**cfg,'status':status,'completed_steps':steps if completed_steps is None else completed_steps,
                 'evaluation_status':'completed' if status=='succeeded' else 'not_run',
                 'checkpoint_sha256':sha256_file(checkpoint),'model_sha256':sha256_file(output/'model.ply')}
        json_write(output/'training.json',summary)
        (output/'evaluation').mkdir(exist_ok=True)
        json_write(output/'evaluation'/'metrics.json',{'schema_version':1,
                   'images':[{'image':validation_image,'psnr':30.,'ssim':.9}]})
        return cfg,summary
    return write


@pytest.fixture
def mock_dispatch_package(monkeypatch):
    """Only for malformed-package runner unit fixtures, never algorithm evidence."""
    from gsstudio.pipeline.training import runner
    from gsstudio.pipeline.training.presets import resolve_settings, effective_settings
    def install(meta):
        def availability(package):
            return {'available':True,'observations':8,'reason':None,
                    'package_sha256':runner.sha256_file(package/'dataset.json')}
        def prepare(package, **options):
            settings=effective_settings(resolve_settings(meta,**options),availability(package))
            digest=runner.sha256_file(package/'dataset.json')
            return package,meta,dict(settings,source_package_sha256=digest,package_sha256=digest)
        monkeypatch.setattr(runner,'prepare_training',prepare)
        monkeypatch.setattr(runner,'depth_availability',availability)
    return install
