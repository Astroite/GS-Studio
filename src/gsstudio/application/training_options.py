"""CPU package preparation and truthful pretraining controls."""
from pathlib import Path

from gsstudio.pipeline.training.presets import prepare_training


def training_options(root, *, import_id=None, location_id=None, scene_id=None,
                     run_id=None, segment=None, duration_seconds=None, **options):
    root = Path(root).resolve()
    if import_id:
        if any(v is not None for v in (location_id, scene_id, run_id, segment, duration_seconds)):
            raise ValueError('Choose an external import or a Run segment')
        from gsstudio.application.external_imports import verify_external_import
        record, project = verify_external_import(root, import_id)
        from gsstudio.infrastructure.runtime.run_lock import run_session
        with run_session(project):
            record, project = verify_external_import(root, import_id)
            package, _, settings = prepare_training(project/'package', source_kind=record['source_kind'], **options)
    else:
        if not all((location_id, scene_id, run_id, segment)):
            raise ValueError('Provide location_id, scene_id, run_id and segment')
        from gsstudio.application._shared import _scene
        from gsstudio.infrastructure.persistence.run_repository import load_run
        from gsstudio.infrastructure.runtime.run_lock import run_session
        from gsstudio.application.experiments import segment_package
        scene = _scene(root, location_id, scene_id)
        with run_session(scene/run_id):
            run = load_run(scene, run_id)
            package, source_kind = segment_package(root, location_id, scene_id, run, segment, duration_seconds)
            package, _, settings = prepare_training(package, source_kind=source_kind, **options)
    return dict(settings, package=str(package))
