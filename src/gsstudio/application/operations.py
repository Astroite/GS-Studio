"""Shared operation surface used by CLI and the independent desktop worker."""
from gsstudio.application.contracts import OperationRequest, OperationResult, OperationEvent, EventSink
from gsstudio.application.projects import create_location, create_scene
from gsstudio.application.captures import (
    create_capture, verify_capture_sources, relink_capture_sources, probe_capture, ingest,
)
from gsstudio.application.runs import (
    create_and_preprocess, resume_preprocess, audit_run, run_stage, cleanup,
)
from gsstudio.application.advance import advance_run
from gsstudio.application.reviews import finalize_masks, review_qa
from gsstudio.application.experiments import segment_choices, train_segment, control_training
from gsstudio.application.delivery import verified_models
from gsstudio.application.external_imports import (
    inspect_external_import, prepare_external_import, train_external_import,
    list_external_imports, verify_external_import, relink_external_import,
)
