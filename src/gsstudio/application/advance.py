"""Advance a Run through verified stages until the next human decision."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from gsstudio.application._shared import _scene
from gsstudio.application.captures import verify_capture_sources
from gsstudio.application.contracts import EventSink, _emit
from gsstudio.application.runs import audit_run, resume_preprocess, run_stage
from gsstudio.infrastructure.persistence.manifests import load_capture_manifest
from gsstudio.infrastructure.persistence.run_repository import load_run
from gsstudio.pipeline.masks.finalize import (
    MaskFinalizationMissingError, validate_mask_finalization,
)


def advance_run(
    root: Path, location_id: str, scene_id: str, run_id: str, *,
    stop_requested: Callable[[], bool] | None = None,
    sink: EventSink | None = None,
) -> dict[str, str]:
    scene = _scene(root, location_id, scene_id)
    run = load_run(scene, run_id)
    capture = load_capture_manifest(scene / "captures" / f"{run.config.capture_id}.yaml")
    verify_capture_sources(capture)
    audit_run(root, location_id, scene_id, run_id)
    if run.status.value == "rejected":
        _emit(sink, "waiting", "qa_rejected", "QA 已拒绝；请创建新 Run 处理素材或配置", run_id)
        return {"run_id": run_id, "next": "qa_rejected"}
    for stage in ("preprocess", "mask", "reconstruct", "qa"):
        run = load_run(scene, run_id)
        if stop_requested and stop_requested():
            _emit(sink, "stopped", stage, "已在阶段边界停止", run_id)
            return {"run_id": run_id, "next": "resume"}
        if run.stages[stage].status.value != "succeeded":
            if stage == "preprocess":
                resume_preprocess(root, location_id, scene_id, run_id, sink)
            else:
                try:
                    run_stage(root, location_id, scene_id, run_id, stage,
                              resume=True, sink=sink)
                except MaskFinalizationMissingError:
                    _emit(sink, "waiting", "mask_review", "请完成当前 attempt 的遮罩审核", run_id)
                    return {"run_id": run_id, "next": "mask_review"}
        run = load_run(scene, run_id)
        if stage == "mask" and run.config.masking.mask_review_required:
            dataset = scene / run_id / "reconstruction-primary"
            try:
                validate_mask_finalization(dataset)
            except MaskFinalizationMissingError:
                _emit(sink, "waiting", "mask_review", "请审核并固定 primary 遮罩", run_id)
                return {"run_id": run_id, "next": "mask_review"}
    run = load_run(scene, run_id)
    next_action = "segment_selection" if run.status.value == "accepted" else "qa_review"
    _emit(sink, "waiting", next_action,
          "请选择合格分段" if next_action == "segment_selection" else "请检查 QA 报告并记录结论",
          run_id)
    return {"run_id": run_id, "next": next_action}
