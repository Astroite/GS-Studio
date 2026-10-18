"""Shared write operations for the CLI and the GS-Studio workbench.

The methods here return domain objects.  They never infer success from an
artifact's existence, and callers must not turn an exception into success.
"""
from __future__ import annotations

import json
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from gsstudio.infrastructure.persistence.manifests import load_capture_manifest, load_model, save_yaml
from gsstudio.pipeline.masks.finalize import MaskFinalizationMissingError, finalize_mask_dataset, validate_mask_finalization
from gsstudio.pipeline.masks.masking import validate_mask_filter
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.domain.models import (
    IMAGE_SUFFIXES, Camera, CaptureManifest, CaptureSource, LocationManifest,
    NormalizationSettings, PreparedInputConfig, Region, Rights, RunConfig,
    RunStatus, SceneManifest, SourceFile, SourceProbe, TimeSelection, utc_now,
)
from gsstudio.infrastructure.paths import host_path, location_dir, scene_dir
from gsstudio.infrastructure.paths import ensure_within
from gsstudio.pipeline.editor.ply import FLOAT_PROPERTIES, read_ply_header
from gsstudio.pipeline.stages import (
    ingest_capture, mask_run, preprocess_run, reconstruct_run, review_run,
    write_qa_report,
)
from gsstudio.pipeline.retention import auto_cleanup, cleanup_run
from gsstudio.infrastructure.runtime.run_lock import run_session
from gsstudio.infrastructure.persistence.run_repository import create_run, load_run, save_run
from gsstudio.pipeline.input.sources import load_prepared_input, prepare_capture_input, probe_capture_source
from gsstudio.pipeline.training.runner import train_package, verify_evaluation
from gsstudio.pipeline.training.data import prepare_segment, validate_package
from gsstudio.pipeline.training.control import submit_control


from gsstudio.application.contracts import EventSink, _emit
from gsstudio.application._shared import _scene, _selected_dataset, _verify_model
from gsstudio.application.captures import verify_capture_sources


def segment_choices(root: Path, location_id: str, scene_id: str,
                    run_id: str) -> dict[str, Any]:
    scene = _scene(root, location_id, scene_id)
    run = load_run(scene, run_id)
    if not run.selected_dataset or run.stages["reconstruct"].status.value != "succeeded":
        raise RuntimeError("No verified reconstruction is selected")
    dataset = _selected_dataset(scene, run)
    report = json.loads((dataset / "segments.json").read_text(encoding="utf-8"))
    return {
        "training_status": report["training_status"],
        "coverage_status": report["coverage_status"],
        "segments": [item for item in report["segments"] if item["status"] == "passed"],
        "report": str(dataset / "segments.json"),
    }


def control_training(root: Path, output: Path, action: str, *, generation_id=None,
                     package_sha256=None) -> dict[str, Any]:
    """Request a safe native-training step-boundary action for one experiment."""
    if action=='resume':
        raise ValueError('Use unpause for hot suspend, or train --resume to restore a saved checkpoint')
    output = ensure_within(output, root).resolve()
    if not output.is_relative_to(root.resolve()):
        raise ValueError("Experiment path resolves outside the Data root")
    dispatch_path = output / "dispatch.json"
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    if dispatch.get("backend") != "gsplat":
        raise ValueError("Live training controls are only available for gsplat")
    if dispatch.get("status") != "prepared":
        raise RuntimeError("Experiment is not currently accepting training controls")
    digest = dispatch.get("package_sha256")
    if not isinstance(digest, str):
        raise RuntimeError("Experiment is missing its training package identity")
    request = submit_control(output, digest, action, generation_id=generation_id,
                             expected_package_sha256=package_sha256)
    return {"output": str(output), "action": action, "request_id": request["request_id"],
            'generation_id':request['generation_id']}


def segment_package(root, location_id, scene_id, run, segment, duration_seconds=None):
    """CPU preparation behind the same actual QA/source gates as dispatch."""
    scene = _scene(root, location_id, scene_id)
    if run.status != RunStatus.ACCEPTED:
        raise RuntimeError('QA must be accepted before GUI training')
    capture = load_capture_manifest(scene/'captures'/f'{run.config.capture_id}.yaml')
    verify_capture_sources(capture)
    choices = segment_choices(root, location_id, scene_id, run.id)
    if segment not in {item['id'] for item in choices['segments']}:
        raise ValueError('Select a passed segment')
    label = run.metrics['selected_attempt']
    work = scene/run.id
    records = [json.loads(line) for line in
               (work/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
    name = segment if duration_seconds is None else f'{segment}-first-{duration_seconds:g}s'
    package = work/'training-data'/name
    prepare_segment(_selected_dataset(scene, run), records, run.config.reconstruction.primary,
                    run.config.segment_qa, segment, package,
                    duration_seconds=duration_seconds, reuse_files=True,
                    source_kind=capture.source.kind)
    return package, capture.source.kind


def train_segment(root: Path, location_id: str, scene_id: str, run_id: str,
                  segment: str, *, backend: str = "gsplat", steps: int | None = None,
                  resume: bool = False, sink: EventSink | None = None,
                  output: Path | None = None, photo_comp: bool | None = None,
                  use_bilateral_grid: bool | None = None,
                  use_sparse_depth: bool | None = None, dry_run: bool = False,
                  duration_seconds: float | None = None,
                  preview_every: int = 0, training_recipe: str = 'quality_v2',
                  sh_degree: int = 3, restart_from_step_zero: bool = False,
                  training_preset: str = 'auto') -> dict[str, Any]:
    scene = _scene(root, location_id, scene_id)
    from gsstudio.pipeline.training.runner import require_backend
    require_backend(backend)
    with run_session(scene / run_id):
        run = load_run(scene, run_id)
        package, source_kind = segment_package(root, location_id, scene_id, run, segment, duration_seconds)
        work = scene / run_id
        _emit(sink, "started", "train", "正在准备训练数据", run_id, segment=segment)
        experiments = run.metrics.setdefault("training_experiments", [])
        previous = next((item for item in reversed(experiments)
                         if item.get("segment") == segment and
                         item.get("backend") == backend and
                         (output is None or item.get("output") == str(output.resolve())) and
                         item.get("status") in {"failed", "preparing", "stopped", "paused"}), None)
        if resume and previous:
            experiment = previous
            output = Path(experiment["output"])
            prior_steps = experiment.get('requested_steps', experiment.get('steps'))
            if steps is not None and prior_steps is not None and steps != prior_steps:
                raise RuntimeError('Experiment identity changed: requested_steps')
            steps = prior_steps if steps is None else steps
            dispatch = output / "dispatch.json"
            if dispatch.is_file():
                recorded = json.loads(dispatch.read_text(encoding="utf-8"))
                if recorded.get("status") == "succeeded":
                    from gsstudio.pipeline.training.runner import verify_dispatch_settings, verify_completed
                    package_meta = verify_dispatch_settings(recorded, package, source_kind=source_kind,
                        training_preset=training_preset, steps=steps, photo_comp=photo_comp,
                        use_bilateral_grid=use_bilateral_grid, use_sparse_depth=use_sparse_depth,
                        training_recipe=training_recipe, sh_degree=sh_degree)
                    verify_completed(output, package_meta, recorded)
                    _verify_model(output / "model.ply", recorded["model_sha256"])
                    experiment.update(recorded)
                    save_run(scene, run)
                    _emit(sink, "completed", "train", "已恢复完成的训练记录", run_id,
                          output=str(output),model_path=str(output/'model.ply'))
                    return {**recorded, "output": str(output), "model_path":str(output/'model.ply')}
        else:
            if resume:
                raise ValueError("No failed or interrupted experiment exists to resume")
            output = output.resolve() if output is not None else (
                work / "experiments" / f"{backend}-{segment}-{uuid.uuid4().hex[:8]}"
            )
            experiment = {"segment": segment, "backend": backend,
                          "output": str(output), "steps": steps,
                          "status": "preparing"}
            experiments.append(experiment)
        from gsstudio.pipeline.training.presets import prepare_training
        actual_package, _, effective = prepare_training(package, source_kind=source_kind,
            training_preset=training_preset,steps=steps,training_recipe=training_recipe,
            photo_comp=photo_comp,use_bilateral_grid=use_bilateral_grid,
            use_sparse_depth=use_sparse_depth,sh_degree=sh_degree)
        _emit(sink, "progress", "train", "训练日志已就绪", run_id,
              log_path=str(output / "train.log"), output=str(output),
              package=str(actual_package), **{k:v for k,v in effective.items() if k!='steps'},
              requested_steps=effective['steps'],backend=backend)
        save_run(scene, run)
        try:
            result = train_package(
                package, output, backend, steps, photo_comp, resume, dry_run,
                use_bilateral_grid=use_bilateral_grid,
                use_sparse_depth=use_sparse_depth,
                preview_every=preview_every,
                training_recipe=training_recipe, sh_degree=sh_degree,
                restart_from_step_zero=restart_from_step_zero,
                source_kind=source_kind, training_preset=training_preset,
            )
            experiment.update(result)
            if dry_run:
                _emit(sink, "completed", "train", "训练输入已准备；尚未训练", run_id)
                return {**result, "output": str(output)}
            if result.get("status") in {"stopped", "paused"}:
                waiting_cancelled = result.get("stop_reason") == "gpu_wait_cancelled"
                checkpoint_preserved = bool(result.get("checkpoint"))
                _emit(sink, "stopped", "train",
                      ("等待 GPU 时已取消；保留已有检查点" if checkpoint_preserved else
                       "等待 GPU 时已取消；尚未开始训练") if waiting_cancelled else
                      "已在训练步边界保存检查点并停止", run_id,
                      output=str(output), checkpoint=result.get("checkpoint"))
                return {**result, "output": str(output)}
            if result.get("status") != "succeeded" or not result.get("model_sha256"):
                raise RuntimeError("Training did not publish a verified PLY")
            _verify_model(output / "model.ply", result["model_sha256"])
            _emit(sink, "completed", "train", "PLY 已校验", run_id,
                  output=str(output),model_path=str(output/'model.ply'))
            return {**result, "output": str(output), "model_path":str(output/'model.ply')}
        except Exception as error:
            experiment.update(status="failed", error=str(error))
            _emit(sink, "failed", "train", str(error), run_id)
            raise
        finally:
            save_run(scene, run)
