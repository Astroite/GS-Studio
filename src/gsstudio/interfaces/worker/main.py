"""Isolated JSON-lines operation worker for the Qt workbench.

The GUI sends one JSON request on stdin. Stdout contains only structured
events; complete engine output remains in the Run's own log files.
"""
from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from gsstudio.application import operations as ops
from gsstudio.application.errors import classify

_PROTOCOL_STREAM = sys.stdout
_REQUEST_ID = None


def emit(kind: str, action: str, message: str, run_id: str | None = None,
         **detail: Any) -> None:
    data = {"kind": kind, "action": action, "message": message,
            "run_id": run_id, "detail": detail}
    if _REQUEST_ID is not None:
        if kind == 'result':
            data = {"type":"result", "id":_REQUEST_ID, "result":detail['result']}
        elif kind == 'error':
            data = {"type":"error", "id":_REQUEST_ID,
                    "error":{"code":detail.get('code','PIPELINE_ERROR'),"message":message,
                             "details":{k:v for k,v in detail.items() if k!='code'}}}
        else:
            data = {"type":"event", "id":_REQUEST_ID, "event":kind, "data":{**data,**detail}}
    _PROTOCOL_STREAM.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    _PROTOCOL_STREAM.flush()


def _event(event: ops.OperationEvent) -> None:
    emit(event.kind, event.action, event.message, event.run_id, **(event.detail or {}))


def _stop_requested(request: dict[str, Any]) -> bool:
    value = request.get("stop_file")
    return bool(value and Path(value).is_file())


def _external_options(request: dict[str, Any]) -> dict[str, Any]:
    return {
        key: Path(request[key]) if request.get(key) else None
        for key in ("source", "images", "model", "masks", "group_file")
    } | {
        "independent_images": bool(request.get("independent_images", False)),
        "white_ignore": bool(request.get("white_ignore", False)),
    }


def execute(request: dict[str, Any]) -> dict[str, Any]:
    from gsstudio.application.studio import execute_additional
    handled, value = execute_additional(request,sink=_event)
    if handled:
        return value
    action = request["action"]
    root = Path(request["root"])
    location = request.get("location_id")
    scene = request.get("scene_id")
    run_id = request.get("run_id")
    if action == "inspect_external_import":
        return ops.inspect_external_import(**_external_options(request))
    if action == "prepare_external_import":
        return ops.prepare_external_import(
            root, request["import_id"], expected_input_identity=request.get("expected_input_identity"),
            sink=_event, **_external_options(request),
        )
    if action == "train_external_import":
        return ops.train_external_import(
            root, request["import_id"], backend=request.get("backend", "gsplat"),
            steps=request.get("steps"), resume=bool(request.get("resume", False)),
            experiment_id=request.get("experiment_id"), sink=_event,
            **{k:request[k] for k in ('photo_comp','use_bilateral_grid','use_sparse_depth',
                                    'preview_every','dry_run','training_recipe','sh_degree',
                                    'restart_from_step_zero','training_preset') if k in request},
        )
    if action == "list_external_imports":
        return ops.list_external_imports(root)
    if action == "relink_external_import":
        return ops.relink_external_import(
            root, request["import_id"],
            new_root=Path(request["new_root"]) if request.get("new_root") else None,
            mapping_file=Path(request["mapping_file"]) if request.get("mapping_file") else None,
        )
    if action == "control_training":
        if not request.get('generation_id') or not request.get('package_sha256'):
            raise ValueError('Training controls require generation_id and package_sha256 from training_state')
        return ops.control_training(root, Path(request["output"]), request["control_action"],
                                    generation_id=request.get('generation_id'),
                                    package_sha256=request.get('package_sha256'))
    if action == "create_location":
        result = ops.create_location(root, location, request["name"],
                                     **{k:request[k] for k in ('capture_date','province','city') if k in request})
        return {"location_id": result.id}
    if action == "create_scene":
        result = ops.create_scene(root, location, scene, request["name"],description=request.get('description'))
        return {"scene_id": result.id}
    if action == "create_capture":
        result = ops.create_capture(
            root, location, scene, request["capture_id"], request["source_type"],
            [Path(value) for value in request["sources"]],
            camera_model=request.get("camera_model", "unknown"),
            fps=request.get("fps"), start_seconds=request.get("start_seconds", 0.0),
            end_seconds=request.get("end_seconds"),
            output_width=request.get("output_width"),
            output_height=request.get("output_height"),
            horizontal_fov=request.get("horizontal_fov", 84.0),
            camera_make=request.get("camera_make", "unknown"),
            normalization=request.get("normalization"),
        )
        return {"capture_id": result.id}
    if action == "relink_capture":
        result = ops.relink_capture_sources(
            root, location, scene, request["capture_id"],
            [Path(value) for value in request["sources"]],
        )
        return {"capture_id": result.id}
    if action == "probe_capture":
        return ops.probe_capture(root, location, scene, request["capture_id"])
    if action == "ingest":
        result = ops.ingest(root, location, scene, request["capture_id"])
        return {"capture_id": result.id}
    if action == "create_run":
        from inspect import signature
        options = {k: v for k, v in request.items() if k in signature(ops.create_and_preprocess).parameters
                   and k not in {'root','location_id','scene_id','capture_id','sink','mask_review_gate'}}
        if options.get('vision_qa'):
            raise ValueError('External vision QA needs explicit image upload authorization; unavailable through automatic RPC')
        run_id = ops.create_and_preprocess(
            root, location, scene, request["capture_id"],
            mask_review_gate=True, sink=_event, **options,
        )
        if not request.get('advance', True):
            return {'run_id':run_id,'status':'preprocessed'}
        return {'run_id':run_id, **ops.advance_run(root, location, scene, run_id,
                               stop_requested=lambda: _stop_requested(request), sink=_event)}
    if action == "resume_run":
        return ops.advance_run(root, location, scene, run_id,
                               stop_requested=lambda: _stop_requested(request), sink=_event)
    if action == "finalize_masks":
        if not request.get('dataset_identity'):
            raise ValueError('Finalize requires dataset_identity from the current mask review page')
        result = ops.finalize_masks(root, location, scene, run_id,
                                    attempt=request.get("attempt", "primary"),
                                    expected_identity=request['dataset_identity'])
        emit("completed", "finalize_masks", "遮罩已固定", run_id)
        return {"finalization_sha256": result["finalization_sha256"]}
    if action == "review_qa":
        result = ops.review_qa(root, location, scene, run_id,
                               request["accepted"], request["notes"])
        return {"status": result.status.value}
    if action == "segment_choices":
        return ops.segment_choices(root, location, scene, run_id)
    if action == "train_segment":
        options = {k:request[k] for k in ('photo_comp','use_bilateral_grid','use_sparse_depth',
                                         'duration_seconds','dry_run','training_recipe','sh_degree',
                                         'restart_from_step_zero','training_preset') if k in request}
        return ops.train_segment(root, location, scene, run_id,
                                 request["segment"],
                                 backend=request.get("backend", "gsplat"),
                                 steps=request.get("steps"),
                                 resume=request.get("resume", False), sink=_event,
                                 output=Path(request["output"]) if request.get("output") else None,
                                 preview_every=request.get('preview_every',500), **options)
    if action == "cleanup":
        return ops.cleanup(root, location, scene, run_id,
                           apply=request.get("apply", False))
    if action == "verified_models":
        return {"models": ops.verified_models(root, location, scene, run_id)}
    raise ValueError(f"Unknown operation: {action}")


def main() -> int:
    global _REQUEST_ID
    # QProcess sends UTF-8 bytes. Frozen Python ignores PYTHONIOENCODING,
    # so the protocol must choose its encoding explicitly on Windows.
    for stream in (sys.stdin, sys.stdout, sys.stderr, _PROTOCOL_STREAM):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="strict")
    try:
        payload = json.loads(sys.stdin.readline())
        if not isinstance(payload, dict):
            raise ValueError("Operation request must be an object")
        if 'method' in payload:
            _REQUEST_ID = payload.get('id')
            if not isinstance(_REQUEST_ID,str) or not isinstance(payload.get('params'),dict):
                raise ValueError('Expected string id and object params')
            method, payload = payload['method'], payload['params']
            if method == 'studio.catalog':
                from gsstudio.application.studio import catalog
                with redirect_stdout(sys.stderr):
                    result = catalog(Path(payload['root']))
                emit('result','catalog','Catalog',result=result)
                return 0
            if method != 'studio.operation':
                raise ValueError(f'Unknown worker method: {method}')
        request = ops.OperationRequest.from_mapping(payload)
        with redirect_stdout(sys.stderr):
            result = ops.OperationResult(request.action, execute(request.to_mapping()))
        emit("result", result.action, "操作完成", result.data.get("run_id"), result=result.data)
        return 0
    except Exception as error:
        classified = classify(error)
        emit("error", "worker", classified.message,
             code=classified.code, path=classified.path)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
