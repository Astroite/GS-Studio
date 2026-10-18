"""Worker contract: Run advancement delegates to application and emits JSON lines."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from gsstudio.application.contracts import OperationEvent, OperationRequest
from gsstudio.interfaces.worker import main as worker


def test_resume_run_uses_application_and_serializes_progress(tmp_path, monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(worker, "_PROTOCOL_STREAM", stream)

    def advance(root, location_id, scene_id, run_id, *, stop_requested, sink):
        assert root == tmp_path
        assert (location_id, scene_id, run_id) == ("loc", "scene", "run")
        assert stop_requested() is False
        sink(OperationEvent("waiting", "mask_review", "请审核遮罩", run_id))
        return {"run_id": run_id, "next": "mask_review"}

    monkeypatch.setattr(worker.ops, "advance_run", advance)
    payload = {"action": "resume_run", "root": str(tmp_path),
               "location_id": "loc", "scene_id": "scene", "run_id": "run"}
    request = OperationRequest.from_mapping(payload)
    result = worker.execute(request.to_mapping())

    assert result == {"run_id": "run", "next": "mask_review"}
    assert json.loads(stream.getvalue()) == {
        "kind": "waiting", "action": "mask_review", "message": "请审核遮罩",
        "run_id": "run", "detail": {},
    }


def test_external_relink_worker_uses_shared_operation(tmp_path, monkeypatch):
    def relink(root, import_id, *, new_root, mapping_file):
        assert (root, import_id, new_root, mapping_file) == (
            tmp_path, "import-one", tmp_path / "moved", None,
        )
        return {"status": "prepared", "id": import_id}

    monkeypatch.setattr(worker.ops, "relink_external_import", relink)
    assert worker.execute({"action": "relink_external_import", "root": str(tmp_path),
                           "import_id": "import-one", "new_root": str(tmp_path / "moved")}) == {
        "status": "prepared", "id": "import-one",
    }


def test_worker_protocol_uses_utf8_even_with_non_utf8_stdio(monkeypatch):
    payload = {"action": "create_location", "root": "C:/中文 Data", "location_id": "loc", "name": "古建筑"}
    incoming = io.TextIOWrapper(io.BytesIO((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")),
                                encoding="cp1252", errors="surrogateescape")
    outgoing_bytes = io.BytesIO()
    outgoing = io.TextIOWrapper(outgoing_bytes, encoding="cp1252")
    monkeypatch.setattr(sys, "stdin", incoming)
    monkeypatch.setattr(sys, "stdout", outgoing)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(worker, "_PROTOCOL_STREAM", outgoing)

    def execute(request):
        assert Path(request["root"]) == Path(payload["root"])
        assert request["name"] == payload["name"]
        return {"location_id": "loc"}

    monkeypatch.setattr(worker, "execute", execute)
    assert worker.main() == 0
    outgoing.flush()
    result = json.loads(outgoing_bytes.getvalue().decode("utf-8"))
    assert result["kind"] == "result" and result["message"] == "操作完成"
