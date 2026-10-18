"""Durable, identity-bound control requests for a native training process."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from gsstudio.pipeline.training.data import json_write


CONTROL_ACTIONS = frozenset({"pause", "resume", "suspend", "unpause", "checkpoint", "stop"})
REQUEST_NAME = "control-request.json"
STATE_NAME = "control-state.json"


def _read_object(path: Path) -> dict | None:
    if not path.is_file():
        return None
    # Windows can briefly deny an open while the writer replaces its record.
    for attempt in range(5):
        try:
            content = path.read_text(encoding="utf-8")
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02)
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid training control record: {path}")
    return value


def read_control_state(output: Path) -> dict | None:
    return _read_object(output / STATE_NAME)


def submit_control(output: Path, package_sha256: str, action: str, *, generation_id=None,
                   expected_package_sha256=None) -> dict:
    """Submit exactly one pending request; stop may supersede any other request."""
    if action not in CONTROL_ACTIONS:
        raise ValueError(f"Unknown training control: {action}")
    if len(package_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in package_sha256):
        raise ValueError("Invalid training package identity")
    output = output.resolve()
    pending = _read_object(output / REQUEST_NAME)
    state = _read_object(output / STATE_NAME)
    if (not state or state.get("package_sha256") != package_sha256 or
            not isinstance(state.get("generation_id"), str)):
        raise RuntimeError("Training is not accepting controls")
    if ((generation_id is not None and state['generation_id'] != generation_id) or
            (expected_package_sha256 is not None and package_sha256 != expected_package_sha256)):
        raise RuntimeError('Training control generation/package identity changed')
    if state and state.get("status") in {"stopping", "stopped", "succeeded", "failed"}:
        raise RuntimeError(f"Training is already {state['status']}")
    if action in {"resume", "unpause"} and state.get("status") not in {"paused", "suspended"}:
        raise RuntimeError("Training is not paused")
    if action in {"pause", "suspend"} and state.get("status") not in {"running", "suspended"}:
        raise RuntimeError("Training is not running")
    if action == "checkpoint" and state.get("status") not in {"running", "paused", "suspended"}:
        raise RuntimeError("Training has not reached a checkpoint boundary")
    if (pending and pending.get("request_id") != (state or {}).get("request_id")
            and action != "stop"):
        raise RuntimeError("A training control request is awaiting a step boundary")
    request = {
        "schema_version": 1,
        "request_id": uuid.uuid4().hex,
        "package_sha256": package_sha256,
        "generation_id": state["generation_id"],
        "action": action,
    }
    json_write(output / REQUEST_NAME, request)
    return request


class TrainingControl:
    """Read controls at safe step boundaries without changing optimizer config."""

    def __init__(self, output: Path, package_sha256: str, *, waiting_for_gpu: bool = False):
        self.output = output.resolve()
        self.package_sha256 = package_sha256
        pending = _read_object(self.output / REQUEST_NAME)
        self.generation_id = uuid.uuid4().hex
        # A request left by an earlier process must not stop a new attempt.
        self.request_id: str | None = pending.get("request_id") if pending else None
        self.status = "waiting_gpu" if waiting_for_gpu else "running"
        self._last_step = 0
        if waiting_for_gpu and (self.output / "checkpoint.pt").is_file():
            try:
                summary = _read_object(self.output / "training.json")
            except (OSError, ValueError, json.JSONDecodeError):
                summary = None
            if summary and isinstance(summary.get("steps"), int) and summary["steps"] >= 0:
                self._last_step = summary["steps"]
        self._publish(self.status, self._last_step, request_id=self.request_id)

    def is_set(self) -> bool:
        """Cancel a GPU-lock wait when a bound stop request arrives."""
        if self.status == "stopping":
            return True
        if self.status != "waiting_gpu":
            return False
        request = self._next_request()
        if request is None:
            return False
        if request["action"] == "stop":
            self._publish("stopping", self._last_step, request_id=request["request_id"],
                          message="Training cancelled before acquiring the GPU")
            print("Training GPU wait cancelled before step 0", flush=True)
            return True
        self._publish("waiting_gpu", self._last_step, request_id=request["request_id"],
                      message="Only stop is available while waiting for the GPU")
        return False

    def start_running(self) -> None:
        """Publish running only after the GPU lock is actually held."""
        if self.status != "waiting_gpu":
            raise RuntimeError(f"Training cannot start from {self.status}")
        self._publish("running", self._last_step, request_id=self.request_id)

    def _publish(self, status: str, step: int, *, request_id: str | None = None,
                 message: str | None = None) -> None:
        self.status = status
        self._last_step = step
        payload = {
            "schema_version": 1,
            "package_sha256": self.package_sha256,
            "generation_id": self.generation_id,
            "status": status,
            "step": step,
            "request_id": request_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if message:
            payload["message"] = message
        json_write(self.output / STATE_NAME, payload)

    def _next_request(self) -> dict | None:
        try:
            request = _read_object(self.output / REQUEST_NAME)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Training control request rejected: {error}", flush=True)
            return None
        if request is None or request.get("request_id") == self.request_id:
            return None
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            print("Training control request rejected: missing request ID", flush=True)
            return None
        self.request_id = request_id
        if (request.get("schema_version") != 1 or
                request.get("package_sha256") != self.package_sha256 or
                request.get("generation_id") != self.generation_id or
                request.get("action") not in CONTROL_ACTIONS):
            self._publish(self.status, self._last_step, request_id=request_id,
                          message="Training control identity or action mismatch")
            print("Training control request rejected: identity or action mismatch", flush=True)
            return None
        return request

    def observe(self, step: int, save: Callable[[], None], *, target: int | None = None) -> str | None:
        while True:
            request = self._next_request()
            if request:
                action, request_id = request["action"], request["request_id"]
                if action == "stop":
                    if target is not None and step >= target:
                        self._publish("running", step, request_id=request_id,
                                      message="Training reached its target before stop")
                        return None
                    self._publish("stopping", step, request_id=request_id)
                    print(f"Training stop requested at step {step}", flush=True)
                    return "stop"
                if action == "checkpoint":
                    was_paused = self.status == "paused"
                    save()
                    self._publish("paused" if was_paused else "running", step,
                                  request_id=request_id, message="checkpoint_saved")
                    print(f"Training checkpoint saved at step {step}", flush=True)
                elif action == "pause":
                    save()
                    self._publish("paused", step, request_id=request_id)
                    print(f"Training paused at step {step}", flush=True)
                elif action == "resume":
                    self._publish("running", step, request_id=request_id)
                    print(f"Training resumed at step {step}", flush=True)
            if self.status != "paused":
                return None
            time.sleep(0.2)

    def complete(self, status: str, step: int) -> None:
        if status not in {"stopped", "succeeded", "failed"}:
            raise ValueError(status)
        self._publish(status, step, request_id=self.request_id)
