from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class OperationRequest:
    action: str
    root: Path
    arguments: dict[str, Any]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "OperationRequest":
        action, root = payload.get("action"), payload.get("root")
        if not isinstance(action, str) or not action:
            raise ValueError("Operation action is required")
        if not isinstance(root, str) or not root:
            raise ValueError("Operation data root is required")
        return cls(action, Path(root), {key: value for key, value in payload.items()
                                      if key not in {"action", "root"}})

    def to_mapping(self) -> dict[str, Any]:
        return {"action": self.action, "root": str(self.root), **self.arguments}


@dataclass(frozen=True)
class OperationResult:
    action: str
    data: dict[str, Any]


@dataclass(frozen=True)
class OperationEvent:
    kind: str
    action: str
    message: str
    run_id: str | None = None
    detail: dict[str, Any] | None = None


EventSink = Callable[[OperationEvent], None]


def _emit(sink: EventSink | None, kind: str, action: str, message: str,
          run_id: str | None = None, **detail: Any) -> None:
    if sink:
        sink(OperationEvent(kind, action, message, run_id, detail or None))
