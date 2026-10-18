from __future__ import annotations

import base64
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from gsstudio.domain.models import StrictModel, VisionQAConfig


class VisionQAVerdict(StrictModel):
    decision: Literal["pass", "fail"]
    confidence: float = Field(ge=0, le=1)
    false_negative_views: list[str] = Field(default_factory=list)
    false_positive_views: list[str] = Field(default_factory=list)
    rationale: str


class LocalVisionQAReview(StrictModel):
    reviewer: Literal["codex-local-llm"]
    contact_sheet_sha256: dict[str, str]
    verdict: VisionQAVerdict


SYSTEM_PROMPT = """You are a strict QA gate for person masks in architectural reconstruction images.
Each contact sheet shows perspective views. Red overlay is the region excluded from both COLMAP and
3DGS training. Pass only when visible people, the camera operator, limbs, and selfie-stick-adjacent
human pixels are covered with reasonable margins, while permanent architecture is not broadly
removed. Do not judge photographic beauty. Return only one JSON object with keys: decision
(pass/fail), confidence (0..1), false_negative_views (filenames), false_positive_views (filenames),
and rationale. If text is too small to name a view, describe its tile position in the list."""


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward the Bearer credential to a redirected URL."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


MAX_INLINE_IMAGE_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BODY_BYTES = 48 * 1024 * 1024


def build_deepseek_request(
    contact_sheets: list[Path], config: VisionQAConfig
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Inspect every tile in these contact sheets. Red is the ignored mask. "
                "Judge only person-mask false negatives and destructive false positives."
            ),
        }
    ]
    for path in contact_sheets[: config.max_contact_sheets]:
        if path.stat().st_size > MAX_INLINE_IMAGE_BYTES:
            raise RuntimeError(
                f"DeepSeek contact sheet exceeds the 32 MiB inline-image limit: {path}"
            )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _data_url(path),
                    "detail": config.image_detail,
                },
            }
        )
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        # Vision-capable reasoning models count hidden reasoning against this
        # budget. 1200 can finish with empty content before emitting the JSON.
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }


def parse_deepseek_verdict(response: dict[str, Any]) -> VisionQAVerdict:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "DeepSeek response does not contain choices[0].message.content"
        ) from error
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    if isinstance(content, dict):
        try:
            return VisionQAVerdict.model_validate(content)
        except Exception as error:
            raise RuntimeError("DeepSeek returned invalid mask-QA JSON") from error
    text = str(content).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return VisionQAVerdict.model_validate_json(text)
    except Exception as direct_error:
        # Some reasoning-capable endpoints prepend prose or <think> blocks even
        # when response_format=json_object is requested. Scan for a complete JSON
        # object, but still validate the exact fail-closed verdict schema.
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
                return VisionQAVerdict.model_validate(candidate)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        raise RuntimeError("DeepSeek returned invalid mask-QA JSON") from direct_error


def validate_deepseek_gate(
    verdict: VisionQAVerdict, config: VisionQAConfig
) -> VisionQAVerdict:
    reasons: list[str] = []
    if verdict.decision != "pass":
        reasons.append("decision is fail")
    if verdict.confidence < config.minimum_confidence:
        reasons.append(
            f"confidence {verdict.confidence:.2f} is below {config.minimum_confidence:.2f}"
        )
    if verdict.false_negative_views:
        reasons.append(f"reported false negatives: {verdict.false_negative_views}")
    if verdict.false_positive_views:
        reasons.append(f"reported false positives: {verdict.false_positive_views}")
    if reasons:
        raise RuntimeError(
            "DeepSeek mask QA did not pass the fail-closed gate: " + "; ".join(reasons)
        )
    return verdict


def load_local_mask_qa_review(
    path: Path,
    contact_sheets: list[Path],
    config: VisionQAConfig,
) -> tuple[LocalVisionQAReview, VisionQAVerdict]:
    """Load a fail-closed local LLM review bound to exact contact-sheet bytes."""
    try:
        review = LocalVisionQAReview.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"Local mask-QA review is invalid: {path}") from error
    expected = {
        sheet.name: hashlib.sha256(sheet.read_bytes()).hexdigest()
        for sheet in contact_sheets
    }
    if review.contact_sheet_sha256 != expected:
        raise RuntimeError(
            "Local mask-QA review does not match the current contact sheets"
        )
    return review, validate_deepseek_gate(review.verdict, config)


def resolve_deepseek_endpoint(base_url: str, endpoint_path: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError("DeepSeek base URL must be an absolute HTTPS URL")
    if parsed.query or parsed.fragment:
        raise RuntimeError("DeepSeek base URL must not contain a query string or fragment")
    return base_url.rstrip("/") + "/" + endpoint_path.lstrip("/")


def run_deepseek_mask_qa(
    contact_sheets: list[Path], config: VisionQAConfig
) -> VisionQAVerdict:
    if not config.enabled:
        raise RuntimeError("DeepSeek vision QA is disabled in this run configuration")
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Vision QA requires {config.api_key_env}; credentials are never stored in YAML or logs"
        )
    if not contact_sheets:
        raise RuntimeError("Vision QA requires at least one contact sheet")
    url = resolve_deepseek_endpoint(config.base_url, config.endpoint_path)
    payload = json.dumps(build_deepseek_request(contact_sheets, config)).encode("utf-8")
    if len(payload) > MAX_REQUEST_BODY_BYTES:
        raise RuntimeError("DeepSeek vision-QA request exceeds the 48 MiB request-body limit")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        opener = urllib.request.build_opener(_RejectRedirectHandler())
        with opener.open(request, timeout=config.timeout_seconds) as response:
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise RuntimeError("DeepSeek mask QA response exceeded 1 MB")
            result = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek mask QA HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"DeepSeek mask QA request failed: {error}") from error
    return validate_deepseek_gate(parse_deepseek_verdict(result), config)
