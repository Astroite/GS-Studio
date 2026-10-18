"""CPU tests for the services shared by the GS-Studio GUI and the gsstudio CLI.

They cover the P1 read-only boundary: status from manifests and evidence,
visible defects instead of silent success, shared check labelling and error
classification. No GPU, no external tools, no writes outside the temp tree.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from typer.testing import CliRunner

from tests.factories import DIGEST, RUN_ID, STRAY_ID, build_tree
from gsstudio.interfaces.cli.app import app
from gsstudio.infrastructure.persistence.manifests import load_yaml, save_yaml
from gsstudio.domain.models import STAGES
from gsstudio.application import browse, deps, errors, run_status

_tree = build_tree


def test_listing_reads_status_from_manifests(tmp_path: Path):
    tree = _tree(tmp_path)
    root = tree["root"]

    locations = browse.list_locations(root)
    assert [item.id for item in locations] == ["loc"]
    assert locations[0].display_name == "燕园"
    assert locations[0].status == "draft"
    assert locations[0].scene_count == 1

    scenes = browse.list_scenes(root, "loc")
    assert [item.id for item in scenes] == ["scene"]
    assert scenes[0].quality_target == "pipeline_proof"

    captures = browse.list_captures(root, "loc", "scene")
    assert captures[0].source_kind == "equirect_video"
    assert captures[0].files_total == 1
    assert captures[0].files_present == 1
    assert captures[0].problems == []

    runs = browse.list_runs(root, "loc", "scene")
    assert [item.id for item in runs] == [RUN_ID]
    assert runs[0].readable is True
    assert runs[0].status == "draft"
    assert runs[0].stages == {name: "pending" for name in STAGES}
    assert runs[0].problems == []


def test_missing_source_file_is_a_visible_problem(tmp_path: Path):
    tree = _tree(tmp_path)
    root = tree["root"]
    Path(tree["capture"].source.files[0].windows_path).unlink()

    entry = browse.list_captures(root, "loc", "scene")[0]
    assert entry.files_present == 0
    assert [problem.code for problem in entry.problems] == [browse.MISSING_SOURCE]
    assert entry.problems[0].path == entry.manifest_path
    assert browse.status_label(entry) == "draft"


def test_config_hash_mismatch_is_reported_and_never_reads_as_success(tmp_path: Path):
    tree = _tree(tmp_path)
    root = tree["root"]
    manifest_path = root / "loc" / "scene" / RUN_ID / "manifest.yaml"
    payload = load_yaml(manifest_path)
    payload["config_hash"] = "f" * 64
    save_yaml(manifest_path, payload)

    entry = browse.list_runs(root, "loc", "scene")[0]
    assert entry.readable is False
    assert entry.status is None
    assert browse.status_label(entry) == "unreadable"
    assert [problem.code for problem in entry.problems] == [browse.CONFIG_HASH_MISMATCH]
    assert entry.problems[0].path == str(manifest_path)


def test_historical_run_schema_is_unreadable_with_its_path(tmp_path: Path):
    tree = _tree(tmp_path)
    root = tree["root"]
    manifest_path = root / "loc" / "scene" / RUN_ID / "manifest.yaml"
    payload = load_yaml(manifest_path)
    payload["config"]["schema_version"] = 5
    save_yaml(manifest_path, payload)

    entry = browse.list_runs(root, "loc", "scene")[0]
    assert entry.readable is False
    assert [problem.code for problem in entry.problems] == [browse.UNREADABLE_MANIFEST]
    assert entry.problems[0].path == str(manifest_path)


def test_stray_run_directory_is_listed_without_inventing_a_status(tmp_path: Path):
    tree = _tree(tmp_path)
    root = tree["root"]
    scene = root / "loc" / "scene"
    (scene / STRAY_ID / "inputs" / "primary").mkdir(parents=True)
    (scene / "logs").mkdir()
    (scene / "logs" / "probe-protocol").write_text("{}", encoding="utf-8")

    runs = browse.list_runs(root, "loc", "scene")
    assert sorted(item.id for item in runs) == sorted([RUN_ID, STRAY_ID])
    stray = next(item for item in runs if item.id == STRAY_ID)
    assert stray.readable is False
    assert stray.status is None
    assert browse.status_label(stray) == "unreadable"
    assert [problem.code for problem in stray.problems] == [browse.STRAY_RUN_DIRECTORY]


def test_classification_preserves_message_and_keeps_the_cause():
    missing = errors.classify(FileNotFoundError("/data/loc/scene.yaml"))
    assert isinstance(missing, errors.NotFoundError)
    assert missing.code == "not_found"
    assert missing.message == "/data/loc/scene.yaml"
    assert isinstance(missing.__cause__, FileNotFoundError)

    invalid = errors.classify(ValueError("end_seconds must be greater than start_seconds"))
    assert invalid.code == "invalid_input"
    assert invalid.detail == "ValueError"

    generic = errors.classify(RuntimeError("Experiment identity changed: backend"))
    assert generic.code == "integrity"
    assert generic.detail == "RuntimeError"
    assert generic.message == "Experiment identity changed: backend"

    already = errors.IntegrityError("mask-final.json is stale or corrupt")
    assert errors.classify(already) is already
    assert already.to_dict() == {
        "code": "integrity",
        "message": "mask-final.json is stale or corrupt",
    }


def test_check_labels_keep_doctor_semantics():
    checks = {
        "ok": False,
        "ffmpeg": (True, "ffmpeg version 9"),
        "mediasdk": (False, "optional unavailable: missing"),
        "torch_cuda": (False, "no CUDA device"),
    }
    views = {view.name: view for view in deps.format_checks(checks, backend="gsplat")}
    assert views["ffmpeg"].label == "PASS"
    assert 'postshot' not in views  # Retired trainer is no longer a doctor dependency.
    assert views["mediasdk"].label == "UNAVAILABLE"
    assert views["torch_cuda"].label == "FAIL"
    assert views["torch_cuda"].required is True

    with pytest.raises(ValueError,match='retired'):
        deps.format_checks(checks,backend='postshot')

    views = {
        view.name: view
        for view in deps.format_checks(checks, require={"mediasdk"}, backend="gsplat")
    }
    assert views["mediasdk"].label == "FAIL"



def test_run_detail_reports_evidence_presence_without_claiming_success(tmp_path: Path):
    tree = _tree(tmp_path)
    root = tree["root"]

    detail = run_status.read_run_detail(root, "loc", "scene", RUN_ID)
    assert detail.status_label == "draft"
    assert [stage.name for stage in detail.stages] == list(STAGES)
    present = {item.name: item.present for item in detail.evidence}
    assert present["prepared-input"] is True
    assert present["qa-report"] is False
    assert detail.problems == []
    assert detail.config_summary["projection"] == {
        "views": 14,
        "size": 1746,
        "crop_bottom": 0.15,
    }
    assert detail.config_summary["mask_discard_threshold"] == 0.005
    assert detail.experiments == []
    # Present artifacts document work; only a stage record reports success.
    assert all(stage.status == "pending" for stage in detail.stages)

    prepared = root / "loc" / "scene" / RUN_ID / "inputs" / "primary" / DIGEST
    (prepared / "dataset.json").unlink()

    detail = run_status.read_run_detail(root, "loc", "scene", RUN_ID)
    present = {item.name: item.present for item in detail.evidence}
    assert present["prepared-input"] is False
    assert detail.problems[0].code == browse.MISSING_INPUT
    assert detail.status_label == "draft"
    assert all(stage.status == "pending" for stage in detail.stages)


def test_run_detail_of_an_unreadable_run_stays_unreadable(tmp_path: Path):
    tree = _tree(tmp_path)
    detail = run_status.read_run_detail(tree["root"], "loc", "scene", "20260922T000000Z-none00")
    assert detail.status_label == "unreadable"
    assert detail.stages == []
    assert detail.problems[0].code == browse.MISSING_MANIFEST


def test_cli_status_exposes_the_shared_service(tmp_path: Path):
    tree = _tree(tmp_path)
    root = tree["root"]
    runner = CliRunner()

    result = runner.invoke(app, ["status", "--json"], env={"GSSTUDIO_DATA_ROOT": str(root)})
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["locations"][0]["id"] == "loc"

    result = runner.invoke(
        app, ["status", "loc", "scene", RUN_ID, "--json"], env={"GSSTUDIO_DATA_ROOT": str(root)}
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["entry"]["id"] == RUN_ID
    assert payload["config_summary"]["capture_id"] == "capture"
    assert payload["config_summary"]["source_kind"] == "equirect_video"

    result = runner.invoke(app, ["status", "loc", "missing"], env={"GSSTUDIO_DATA_ROOT": str(root)})
    assert result.exit_code == 1
    assert "not_found" in result.output
    assert "Scene does not exist" in result.output
