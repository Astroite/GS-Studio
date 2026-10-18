from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from gsstudio.infrastructure.persistence.catalog import build_catalog
from gsstudio.application.doctor import run_doctor
from gsstudio.pipeline.masks.review import create_mask_review_server
from gsstudio.interfaces.cli.locking import run_cli_locked
from gsstudio.interfaces.cli.paths import data_root as _data_root, scene_path as _scene
from gsstudio.application import classify, format_checks
from gsstudio.application import browse, run_status
from gsstudio.application import operations as ops
from gsstudio.infrastructure.paths import host_path
from gsstudio.infrastructure.persistence.run_repository import load_run
from gsstudio.pipeline.input.sharp_select import run_select_sharp
from gsstudio.interfaces.cli.external_import import app as external_import_app


app = typer.Typer(
    no_args_is_help=True,
    help=(
        "360, perspective video and photos to Gaussian Splatting workflow and catalog. "
        "Windows path: doctor → location/scene/capture → media/ingest → preprocess → "
        "mask → reconstruct → qa → train --segment → cleanup."
    ),
)
location_app = typer.Typer(no_args_is_help=True, help="Manage location manifests")
scene_app = typer.Typer(no_args_is_help=True, help="Manage scene manifests")
catalog_app = typer.Typer(no_args_is_help=True, help="Build and inspect the SQLite catalog")
qa_app = typer.Typer(no_args_is_help=True, help="Generate and review QA reports")
capture_app = typer.Typer(no_args_is_help=True, help="Manage capture manifests")
media_app = typer.Typer(no_args_is_help=True, help="Probe source media")
app.add_typer(location_app, name="location")
app.add_typer(scene_app, name="scene")
app.add_typer(catalog_app, name="catalog")
app.add_typer(qa_app, name="qa")
app.add_typer(capture_app, name="capture")
app.add_typer(media_app, name="media")
app.add_typer(external_import_app, name="external")
console = Console()

def _fatal(error: Exception) -> None:
    classified = classify(error)
    line = Text(f"Error[{classified.code}]: ", style="red")
    line.append(classified.message)
    console.print(line)
    raise typer.Exit(1)


@app.command()
def doctor(
    minimum_free_gib: Annotated[float, typer.Option(help="Required free-space reserve")] = 20.0,
    backend: Annotated[str, typer.Option(help="Training backend to check: gsplat or all")] = "gsplat",
    require: Annotated[
        list[str] | None,
        typer.Option("--require", help="Require additional component: mediasdk or colmap"),
    ] = None,
) -> None:
    """Check Windows reconstruction deps and the training backend."""
    try:
        checks = run_doctor(
            _data_root(), minimum_free_gib=minimum_free_gib, require=set(require or []), backend=backend
        )
    except Exception as error:
        _fatal(error)
    table = Table("Check", "Result", "Detail")
    for view in format_checks(checks, require=set(require or []), backend=backend):
        table.add_row(
            view.name,
            view.label,
            view.detail,
            style=None if view.ok or not view.required else "red",
        )
    console.print(table)
    if not checks["ok"]:
        raise typer.Exit(1)


def _emit(payload: dict, as_json: bool, render) -> None:
    if as_json:
        # Plain stdout: scripts must get clean JSON regardless of console width.
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        render()


def _print_problem(subject: str, problem) -> None:
    where = f" ({escape(problem.path)})" if problem.path else ""
    console.print(
        f"[yellow]{escape(problem.code)}[/yellow] {escape(subject)}: "
        f"{escape(problem.message)}{where}"
    )


def _print_problems(entries) -> None:
    for entry in entries:
        for problem in entry.problems:
            _print_problem(entry.id, problem)


def _print_locations(entries) -> None:
    table = Table("Location", "Name", "Status", "Scenes", "Capture date")
    for item in entries:
        table.add_row(
            item.id,
            item.display_name or "-",
            browse.status_label(item),
            str(item.scene_count),
            item.capture_date or "-",
        )
    console.print(table)
    _print_problems(entries)


def _print_scenes(entries) -> None:
    table = Table("Scene", "Name", "Status", "Quality target", "Metric scale")
    for item in entries:
        table.add_row(
            item.id,
            item.display_name or "-",
            browse.status_label(item),
            item.quality_target or "-",
            "yes" if item.metric_scale else "no",
        )
    console.print(table)
    _print_problems(entries)


def _print_scene(captures, runs) -> None:
    table = Table("Capture", "Source", "Files", "Media", "Status", "Selection")
    for item in captures:
        media = item.media or {}
        size = (
            f"{media.get('width')}x{media.get('height')}"
            if media.get("width")
            else "-"
        )
        selection = (
            f"{item.selection['start_seconds']:.2f}-{item.selection['end_seconds']:.2f}s"
            if item.selection
            else "-"
        )
        table.add_row(
            item.id,
            item.source_kind or "-",
            f"{item.files_present}/{item.files_total}",
            size,
            browse.status_label(item),
            selection,
        )
    console.print(table)
    _print_problems(captures)
    run_table = Table("Run", "Status", "Stages", "Selected", "Updated")
    for item in runs:
        stages = ", ".join(f"{name}={value}" for name, value in item.stages.items()) or "-"
        run_table.add_row(
            item.id,
            browse.status_label(item),
            stages,
            item.selected_dataset or "-",
            item.updated_at or "-",
        )
    console.print(run_table)
    _print_problems(runs)


def _print_run_detail(detail) -> None:
    entry = detail.entry
    console.print(
        f"Run [green]{escape(entry.id)}[/green] in {escape(entry.location_id)}/{escape(entry.scene_id)}: "
        f"status={escape(detail.status_label)}"
    )
    if detail.config_summary:
        console.print("[bold]Configuration[/bold]")
        for key, value in detail.config_summary.items():
            if isinstance(value, (dict, list)):
                shown = json.dumps(value, ensure_ascii=False)
            else:
                shown = str(value)
            console.print(f"  {escape(key)}: {escape(shown)}")
    stages = Table("Stage", "Status", "Elapsed", "Message", "Log")
    for stage in detail.stages:
        elapsed = "-" if stage.elapsed_seconds is None else f"{stage.elapsed_seconds:.1f}s"
        stages.add_row(
            stage.name,
            stage.status,
            elapsed,
            stage.message or "-",
            stage.log_path if stage.log_path and stage.log_present else "-",
        )
    console.print(stages)
    evidence = Table("Evidence", "Present", "Path")
    for item in detail.evidence:
        evidence.add_row(item.name, "yes" if item.present else "no", escape(item.path))
    console.print(evidence)
    if detail.logs:
        logs = Table("Log", "Bytes", "Path")
        for item in detail.logs:
            logs.add_row(escape(item.name), str(item.size_bytes), escape(item.path))
        console.print(logs)
    if detail.experiments:
        experiments = Table("Segment", "Backend", "Status", "Output", "Error")
        for item in detail.experiments:
            experiments.add_row(
                item.segment or "-",
                item.backend or "-",
                item.status or "-",
                escape(item.output or "-"),
                escape(item.error or "-"),
            )
        console.print(experiments)
    if detail.review_notes:
        console.print(f"Review notes: {escape(detail.review_notes)}")
    _print_problems(detail.entry.problems + detail.problems)


@app.command()
def status(
    location_id: Annotated[str | None, typer.Argument(help="Location to expand")] = None,
    scene_id: Annotated[str | None, typer.Argument(help="Scene to expand")] = None,
    run_id: Annotated[str | None, typer.Argument(help="Run to inspect")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output for scripts")] = False,
) -> None:
    """Read-only project, stage, evidence and log status.

    Status comes from the manifests and the evidence they name: a directory, a
    PLY or a report file never reports a stage as succeeded. Writes nothing.
    """
    try:
        root = _data_root()
        if run_id is not None:
            if not location_id or not scene_id:
                raise ValueError("Inspecting a run needs a location and a scene")
            detail = run_status.read_run_detail(root, location_id, scene_id, run_id)
            _emit(detail.to_dict(), as_json, lambda: _print_run_detail(detail))
            return
        if scene_id is not None:
            if not location_id:
                raise ValueError("Listing a scene needs a location")
            captures = browse.list_captures(root, location_id, scene_id)
            runs = browse.list_runs(root, location_id, scene_id)
            _emit(
                {
                    "location_id": location_id,
                    "scene_id": scene_id,
                    "captures": [item.to_dict() for item in captures],
                    "runs": [item.to_dict() for item in runs],
                },
                as_json,
                lambda: _print_scene(captures, runs),
            )
            return
        if location_id is not None:
            scenes = browse.list_scenes(root, location_id)
            _emit(
                {"location_id": location_id, "scenes": [item.to_dict() for item in scenes]},
                as_json,
                lambda: _print_scenes(scenes),
            )
            return
        locations = browse.list_locations(root)
        _emit(
            {"locations": [item.to_dict() for item in locations]},
            as_json,
            lambda: _print_locations(locations),
        )
    except Exception as error:
        _fatal(error)


@capture_app.command("init")
def capture_init(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
    source_type: Annotated[
        str,
        typer.Option(
            "--source-type",
            help="equirect_video, equirect_sequence, insta360_insv, perspective_video, or perspective_images",
        ),
    ],
    source: Annotated[list[Path], typer.Option("--source", help="Explicit source path")],
    camera_model: Annotated[str, typer.Option("--camera-model")] = "unknown",
    camera_make: Annotated[str, typer.Option("--camera-make")] = "Insta360",
    fps: Annotated[float | None, typer.Option(help="Image-sequence FPS")] = None,
    start_seconds: Annotated[float, typer.Option()] = 0.0,
    end_seconds: Annotated[float | None, typer.Option()] = None,
    output_width: Annotated[int | None, typer.Option()] = None,
    output_height: Annotated[int | None, typer.Option()] = None,
    horizontal_fov: Annotated[
        float,
        typer.Option(help="Horizontal FOV in degrees; perspective_video focal prior"),
    ] = 84.0,
) -> None:
    """Create and fingerprint a panorama, perspective video or native photo collection."""
    try:
        created = ops.create_capture(
            _data_root(), location_id, scene_id, capture_id, source_type, source,
            camera_model=camera_model, camera_make=camera_make, fps=fps,
            start_seconds=start_seconds, end_seconds=end_seconds,
            output_width=output_width, output_height=output_height,
            horizontal_fov=horizontal_fov,
        )
        console.print(
            f"Created [green]{location_id}/{scene_id}/captures/{capture_id}.yaml[/green]; "
            f"source={source_type}; files={len(created.source.files)}"
        )
    except Exception as error:
        _fatal(error)


@capture_app.command("relink")
def capture_relink(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
    source: Annotated[list[Path], typer.Option("--source", help="New path for each original source, in recorded order")],
) -> None:
    """Relocate external source files only after exact size and hash checks."""
    try:
        result = ops.relink_capture_sources(
            _data_root(), location_id, scene_id, capture_id, source
        )
        console.print(f"Relinked [green]{result.id}[/green]; files={len(result.source.files)}")
    except Exception as error:
        _fatal(error)


@media_app.command("probe")
def media_probe(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
) -> None:
    """Probe a capture without preparing frames or changing its manifest."""
    try:
        result = ops.probe_capture(_data_root(), location_id, scene_id, capture_id)
        console.print_json(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@location_app.command("init")
def location_init(
    location_id: Annotated[str, typer.Argument(help="Stable ASCII slug")],
    display_name: Annotated[str, typer.Option("--name", help="Human-readable name")],
    capture_date: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    province: Annotated[str | None, typer.Option()] = None,
    city: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Create a location manifest without any media files."""
    try:
        ops.create_location(_data_root(), location_id, display_name, capture_date, province, city)
        console.print(f"Created [green]{location_id}/location.yaml[/green]")
    except Exception as error:
        _fatal(error)


@scene_app.command("init")
def scene_init(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument(help="Stable ASCII slug")],
    display_name: Annotated[str, typer.Option("--name")],
    description: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Create an independently reconstructable scene under a location."""
    try:
        ops.create_scene(_data_root(), location_id, scene_id, display_name, description)
        console.print(f"Created [green]{location_id}/{scene_id}/scene.yaml[/green]")
    except Exception as error:
        _fatal(error)


@app.command()
def ingest(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
) -> None:
    """Re-probe the registered source set and refresh its recorded media facts."""
    try:
        result = ops.ingest(_data_root(), location_id, scene_id, capture_id)
        probe = result.source.probe
        console.print(
            f"Ingested [green]{probe.width}x{probe.height}[/green] "
            f"{probe.duration_seconds:.2f}s; files={len(result.source.files)}"
        )
    except Exception as error:
        _fatal(error)


@app.command()
def preprocess(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    capture_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str | None, typer.Option(help="Existing run ID when resuming")] = None,
    resume: Annotated[bool, typer.Option(help="Reuse completed work in the same run")] = False,
    candidate_fps: Annotated[float, typer.Option(help="Candidate frame sampling rate")] = 1.0,
    selected_per_second: Annotated[
        int, typer.Option(help="Best candidates retained in each one-second bucket")
    ] = 1,
    primary_per_second: Annotated[
        int, typer.Option(help="Temporal ranks used by the primary attempt")
    ] = 1,
    primary_fov: Annotated[
        float, typer.Option(help="Horizontal FOV of each projected view in degrees")
    ] = 110.0,
    primary_projection_size: Annotated[int, typer.Option(help="Square projected view size")] = 1746,
    reconstruction_backend: Annotated[str, typer.Option(help="realityscan (default) or colmap")] = "realityscan",
    keep_intermediates: Annotated[bool, typer.Option(help="Keep intermediate files for debugging")] = False,
    mask_device: Annotated[str, typer.Option(help="Person segmenter device: cuda or cpu")] = "cuda",
    mask_score_threshold: Annotated[
        float, typer.Option(help="Mask R-CNN person confidence threshold")
    ] = 0.25,
    mask_probability_threshold: Annotated[
        float, typer.Option(help="Per-pixel probability threshold inside a person instance")
    ] = 0.50,
    mask_gamma: Annotated[
        float, typer.Option(help="Gamma used only for night-time segmentation inference")
    ] = 0.75,
    mask_dilation_pixels: Annotated[
        int, typer.Option(help="Pixel radius added around detected people")
    ] = 24,
    mask_discard_threshold: Annotated[
        float, typer.Option(help="Delete a perspective view when its ignored fraction exceeds this value")
    ] = 0.005,
    mask_review_gate: Annotated[
        bool,
        typer.Option("--mask-review-gate/--no-mask-review-gate", help="Require human mask finalization"),
    ] = False,
    mask_qa_sample_count: Annotated[
        int, typer.Option(help="Perspective views sampled for person-mask QA")
    ] = 16,
    mask_classes: Annotated[
        str,
        typer.Option(help="Comma-separated COCO dynamic classes; default person"),
    ] = "person",
    loop_closure: Annotated[
        bool, typer.Option("--loop-closure/--no-loop-closure")
    ] = False,
    vocabulary_tree: Annotated[
        Path | None,
        typer.Option(help="Explicit local COLMAP vocabulary tree; never downloaded"),
    ] = None,
    loop_period: Annotated[int, typer.Option()] = 20,
    loop_num_images: Annotated[int, typer.Option()] = 5,
    vision_qa: Annotated[
        bool,
        typer.Option(
            "--vision-qa/--no-vision-qa",
            help="Gate masks through DeepSeek V4 Flash Vision",
        ),
    ] = False,
    smoke: Annotated[
        bool,
        typer.Option(
            "--smoke",
            help=(
                "Process only the first 5 seconds at the same 1/1 temporal density, "
                "with smaller projections; does not start training"
            ),
        ),
    ] = False,
) -> None:
    """Create a current-schema Run or resume its prepared input."""
    try:
        if run_id:
            ops.resume_preprocess(_data_root(), location_id, scene_id, run_id, resume=resume)
            result_id = run_id
        else:
            result_id = ops.create_and_preprocess(
                _data_root(), location_id, scene_id, capture_id,
                candidate_fps=candidate_fps,
                selected_per_second=selected_per_second,
                primary_per_second=primary_per_second,
                projection_fov=primary_fov,
                projection_size=primary_projection_size,
                reconstruction_backend=reconstruction_backend,
                mask_review_gate=mask_review_gate,
                keep_intermediates=keep_intermediates,
                mask_device=mask_device,
                mask_score_threshold=mask_score_threshold,
                mask_probability_threshold=mask_probability_threshold,
                mask_gamma=mask_gamma,
                mask_dilation_pixels=mask_dilation_pixels,
                mask_discard_threshold=mask_discard_threshold,
                mask_qa_sample_count=mask_qa_sample_count,
                mask_classes=mask_classes,
                loop_closure=loop_closure,
                vocabulary_tree=vocabulary_tree,
                loop_period=loop_period,
                loop_num_images=loop_num_images,
                vision_qa=vision_qa,
                smoke=smoke,
            )
        run = load_run(_scene(location_id, scene_id), result_id)
        console.print(f"Run [green]{result_id}[/green]: preprocess={run.stages['preprocess'].status.value}")
        console.print(f"RUN_ID={result_id}", markup=False)
    except Exception as error:
        _fatal(error)


@app.command("mask")
def mask_command(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Project panoramas, segment people, and create the alignment masks."""
    try:
        run = ops.run_stage(_data_root(), location_id, scene_id, run_id, "mask", resume=resume)
        metrics = run.metrics.get("masking", {}).get("primary", {})
        console.print(
            f"Run [green]{run.id}[/green]: mask={run.stages['mask'].status.value}; "
            f"views={metrics.get('planar_images', 0)}"
        )
    except Exception as error:
        _fatal(error)


@app.command()
def reconstruct(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Align masked images with RealityScan and apply the single bounded repair when required."""
    try:
        run = ops.run_stage(_data_root(), location_id, scene_id, run_id, "reconstruct", resume=resume)
        console.print(f"Run [green]{run.id}[/green]: selected={run.selected_dataset}")
    except Exception as error:
        _fatal(error)


def _mask_attempt_path(path: Path, run_id: str, attempt: str) -> Path:
    if attempt not in {"primary", "repair"}:
        raise ValueError("Attempt must be primary or repair")
    dataset = path / run_id / f"reconstruction-{attempt}"
    if not (dataset / "images").is_dir() or not (dataset / "masks").is_dir():
        raise FileNotFoundError(f"Masked {attempt} dataset is missing: {dataset}")
    return dataset


@app.command("mask-review")
@run_cli_locked
def mask_review(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str, typer.Option(help="primary or repair")] = "primary",
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8765,
) -> None:
    """Review reconstruction masks; labels are bound to each current mask hash."""
    server = None
    try:
        path = _scene(location_id, scene_id)
        dataset = _mask_attempt_path(path, run_id, attempt)
        server = create_mask_review_server(dataset, host=host, port=port)
        actual_host, actual_port = server.server_address[:2]
        browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
        console.print(
            f"Mask review: [green]http://{browser_host}:{actual_port}[/green]\n"
            f"Dataset: {dataset.absolute()}\nPress Ctrl+C to stop."
        )
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            console.print("\nMask review stopped.")
    except Exception as error:
        _fatal(error)
    finally:
        if server is not None:
            server.server_close()


@app.command("mask-finalize")
@run_cli_locked
def mask_finalize(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str, typer.Option(help="primary or repair")] = "primary",
) -> None:
    """Freeze the reviewed mask set and exclusion list for reconstruction."""
    try:
        result = ops.finalize_masks(_data_root(), location_id, scene_id, run_id, attempt)
        console.print(
            f"Mask final [green]passed[/green]: masks={result['counts']['masks']}; "
            f"excluded={result['counts']['excluded']}; "
            f"sha256={result['finalization_sha256']}"
        )
    except Exception as error:
        _fatal(error)


@app.command("postshot-review")
def postshot_review(
    dataset: Annotated[
        Path,
        typer.Option(help="Prepared Postshot input directory, e.g. .../postshot-input"),
    ],
    host: Annotated[
        str, typer.Option(help="Review server bind address; keep loopback for local use")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=0, max=65535)] = 8765,
) -> None:
    """Review Postshot images and occluder masks in a local web interface."""
    server = None
    try:
        target = host_path(dataset)
        server = create_mask_review_server(target, host=host, port=port)
        actual_host, actual_port = server.server_address[:2]
        browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
        console.print(
            f"Mask review: [green]http://{browser_host}:{actual_port}[/green]\n"
            f"Dataset: {target.absolute()}\nPress Ctrl+C to stop."
        )
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            console.print("\nMask review stopped.")
    except Exception as error:
        _fatal(error)
    finally:
        if server is not None:
            server.server_close()


@qa_app.command("report")
def qa_report(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    """Generate the review checklist and move the run to needs_review."""
    try:
        report = ops.run_stage(_data_root(), location_id, scene_id, run_id, "qa", resume=resume)
        console.print(f"QA report: [green]{report}[/green]")
    except Exception as error:
        _fatal(error)


@qa_app.command("approve")
def qa_approve(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    notes: Annotated[str, typer.Option(help="Required human review note")],
) -> None:
    """Mark a human-reviewed artifact accepted."""
    try:
        run = ops.review_qa(_data_root(), location_id, scene_id, run_id, True, notes)
        console.print(f"Run [green]{run.id}[/green] accepted")
    except Exception as error:
        _fatal(error)


@qa_app.command("reject")
def qa_reject(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    notes: Annotated[str, typer.Option(help="Required rejection reason")],
) -> None:
    """Reject a human-reviewed artifact without deleting it."""
    try:
        run = ops.review_qa(_data_root(), location_id, scene_id, run_id, False, notes)
        console.print(f"Run [yellow]{run.id}[/yellow] rejected")
    except Exception as error:
        _fatal(error)


@catalog_app.command("build")
def catalog_build() -> None:
    """Atomically rebuild catalog/catalog.sqlite from YAML manifests."""
    try:
        root = _data_root()
        counts = build_catalog(root)
        console.print("Catalog rebuilt: " + json.dumps(counts, ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@qa_app.command("segments")
def qa_segments(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    attempt: Annotated[str | None, typer.Option(help="primary or repair; defaults to the selected attempt")] = None,
    output: Annotated[Path | None, typer.Option(help="Report path; defaults to the dataset")] = None,
) -> None:
    """Recompute auditable segment QA without changing reconstruction stage status."""
    from gsstudio.pipeline.quality.segments import write_segments
    from gsstudio.pipeline.masks.masking import validate_mask_filter
    from gsstudio.pipeline.masks.finalize import validate_mask_finalization
    try:
        scene=_scene(location_id,scene_id)
        run=load_run(scene,run_id)
        label=attempt or run.metrics.get('selected_attempt','primary')
        if label not in ('primary','repair'):
            raise ValueError('Unknown reconstruction attempt')
        work=scene/run_id
        dataset=work/f'reconstruction-{label}'
        records=[json.loads(line) for line in (work/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if line]
        filtered=validate_mask_filter(dataset,verify_hashes=True)
        included={r['image'] for r in filtered['accepted']}
        final=validate_mask_finalization(dataset,included)
        included-=set(final.get('excluded_images',[]))
        report=write_segments(dataset,records,included,run.config.reconstruction.primary,run.config.segment_qa,output)
        console.print(json.dumps({k:report[k] for k in ('integrity','training_status','coverage_status')},ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@app.command("train")
@run_cli_locked
def train_segment(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    segment: Annotated[str, typer.Option(help="Passed segment ID, e.g. segment-001")],
    output: Annotated[Path, typer.Option(help="New training experiment directory")],
    backend: Annotated[str, typer.Option(help="gsplat (independent GS Studio engine)")] = "gsplat",
    steps: Annotated[int | None, typer.Option(min=1)] = None,
    photo_comp: Annotated[bool | None, typer.Option("--photo-comp/--no-photo-comp", help="Photometric master enable; override source preset")] = None,
    use_bilateral_grid: Annotated[bool | None, typer.Option("--use-bilateral-grid/--no-use-bilateral-grid", help="Select Bilateral Grid; requires photometric master enabled")] = None,
    use_sparse_depth: Annotated[bool | None, typer.Option("--use-sparse-depth/--no-use-sparse-depth", help="Override preset; effective only with validated anchors")] = None,
    resume: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option(help="Prepare validated input and command without training")] = False,
    training_preset: Annotated[str,typer.Option(help='auto (source-aware), photo or gs')] = 'auto',
    training_recipe: Annotated[str,typer.Option(help='quality_v2; observed_v5 released; observed_v6 experimental')] = 'quality_v2',
    preview_every: Annotated[int,typer.Option(min=0,help='Paired source/render preview interval')] = 500,
    restart_from_step_zero: Annotated[bool,typer.Option(help='Explicit retry without a checkpoint; requires --resume')] = False,
    duration_seconds: Annotated[float | None, typer.Option(min=0.001,help="Earliest fully validated window within the segment")] = None,
) -> None:
    """Train a QA-approved segment with the independent GS Studio engine."""
    try:
        result = ops.train_segment(
            _data_root(), location_id, scene_id, run_id, segment,
            backend=backend, steps=steps, resume=resume, output=output,
            photo_comp=photo_comp, use_bilateral_grid=use_bilateral_grid,
            use_sparse_depth=use_sparse_depth, dry_run=dry_run,
            duration_seconds=duration_seconds,
            training_recipe=training_recipe,training_preset=training_preset,preview_every=preview_every,
            restart_from_step_zero=restart_from_step_zero,
        )
        console.print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        _fatal(error)


@app.command("train-control")
def train_control(
    output: Annotated[Path, typer.Argument(help="Existing gsplat experiment directory")],
    action: Annotated[str, typer.Argument(help="pause (save/release), suspend, unpause, checkpoint or stop")],
) -> None:
    """Request a safe step-boundary action in an active gsplat experiment."""
    try:
        result = ops.control_training(_data_root(), output, action)
        console.print_json(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        _fatal(error)


@app.command('train-options')
def train_options(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    segment: Annotated[str, typer.Option()],
    training_preset: Annotated[str, typer.Option(help='auto, photo or gs')] = 'auto',
    training_recipe: Annotated[str, typer.Option(help='quality_v2; observed_v5 released; observed_v6 experimental')] = 'quality_v2',
    duration_seconds: Annotated[float | None, typer.Option(min=0.001)] = None,
) -> None:
    """Prepare CPU input and report actual defaults/depth availability before training."""
    try:
        from gsstudio.application.training_options import training_options
        console.print_json(json.dumps(training_options(_data_root(), location_id=location_id,
            scene_id=scene_id, run_id=run_id, segment=segment, training_preset=training_preset,
            training_recipe=training_recipe, duration_seconds=duration_seconds)))
    except Exception as error:
        _fatal(error)


@app.command("cleanup")
def cleanup_command(
    location_id: Annotated[str, typer.Argument()],
    scene_id: Annotated[str, typer.Argument()],
    run_id: Annotated[str, typer.Argument()],
    apply: Annotated[bool, typer.Option(help="Execute verified cleanup; default is preview")] = False,
) -> None:
    """Preview or apply verified retention cleanup for a Run."""
    try:
        console.print_json(json.dumps(ops.cleanup(_data_root(), location_id, scene_id, run_id, apply=apply)))
    except Exception as error:
        _fatal(error)


@app.command("select-sharp")
def select_sharp(
    source: Annotated[list[Path], typer.Option("--source", help="Source media file; repeatable")],
    out: Annotated[Path, typer.Option(help="Output directory for the filtered frames")],
    kind: Annotated[
        str,
        typer.Option(help="equirect_video, equirect_sequence, insta360_insv, perspective_video, or perspective_images"),
    ] = "perspective_video",
    stage: Annotated[
        str,
        typer.Option(help="probe, extract, select, project, mask, filter, or all"),
    ] = "all",
    window: Annotated[
        str, typer.Option(help="Time window as start..end seconds; default is the whole source")
    ] = "",
    candidate_fps: Annotated[float, typer.Option(help="Candidate frame sampling rate")] = 5.0,
    selected_per_second: Annotated[int, typer.Option(help="Frames kept in each one-second bucket")] = 1,
    mask_threshold: Annotated[
        float, typer.Option(help="Reject a view when its ignored fraction exceeds this value")
    ] = 0.01,
    views: Annotated[int, typer.Option(help="Projected views per panorama: 8 or 14")] = 14,
    view_fov: Annotated[float, typer.Option(help="Horizontal FOV of each projected view")] = 110.0,
    view_size: Annotated[int, typer.Option(help="Square projected view size")] = 1746,
    crop_bottom: Annotated[float, typer.Option(help="Fraction of the panorama cropped from below")] = 0.15,
    target: Annotated[
        str,
        typer.Option(help="Image set for mask/filter: auto, flat, planar, or all"),
    ] = "auto",
    keep_candidates: Annotated[bool, typer.Option(help="Keep the extracted candidates")] = False,
    jpeg_quality: Annotated[int, typer.Option(min=1, max=100)] = 95,
    device: Annotated[str, typer.Option(help="Person segmenter device: cuda or cpu")] = "cuda",
) -> None:
    """Extract the sharpest frames from explicit sources and drop person-heavy views.

    A workbench tool outside the Run pipeline: no capture manifest, no run directory,
    no training. Interrupted work resumes by repeating the same command. Not a
    quality acceptance path.
    """
    try:
        run_select_sharp(
            sources=[host_path(item) for item in source],
            out=host_path(out),
            kind=kind,
            stage=stage,
            window=window,
            candidate_fps=candidate_fps,
            selected_per_second=selected_per_second,
            mask_threshold=mask_threshold,
            views=views,
            view_fov=view_fov,
            view_size=view_size,
            crop_bottom=crop_bottom,
            target=target,
            keep_candidates=keep_candidates,
            jpeg_quality=jpeg_quality,
            device=device,
        )
    except Exception as error:
        _fatal(error)


if __name__ == "__main__":
    app()
