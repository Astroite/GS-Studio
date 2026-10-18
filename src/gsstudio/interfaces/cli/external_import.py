"""CLI adapter for the same independent import operations as the Qt workbench."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from gsstudio.application import external_imports
from gsstudio.application.errors import classify
from gsstudio.interfaces.cli.paths import data_root


app = typer.Typer(no_args_is_help=True, help='Inspect and train an external COLMAP source')


def _source_options(package: Path | None, images: Path | None, model: Path | None,
                    masks: Path | None, groups: Path | None, independent: bool,
                    white_ignore: bool) -> dict:
    return {'source': package, 'images': images, 'model': model, 'masks': masks,
            'group_file': groups, 'independent_images': independent,
            'white_ignore': white_ignore}


def _show(call) -> None:
    try:
        print(json.dumps(call(), ensure_ascii=False, indent=2))
    except Exception as error:
        classified = classify(error)
        typer.echo(f'Error[{classified.code}]: {classified.message}', err=True)
        raise typer.Exit(1) from error


@app.command('inspect')
def inspect(
    package: Annotated[Path | None, typer.Option(help='Existing verified dataset.json package')] = None,
    images: Annotated[Path | None, typer.Option(help='External source images directory')] = None,
    model: Annotated[Path | None, typer.Option(help='Selected COLMAP BIN/TXT model directory')] = None,
    masks: Annotated[Path | None, typer.Option(help='Optional binary mask directory')] = None,
    groups: Annotated[Path | None, typer.Option(help='JSON image-name to group-ID mapping')] = None,
    independent: Annotated[bool, typer.Option(help='Explicitly confirm every image is its own group')] = False,
    white_ignore: Annotated[bool, typer.Option(help='Treat white pixels in source masks as ignored')] = False,
) -> None:
    """Read-only inspection, including grouping and image/model problems."""
    options = _source_options(package, images, model, masks, groups, independent, white_ignore)
    _show(lambda: external_imports.inspect_external_import(**options))


@app.command('prepare')
def prepare(
    import_id: Annotated[str, typer.Argument(help='New external project ID')],
    package: Annotated[Path | None, typer.Option(help='Existing verified dataset.json package')] = None,
    images: Annotated[Path | None, typer.Option(help='External source images directory')] = None,
    model: Annotated[Path | None, typer.Option(help='Selected COLMAP BIN/TXT model directory')] = None,
    masks: Annotated[Path | None, typer.Option(help='Optional binary mask directory')] = None,
    groups: Annotated[Path | None, typer.Option(help='JSON image-name to group-ID mapping')] = None,
    independent: Annotated[bool, typer.Option(help='Explicitly confirm every image is its own group')] = False,
    white_ignore: Annotated[bool, typer.Option(help='Treat white pixels in source masks as ignored')] = False,
    expected_input_identity: Annotated[str | None, typer.Option(help='Identity returned by inspect')] = None,
) -> None:
    """Prepare a hashed package under Data/.external-imports, outside Runs."""
    options = _source_options(package, images, model, masks, groups, independent, white_ignore)
    _show(lambda: external_imports.prepare_external_import(
        data_root(), import_id, expected_input_identity=expected_input_identity, **options))


@app.command('train')
def train(
    import_id: Annotated[str, typer.Argument()],
    backend: Annotated[str, typer.Option(help='gsplat (independent GS Studio engine)')] = 'gsplat',
    steps: Annotated[int | None, typer.Option(min=1)] = None,
    resume: Annotated[bool, typer.Option()] = False,
    experiment_id: Annotated[str | None, typer.Option(help='Existing failed experiment for resume')] = None,
    dry_run: Annotated[bool, typer.Option(help='Prepare command and inputs only')] = False,
    photo_comp: Annotated[bool | None,typer.Option("--photo-comp/--no-photo-comp", help="Photometric master enable; override source preset")] = None,
    use_bilateral_grid: Annotated[bool | None,typer.Option("--use-bilateral-grid/--no-use-bilateral-grid", help="Select Bilateral Grid; requires photometric master enabled")] = None,
    use_sparse_depth: Annotated[bool | None,typer.Option("--use-sparse-depth/--no-use-sparse-depth", help="Override preset; effective only with validated anchors")] = None,
    training_preset: Annotated[str,typer.Option(help='auto (source-aware), photo or gs')] = 'auto',
    training_recipe: Annotated[str,typer.Option(help='quality_v2; observed_v5 released; observed_v6 experimental')] = 'quality_v2',
    preview_every: Annotated[int,typer.Option(min=0)] = 500,
    restart_from_step_zero: Annotated[bool,typer.Option(help='Explicit retry without checkpoint; requires resume')] = False,
) -> None:
    """Train an independent external import without a Run QA assertion."""
    _show(lambda: external_imports.train_external_import(
        data_root(), import_id, backend=backend, steps=steps, resume=resume,
        experiment_id=experiment_id, dry_run=dry_run,photo_comp=photo_comp,
        use_bilateral_grid=use_bilateral_grid,use_sparse_depth=use_sparse_depth,
        training_recipe=training_recipe,training_preset=training_preset,preview_every=preview_every,restart_from_step_zero=restart_from_step_zero))


@app.command('relink')
def relink(
    import_id: Annotated[str, typer.Argument()],
    new_root: Annotated[Path | None, typer.Option(help='Directory containing the relocated source tree')] = None,
    mapping_file: Annotated[Path | None, typer.Option(help='JSON original-absolute-path to current-absolute-path mapping')] = None,
) -> None:
    """Relocate read-only source references after exact per-file hash checks."""
    _show(lambda: external_imports.relink_external_import(
        data_root(), import_id, new_root=new_root, mapping_file=mapping_file))


@app.command('options')
def options(
    import_id: Annotated[str, typer.Argument()],
    training_preset: Annotated[str, typer.Option(help='auto, photo or gs')] = 'auto',
    training_recipe: Annotated[str, typer.Option(help='quality_v2; observed_v5 released; observed_v6 experimental')] = 'quality_v2',
) -> None:
    """Prepare CPU input and report preset/depth availability without training."""
    from gsstudio.application.training_options import training_options
    _show(lambda: training_options(data_root(), import_id=import_id,
          training_preset=training_preset, training_recipe=training_recipe))


@app.command('list')
def list_imports() -> None:
    _show(lambda: external_imports.list_external_imports(data_root()))
