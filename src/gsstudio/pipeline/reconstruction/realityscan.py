"""RealityScan alignment result conversion and Run integration."""
from __future__ import annotations
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any
from gsstudio.domain.models import ReconstructionAttempt, ReconstructionConfig
from gsstudio.infrastructure.paths import host_path
from gsstudio.infrastructure.runtime.processes import run_logged
from gsstudio.infrastructure.adapters.realityscan import (
    ReconstructionAttemptSettings, ReconstructionSettings,
    build_realityscan_align_command, export_params_file, realityscan_executable,
)
from gsstudio.pipeline.masks.masking import image_files
from gsstudio.pipeline.reconstruction.core import (
    _recorded_cross_view_pairs, _selected_model_dir, _write_colmap_binary_model,
    attach_frame_masks, validate_frame_masks,
)


def convert_colmap_text_to_binary(
    text_dir: Path,
    output_dir: Path,
    *,
    image_name_map: dict[str, str] | None = None,
) -> None:
    """Convert RealityScan's exported COLMAP Text Format into the binary triplet
    postshot.py and the rest of gsstudio's pipeline already consume.

    RealityScan's own internal model of "a COLMAP file" is the .txt triplet --
    -exportRegistration has no binary-COLMAP output option, and there is no
    need to invoke `model_converter`. This uses the locally vendored COLMAP
    parser from the accepted Nerfstudio baseline, whose Camera/Image/Point3D
    namedtuples already match
    reconstruction._write_colmap_binary_model()'s expected input shape exactly.
    """
    from gsstudio._vendor.nerfstudio.colmap_parsing_utils import (
        read_cameras_text,
        read_images_text,
        read_points3D_text,
    )

    cameras = read_cameras_text(text_dir / "cameras.txt")
    images = read_images_text(text_dir / "images.txt")
    points_path = text_dir / "points3D.txt"
    points = read_points3D_text(points_path) if points_path.is_file() else {}
    if image_name_map is not None:
        remapped_images = {}
        for image_id, image in images.items():
            exported_name = image.name.replace("\\", "/")
            original_name = image_name_map.get(exported_name)
            if original_name is None:
                raise RuntimeError(
                    "RealityScan exported an image outside its staged input map: "
                    f"{exported_name}"
                )
            remapped_images[image_id] = image._replace(name=original_name)
        images = remapped_images
    for camera in cameras.values():
        if camera.model != "PINHOLE" or len(camera.params) != 4:
            raise RuntimeError(
                "RealityScan exported a non-PINHOLE camera model "
                f"({camera.model}, {len(camera.params)} params); gsstudio's reprojected "
                "views are distortion-free by construction (equirect2persp is a pure "
                "pinhole projection), so a non-PINHOLE result means either the export "
                "settings need adjusting or RealityScan estimated real distortion that "
                "needs investigating -- not something to silently coerce."
            )
    _write_colmap_binary_model(output_dir, cameras, images, points)


def realityscan_reconstruction_metrics(
    dataset: Path,
    attempt: ReconstructionAttemptSettings,
    settings: ReconstructionSettings | None = None,
    *,
    included_images: set[str],
    projected_image_count: int | None = None,
) -> dict[str, Any]:
    """Measure the single maximal component exported by RealityScan."""
    transforms_path = dataset / "transforms.json"
    if not transforms_path.is_file():
        raise RuntimeError(f"RealityScan conversion did not create {transforms_path}")
    try:
        selected_registered = validate_frame_masks(transforms_path, dataset / "masks")
    except (OSError, RuntimeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(f"RealityScan frame-mask validation failed: {error}") from error

    selection_path = dataset / "colmap" / "selected-attempt.json"
    if not selection_path.is_file():
        raise RuntimeError(f"RealityScan model selection is missing: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model = _selected_model_dir(dataset)

    from gsstudio._vendor.nerfstudio.colmap_parsing_utils import read_images_binary

    registered_model = read_images_binary(model / "images.bin")
    registered_names = {image.name.replace("\\", "/") for image in registered_model.values()}
    registered = len(registered_names)
    normalized_included = {name.replace("\\", "/") for name in included_images}
    unexpected = registered_names - normalized_included
    if unexpected:
        raise RuntimeError(
            "RealityScan registered images outside the final inclusion manifest: "
            + ", ".join(sorted(unexpected)[:5])
        )
    expected = len(normalized_included)
    written = (
        int(projected_image_count)
        if projected_image_count is not None
        else len(image_files(dataset / "images"))
    )
    excluded_count = max(0, written - expected)
    registration_ratio = registered / expected if expected else 0.0
    return {
        "expected_planar_images": expected,
        "written_planar_images": written,
        "excluded_images": excluded_count,
        "registered_images": registered,
        "selected_component_images": selected_registered,
        "registration_ratio": registration_ratio,
        "largest_component_coverage": registration_ratio,
        "component_count": 1 if registered else 0,
        "component_sizes": [registered] if registered else [],
        "sift_gpu": bool(getattr(settings, "use_gpu_sift", False)),
        "fixed_intrinsics": bool(getattr(settings, "fix_intrinsics", False)),
        "cross_view_pairs": _recorded_cross_view_pairs(dataset),
        "rig": selection.get("rig", {"enabled": False}),
    }


def run_realityscan_alignment(
    dataset: Path,
    attempt: ReconstructionAttemptSettings,
    log_dir: Path,
    settings: ReconstructionSettings | None = None,
    *,
    included_images: set[str],
    projected_image_count: int | None = None,
) -> dict[str, Any]:
    """Align the masked views with RealityScan and export a binary COLMAP model.

    Returns the usual registration metrics, lands a valid model under
    ``dataset/colmap`` (recorded via ``selected-attempt.json``, which
    ``_selected_model_dir()`` parses) and produces ``dataset/transforms.json``
    through the vendored ``colmap_to_json``. ``included_images`` is the frozen
    input inventory the alignment must not exceed.
    """
    transforms = dataset / "transforms.json"
    alignment_masks = bool(getattr(settings, "alignment_masks", False))
    if transforms.is_file():
        try:
            existing_metrics = realityscan_reconstruction_metrics(
                dataset,
                attempt,
                settings,
                included_images=included_images,
                projected_image_count=projected_image_count,
            )
            threshold = float(getattr(settings, "registration_threshold", 0.70))
            mask_audit = dataset / "alignment-masks.json"
            mask_compatible = not alignment_masks
            if alignment_masks and mask_audit.is_file():
                from gsstudio.infrastructure.adapters.media import sha256_file
                audit = json.loads(mask_audit.read_text(encoding="utf-8"))
                audit_records = audit.get('records',[])
                mask_compatible = audit.get('completed') is True and {r['image'] for r in audit_records} == set(included_images)
                if mask_compatible:
                    mask_compatible = all(sha256_file(dataset/'masks'/f"{r['image']}.png") == r['mask_sha256'] and
                                          sha256_file(dataset/'images'/r['image']) == r['image_sha256'] for r in audit_records)
            if mask_compatible and (alignment_masks or (
                min(
                    existing_metrics["registration_ratio"],
                    existing_metrics["largest_component_coverage"],
                )
                >= threshold
            )):
                return existing_metrics
        except (RuntimeError, json.JSONDecodeError, KeyError, TypeError):
            # Reuse the completed alignment below and regenerate transforms/masks
            # rather than trusting a possibly-partial previous run.
            pass

    colmap_root = dataset / "colmap"
    log_dir.mkdir(parents=True, exist_ok=True)
    colmap_root.mkdir(parents=True, exist_ok=True)

    export_dir = colmap_root / "realityscan-export"
    # Never allow files from a failed export to satisfy a later resume check.
    shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset / "images"
    staged_images_dir: Path | None = None
    image_name_map: dict[str, str] | None = None
    if included_images is not None:
        available = {
            path.relative_to(images_dir).as_posix(): path
            for path in image_files(images_dir)
        }
        normalized_included = {name.replace("\\", "/") for name in included_images}
        missing = normalized_included - set(available)
        if missing:
            raise RuntimeError(
                "Final RealityScan inclusion manifest references missing images: "
                + ", ".join(sorted(missing)[:5])
            )
        # RealityScan's COLMAP writer drops input subdirectories and records only
        # basenames.  Projection trees deliberately repeat frame filenames under
        # view_XX/, so feed RealityScan unique flat names and restore the original
        # relative names immediately after export.
        staged_images_dir = dataset / "alignment-images"
        shutil.rmtree(staged_images_dir, ignore_errors=True)
        staged_images_dir.mkdir(parents=True, exist_ok=True)
        image_name_map = {}
        mask_records = []
        for index, name in enumerate(sorted(normalized_included)):
            suffix = available[name].suffix.lower() or ".jpg"
            staged_name = f"gsdb_{index:06d}{suffix}"
            destination = staged_images_dir / staged_name
            try:
                os.link(available[name], destination)
            except OSError as error:
                raise RuntimeError(
                    "RealityScan's unique input tree requires same-volume hardlinks; "
                    f"could not link {name}: {error}"
                ) from error
            image_name_map[staged_name] = name
            if alignment_masks:
                from gsstudio.pipeline.masks.masking import mask_path_for_image, image_dimensions
                from gsstudio.infrastructure.adapters.media import sha256_file
                import cv2
                from gsstudio.infrastructure.adapters.image_io import read_cv_image
                import numpy as np
                source_mask = mask_path_for_image(dataset / "masks", available[name])
                pixels = read_cv_image(str(source_mask), cv2.IMREAD_GRAYSCALE)
                if pixels is None or pixels.shape != image_dimensions(available[name]):
                    raise RuntimeError(f"Invalid alignment mask: {source_mask}")
                if not np.all((pixels == 0) | (pixels == 255)):
                    raise RuntimeError(f"Nonbinary alignment mask: {source_mask}")
                target_mask = staged_images_dir / f"{staged_name}.mask.png"
                shutil.copy2(source_mask, target_mask)
                mask_records.append({"image": name, "staged_image": staged_name,
                                     "mask_sha256": sha256_file(source_mask), "image_sha256": sha256_file(available[name])})
        images_dir = staged_images_dir

    command = build_realityscan_align_command(
        realityscan_executable(),
        images_dir,
        log_dir / "crash-reports",
        # The first argument is the export job's chosen filename, not the
        # COLMAP triplet's images.txt member.  Using that reserved member name
        # causes RealityScan 2.2 to omit it while still reporting success.
        export_dir / "registration.txt",
        colmap_root / "project.rsproj",
        projection_fov_degrees=float(attempt.projection_fov_degrees),
        fix_intrinsics=bool(getattr(settings, "fix_intrinsics", True)),
        **({"alignment_masks": True} if alignment_masks else {}),
    )
    if alignment_masks:
        (dataset / "alignment-masks.json").write_text(json.dumps({
            "completed": False, "polarity": "white_keep", "inpMaskOpts": 1,
            "records": mask_records,
        }, indent=2), encoding="utf-8")
    run_logged(command, log_dir / "realityscan-align.log")

    if not (export_dir / "images.txt").is_file():
        raise RuntimeError(
            f"RealityScan did not produce a registration export in {export_dir}; "
            f"see {log_dir / 'realityscan-align.log'}"
        )
    convert_colmap_text_to_binary(
        export_dir,
        colmap_root,
        image_name_map=image_name_map,
    )
    shutil.rmtree(export_dir, ignore_errors=True)
    (colmap_root / ".mapping-complete").write_text("complete\n", encoding="utf-8")

    from gsstudio._vendor.nerfstudio.colmap_utils import colmap_to_json

    colmap_to_json(colmap_root, dataset, use_single_camera_mode=False)
    attach_frame_masks(transforms, dataset / "masks")

    (colmap_root / "selected-attempt.json").write_text(
        json.dumps(
            {
                "model": ".",
                "rig": {"enabled": False, "backend": "realityscan"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (colmap_root / ".conversion-complete").write_text("complete\n", encoding="utf-8")
    if alignment_masks:
        audit_path = dataset / "alignment-masks.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["completed"] = True
        audit["verification"] = "paired_binary_masks_staged_and_alignment_command_enabled"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metrics = realityscan_reconstruction_metrics(
        dataset,
        attempt,
        settings,
        included_images=included_images,
        projected_image_count=projected_image_count,
    )
    if staged_images_dir is not None:
        shutil.rmtree(staged_images_dir, ignore_errors=True)
    return metrics
