"""RealityScan executable and command-line adapter."""
from __future__ import annotations
import os
import math
from pathlib import Path
from gsstudio.domain.models import ReconstructionAttempt, ReconstructionConfig
from gsstudio.infrastructure.paths import find_app_root, host_path

REALITYSCAN_CLI_ENV = "GSSTUDIO_REALITYSCAN_CLI"
REALITYSCAN_EXPORT_PARAMS_ENV = "GSSTUDIO_REALITYSCAN_EXPORT_PARAMS"
ReconstructionSettings = ReconstructionConfig
ReconstructionAttemptSettings = ReconstructionAttempt


def realityscan_executable() -> Path:
    from gsstudio.infrastructure.runtime.configuration import runtime_config
    configured = os.environ.get(REALITYSCAN_CLI_ENV, "").strip() or runtime_config().get('realityscan')
    if configured and host_path(configured).is_file():
        return host_path(configured)
    raise RuntimeError("RealityScan is unavailable; configure an explicit realityscan path in runtime override JSON or GSSTUDIO_REALITYSCAN_CLI")


def _default_export_params_path() -> Path:
    return find_app_root() / "tools" / "realityscan-setup" / "colmap-export-params.xml"


def export_params_file() -> Path:
    """The one-time, GUI-exported 'COLMAP' registration-export settings file.

    See this module's docstring: -exportRegistration has no CLI-native way to
    generate this. Checked first as an env var override (for local testing
    against a not-yet-committed file), then the committed default asset.
    """
    from gsstudio.infrastructure.runtime.configuration import runtime_config
    configured = os.environ.get(REALITYSCAN_EXPORT_PARAMS_ENV, "").strip() or runtime_config().get("realityscan_export_params")
    candidate = host_path(configured) if configured else _default_export_params_path()
    if not candidate.is_file():
        raise RuntimeError(
            f"RealityScan export params file is missing: {candidate}. Generate it once "
            "via RealityScan's GUI (Export Registration -> COLMAP format -> configure "
            "settings -> Save) -- see tools/realityscan-setup/README.md -- then either "
            f"commit it to that path or point {REALITYSCAN_EXPORT_PARAMS_ENV} at it."
        )
    return candidate


def _windows_argument(path: Path) -> str:
    absolute = path.absolute().resolve(strict=False)
    converted = str(absolute)
    if os.name != "nt" and converted == str(absolute):
        raise RuntimeError(f"RealityScan paths must be Windows-accessible: {absolute}")
    if converted.startswith("\\\\"):
        raise RuntimeError(f"RealityScan paths may not use UNC storage: {converted}")
    return converted


def build_realityscan_align_command(
    executable: Path,
    images_dir: Path,
    crash_dir: Path,
    export_txt: Path,
    project_path: Path,
    *,
    projection_fov_degrees: float,
    fix_intrinsics: bool = True,
    alignment_masks: bool = False,
) -> list[str]:
    # equirect2persp emits an ideal rectilinear (pinhole) image.  RealityScan
    # otherwise starts from an approximate no-distortion prior and is free to
    # optimize it into Brown3, which COLMAP exports as FULL_OPENCV.  Besides
    # being physically wrong for these generated views, that also leaves the
    # export behind a warning prompt in headless mode.  Pin the known lens model
    # and, by default, the analytically known focal length instead.
    if not 0.0 < projection_fov_degrees < 180.0:
        raise ValueError("projection_fov_degrees must be between 0 and 180")
    focal_35mm = 36.0 / (
        2.0 * math.tan(math.radians(projection_fov_degrees) / 2.0)
    )
    calibration_prior = "2" if fix_intrinsics else "1"
    command = [
        str(executable),
        "-headless",
        "-silent",
        _windows_argument(crash_dir),
        "-set",
        "appQuitOnError=true",
        "-newScene",
        "-set",
        "appIncSubdirs=true",
        "-addFolder",
        _windows_argument(images_dir),
        "-selectAllImages",
        "-editInputSelection",
        "inpCalibrationGroup=0",
        "-editInputSelection",
        f"inpCalibration={calibration_prior}",
        "-editInputSelection",
        f"inpFocal={focal_35mm:.12g}",
        "-editInputSelection",
        "inpPPX=0",
        "-editInputSelection",
        "inpPPY=0",
        "-editInputSelection",
        "inpLensGroup=0",
        "-editInputSelection",
        "inpDistortion=2",
        "-editInputSelection",
        "inpDistortionModel=0",
        # Clear the input selection used for editing.  Leaving it active makes
        # exportRegistration take the selected-input export path, which writes
        # cameras.txt/points3D.txt but (in RealityScan 2.2) omits images.txt.
        "-deselectAllImages",
        "-align",
        "-selectMaximalComponent",
        # Confirm that exportRegistration should export the currently selected
        # maximal component rather than all components or cancel.  In silent
        # mode an unset choice can report a successful export while omitting
        # images.txt (only cameras.txt/points3D.txt are written).
        "-set",
        "PUS-1-323292754=1",
        # RealityScan records confirmation choices as PUS keys.  This is the
        # documented/observed "continue with current COLMAP export" prompt.  It
        # is normally dormant after fixing the pinhole model, but keeping it in
        # the command makes recovery deterministic if RealityScan warns again.
        "-set",
        "PUS-1-323301212=1",
        "-exportRegistration",
        _windows_argument(export_txt),
        _windows_argument(export_params_file()),
        "-save",
        _windows_argument(project_path),
        "-quit",
    ]
    if alignment_masks:
        position = command.index("-deselectAllImages")
        command[position:position] = ["-editInputSelection", "inpMaskOpts=1"]
    return command
