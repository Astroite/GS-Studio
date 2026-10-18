from __future__ import annotations
import ntpath
from pathlib import PureWindowsPath
from typing import Literal
from pydantic import Field, model_validator
from gsstudio.domain.common import StrictModel, RunStatus, SLUG_PATTERN, CAPTURE_SCHEMA_VERSION


class Camera(StrictModel):
    make: str
    model: str
    firmware: str | None = None


class TimeSelection(StrictModel):
    start_seconds: float = Field(default=0.0, ge=0)
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "TimeSelection":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class SourceFile(StrictModel):
    """One immutable source file owned outside the prepared-input cache."""

    windows_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)


class SourceProbe(StrictModel):
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    duration_seconds: float | None = Field(default=None, gt=0)
    codec: str | None = None
    camera_model: str | None = None
    helper_version: str | None = None
    sdk_version: str | None = None


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

SOURCE_PROJECTIONS = {
    "insta360_insv": "dual_fisheye",
    "perspective_video": "perspective",
    "perspective_images": "perspective",
}


class CaptureSource(StrictModel):
    """The registered raw media. Panorama kinds are projected into views; a
    perspective_video source is used frame for frame."""

    kind: Literal["equirect_video", "equirect_sequence", "insta360_insv", "perspective_video", "perspective_images"]
    files: list[SourceFile] = Field(min_length=1)
    projection: Literal["equirectangular", "dual_fisheye", "perspective"]
    probe: SourceProbe = Field(default_factory=SourceProbe)

    @property
    def is_perspective(self) -> bool:
        return self.kind in {"perspective_video", "perspective_images"}

    @model_validator(mode="after")
    def validate_source_shape(self) -> "CaptureSource":
        paths = [
            ntpath.normcase(ntpath.normpath(item.windows_path.replace("/", "\\")))
            for item in self.files
        ]
        if len(set(paths)) != len(paths):
            raise ValueError("source.files must list distinct paths explicitly")
        suffixes = {
            PureWindowsPath(item.windows_path.replace("/", "\\")).suffix.casefold()
            for item in self.files
        }
        expected_suffixes = IMAGE_SUFFIXES if self.kind in {"equirect_sequence", "perspective_images"} else (
            {".insv"} if self.kind == "insta360_insv" else VIDEO_SUFFIXES
        )
        if not suffixes.issubset(expected_suffixes):
            raise ValueError(f"{self.kind} source must use supported {sorted(expected_suffixes)} files")
        limits = {
            "equirect_video": (1, 1),
            "perspective_video": (1, 1),
            "equirect_sequence": (1, 100000),
            "perspective_images": (2, 100000),
            "insta360_insv": (1, 2),
        }[self.kind]
        if not limits[0] <= len(self.files) <= limits[1]:
            raise ValueError(f"{self.kind} requires between {limits[0]} and {limits[1]} source files")
        expected = SOURCE_PROJECTIONS.get(self.kind, "equirectangular")
        if self.projection != expected:
            raise ValueError(f"{self.kind} source projection must be {expected}")
        return self


class NormalizationSettings(StrictModel):
    """Output size and stitching for prepared frames.

    Panorama sources normalise to a 2:1 equirectangular frame; a perspective_video
    source keeps its native dimensions and leaves width/height unset.
    """

    projection: Literal["equirectangular"] = "equirectangular"
    aspect_ratio: Literal["2:1"] = "2:1"
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    image_format: Literal["jpeg"] = "jpeg"
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    stitching: Literal["dynamic"] = "dynamic"
    flowstate: Literal[True] = True
    direction_lock: Literal[False] = False
    ai_denoise: Literal[False] = False
    sharpening: Literal[False] = False
    color_processing: Literal[False] = False

    @model_validator(mode="after")
    def validate_dimensions(self) -> "NormalizationSettings":
        if (self.width is None) != (self.height is None):
            raise ValueError("normalization width and height must be set together")
        if self.width is not None and self.width != self.height * 2:
            raise ValueError("normalization dimensions must be exactly 2:1")
        return self


class CaptureManifest(StrictModel):
    schema_version: Literal[2] = CAPTURE_SCHEMA_VERSION
    id: str = Field(pattern=SLUG_PATTERN)
    location_id: str = Field(pattern=SLUG_PATTERN)
    scene_id: str = Field(pattern=SLUG_PATTERN)
    status: RunStatus = RunStatus.DRAFT
    source: CaptureSource
    camera: Camera
    selection: TimeSelection
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    # Focal prior for RealityScan only; the training intrinsics come from the
    # exported COLMAP model.
    horizontal_fov_degrees: float | None = Field(default=None, gt=0, lt=180)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_source_settings(self) -> "CaptureManifest":
        if self.source.is_perspective:
            if self.horizontal_fov_degrees is None:
                raise ValueError("perspective_video requires horizontal_fov_degrees")
            if self.normalization.width is not None:
                raise ValueError(
                    "perspective_video keeps the source frame size; normalization dimensions must be unset"
                )
        else:
            if self.horizontal_fov_degrees is not None:
                raise ValueError("horizontal_fov_degrees applies only to perspective_video")
        return self
