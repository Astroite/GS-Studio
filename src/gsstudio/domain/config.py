from __future__ import annotations
from typing import Literal
from pydantic import Field, field_validator, model_validator
from gsstudio.domain.common import StrictModel, RUN_SCHEMA_VERSION, SLUG_PATTERN
from gsstudio.domain.capture import SourceProbe, NormalizationSettings, TimeSelection


class PreprocessConfig(StrictModel):
    candidate_fps: float = Field(default=1.0, ge=1, le=120)
    selected_per_second: int = Field(default=1, ge=1, le=120)
    minimum_free_gib: float = Field(default=20.0, ge=1)


COCO_DYNAMIC_CLASS_IDS = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 6,
    "truck": 8,
}


class MaskingConfig(StrictModel):
    enabled: bool = True
    model: Literal["maskrcnn_resnet50_fpn_v2"] = "maskrcnn_resnet50_fpn_v2"
    weights: Literal["DEFAULT"] = "DEFAULT"
    device: Literal["cuda", "cpu"] = "cuda"
    person_class_id: int = 1
    score_threshold: float = Field(default=0.25, ge=0, le=1)
    probability_threshold: float = Field(default=0.50, ge=0, le=1)
    inference_gamma: float = Field(default=0.75, gt=0, le=2)
    dilation_pixels: int = Field(default=24, ge=0, le=256)
    closing_pixels: int = Field(default=7, ge=0, le=255)
    mask_discard_threshold: float = Field(default=0.005, gt=0, lt=1)
    qa_sample_count: int = Field(default=16, ge=1, le=256)
    classes: list[Literal["person", "car", "bus", "truck", "bicycle", "motorcycle"]] = Field(
        default_factory=lambda: ["person"], min_length=1
    )
    mask_review_required: bool = False

    @field_validator("closing_pixels")
    @classmethod
    def validate_closing_kernel(cls, value: int) -> int:
        if value not in (0, 1) and value % 2 == 0:
            raise ValueError("closing_pixels must be odd (or 0/1 to disable closing)")
        return value

    @field_validator("classes")
    @classmethod
    def validate_unique_classes(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("masking classes must be unique")
        return value


class VisionQAConfig(StrictModel):
    enabled: bool = False
    provider: Literal["deepseek"] = "deepseek"
    model: Literal["deepseek-v4-flash-vision-exp"] = "deepseek-v4-flash-vision-exp"
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: Literal["https://api.deepseek.com"] = "https://api.deepseek.com"
    endpoint_path: str = "/chat/completions"
    image_detail: Literal["original"] = "original"
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    max_contact_sheets: int = Field(default=4, ge=1, le=16)
    minimum_confidence: float = Field(default=0.80, ge=0, le=1)


class LoopClosureConfig(StrictModel):
    enabled: bool = False
    period: int = Field(default=20, ge=1)
    num_images: int = Field(default=5, ge=1)
    vocabulary_tree_path: str | None = None
    vocabulary_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_vocabulary_tree(self) -> "LoopClosureConfig":
        if self.enabled and not self.vocabulary_tree_path:
            raise ValueError("loop closure requires an explicit vocabulary_tree_path")
        return self


class PanoramaProjection(StrictModel):
    """How one prepared panorama becomes training views."""

    views: Literal[8, 14]
    size: int = Field(ge=256, le=8192)
    crop_bottom: float = Field(ge=0, lt=0.5)


class ReconstructionAttempt(StrictModel):
    temporal_rank_limit: int = Field(ge=1, le=120)
    # Equirect sources: horizontal FOV of each projected view. Perspective
    # sources: the capture's horizontal FOV, used as the RealityScan prior.
    projection_fov_degrees: float = Field(gt=0, lt=180)
    # None means the prepared frames are already the training views
    # (perspective_video): one frame is one view.
    panorama: PanoramaProjection | None = None


class ReconstructionConfig(StrictModel):
    backend: Literal["realityscan", "colmap"] = "realityscan"
    use_gpu_sift: bool = True
    fix_intrinsics: bool = True
    registration_threshold: float = Field(default=0.70, gt=0, le=1)
    rig_center_spread_ratio_limit: float = Field(default=0.001, gt=0, le=0.1)
    alignment_masks: bool = True
    primary: ReconstructionAttempt = Field(
        default_factory=lambda: ReconstructionAttempt(
            temporal_rank_limit=1,
            projection_fov_degrees=110.0,
            panorama=PanoramaProjection(views=14, size=1746, crop_bottom=0.15),
        )
    )
    # One bounded local repair round after the primary alignment.
    repair_padding_seconds: float = Field(default=2.0, ge=0)
    repair_attempts: Literal[1] = 1
    repair_candidate_fps: float = Field(default=2.0, ge=1, le=120)
    loop_closure: LoopClosureConfig = Field(default_factory=LoopClosureConfig)


class SegmentQAConfig(StrictModel):
    max_gap_seconds: float = Field(default=2.0, gt=0)
    speed_multiplier: float = Field(default=10.0, gt=1)
    minimum_samples: int = Field(default=10, ge=2)
    minimum_positions: int = Field(default=3, ge=2)
    minimum_views: int = Field(default=3, ge=1)
    registration_threshold: float = Field(default=0.70, gt=0, le=1)
    center_spread_step_ratio: float = Field(default=3.0, gt=0)


class RetentionConfig(StrictModel):
    mode: Literal["minimal", "keep"] = "minimal"


class PreparedInputConfig(StrictModel):
    """Rate-sampled candidate frames prepared from the capture sources."""

    source_kind: Literal["equirect_video", "equirect_sequence", "insta360_insv", "perspective_video", "perspective_images"]
    source_sha256: list[str] = Field(min_length=1)
    source_probe: SourceProbe
    normalization: NormalizationSettings
    selection: TimeSelection
    candidate_frame_indices: list[int] = Field(min_length=2)
    candidate_fps: float = Field(default=1.0, gt=0, le=120)
    helper_version: str | None = None
    sdk_version: str | None = None


class RunConfig(StrictModel):
    schema_version: Literal[6] = RUN_SCHEMA_VERSION
    capture_id: str = Field(pattern=SLUG_PATTERN)
    input_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Run-owned candidates, never Capture-shared: inputs/primary/<preparation hash>.
    input_relative_path: str
    input: PreparedInputConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    vision_qa: VisionQAConfig = Field(default_factory=VisionQAConfig)
    reconstruction: ReconstructionConfig = Field(default_factory=ReconstructionConfig)
    segment_qa: SegmentQAConfig = Field(default_factory=SegmentQAConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    @field_validator("input_relative_path")
    @classmethod
    def validate_owned_input(cls, value: str) -> str:
        from pathlib import PurePosixPath

        parts = PurePosixPath(value.replace("\\", "/")).parts
        if len(parts) != 3 or parts[:2] != ("inputs", "primary") or len(parts[2]) != 64:
            raise ValueError("Run input must live at inputs/primary/<preparation hash>")
        if any(character not in "0123456789abcdef" for character in parts[2]):
            raise ValueError("Invalid preparation hash")
        return "/".join(parts)

    @model_validator(mode="after")
    def validate_temporal_sampling(self) -> "RunConfig":
        primary = self.reconstruction.primary.temporal_rank_limit
        if not 1 <= primary <= self.preprocess.selected_per_second <= self.preprocess.candidate_fps:
            raise ValueError(
                "temporal sampling requires 1 <= primary <= selected_per_second <= candidate_fps"
            )
        if abs(self.input.candidate_fps - self.preprocess.candidate_fps) > 1e-9:
            raise ValueError("prepared input candidate_fps must match preprocess.candidate_fps")
        return self
