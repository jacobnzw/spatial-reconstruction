"""Configuration for Structure from Motion pipeline."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict

import numpy as np

from utils.features import FeatureExtractorConfig, MatcherConfig
from utils.view import FrameLoaderConfig


def frame_loader_preset(id: str) -> Dict:
    if id == "corridor":
        return {
            "calib_file": "data/calibration/tumvi/calib.yaml",
            "data_path": f"data/raw/{id}",
        }
    elif id == "statue_orbit":
        return {
            "calib_file": "data/calibration/redmi/calib.yaml",
            "data_path": f"data/raw/{id}",
        }
    else:
        raise ValueError(f"Unknown preset {id=}")


@dataclass
class BaseConfig:
    """Common config for both pipelines."""

    loader: FrameLoaderConfig
    """Frame Loader Configuration."""

    features: FeatureExtractorConfig
    """Keypoint Descriptor Configuration."""

    matcher: MatcherConfig
    """Keypoint Matcher Configuration."""

    depth_threshold: float = 0.3
    """Minimum depth (along z-axis) in camera frame for the triangulated points."""

    out_name: str | None = None
    """Override basename for the pipeline output files (e.g. point cloud saved to 'out_basename.ply')."""

    @property
    def out_basename(self):
        """Default output basename for all output files, if override not specified via 'out_name'."""
        return (
            self.out_name
            if self.out_name is not None
            else f"{self.loader.dataset}_{self.features.type}_{self.matcher.type}"
        )

    @property
    def out_dir(self):
        """Output directory where all pipeline output files are written to."""
        return Path("data") / "out" / self.loader.dataset


@dataclass
class SfMConfig(BaseConfig):
    """Configuration for Structure from Motion pipeline."""

    features: FeatureExtractorConfig = field(default_factory=lambda: FeatureExtractorConfig(type="disk"))

    matcher: MatcherConfig = field(default_factory=lambda: MatcherConfig(type="lg"))

    # SfM-specific fields
    min_inliers: int = 50
    """Minimum number of inliers to consider two views as overlapping"""

    run_ba: bool = True
    """Run bundle adjustment optimization after initial reconstruction"""

    fix_first_camera: bool = True
    """Fix the first camera during bundle adjustment"""

    dump_sfm_debug: bool = False
    """Dump the SFM structs (image_store, point_cloud, track_manager) to disk for debugging/inspection"""

    save_gsplat: bool = False
    """Save tensors for gsplat (without BA)"""


@dataclass
class SLAMConfig(BaseConfig):
    """Configuration for GTSAM ISAM2 SLAM pipeline."""

    loader: FrameLoaderConfig = field(
        default_factory=lambda: FrameLoaderConfig(
            **frame_loader_preset(id="corridor"),
            max_read_frames=800,
            max_size=512,
        )
    )
    features: FeatureExtractorConfig = field(
        default_factory=lambda: FeatureExtractorConfig(feature_type="disk", num_features=1_000)
    )
    matcher: MatcherConfig = field(default_factory=lambda: MatcherConfig(matcher_type="lg"))

    # Keyframe selection
    min_inliers: int = 30
    """Minimum number of inliers to consider when finding reference keyframe for a new keyframe"""

    max_window_keyframes: int = 10
    """Maximum number of recent keyframes to keep in the sliding window for optimization"""

    max_motion_matches: int = 125
    """Maximum number of keypoint matches to judge the motion between frames (if too high, we might add redundant keyframes with little motion)"""


def write_config_to_json(cfg: SfMConfig | SLAMConfig, file: str):
    """Write dataclass config to JSON file."""

    def config_serializer(obj):
        """Serializer for numpy arrays and Enums."""

        if isinstance(obj, np.ndarray):
            return np.array2string(obj)
        if isinstance(obj, Enum):
            return obj.value

        raise TypeError(f"Object of type {type(obj)} is not serializable")

    Path(file).write_text(json.dumps(asdict(cfg), indent=2, default=config_serializer))
