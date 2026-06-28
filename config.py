"""Configuration for Structure from Motion pipeline."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from utils.camera import CameraModel, CameraType, calibrate_camera
from utils.features import FeatureExtractorConfig, MatcherConfig
from utils.view import FrameLoaderConfig


def _default_camera(camera_params_file: str) -> CameraModel:
    K, dist = calibrate_camera(Path(camera_params_file))
    return CameraModel(CameraType.PINHOLE, K, dist)


def _tumvi_camera(calib_yaml_path: str, cam_key: str = "cam0") -> CameraModel:

    camchain_file = yaml.safe_load(Path(calib_yaml_path).open())
    k, dist = np.array(camchain_file[cam_key]["intrinsics"]), np.array(camchain_file[cam_key]["distortion_coeffs"])
    fx, fy, cx, cy = k

    return CameraModel(
        CameraType.FISHEYE,
        K=np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        ),
        dist=dist,
    )


def preset(id: str, dataset: str) -> Dict:
    if id == "tumvi":
        return {
            "camera_model": _tumvi_camera("data/calibration/tumvi/camchain.yaml"),
            "pre_path": "data/raw",
            "dataset": dataset,
            "post_path": "",
            "ext": "png",
        }
    elif id == "default":
        return {
            "camera_model": _default_camera("data/calibration/redmi/calibration_params.npz"),
            "pre_path": "data/raw",
            "dataset": dataset,
            "post_path": "",
            "ext": "jpg",
        }
    else:
        raise ValueError(f"Unknown preset {id=}")


@dataclass
class BaseConfig:
    """Common config for both pipelines."""

    loader: FrameLoaderConfig

    features: FeatureExtractorConfig

    matcher: MatcherConfig

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
            else f"{self.loader.dataset}_{self.features.feature_type}_{self.matcher.matcher_type}"
        )

    @property
    def out_dir(self):
        """Output directory where all pipeline output files are written to."""
        return Path("data") / "out" / self.loader.dataset


@dataclass
class SfMConfig(BaseConfig):
    """Configuration for Structure from Motion pipeline."""

    loader: FrameLoaderConfig = field(
        default_factory=lambda: FrameLoaderConfig(
            # **preset(id="default", dataset="statue_orbit")
            **preset(id="tumvi", dataset="corridor")
        )
    )
    features: FeatureExtractorConfig = field(default_factory=lambda: FeatureExtractorConfig(feature_type="disk"))
    matcher: MatcherConfig = field(default_factory=lambda: MatcherConfig(matcher_type="lg"))

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
            **preset(id="tumvi", dataset="dataset-corridor4_512_16"),
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


def config_serializer(obj):
    """Serializer for numpy arrays and Enums."""

    if isinstance(obj, np.ndarray):
        return np.array2string(obj)
    if isinstance(obj, Enum):
        return obj.value

    raise TypeError(f"Object of type {type(obj)} is not serializable")


def write_config_to_json(cfg: SfMConfig | SLAMConfig, file: str):
    """Write dataclass config to JSON file."""
    Path(file).write_text(json.dumps(asdict(cfg), indent=2, default=config_serializer))
