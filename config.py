"""Configuration for Structure from Motion pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from utils.camera import CameraModel, CameraType, calibrate_camera
from utils.features import FeatureExtractorConfig, MatcherConfig
from utils.view import FrameLoaderConfig

# TODO: add SfM fields for depth-filter, pre-triangulation geometric masking, etc.


def _default_camera() -> CameraModel:
    K, dist = calibrate_camera()
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
            "camera_model": _tumvi_camera("data/tum/dataset-corridor4_512_16/dso/camchain.yaml"),
            "pre_path": "data/tum/",
            "dataset": dataset,
            "post_path": "dso/cam0/images",
            "ext": "png",
        }
    elif id == "default":
        return {
            "camera_model": _default_camera(),
            "pre_path": "data/raw",
            "dataset": dataset,
            "post_path": "",
            "ext": "jpg",
        }
    else:
        raise ValueError(f"Unknown preset {id=}")


@dataclass
class SfMConfig:
    """Configuration for Structure from Motion pipeline."""

    loader: FrameLoaderConfig = field(
        default_factory=lambda: FrameLoaderConfig(
            **preset(id="default", dataset="statue")
            # camera_model=_tumvi_camera("data/tum/dataset-corridor4_512_16/dso/camchain.yaml"),
            # pre_path="data/raw",
            # dataset="corridor",
            # post_path="",
            # ext="png",
        )
    )
    features: FeatureExtractorConfig = field(default_factory=lambda: FeatureExtractorConfig(feature_type="sift"))
    matcher: MatcherConfig = field(default_factory=lambda: MatcherConfig(matcher_type="bf"))

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
class SLAMConfig:
    """Configuration for GTSAM ISAM2 SLAM pipeline."""

    loader: FrameLoaderConfig = field(
        default_factory=lambda: FrameLoaderConfig(
            **preset(id="tumvi", dataset="dataset-corridor4_512_16"),
            max_read_frames=800,
            max_size=512,
        )
    )
    features: FeatureExtractorConfig = field(default_factory=lambda: FeatureExtractorConfig(num_features=1_000))
    matcher: MatcherConfig = field(default_factory=lambda: MatcherConfig())

    # Keyframe selection
    min_inliers: int = 30
    """Minimum number of inliers to consider when finding reference keyframe for a new keyframe"""

    max_window_keyframes: int = 10
    """Maximum number of recent keyframes to keep in the sliding window for optimization"""

    max_motion_matches: int = 125
    """Maximum number of keypoint matches to judge the motion between frames (if too high, we might add redundant keyframes with little motion)"""
