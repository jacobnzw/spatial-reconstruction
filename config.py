"""Configuration for Structure from Motion pipeline."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SfMConfig:
    """Configuration for Structure from Motion pipeline.

    Modify the default values here for experimentation.
    Command-line overrides: --cfg.param_name value
    """

    # Feature extraction
    feature_type: Literal["sift", "disk"] = "sift"
    """Feature extraction method: 'sift' or 'disk'"""

    num_features: int = 10_000
    """Maximum number of features to extract per image"""

    max_size: int = 1024
    """Maximum image dimension (images will be resized if larger)"""

    # Keypoint matching
    matcher_type: Literal["bf", "lightglue"] = "bf"
    """Keypoint matching method: 'bf' (brute-force) or 'lightglue'"""

    lowe_ratio: float = 0.75
    """Lowe's ratio test threshold for BF matcher (only used when matcher='bf' and cross_check=False)"""

    cross_check: bool = True
    """Whether to use cross-checking for BF matcher (only used when matcher='bf')"""

    min_dist: float = 0.0
    """Minimum distance threshold for LightGlue matcher (only used when matcher='lightglue')"""

    # Dataset
    dataset: str = "statue"
    """Dataset name (subdirectory in data/raw/)"""

    # View graph construction
    min_inliers: int = 50
    """Minimum number of inliers to consider two views as overlapping"""

    # Optimization
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
    """Configuration for GTSAM ISAM2 SLAM pipeline.

    Modify the default values here for experimentation.
    Command-line overrides: --cfg.param_name value
    """

    # Feature extraction
    feature_type: Literal["sift", "disk"] = "sift"
    """Feature extraction method: 'sift' or 'disk'"""

    num_features: int = 1_000
    """Maximum number of features to extract per image"""

    max_size: int = 512
    """Maximum image dimension (images will be resized if larger)"""

    # Keypoint matching
    matcher_type: Literal["bf", "lightglue"] = "bf"
    """Keypoint matching method: 'bf' (brute-force) or 'lightglue'"""

    lowe_ratio: float = 0.75
    """Lowe's ratio test threshold for BF matcher (only used when matcher='bf' and cross_check=False)"""

    cross_check: bool = True
    """Whether to use cross-checking for BF matcher (only used when matcher='bf')"""

    min_dist: float = 0.0
    """Minimum distance threshold for LightGlue matcher (only used when matcher='lightglue')"""

    # Dataset
    dataset: str = "dataset-corridor4_512_16"
    """Dataset name (subdirectory in data/tum/)"""

    # View graph construction
    min_inliers: int = 50
    """Minimum number of inliers to consider two views as overlapping"""

    max_read_frames: int | None = None
    """Maximum number of frames to process from the dataset"""

    max_window_keyframes: int = 10
    """Maximum number of recent keyframes to keep in the sliding window for optimization"""

    max_motion_matches: int = 85
    """Maximum number of keypoint matches to judge the motion between frames (if too high, we might add redundant keyframes with little motion)"""
