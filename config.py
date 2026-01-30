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
    feature_type: Literal["sift", "disk"] = "disk"
    """Feature extraction method: 'sift' or 'disk'"""

    num_features: int = 2048
    """Maximum number of features to extract per image"""

    max_size: int = 4080
    """Maximum image dimension (images will be resized if larger)"""

    # Keypoint matching
    matcher_type: Literal["bf", "lightglue"] = "lightglue"
    """Keypoint matching method: 'bf' (brute-force) or 'lightglue'"""

    lowe_ratio: float = 0.75
    """Lowe's ratio test threshold for BF matcher (only used when matcher='bf')"""

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

    fix_first_camera: bool = False
    """Fix the first camera during bundle adjustment"""
