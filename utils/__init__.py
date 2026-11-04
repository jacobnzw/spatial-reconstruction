"""
Utils package for spatial reconstruction.

This package provides modular utilities for camera models, feature extraction,
tracking, point clouds, view graphs, and I/O operations.

All public APIs are re-exported at the package level for backward compatibility.
"""

# Type aliases and common types
from .camera import CameraModel, CameraType, NDArrayFloat, NDArrayInt, calibrate_camera

# I/O module
from .colmap import ColmapAdapter

# Features module
from .features import FeatureExtractor, FeatureStore, KeypointMatcher, KPKey, MatcherResult

# View graph module
from .graph import ViewGraph

# Logging module
from .logging import ReRunLogger, build_track_length_histogram, dump_sfm_debug, log_wandb_artifacts

# Point cloud module
from .pointcloud import Point3D, PointCloud

# Tracks module
from .tracks import TrackManager

# View module
from .view import FrameLoader, ViewData

__all__ = [
    # Camera
    "CameraType",
    "CameraModel",
    "calibrate_camera",
    "NDArrayFloat",
    "NDArrayInt",
    # View
    "ViewData",
    "FrameLoader",
    # Features
    "FeatureExtractor",
    "FeatureStore",
    "KeypointMatcher",
    "MatcherResult",
    "KPKey",
    # Tracks
    "TrackManager",
    # Point cloud
    "PointCloud",
    "Point3D",
    # Graph
    "ViewGraph",
    "ViewEdge",
    "has_overlap",
    "construct_view_graph",
    # Colmap adapter
    "ColmapReconstructionAdapter",
    # Logging
    "build_track_length_histogram",
    "log_wandb_artifacts",
    "dump_sfm_debug",
    "ReRunLogger",
]
