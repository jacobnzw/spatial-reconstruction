"""
Utils package for spatial reconstruction.

This package provides modular utilities for camera models, feature extraction,
tracking, point clouds, view graphs, and I/O operations.

All public APIs are re-exported at the package level for backward compatibility.
"""

# Type aliases and common types
from .camera import CameraModel, CameraType, NDArrayFloat, NDArrayInt, calibrate_camera

# Features module
from .features import FeatureExtractor, FeatureStore, KPKey, device, make_keypoint_matcher

# View graph module
from .graph import ViewEdge, ViewGraph, construct_view_graph, has_overlap

# I/O module
from .io import ReconIO

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
    "make_keypoint_matcher",
    "device",
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
    # I/O
    "ReconIO",
]
