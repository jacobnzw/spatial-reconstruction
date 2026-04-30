from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import RigidTransform as SE3Pose
from scipy.spatial.transform import Rotation

from .camera import CameraModel, CameraType, NDArrayFloat


@dataclass
class FrameLoaderConfig:
    camera_model: CameraModel

    # Dataset
    pre_path: str
    """Path pre-fix"""
    dataset: str
    """Dataset name"""
    post_path: str
    """Path post-fix"""
    ext: str
    """Image file extension to expect in img_dir_path directory"""

    @property
    def img_dir(self) -> Path:
        return Path(self.pre_path) / self.dataset / self.post_path

    @property
    def img_paths(self) -> list[Path]:
        return sorted(list(Path(self.img_dir).glob(f"*.{self.ext}")))

    max_read_frames: int | None = None
    """Maximum number of frames to process from the dataset"""

    offset_frames: int | None = None
    """Index of a frame to from which to progressively start loading the dataset."""

    undistort: bool = True
    """Whether to undistort images using the provided camera intrinsics and distortion coefficients"""

    max_size: int = 1024
    """Maximum image dimension (images will be resized if larger)"""


@dataclass
class ViewData:
    """Represents a single image with extracted features and estimated camera pose.

    Stores image metadata, pixel data, extracted keypoints and descriptors,
    and the estimated camera extrinsics (world-to-camera transformation).

    Attributes:
        idx: Unique image index in the reconstruction.
        path: Path to the image file.
        pixels: RGB pixel data for rendering and debugging (H, W, 3).
        camera_model: Camera intrinsic model containing K and distortion coefficients.
        kp: Extracted keypoint locations as (N, 2) array of (x, y) coordinates.
        des: Feature descriptors as (N, D) array where D is descriptor dimension.
        cam_T_world: World-to-camera transformation (camera extrinsics). Transforms world points to camera frame: X_cam = R @ X_world + t
    """

    idx: int
    path: Path
    # Useful for rendering and debugging
    pixels: NDArray[Any]  # GRAYs and RBGs as (H, W, C) unit8
    # Camera model
    camera_model: CameraModel
    # Extracted keypoints and descriptors
    kp: NDArrayFloat | None = None
    des: NDArrayFloat | None = None
    # Estimated camera extrinsics, i.e world-to-camera transform; output of cv.solvePnP etc.
    cam_T_world: SE3Pose | None = None

    def _check_pose(self):
        if not self.has_pose:
            raise ValueError("Pose not set for this image")

    @property
    def has_pose(self) -> bool:
        return self.cam_T_world is not None

    @property
    def world_T_cam(self) -> SE3Pose:
        """Camera's pose in world frame."""
        return self.cam_T_world.inv()  # ty:ignore[possibly-missing-attribute]

    @property
    def R(self) -> NDArrayFloat:
        self._check_pose()
        return self.cam_T_world.rotation.as_matrix()  # ty:ignore[possibly-missing-attribute]

    @property
    def t(self) -> NDArrayFloat:
        self._check_pose()
        return self.cam_T_world.translation.squeeze()  # ty:ignore[possibly-missing-attribute]

    @property
    def rvec(self) -> NDArrayFloat:
        self._check_pose()
        # cv.Rodrigues(R)[0]
        return self.cam_T_world.rotation.as_rotvec().squeeze()  # ty:ignore[possibly-missing-attribute]

    @property
    def pose_matrix(self) -> NDArrayFloat:
        self._check_pose()
        return self.cam_T_world.as_matrix()[:3, :]  # 3x4 [R | t]  # ty:ignore[possibly-missing-attribute]

    @property
    def projection_matrix(self) -> NDArrayFloat:
        """Get the 3x4 projection matrix P = K [R | t] for this image.

        Note: Effectively camera_P_world
          - transformation of points from world coordinates to camera image plane coordinates
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.html
        """
        K = self.camera_model.get_camera_matrix(rescaled=True)
        return K @ self.pose_matrix

    def set_extrinsics(self, R, t):
        """Set camera extrinsics, i.e. cam_T_world."""
        rotation = Rotation.from_matrix(R)
        self.cam_T_world = SE3Pose.from_components(t.squeeze(), rotation)

    def set_pose(self, R, t):
        """Set pose of camera in world frame, i.e. world_T_cam.

        Args:
            R: rotation matrix of the camera in world frame.
            t: translation vector of the camera in world frame.
        """
        # The input R and t in relate to the camera pose, not the extrinsics!
        # But since we store only extrinsics, we need .inv()
        rotation = Rotation.from_matrix(R)
        world_T_cam = SE3Pose.from_components(t.squeeze(), rotation)
        self.cam_T_world = world_T_cam.inv()

    def get_camera_center(self) -> NDArrayFloat:
        """Get the camera center in world coordinates."""
        self._check_pose()
        return self.world_T_cam.translation  # -R.T @ t

    def get_undistorted_keypoints(self) -> NDArrayFloat:
        """Undistort keypoint pixel coordinates.

        Returns:
            (N, 2) array of undistorted keypoint coordinates in pixel space.
        """
        K, dist = self.camera_model.get_camera_matrix(), self.camera_model.dist

        if self.camera_model.model_type == CameraType.FISHEYE:
            # undistortPoints returns normalized coords, need to reproject to pixels
            kp_normalized = cv.fisheye.undistortPoints(self.kp, K, dist, R=None, P=K)
            return kp_normalized.squeeze()  # (N, 1, 2) -> (N, 2)
        else:  # PINHOLE
            # Same for pinhole: P=K to get pixel coordinates back
            kp_normalized = cv.undistortPoints(self.kp, K, dist, R=None, P=K)
            return kp_normalized.squeeze()  # (N, 1, 2) -> (N, 2)

    def transform_to_camera_frame(self, world_pts: NDArrayFloat) -> NDArrayFloat:
        """Transform points from world coordinates to this camera's coordinate frame."""
        self._check_pose()
        # R @ world_pts.T + t
        return self.cam_T_world.apply(world_pts)  # ty:ignore[possibly-missing-attribute]

    def transform_to_world_frame(self, camera_pts: NDArrayFloat) -> NDArrayFloat:
        """Transform points from this camera's coordinate frame to world coordinates.

        Args:
            camera_pts: (N, 3) array of points in the camera's coordinate frame.
        """
        self._check_pose()
        # self.R.T @ (camera_pts.T - self.t)
        return self.world_T_cam.apply(camera_pts)

    def project_to_image_plane(self, world_pts: NDArrayFloat) -> NDArrayFloat:
        """Project 3D world points to this camera's image plane (2D pixel coordinates).

        Args:
            world_pts: (N, 3) array of points in world coordinates.

        Returns:
            (N, 2) array of projected points in pixel coordinates.
        """
        self._check_pose()
        K, dist = self.camera_model.get_camera_matrix(), self.camera_model.dist
        points_2d, _ = cv.projectPoints(world_pts, self.rvec, self.t, K, dist)
        return points_2d.squeeze()  # (N, 1, 2) -> (N, 2)  # ty:ignore[invalid-return-type]


class FrameLoader:
    """Loads images from a directory and applies preprocessing such as scaling.

    Args:
        img_dir: Directory containing input images.
        camera_model: camera model to assign to each frame.
        max_size: Maximum size (in pixels) for the longest edge of the image. Images larger than this will be downscaled.
        max_frames: Optional maximum number of frames to load. If None, loads all frames in the directory.
        offset_frames: Optional skip first offset_frames images in dataset.
        ext: Image file extension to look for (default "png").
        undistort: If True, applies undistortion to images using the provided camera model parameters.
    """

    def __init__(self, cfg: FrameLoaderConfig):
        img_paths = cfg.img_paths
        if not img_paths:
            raise ValueError(f"No *.{cfg.ext} images found in {cfg.img_dir}")

        self.img_paths = img_paths
        self.max_frames = cfg.max_read_frames
        self.offset_frames = cfg.offset_frames if cfg.offset_frames is not None else 0
        self.max_size = cfg.max_size
        self.scale = 1.0
        self.camera_model = cfg.camera_model
        self.undistort = cfg.undistort

    def __call__(self, idx: int) -> ViewData:
        """Load frame at given index in internally stored list of image paths."""
        path = self.img_paths[idx]
        # Grayscale loaded as (H, W, 3) with identical channels, color loaded as (H, W, 3) in RGB order
        img = cv.imread(str(path), cv.IMREAD_COLOR_RGB)
        if img is None:
            raise FileNotFoundError(f"FrameLoader: Failed to load image: {path}")
        return ViewData(idx, path, img, self.camera_model)

    def iter_frames(self) -> Iterable[ViewData]:
        """Yields images as ImageData objects.

        Returns:
            ImageData object containing the image and its metadata.

            ImageData.pixels contains the loaded image as a (H, W, 3) uint8 array in RGB format,
            regardless of original format.
        """
        for idx, path in enumerate(self.img_paths[self.offset_frames :], start=self.offset_frames):
            if self.max_frames and idx >= self.max_frames:
                print(f"FrameLoader: Reached max_frames={self.max_frames}, stopping further loading.")
                break

            # Grayscale loaded as (H, W, 3) with identical channels, color loaded as (H, W, 3) in RGB order
            img = cv.imread(str(path), cv.IMREAD_COLOR_RGB)
            if img is None:
                raise FileNotFoundError(f"FrameLoader: Failed to load image: {path}")

            camera_model = self.camera_model
            if self.undistort:
                if self.camera_model.model_type == CameraType.FISHEYE:
                    img, K_undistorted = self._undistort_fisheye(img)
                elif self.camera_model.model_type == CameraType.PINHOLE:
                    img, K_undistorted = self._undistort_pinhole(img)
                else:
                    raise ValueError(f"Uknown {camera_model=}! Only PINHOLE and FISHEYE supported.")

                # After undistortion, it's pinhole camera with new intrinsics K_undistorted and no distortion
                camera_model = CameraModel(
                    model_type=CameraType.PINHOLE, K=K_undistorted, dist=np.zeros(len(self.camera_model.dist))
                )

            # Compute scale based on first image
            # Assumption: all images have the same resolution and thus the same scale factor applies to all
            if idx == self.offset_frames and self.max_size is not None:
                h, w = img.shape[:2]
                self.scale = self.max_size / max(h, w) if max(h, w) > self.max_size else 1.0

            # Apply scaling if needed
            if self.scale < 1.0:
                h, w = img.shape[:2]
                new_w, new_h = int(w * self.scale), int(h * self.scale)
                img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
                camera_model.scale = self.scale
                # NOTE: camera_model.get_camera_matrix() will handle rescaling K based on self.scale

            yield ViewData(idx, path, img, camera_model=camera_model)

    def _undistort_fisheye(self, img: NDArray[Any], balance=0.0, fov_scale=1.0) -> tuple[NDArray[Any], NDArrayFloat]:
        """Undistortion for equidistant fisheye.

        Args:
            balance: float A value of 0.0 crops aggressively to remove all black borders, keeping only the "good"
            pixels. A value of 1.0 tries to preserve the entire original field of view, resulting in a larger,
            more zoomed-out image with significant black areas in the corners.
        """
        h, w = img.shape[:2]

        # Create the undistortion + rectification map once (or cache it)
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.camera_model.K, self.camera_model.dist, (w, h), np.eye(3), balance=balance, fov_scale=fov_scale
        )
        K, dist = self.camera_model.get_camera_matrix(), self.camera_model.dist
        img_undist = cv.fisheye.undistortImage(img, K, dist, Knew=new_K, new_size=(w, h))
        return img_undist, new_K  # ty:ignore[invalid-return-type]

    def _undistort_pinhole(self, img: NDArray[Any], alpha=0.0) -> tuple[NDArray[Any], NDArrayFloat]:
        """Undistortion for pinhole camera model.

        Args:
            alpha: float  If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels.
            So it may even remove some pixels at image corners. If alpha=1, all pixels are retained with some extra
            black borders.
        """
        h, w = img.shape[:2]
        K, dist = self.camera_model.get_camera_matrix(), self.camera_model.dist
        new_K, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=alpha, newImgSize=(w, h))
        img_undist = cv.undistort(img, K, dist, newCameraMatrix=new_K)

        return img_undist, new_K  # ty:ignore[invalid-return-type]
