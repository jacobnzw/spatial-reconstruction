from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Literal

import cv2 as cv
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.transform import RigidTransform as SE3Pose
from scipy.spatial.transform import Rotation

from config import SfMConfig, SLAMConfig

device = K.utils.get_cuda_or_mps_device_if_available()

NDArrayFloat = NDArray[np.floating[Any]]
NDArrayInt = NDArray[np.integer[Any]]
Point3D = Annotated[NDArrayFloat, Literal[3]]
KPKey = tuple[int, int]  # Keypoint observation (img_id, kp_idx)


class CameraType(Enum):
    """Enum for different camera models."""

    PINHOLE = "pinhole"
    FISHEYE = "fisheye"


@dataclass
class CameraModel:
    """Camera intrinsic parameters and distortion model.

    Attributes:
        model_type: Type of camera model (pinhole or fisheye).
        K: Camera intrinsic matrix (3x3).
        dist: Distortion coefficients.
    """

    model_type: CameraType
    K: NDArrayFloat
    dist: NDArrayFloat
    scale: float = 1.0  # Scaling factor applied to the image (1.0 means no scaling)

    def get_camera_matrix(self, rescaled: bool = True) -> NDArrayFloat:
        """Get camera matrix K, rescaled if necessary.

        Args:
            rescaled: If True, return the rescaled K based on the current scale factor. If False, return the original K.
        """

        if rescaled and self.scale < 1.0:
            K_rescaled = self.K.copy()
            K_rescaled[0, :] *= self.scale
            K_rescaled[1, :] *= self.scale
            return K_rescaled
        return self.K


def calibrate_camera(img_dir: Path = Path("data/calibration")):
    """Compute camera intrinsics given a sample of checkerboard photos."""
    # Try to load cached calibration parameters
    CALIBRATION_FILENAME = "calibration_params.npz"
    CALIBRATION_PATH = img_dir / CALIBRATION_FILENAME
    if CALIBRATION_PATH.exists():
        print(f"Loading cached calibration parameters from: {CALIBRATION_PATH}")
        with np.load(CALIBRATION_PATH) as data:
            return data["K"], data["dist"]

    print("Calibrating camera...")
    # Checkerboard parameters
    CHECKERBOARD = (8, 6)  # inner corners (width, height)
    SQUARE_SIZE = 0.025  # meters (example)

    # Prepare object points (0,0,0), (1,0,0), ...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    images = list(img_dir.glob("*.jpg"))

    for fname in images:
        img = cv.imread(str(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # ty:ignore[no-matching-overload]

        ret, corners = cv.findChessboardCorners(
            gray,
            CHECKERBOARD,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            corners_refined = cv.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )

            objpoints.append(objp)
            imgpoints.append(corners_refined)

    # Camera calibration
    ret, K, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  # ty:ignore[no-matching-overload]

    # Cache the calibration parameters
    np.savez(CALIBRATION_PATH, K=K, dist=dist)

    return K, dist


@dataclass
class ViewData:
    """Represents a single image with extracted features and estimated camera pose.

    Stores image metadata, pixel data, extracted keypoints and descriptors,
    and the estimated camera pose (rotation and translation).

    Attributes:
        idx: Unique image index in the reconstruction.
        path: Path to the image file.
        pixels: RGB pixel data for rendering and debugging (H, W, 3).
        camera_model: Camera intrinsic model containing K and distortion coefficients.
        kp: Extracted keypoint locations as (N, 2) array of (x, y) coordinates.
        des: Feature descriptors as (N, D) array where D is descriptor dimension.
        cam_T_world: Pose of the world in camera frame.
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
    # Estimated camera pose; world-to-camera transform
    cam_T_world: SE3Pose | None = None

    def _check_pose(self):
        if not self.has_pose:
            raise ValueError("Pose not set for this image")

    @property
    def has_pose(self) -> bool:
        return self.cam_T_world is not None

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

    def set_pose(self, R, t):
        rotation = Rotation.from_matrix(R)
        self.cam_T_world = SE3Pose.from_components(t.squeeze(), rotation)

    def get_camera_center(self) -> NDArrayFloat:
        """Get the camera center in world coordinates."""
        self._check_pose()
        cam_center = np.zeros((3,))  # origin in camera coordinates
        # -R.T @ t
        return self.cam_T_world.inv().apply(cam_center)  # ty:ignore[possibly-missing-attribute]

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
        return self.cam_T_world.inv().apply(camera_pts)  # ty:ignore[possibly-missing-attribute]

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
        ext: Image file extension to look for (default "png").
        undistort: If True, applies undistortion to fisheye images using the provided camera model parameters.
        Only applicable if camera_model.model_type is FISHEYE.
    """

    def __init__(
        self,
        img_dir: Path,
        camera_model: CameraModel,
        max_size: int | None = None,
        max_frames: int | None = None,
        # TODO: add offset to skip first N images in dataset (skip calibration phase)
        ext: str = "png",
        undistort: bool = True,
    ):
        img_paths = sorted(list(Path(img_dir).glob(f"*.{ext}")))
        if not img_paths:
            raise ValueError(f"No *.{ext} images found in {img_dir}")

        self.img_paths = img_paths
        self._max_frames = max_frames
        self.max_size = max_size
        self.camera_model = camera_model
        self.undistort = undistort

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
        for idx, path in enumerate(self.img_paths):
            if self._max_frames and idx >= self._max_frames:
                print(f"FrameLoader: Reached max_frames={self._max_frames}, stopping further loading.")
                break

            # Grayscale loaded as (H, W, 3) with identical channels, color loaded as (H, W, 3) in RGB order
            img = cv.imread(str(path), cv.IMREAD_COLOR_RGB)
            if img is None:
                raise FileNotFoundError(f"FrameLoader: Failed to load image: {path}")

            # TODO: add undistort for pinhole
            camera_model = self.camera_model
            if self.undistort and self.camera_model.model_type == CameraType.FISHEYE:
                img, K_undistorted = self._undistort_fisheye(img)
                # After undistortion, it's pinhole camera with new intrinsics K_undistorted and no distortion
                camera_model = CameraModel(model_type=CameraType.PINHOLE, K=K_undistorted, dist=np.zeros(4))

            # Compute scale based on first image
            # Assumption: all images have the same resolution and thus the same scale factor applies to all
            if idx == 0 and self.max_size is not None:
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
        """Undistortion for equidistant fisheye."""
        h, w = img.shape[:2]

        # Create the undistortion + rectification map once (or cache it)
        # new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
        #     self.camera_model.K, self.camera_model.dist, (w, h), np.eye(3), balance=balance, fov_scale=fov_scale
        # )
        K, dist = self.camera_model.get_camera_matrix(), self.camera_model.dist
        new_K = K
        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, dist, np.eye(3), new_K, (w, h), cv.CV_16SC2)

        undist = cv.remap(img, map1, map2, cv.INTER_LINEAR)
        return undist, new_K


class FeatureExtractor:
    """Feature extractor class that curries the extraction function based on config."""

    def __init__(self, cfg: SfMConfig | SLAMConfig, loader: FrameLoader):
        self.cfg = cfg
        self.loader = loader

        if cfg.feature_type == "sift":
            sift = cv.SIFT_create(nfeatures=cfg.num_features)  # ty:ignore[unresolved-attribute]
            self._extract_fn = partial(self._extract_sift, sift=sift)
        elif cfg.feature_type == "disk":
            disk_model = KF.DISK.from_pretrained("depth", device=device).eval()
            self._extract_fn = partial(self._extract_disk, disk_model=disk_model)
        else:
            raise ValueError(f"FeatureExtractor: Unknown feature type {cfg.feature_type} in config!")

    def __call__(self, frame: ViewData) -> ViewData:
        """Extract features from a single frame using the curried extraction function."""
        return self._extract_fn(frame)

    def _extract_sift(self, frame: ViewData, sift: cv.SIFT) -> ViewData:
        """Extract SIFT features from a single image."""
        img_arr = frame.pixels  # (H, W, C)

        # Ensure grayscale is uint8 for SIFT
        if len(img_arr.shape) == 3:  # RGB image, convert to grayscale
            gray = cv.cvtColor(img_arr, cv.COLOR_RGB2GRAY)
        else:
            gray = img_arr

        keypoints, descriptors = sift.detectAndCompute(gray.astype(np.uint8), mask=None)

        # Convert list[cv.KeyPoint] to NDArray (N, 2)
        frame.kp = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        frame.des = descriptors  # ty:ignore[invalid-assignment]

        return frame

    def _extract_disk(self, frame: ViewData, disk_model: torch.nn.Module) -> ViewData:
        """Extract DISK features from a single image."""
        img_arr = frame.pixels  # (H, W, C)

        # Convert to float32 in [0, 1]
        max_val = np.iinfo(img_arr.dtype).max if img_arr.dtype != np.float32 else 1.0
        img_float = img_arr.astype(np.float32) / max_val

        # Convert to tensor and add batch dimension (H, W, C) -> (1, C, H, W)
        img_tensor = K.utils.image_to_tensor(img_float, keepdim=False).to(device=device)
        with torch.inference_mode():
            features = disk_model(img_tensor, self.cfg.num_features, pad_if_not_divisible=True)[0]

        frame.kp = features.keypoints.cpu().numpy()  # (N, 2)
        frame.des = features.descriptors.cpu().numpy()  # (N, D)

        # Memory management
        # del img_tensor, features
        # torch.cuda.empty_cache()
        return frame

    def iter_frames_with_features(self) -> Iterable[ViewData]:
        """Yields ImageData objects with extracted features."""
        for frame in self.loader.iter_frames():
            yield self(frame)


class FeatureStore:
    """Manages feature extraction and storage for multiple images.

    Handles loading images from a directory, extracting keypoints and descriptors
    using a provided FeatureExtractor, and storing the results. Camera intrinsics
    are stored in each ImageData object.

    Attributes:
        _store: List of ImageData objects containing extracted features for each image.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
    ):
        self._store: list[ViewData] = list(feature_extractor.iter_frames_with_features())

    def get_pixels(self, kp_keys: list[KPKey]) -> NDArray[np.uint8]:
        """Get pixel color for a given keypoint in an image."""
        pixels = np.zeros((len(kp_keys), 3), dtype=np.uint8)
        for i, (img_idx, kp_idx) in enumerate(kp_keys):
            kp_uv = self._store[img_idx].kp[kp_idx].astype(np.uint16)  # ty:ignore[not-subscriptable]
            pixels[i] = self._store[img_idx].pixels[*np.flip(kp_uv)]  # flip (x, y) to (row, col) for indexing
        return pixels

    @property
    def size(self) -> int:
        return len(self._store)

    def __getitem__(self, img_idx: int) -> ViewData:
        return self._store[img_idx]

    def get_keypoint(self, kp_key: KPKey) -> NDArrayFloat:
        """Get keypoint for a given keypoint key."""
        img_idx, kp_idx = kp_key
        return self._store[img_idx].kp[kp_idx]  # ty:ignore[not-subscriptable]

    def get_keypoints(self) -> list[NDArrayFloat]:
        """Get keypoints of all images."""
        return [img_data.kp for img_data in self._store]  # ty:ignore[invalid-return-type]

    def get_descriptors(self) -> list[NDArrayFloat]:
        """Get descriptors of all images."""
        return [img_data.des for img_data in self._store]  # ty:ignore[invalid-return-type]

    def iter_images_with_pose(self) -> Iterable[ViewData]:
        """Yield images for which we have a pose estimate."""
        yield from (img_data for img_data in self._store if img_data.has_pose)


@dataclass
class ViewEdge:
    i: int
    j: int
    inliers_ij: int  # matches i -> j
    inliers_ji: int  # matches j -> i

    @property
    def weight(self) -> int:
        # symmetric weight used for ranking
        # FIXME: # matches maximized when images identical, so this won't result in good baseline for triangulation
        # need large-enough relative translation for good baseline + enough plausible matches after geometric verification
        # see has_overlap()
        return min(self.inliers_ij, self.inliers_ji)


class ViewGraph:
    """
    Undirected weighted view graph with asymmetric match support.
    """

    def __init__(self):
        self.edges = []  # list of ViewEdge (global access)

    def add_edge(self, i, j, inliers_ij, inliers_ji):
        """
        Add or update an undirected edge between image i and j.
        """
        if i == j:
            return

        edge = ViewEdge(i, j, inliers_ij, inliers_ji)

        self.edges.append(edge)

    def best_edge(self) -> ViewEdge | None:
        """
        Return the edge with maximum symmetric weight.
        """
        return max(self.edges, key=lambda e: e.weight, default=None)


def _match_lightglue(
    img_from: ViewData,
    img_to: ViewData,
    lg_matcher: KF.LightGlueMatcher,
    min_dist: float | None = None,
) -> tuple[NDArrayFloat, NDArrayInt]:
    kp_from = torch.from_numpy(img_from.kp).to(device)
    des_from = torch.from_numpy(img_from.des).to(device)
    kp_to = torch.from_numpy(img_to.kp).to(device)
    des_to = torch.from_numpy(img_to.des).to(device)

    lafs_from = KF.laf_from_center_scale_ori(kp_from[None], torch.ones(1, len(kp_from), 1, 1, device=device))
    lafs_to = KF.laf_from_center_scale_ori(kp_to[None], torch.ones(1, len(kp_to), 1, 1, device=device))

    with torch.inference_mode():
        dists, idxs = lg_matcher(des_from, des_to, lafs_from, lafs_to)

        if min_dist:  # not None and > 0.0
            if min_dist < 0.0 or min_dist > 1.0:
                raise ValueError(f"min_dist must be in [0, 1], got {min_dist}")
            mask = (dists > min_dist).squeeze()
            dists, idxs = dists[mask], idxs[mask]

        # min_dist=0.0 is valid (retain all matches)
        return dists.detach().cpu().numpy(), idxs.detach().cpu().numpy()


def _match_brute_force(
    img_from: ViewData,
    img_to: ViewData,
    lowe_ratio: float | None = None,
    cross_check: bool = False,
) -> tuple[NDArrayFloat, NDArrayInt]:
    """Match descriptors using brute-force matcher with optional Lowe's ratio test.

    Args:
        img_from: Query image.
        img_to: Train image.
        lowe_ratio: Ratio threshold for Lowe's ratio test. If None, no filtering is applied.
        cross_check: If True, only keep matches that are mutual best matches.

    Returns:
        Tuple of (distances, matches):
            - distances: Array of match distances (N,).
            - matches: Array of match indices (N, 2) where each row is (queryIdx, trainIdx).

    Note:
        Using crossCheck=False means that multiple KPs in img_from can match to the same KP in img_to. Consequently,
        this might result in one KP in img_to triangulating to multiple 3D points.
        Using crossCheck=True would remove this ambiguity, but might also remove valid matches.
    """
    des_from, des_to = img_from.des, img_to.des
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=cross_check)
    if cross_check:
        matches = bf.match(des_from, des_to)  # ty:ignore[no-matching-overload]
    else:
        matches = bf.knnMatch(des_from, des_to, k=2)  # ty:ignore[no-matching-overload]
        if lowe_ratio:
            matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]
        else:
            matches = [m for m, n in matches]

    dist = np.array([m.distance for m in matches])  # (N,)
    return dist, np.array([(m.queryIdx, m.trainIdx) for m in matches])  # (N, 2)


def make_keypoint_matcher(
    cfg: SfMConfig | SLAMConfig,
) -> Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]]:
    if cfg.matcher_type == "bf":
        return partial(_match_brute_force, lowe_ratio=cfg.bf_lowe_ratio, cross_check=cfg.bf_cross_check)
    if cfg.matcher_type == "lg":
        lightglue_matcher = KF.LightGlueMatcher("disk").eval().to(device)
        return partial(_match_lightglue, min_dist=cfg.lg_min_dist, lg_matcher=lightglue_matcher)
    else:
        ValueError(f"Unknown matcher type {cfg.matcher_type=} in config!")


def has_overlap(
    img_from: ViewData,
    img_to: ViewData,
    matcher_fn: Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]] | None = None,
    min_inliers: int = 50,
) -> tuple[bool, int | None, NDArray | None]:
    """Returns True if there is sufficient overlap between two images.

    Args:
        img_from: Source image.
        img_to: Target image.
        K: Camera intrinsic matrix. If None, uses img_from.camera_model.K.
        matcher_fn: Keypoint matcher function. Required parameter.
        min_inliers: Minimum number of inliers to consider overlap sufficient.
    """
    if matcher_fn is None:
        raise ValueError("matcher_fn is required")

    K = img_from.camera_model.get_camera_matrix()

    _, good = matcher_fn(img_from, img_to)

    if len(good) < min_inliers:
        return False, None, None

    # geometric validation: rejects matches that cannot arise from a rigid 3D scene
    # [:, 0] = queryIdx; [:, 1] = trainIdx
    pts1, pts2 = img_from.kp[good[:, 0]], img_to.kp[good[:, 1]]  # ty:ignore[not-subscriptable]

    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0)

    if E is None:
        return False, None, None

    inliers = int((mask > 0).sum())
    if inliers < min_inliers:
        return False, None, None

    return True, inliers, good


def construct_view_graph(
    image_store: FeatureStore,
    matcher_fn: Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]],
    min_inliers: int = 50,
):
    view_graph = ViewGraph()
    kp, des = image_store.get_keypoints(), image_store.get_descriptors()
    assert len(kp) == len(des)

    N = len(kp)
    for i in range(N):
        for j in range(i + 1, N):
            img_i, img_j = image_store[i], image_store[j]
            # TODO: 1 direction enough, when matches symmetrical (e.g. crossCheck=True)
            ok_ij, inliers_ij, _ = has_overlap(img_i, img_j, matcher_fn, min_inliers)
            ok_ji, inliers_ji, _ = has_overlap(img_j, img_i, matcher_fn, min_inliers)
            # ASK: why the matches should not be preserved ???
            if ok_ij and ok_ji:
                view_graph.add_edge(i, j, inliers_ij, inliers_ji)

    return view_graph


class TrackManager:
    """
    Manages the mapping between 2D keypoint observations and 3D point tracks.

    A track represents a single 3D point observed across multiple images. Each track
    is identified by a unique track_id and contains a list of keypoint observations
    (KPKey = (img_id, kp_idx)) that correspond to the same 3D point.

    The class maintains a bidirectional mapping:
    - kp_to_track: maps each keypoint observation to its track_id; 1-to-1
    - track_to_kps: maps each track_id to all keypoint observations in that track; 1-to-N

    Note: This assumes symmetric keypoint matches (e.g., BFMatcher with crossCheck=True).
    If crossCheck=False, multiple keypoints in one image could match to the same keypoint
    in another image, which would violate the 1-to-1 constraint and is geometrically invalid.
    """

    def __init__(self):
        self.next_track_id = 0
        self.kp_to_track: dict[KPKey, int] = {}
        self.track_to_kps: dict[int, list[KPKey]] = {}

    def _register_keypoint_track(self, kp_key: KPKey, track_id: int):
        self.kp_to_track[kp_key] = track_id
        self.track_to_kps.setdefault(track_id, []).append(kp_key)

    def get_track(self, kp_key: KPKey) -> int | None:
        return self.kp_to_track.get(kp_key, None)

    def get_keypoints(self, track_id: int, img_idx: int | None = None) -> list[KPKey]:
        kp_keys = self.track_to_kps.get(track_id, [])
        if img_idx is not None:
            kp_keys = [kp_key for kp_key in kp_keys if kp_key[0] == img_idx]
        return kp_keys

    def get_triangulated_view_keypoints(self, image_idx: int) -> list[KPKey]:
        """Get keys of keypoints triangulated from a given view.
        image_idx: int Index of the camera view (image).
        """
        # return filter(lambda kpkey: kpkey[0] == image_idx, self.kp_to_track.keys())
        return [kpkey for kpkey in self.kp_to_track.keys() if kpkey[0] == image_idx]

    def get_triangulated_view_tracks(self, image_idx: int) -> list[int]:
        """Get track_ids of tracks triangulated from a given view."""
        # FIXME: not quite true; some of these were triangulated from other views: more like "tracks_in_view"
        kp_keys = self.get_triangulated_view_keypoints(image_idx)
        return [tid for kp_key in kp_keys if (tid := self.get_track(kp_key)) is not None]

    def add_new_tracks(self, kp_pairs: list[tuple[KPKey, KPKey]]):
        """Create new track for every given pair of KP observations."""
        added_track_ids = []
        for kp_key_pair in kp_pairs:  # kp_key = (img_idx, kp_idx)
            tid = self.next_track_id
            self._register_keypoint_track(kp_key_pair[0], tid)
            self._register_keypoint_track(kp_key_pair[1], tid)
            added_track_ids.append(tid)
            self.next_track_id += 1
        return added_track_ids

    def add_keypoints_to_tracks(self, kp_keys: list[KPKey], track_id: list[int]):
        """Add given KP observations to the given tracks."""
        for kp_key, tid in zip(kp_keys, track_id):
            self._register_keypoint_track(kp_key, tid)

    def get_track_observations_for_view(self, img_idx_ref: int, ref2new_matches: NDArrayInt):
        """Get track observations for a given view.

        Partitions supplied matches in `ref2new_matches` into:
          - tracked, which join KPs in reference view that have tracks to the KPs in the new view, and
          - untracked, which join KPs between the reference and the new view for which there are no tracks yet.

        Used for PnP pose estimation of the new view, where 2D-to-3D correspondences are needed.
        The 3D points are represented by `track_ids`, and the 2D points are represented by `tracked_kp_idxs_new`,
        which can be extracted from the returned `tracked_matches` by

        ```
        tracked_kp_idxs_new = tracked_matches[:, 1]
        ```

        Args:
            img_idx_ref: int Index of the reference image (view).
            ref2new_matches: NDArrayInt of shape (N, 2) containing matches between the reference image and the new image.

        Returns:
            track_ids: NDArray of shape (M,) containing track IDs visible from both the reference and the new view.
            These are the tracks that can be used for PnP pose estimation of the new view.
            tracked_matches: NDArray of shape (M, 2) containing matches of tracked keypoints between the reference
            and the new view. Used for PnP pose estimation of the new view.
            untracked_matches: NDArray of shape (K, 2) containing matches that do not correspond to any existing track
            (i.e., new tracks that can be triangulated from these matches).
        """
        track_ids, tracked_matches, untracked_matches = [], [], []
        for match in ref2new_matches:
            kp_idx_ref, _ = match
            kp_key_ref = (img_idx_ref, kp_idx_ref)
            if (track_id := self.get_track(kp_key_ref)) is not None:
                # tracked match indicates the same 3D world point is now observed by a new 2D KP
                track_ids.append(track_id)
                tracked_matches.append(match)
            else:  # reference view KP is untracked and therefore its matching KP from the new view is also untracked
                untracked_matches.append(match)

        return np.array(track_ids), np.array(tracked_matches), np.array(untracked_matches)

    def is_valid(self) -> bool:
        """Check consistency of the bi-directional map between track_id and KP_key.

        One track_id can map to multiple KPKeys, but one KPKey can only map to one track_id.
        Returns True if the mapping is consistent, False otherwise. Only True if KP matches are symmetrical
        (e.g. BFMatcher cross_check=True).
        """
        for track_id, kp_keys in self.track_to_kps.items():
            for kp_key in kp_keys:
                if self.kp_to_track[kp_key] != track_id:
                    return False
        return True


class PointCloud:
    def __init__(self):
        self._data: dict[int, Point3D] = {}  # track_id -> np.array([x, y, z])

    @property
    def size(self):
        return len(self._data)

    def add_points(self, track_ids: list[int], xyz: NDArray[Any]) -> None:
        assert len(track_ids) == xyz.shape[0], "Number of track_ids must match number of 3D points"
        for track_id, pt in zip(track_ids, xyz):
            self._data[track_id] = pt

    def set_point(self, track_id: int, xyz: Point3D):
        self._data[track_id] = xyz

    def get_point(self, track_id: int) -> Point3D | None:
        return self._data.get(track_id, None)

    def get_points_as_array(self, track_ids: list[int] | None = None) -> NDArrayFloat:
        """Returns array of 3D points corresponding to the given track_ids.

        Missing points are returned as np.nan.
        """
        if track_ids is None:
            return np.array(list(self._data.values()))
        return np.array([self._data.get(track_id, np.nan) for track_id in track_ids])

    def items(self) -> Iterable[tuple[int, Point3D]]:
        yield from self._data.items()


class ReconIO:
    """Handles saving and loading of reconstruction data (PLY files and gsplat tensors)."""

    Vertex = tuple[float, float, float]  # (x, y, z)
    Edge = tuple[int, int]  # (v1, v2)

    def __init__(self, point_cloud: PointCloud, images: FeatureStore, track_manager: TrackManager):
        self.point_cloud = point_cloud
        self.images = images
        self.track_manager = track_manager

    @staticmethod
    def _camera_frustum_points(R: NDArray[np.float32], t: NDArray[np.float32], scale: float = 0.1) -> list:
        """
        Returns 5 world-space points for a tiny camera frustum
        """
        # World --> Camera:   Xc = R Xw + t
        # Camera --> World:   Xw = Rᵀ (Xc − t)
        # Camera center in world coordinates: Xc=0 --> Xw = Rᵀ (0 − t) = -Rᵀ t
        C = -R.T @ t.squeeze()  # R.T @ (-t)

        # Camera axes in world frame
        right = R.T @ np.array([1, 0, 0])
        up = R.T @ np.array([0, 1, 0])
        forward = R.T @ np.array([0, 0, 1])

        # Image plane center
        P = C + scale * forward

        # Image plane corners
        s = scale * 0.5
        corners = [
            P + s * (right + up),
            P + s * (right - up),
            P + s * (-right - up),
            P + s * (-right + up),
        ]

        return [C] + corners

    def _add_camera_frustum(
        self,
        vertices: list[Vertex],
        edges: list[Edge],
        R: NDArray[np.float32],
        t: NDArray[np.float32],
        scale: float = 0.1,
    ):
        base_idx = len(vertices)

        pts = self._camera_frustum_points(R, t, scale)
        for p in pts:
            vertices.append((p[0], p[1], p[2]))

        # center → corners
        for i in range(1, 5):
            edges.append((base_idx, base_idx + i))

        # square around image plane
        edges += [
            (base_idx + 1, base_idx + 2),
            (base_idx + 2, base_idx + 3),
            (base_idx + 3, base_idx + 4),
            (base_idx + 4, base_idx + 1),
        ]

    def _get_point_colors(self) -> NDArray[np.uint8]:
        """Returns an array of RGB colors for each 3D point."""
        colors = np.zeros((self.point_cloud.size, 3), dtype=np.uint8)
        for track_id, pt in self.point_cloud.items():
            kp_keys = self.track_manager.track_to_kps[track_id]
            # average the colors of all KPs in the track
            colors[track_id] = self.images.get_pixels(kp_keys).mean(axis=0)
        return colors

    def save_for_gsplat(self, filename: Path):
        """Save SfM reconstruction as tensors for gsplat training.

        Saves:
            - poses: (N, 4, 4) camera-to-world transformation matrices
            - images: (N, H, W, 3) RGB images (float32, range [0, 1])
            - points: (M, 3) 3D point positions
            - colors: (M, 3) RGB colors for 3D points (float32, range [0, 1])
            - intrinsics: (3, 3) camera intrinsic matrix
        """
        filename.parent.mkdir(exist_ok=True, parents=True)

        # Collect camera poses as 4x4 matrices
        poses_list = []
        images_list = []

        for img_data in self.images.iter_images_with_pose():
            # Create 4x4 pose matrix [R | t; 0 0 0 1]
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :3] = img_data.R
            pose_4x4[:3, 3:4] = img_data.t[..., None]
            poses_list.append(pose_4x4)
            images_list.append(img_data.pixels)

        # Stack into tensors
        poses_tensor = torch.from_numpy(np.stack(poses_list, axis=0))  # (N, 4, 4)
        images_tensor = (
            torch.from_numpy(np.stack(images_list, axis=0)).float() / 255.0
        )  # (N, H, W, 3), normalized to [0, 1]

        # Get 3D points and colors (reuse existing method)
        points_3d = self.point_cloud.get_points_as_array()  # (M, 3)
        colors = self._get_point_colors()  # (M, 3) uint8

        points_tensor = torch.from_numpy(points_3d).float()  # (M, 3)
        colors_tensor = torch.from_numpy(colors).float() / 255.0  # (M, 3), normalized to [0, 1]

        # Get camera intrinsics (rescaled) from first image with pose
        K = self.images[0].camera_model.get_camera_matrix()
        intrinsics_tensor = torch.from_numpy(K).float()  # (3, 3)

        # Save as .pt file
        torch.save(
            {
                "poses": poses_tensor,
                "images": images_tensor,
                "points": points_tensor,
                "colors": colors_tensor,
                "intrinsics": intrinsics_tensor,
            },
            filename,
        )

        print("Saved reconstruction for gsplat:")
        print(f"  - {len(poses_list)} camera poses: {poses_tensor.shape}")
        print(f"  - {len(images_list)} images: {images_tensor.shape}")
        print(f"  - {self.point_cloud.size} 3D points: {points_tensor.shape}")
        print(f"  - Colors: {colors_tensor.shape}")
        print(f"  - Intrinsics: {intrinsics_tensor.shape}")
        print(f"  -> {filename}")

    @staticmethod
    def load_for_gsplat(
        filename: Path, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Load SfM reconstruction tensors saved for gsplat training.

        Returns:
            Tuple of (poses, images, points, colors, intrinsics, width, height):
                - poses: (N, 4, 4) camera-to-world transformation matrices
                - images: (N, H, W, 3) RGB images (float32, range [0, 1])
                - points: (M, 3) 3D point positions
                - colors: (M, 3) RGB colors for 3D points (float32, range [0, 1])
                - intrinsics: (N, 3, 3) camera intrinsic matrices (tiled for each camera)
                - width: image width (int)
                - height: image height (int)
        """
        data = torch.load(filename)

        poses = data["poses"]  # (N, 4, 4)
        images = data["images"]  # (N, H, W, 3)
        points = data["points"]  # (M, 3)
        colors = data["colors"]  # (M, 3)
        intrinsics_single = data["intrinsics"]  # (3, 3)

        # Get image dimensions from the images tensor
        N, H, W, _ = images.shape

        # Tile intrinsics to match number of cameras (N, 3, 3)
        intrinsics = intrinsics_single.unsqueeze(0).expand(N, 3, 3)

        print(f"Loaded reconstruction from {filename}:")
        print(f"  - {N} camera poses: {poses.shape}")
        print(f"  - {N} images: {images.shape}")
        print(f"  - {points.shape[0]} 3D points: {points.shape}")
        print(f"  - Colors: {colors.shape}")
        print(f"  - Intrinsics (tiled): {intrinsics.shape}")
        print(f"  - Image size: {W} x {H}")

        return poses.to(device), images.to(device), points.to(device), colors.to(device), intrinsics.to(device), W, H

    def save_ply(self, filename: Path = Path("point_cloud.ply")):
        import pandas as pd

        # Convert to DataFrame
        xyz = self.point_cloud.get_points_as_array()
        df = pd.DataFrame(xyz, columns=["x", "y", "z"])

        # --- OUTLIER REMOVAL ---
        # Calculate the distance from the median to find the "main cluster"
        median = df.median()
        distance = np.sqrt(((df - median) ** 2).sum(axis=1))

        # Keep only points within the 95th percentile of distance
        # This removes the "points at infinity" that squash your visualization
        distance_mask = distance < distance.quantile(0.95)
        df_filtered = df[distance_mask]
        colors_filtered = self._get_point_colors()[distance_mask]

        # Add camera frustums
        vertices = []
        edges = []
        for img in self.images.iter_images_with_pose():
            self._add_camera_frustum(vertices, edges, img.R, img.t)
        # edge indices offset by number of 3D points
        camera_vertex_offset = len(df_filtered)
        edges = [(e[0] + camera_vertex_offset, e[1] + camera_vertex_offset) for e in edges]

        num_vertices = len(df_filtered) + len(vertices)
        num_edges = len(edges)
        # --- DEBUG --- sanity checks
        # TODO: nicer stats
        print()
        print(f"{df.shape = }")
        print(f"# nans: {df.isna().sum().sum()}")
        print(f"# infs: {np.isinf(df.values).sum()}")
        norms = np.linalg.norm(df.values, axis=1)
        print(f"{norms.min() = }\n{norms.max() = }")
        print()
        print(f"{df_filtered.shape = }")

        print(f"Writing {len(df_filtered)} points to {filename}")
        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            # camera frustums: colored vertices + edges
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element edge {num_edges}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            # point cloud as white points
            for (x, y, z), (r, g, b) in zip(df_filtered.values, colors_filtered):
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            # camera frustums: red vertices + edges
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]} 255 0 0\n")
            for e in edges:  # TODO: edges don't work in my viewer
                f.write(f"{e[0]} {e[1]} 255 0 0\n")
