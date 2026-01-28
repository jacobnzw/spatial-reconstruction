from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Iterable, Literal

import cv2 as cv
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from numpy.typing import NDArray

device = K.utils.get_cuda_or_mps_device_if_available()

NDArrayFloat = NDArray[np.floating[Any]]
NDArrayInt = NDArray[np.integer[Any]]
Point3D = Annotated[NDArrayFloat, Literal[3]]


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
class ImageData:
    idx: int
    path: Path
    # Useful for rendering and debugging
    pixels: NDArray[Any]
    # Extracted keypoints and descriptors
    kp: NDArrayFloat
    des: NDArrayFloat
    # Estimated pose
    R: NDArrayFloat | None = None
    t: NDArrayFloat | None = None

    def set_pose(self, R, t):
        self.R = R
        self.t = t

    @property
    def has_pose(self) -> bool:
        return self.R is not None and self.t is not None

    @property
    def pose_matrix(self) -> NDArray[np.float32]:
        return np.hstack((self.R, self.t))  # ty:ignore[no-matching-overload]

    @property
    def rvec(self) -> NDArray[np.float32]:
        return cv.Rodrigues(self.R)[0].ravel()  # ty:ignore[no-matching-overload]


class FeatureStore:
    def __init__(self, img_dir: Path, method: Literal["sift", "disk"] = "sift", num_features: int = 2048):
        self._store: list[ImageData] = []
        self._disk_model = None  # Lazy load DISK model
        self._load_dir(img_dir, method=method, num_features=num_features)

    def _extract_sift(self, img_path: Path, num_features: int) -> tuple[NDArrayFloat, NDArrayFloat, NDArray[Any]]:
        """Extract SIFT features from a single image."""
        img = cv.imread(str(img_path))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # ty:ignore[no-matching-overload]

        sift = cv.SIFT_create(nfeatures=num_features)  # ty:ignore[unresolved-attribute]
        kps, des = sift.detectAndCompute(gray, None)

        # Convert list[cv.KeyPoint] to NDArray (N, 2)
        kp = np.array([kp.pt for kp in kps], dtype=np.float32)

        return kp, des, img

    def _extract_disk(self, img_path: Path, num_features: int) -> tuple[NDArrayFloat, NDArrayFloat, NDArray[Any]]:
        """Extract DISK features from a single image."""
        # Lazy load DISK model
        if self._disk_model is None:
            self._disk_model = KF.DISK.from_pretrained("depth").to(device)

        # Load image
        img_tensor = K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device=device)[None, ...]

        # Extract features
        with torch.inference_mode():
            features = self._disk_model(img_tensor, num_features, pad_if_not_divisible=True)[0]

        kp = features.keypoints.cpu().numpy()  # (N, 2)
        des = features.descriptors.cpu().numpy()  # (N, D)
        img = img_tensor[0].cpu().numpy()  # (C, H, W) -> convert to numpy

        return kp, des, img

    def _load_dir(self, img_dir: Path, method: Literal["sift", "disk"], num_features: int, ext: str = "jpg"):
        """Load all images from directory and extract features using specified method."""
        img_paths = sorted(list(img_dir.glob(f"*.{ext}")))

        if not img_paths:
            raise ValueError(f"No *.{ext} images found in {img_dir}")

        for idx, filepath in enumerate(img_paths):
            if method == "sift":
                kp, des, img = self._extract_sift(filepath, num_features)
            else:  # disk
                kp, des, img = self._extract_disk(filepath, num_features)

            self._store.append(ImageData(idx, filepath, img, kp, des))

    @property
    def size(self) -> int:
        return len(self._store)

    def __getitem__(self, img_idx: int) -> ImageData:
        return self._store[img_idx]

    def get_keypoints(self) -> list[NDArrayFloat]:
        """Get keypoints of all images."""
        return [img_data.kp for img_data in self._store]

    def get_descriptors(self) -> list[NDArrayFloat]:
        """Get descriptors of all images."""
        return [img_data.des for img_data in self._store]

    def iter_images_with_pose(self) -> Iterable[ImageData]:
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
        return min(self.inliers_ij, self.inliers_ji)


class ViewGraph:
    """
    Undirected weighted view graph with asymmetric match support.
    """

    def __init__(self):
        self.adj = defaultdict(dict)  # adj[i][j] = ViewEdge
        self.edges = []  # list of ViewEdge (global access)

    def add_edge(self, i, j, inliers_ij, inliers_ji):
        """
        Add or update an undirected edge between image i and j.
        """
        if i == j:
            return

        edge = ViewEdge(i, j, inliers_ij, inliers_ji)

        self.adj[i][j] = edge
        self.adj[j][i] = edge
        self.edges.append(edge)

    def neighbors(self, i) -> dict[int, ViewEdge]:
        """
        Return neighbors of image i with weights.
        """
        return self.adj[i]

    def best_edge(self) -> ViewEdge | None:
        """
        Return the edge with maximum symmetric weight.
        """
        return max(self.edges, key=lambda e: e.weight, default=None)


def _match_lightglue_disk(
    kp_from: torch.Tensor,
    des_from: torch.Tensor,
    kp_to: torch.Tensor,
    des_to: torch.Tensor,
    lg_matcher: KF.LightGlueMatcher,
    min_dist: float | None = None,
) -> tuple[NDArrayFloat, NDArrayInt]:
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


def _match_bf(
    des_from: NDArrayFloat, des_to: NDArrayFloat, lowe_ratio: float | None = None
) -> tuple[NDArrayFloat, NDArrayInt]:
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_from, des_to, k=2)
    if lowe_ratio:
        matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]
    else:
        matches = [m for m, n in matches]

    dist = np.array([m.distance for m in matches])  # (N,)
    return dist, np.array([(m.queryIdx, m.trainIdx) for m in matches])  # (N, 2)


def compute_matches(
    img_from: ImageData,
    img_to: ImageData,
    method: Literal["lightglue", "bf"] = "bf",
    lowe_ratio: float | None = None,
    min_dist: float | None = None,
    lightglue_matcher: KF.LightGlueMatcher | None = None,
) -> tuple[NDArrayFloat, NDArrayInt]:
    if method == "bf":
        return _match_bf(img_from.des, img_to.des, lowe_ratio)
    elif method == "lightglue":
        if lightglue_matcher is None:
            raise ValueError("lightglue_matcher must be provided when method='lightglue'")
        return _match_lightglue_disk(
            torch.from_numpy(img_from.kp).to(device),
            torch.from_numpy(img_from.des).to(device),
            torch.from_numpy(img_to.kp).to(device),
            torch.from_numpy(img_to.des).to(device),
            lightglue_matcher,
            min_dist,
        )
    else:
        raise ValueError(f"Unknown KP matching method: {method}")


def has_overlap(
    img_from: ImageData,
    img_to: ImageData,
    K: NDArray,
    matcher_fn: Callable[[ImageData, ImageData], tuple[NDArrayFloat, NDArrayInt]],
    min_inliers: int,
) -> tuple[bool, int | None, NDArray | None]:
    """Returns True if there is sufficient overlap between two images."""
    # TODO: pass in matcher_fn!
    _, good = compute_matches(img_from, img_to, lowe_ratio=0.75)

    if len(good) < min_inliers:
        return False, None, None

    # geometric validation: rejects matches that cannot arise from a rigid 3D scene
    pts1 = img_from.kp[good[:, 0]]  # [:, 0] = queryIdx; [:, 1] = trainIdx
    pts2 = img_to.kp[good[:, 1]]

    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0)

    if E is None:
        return False, None, None

    inliers = int((mask > 0).sum())
    if inliers < min_inliers:
        return False, None, None

    return True, inliers, good


def construct_view_graph(
    image_store: FeatureStore,
    K: NDArray,
    matcher_fn: Callable[[ImageData, ImageData], tuple[NDArrayFloat, NDArrayInt]],
    min_inliers: int = 50,
):
    view_graph = ViewGraph()
    kp, des = image_store.get_keypoints(), image_store.get_descriptors()
    assert len(kp) == len(des)

    N = len(kp)
    for i in range(N):
        for j in range(i + 1, N):
            img_i, img_j = image_store[i], image_store[j]
            ok_ij, inliers_ij, _ = has_overlap(img_i, img_j, K, matcher_fn, min_inliers)
            ok_ji, inliers_ji, _ = has_overlap(img_j, img_i, K, matcher_fn, min_inliers)
            # ASK: why the matches should not be preserved ???
            if ok_ij and ok_ji:
                view_graph.add_edge(i, j, inliers_ij, inliers_ji)

    return view_graph


class TrackManager:
    KPKey = tuple[int, int]  # (img_id, kp_idx)

    def __init__(self):
        self.next_track_id = 0
        self.kp_to_track: dict[self.KPKey, int] = {}
        self.track_to_kps: dict[int, list[self.KPKey]] = {}

    def _register_keypoint_track(self, kp_key: KPKey, track_id: int):
        self.kp_to_track[kp_key] = track_id
        self.track_to_kps.setdefault(track_id, []).append(kp_key)

    def get_track(self, kp_key: KPKey):
        return self.kp_to_track.get(kp_key, None)

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

    def update_tracks(self, img_idx_new: int, img_idx_ref: int, ref2new_matches: NDArrayInt):
        """Update tracks with new KP observations from img_new.

        img_ref is the reference image for which we already have 2D-3D pt correspondence in track_manager.
        Some of the KPs in img_ref have tracks, some don't. The KPs in img_new that match the tracked KPs in img_ref
        are added to those tracks. The KPs in img_new that match the untracked KPs in img_ref are added to new tracks.
        """
        # for each match, if ref KP has a track, add new img KP to track; else create new track
        kp_idx_new_tracked, track_ids_tracked = [], []
        matches_untracked = []
        # I wanna add the new KPs (that have matches to tracked ref KPs) to current tracks
        for match in ref2new_matches:
            kp_idx_ref, kp_idx_new = match
            # if ref KP has a track, add the matching new KP to the same track
            # this means the same 3D object point is now observed by a new 2D KP
            kp_key_new = (img_idx_new, kp_idx_new)
            kp_key_ref = (img_idx_ref, kp_idx_ref)
            if (track_id := self.get_track(kp_key_ref)) is not None:  # ty:ignore[invalid-argument-type]
                track_ids_tracked.append(track_id)
                kp_idx_new_tracked.append(kp_idx_new)
                self._register_keypoint_track(kp_key_new, track_id)  # ty:ignore[invalid-argument-type]
            else:  # ref KP is untracked and therefore its matching KP from img_new is also untracked
                matches_untracked.append(match)

        # return tracked KPs in new img and ref2new matches for untracked KPs for triangulation
        return track_ids_tracked, kp_idx_new_tracked, np.array(matches_untracked)


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


class ReconExporter:
    """Exports a reconstruction to a PLY file."""

    Vertex = tuple[float, float, float]  # (x, y, z)
    Edge = tuple[int, int]  # (v1, v2)

    def __init__(self, point_cloud: PointCloud, images: FeatureStore):
        self.point_cloud = point_cloud
        self.images = images

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
        df_filtered = df[distance < distance.quantile(0.95)]

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
            for x, y, z in df_filtered.values:
                f.write(f"{x} {y} {z} 255 255 255\n")
            # camera frustums: red vertices + edges
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]} 255 0 0\n")
            for e in edges:
                f.write(f"{e[0]} {e[1]} 255 0 0\n")
