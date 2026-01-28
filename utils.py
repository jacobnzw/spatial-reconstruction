from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal

import cv2 as cv
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from numpy.typing import NDArray

device = K.utils.get_cuda_or_mps_device_if_available()

NDArrayFloat = NDArray[np.floating[Any]]
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


def extract_sift(img_path: Path):
    img = cv.imread(img_path)  # ty:ignore[no-matching-overload]
    # img = cv.resize(img, (800, 600))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create(nfeatures=10_000)  # ty:ignore[unresolved-attribute]
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des, img


def extract_disk(img_dir: Path, num_features: int = 2048) -> tuple[list[KF.DISKFeatures], torch.Tensor]:
    """Extract DISK features from all images in img_dir."""
    # Load all images
    # TODO: generalize to processing batch_size images at a time if img_dir has > batch_size images
    img_files = list(img_dir.glob("*.jpg"))
    imgs = []
    for img_path in img_files:
        imgs.append(K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device=device)[None, ...])
    imgs = torch.cat(imgs, dim=0)

    # Extract features
    disk = KF.DISK.from_pretrained("depth").to(device)
    features = disk(imgs, num_features, pad_if_not_divisible=True)
    return features, imgs


def match_disk(features1: KF.DISKFeatures, features2: KF.DISKFeatures, min_dist: float | None = None):
    """Match DISK features between two images using LightGlue matcher.

    Returns:
        dists: Distances between descriptors of matched keypoints of shape (M, 1)
        idxs: Match indices of shape (M, 2) into descriptors of features1 and features2, respectively.
    """
    lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)

    kps1, descs1 = features1.keypoints, features1.descriptors
    kps2, descs2 = features2.keypoints, features2.descriptors
    lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
    lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))
    # hw1 = torch.tensor(img1.shape[2:], device=device)  # optional image height and width for LightGlue
    # hw2 = torch.tensor(img2.shape[2:], device=device)
    dists, idxs = lg_matcher(descs1, descs2, lafs1, lafs2)
    if min_dist is not None:
        if min_dist <= 0.0 or min_dist > 1.0:
            raise ValueError(f"min_dist must be in (0, 1], got {min_dist}")
        mask = dists > min_dist
        return dists[mask], idxs[mask]
    return dists, idxs


@dataclass
class ImageData:
    idx: int
    path: Path
    # TODO; rename to "pixels"
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
    def __init__(self, img_dir: Path, method: Literal["sift", "disk"] = "sift"):
        self._store: list[ImageData] = []
        if method == "sift":
            self._load_dir_sift(img_dir)
        else:
            self._load_dir_disk(img_dir)

    def _load_dir_sift(self, img_dir: Path, ext: str = "jpg"):
        img_paths = sorted(list(img_dir.glob(f"*.{ext}")))

        if not img_paths:
            raise ValueError(f"No *.{ext} images found in {img_dir}")

        for idx, filepath in enumerate(img_paths):
            # TODO: use SIFT from kornia to match better with DISK
            kps, des, img = extract_sift(filepath)
            kp = np.array([kp.pt for kp in kps])  # list[cv.KeyPoint] -> NDArray (N, 2)
            self._store.append(ImageData(idx, filepath, img, kp, des))

    # TODO: move feature extraction to this class; join w/ _load_dir
    def _load_dir_disk(self, img_dir: Path):
        features, imgs = extract_disk(img_dir)
        for idx, filepath in enumerate(img_dir.glob("*.jpg")):
            self._store.append(
                ImageData(
                    idx,
                    filepath,
                    imgs[idx].cpu().numpy(),
                    features[idx].keypoints.cpu().numpy(),  # (N, 2)
                    features[idx].descriptors.cpu().numpy(),  # (N, D)
                )
            )

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


def compute_matches(
    des_from: NDArrayFloat,
    des_to: NDArrayFloat,
    lowe_ratio: float | None = None,
):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_from, des_to, k=2)
    if lowe_ratio:
        matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]
    return matches


def has_overlap(
    kp1: NDArrayFloat,
    des1: NDArrayFloat,
    kp2: NDArrayFloat,
    des2: NDArrayFloat,
    K: NDArray,
    min_inliers: int = 50,
):
    good = compute_matches(des1, des2, lowe_ratio=0.75)

    if len(good) < min_inliers:
        return False, None, None

    # geometric validation: rejects matches that cannot arise from a rigid 3D scene
    pts1 = kp1[[m.queryIdx for m in good]]
    pts2 = kp2[[m.trainIdx for m in good]]

    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0)

    if E is None:
        return False, None, None

    inliers = int(mask.sum())
    if inliers < min_inliers:
        return False, None, None

    return True, inliers, good


def construct_view_graph(kp: list[NDArrayFloat], des: list[NDArrayFloat], K: NDArray):
    view_graph = ViewGraph()
    assert len(kp) == len(des)
    N = len(kp)
    for i in range(N):
        for j in range(i + 1, N):
            ok_ij, inliers_ij, _ = has_overlap(kp[i], des[i], kp[j], des[j], K)
            ok_ji, inliers_ji, _ = has_overlap(kp[j], des[j], kp[i], des[i], K)
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

    def update_tracks(self, img_idx_new: int, img_idx_ref: int, ref2new_matches: list[cv.DMatch]):
        """Update tracks with new KP observations from img_new.

        img_ref is the reference image for which we already have 2D-3D pt correspondence in track_manager.
        Some of the KPs in img_ref have tracks, some don't. The KPs in img_new that match the tracked KPs in img_ref
        are added to those tracks. The KPs in img_new that match the untracked KPs in img_ref are added to new tracks.
        """
        # for each match, if ref KP has a track, add new img KP to track; else create new track
        kp_idx_new_tracked, track_ids_tracked = [], []
        matches_untracked = []
        # I wanna add the new KPs (that have matches to tracked ref KPs) to current tracks
        for m in ref2new_matches:
            kp_idx_ref, kp_idx_new = m.queryIdx, m.trainIdx
            # if ref KP has a track, add the matching new KP to the same track
            # this means the same 3D object point is now observed by a new 2D KP
            kp_key_new = (img_idx_new, kp_idx_new)
            kp_key_ref = (img_idx_ref, kp_idx_ref)
            if (track_id := self.get_track(kp_key_ref)) is not None:
                track_ids_tracked.append(track_id)
                kp_idx_new_tracked.append(kp_idx_new)
                self._register_keypoint_track(kp_key_new, track_id)
            else:  # ref KP is untracked and therefore its matching KP from img_new is also untracked
                matches_untracked.append(m)

        # return tracked KPs in new img and ref2new matches for untracked KPs for triangulation
        return track_ids_tracked, kp_idx_new_tracked, matches_untracked


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
        self.xyz = point_cloud.get_points_as_array()
        self.images = images

    @staticmethod
    def _camera_frustum_points(R: NDArray[np.float32], t: NDArray[np.float32], scale: float = 0.1) -> list:
        """
        Returns 5 world-space points for a tiny camera frustum
        """
        # World --> Camera:   Xc = R Xw + t
        # Camera --> World:   Xw = Rᵀ (Xc − t)
        # Camera center in world coordinates: Xc=0 --> Xw = Rᵀ (0 − t) = -Rᵀ t
        C = -R.T @ t  # R.T @ (-t)

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
        df = pd.DataFrame(self.xyz, columns=["x", "y", "z"])

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
