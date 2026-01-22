from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal

import cv2 as cv
import numpy as np
import pyceres
import pycolmap
from pycolmap import cost_functions
from numpy.typing import NDArray


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

    sift = cv.SIFT_create(nfeatures=10_000)
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des, img


@dataclass
class ImageData:
    idx: int
    path: Path
    img: NDArray[Any]
    kp: list[cv.KeyPoint]
    des: NDArray[np.floating[Any]]
    R: NDArray[np.floating[Any]] | None = None
    t: NDArray[np.floating[Any]] | None = None

    def set_pose(self, R, t):
        self.R = R
        self.t = t

    @property
    def has_pose(self) -> bool:
        return self.R is not None and self.t is not None

    @property
    def pose_matrix(self) -> NDArray[np.float32]:
        return np.hstack((self.R, self.t))

    @property
    def rvec(self) -> NDArray[np.float32]:
        return cv.Rodrigues(self.R)[0].ravel()


class FeatureStore:
    def __init__(self, img_dir: Path):
        self._store: list[ImageData] = []
        self._load_dir(img_dir)

    def _load_dir(self, img_dir: Path, ext: str = "jpg"):
        img_paths = sorted(list(img_dir.glob(f"*.{ext}")))

        if not img_paths:
            raise ValueError(f"No *.{ext} images found in {img_dir}")

        for idx, filepath in enumerate(img_paths):
            kp, des, img = extract_sift(filepath)
            self._store.append(ImageData(idx, filepath, img, kp, des))

    @property
    def size(self) -> int:
        return len(self._store)

    def __getitem__(self, img_idx: int) -> ImageData:
        return self._store[img_idx]

    def get_keypoints(self) -> list[list[cv.KeyPoint]]:
        """Get keypoints of all images."""
        return [img_data.kp for img_data in self._store]

    def get_descriptors(self) -> list[NDArray[np.floating[Any]]]:
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
    des_from: NDArray[np.float32],
    des_to: NDArray[np.float32],
    lowe_ratio: float | None = None,
):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_from, des_to, k=2)
    if lowe_ratio:
        matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]
    return matches


def has_overlap(kp1, des1, kp2, des2, K, min_inliers=50):
    good = compute_matches(des1, des2, lowe_ratio=0.75)

    if len(good) < min_inliers:
        return False, None, None

    # geometric validation: rejects matches that cannot arise from a rigid 3D scene
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0)  # ty:ignore[no-matching-overload]

    if E is None:
        return False, None, None

    inliers = int(mask.sum())
    if inliers < min_inliers:
        return False, None, None

    return True, inliers, good


def construct_view_graph(kp: list, des: list, K):
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


# Type alias for 3D points in Euclidean space
Point3D = Annotated[NDArray[np.floating[Any]], Literal[3]]


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

    def get_points_as_array(self, track_ids: list[int]) -> NDArray[np.floating[Any]]:
        """Returns array of 3D points corresponding to the given track_ids.

        Missing points are returned as np.nan.
        """
        return np.array([self._data.get(track_id, np.nan) for track_id in track_ids])

    def items(self) -> Iterable[tuple[int, Point3D]]:
        yield from self._data.items()

    def save_ply(self, filename: Path = Path("point_cloud.ply")):
        import pandas as pd

        # Convert to DataFrame
        df = pd.DataFrame(self._data.values(), columns=["x", "y", "z"])

        # --- OUTLIER REMOVAL ---
        # Calculate the distance from the median to find the "main cluster"
        median = df.median()
        distance = np.sqrt(((df - median) ** 2).sum(axis=1))

        # Keep only points within the 95th percentile of distance
        # This removes the "points at infinity" that squash your visualization
        df_filtered = df[distance < distance.quantile(0.95)]

        # --- DEBUG --- sanity checks
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
            f.write(f"element vertex {self.size}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for x, y, z in df_filtered.values:
                f.write(f"{x} {y} {z}\n")


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

    def add_new_tracks(self, kp_obs: list[KPKey]):
        """Create new track for every given KP observation."""
        added_track_ids = []
        for kp_key in kp_obs:  # kp_key = (img_idx, kp_idx)
            tid = self.next_track_id
            self._register_keypoint_track(kp_key, tid)
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
        kp_keys_pending, matches_untracked = [], []
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
                # postpone creating new tracks until after triangulation; return the pending KPs to be added later
                kp_keys_pending.append(kp_key_new)

        # return tracked KPs in new img and ref2new matches for untracked KPs for triangulation
        return track_ids_tracked, kp_idx_new_tracked, kp_keys_pending, matches_untracked


def compute_baseline_estimate(
    img_0: ImageData, img_1: ImageData, K, track_manager: TrackManager, point_cloud: PointCloud
):
    """Computes two-view baseline estimate of 3D points and poses

    First image is at the origin.
    """

    # Match key points (via descriptors)
    print(f"baseline: Computing matches from {img_0.idx}:{img_0.path.name} to {img_1.idx}:{img_1.path.name}")
    matches = compute_matches(img_0.des, img_1.des, lowe_ratio=0.75)

    # extract corresponding pixel coordinates
    pts0 = np.array([img_0.kp[m.queryIdx].pt for m in matches]).astype(np.float32)
    pts1 = np.array([img_1.kp[m.trainIdx].pt for m in matches]).astype(np.float32)

    # compute Essential matrix using camera intrinsics; mask indicates inliers
    E, mask = cv.findEssentialMat(pts0, pts1, K, method=cv.RANSAC, prob=0.999, threshold=1.0)

    # estimate camera pose & triangulate 3D points; mask refined to include only triangulatable points
    retval, R, t, mask, points_4d = cv.recoverPose(
        E=E,
        points1=pts0,
        points2=pts1,
        cameraMatrix=K,
        distanceThresh=50.0,  # mandatory for triangulation
        mask=mask,  # input mask selects 2D points to include in triangulation
    )

    # Homogeneous --> Euclidean; filter out outliers
    inliers = mask.ravel() > 0
    points_3d = (points_4d[:3, inliers] / points_4d[3, inliers]).T
    matches = [m for m, inlier in zip(matches, inliers) if inlier]

    # Create new tracks for the triangulated 3D object points
    # first create tracks for KPs in img_0, then add KPs in img_1 that match to KPs in img_0
    track_ids_added = track_manager.add_new_tracks([(img_0.idx, m.queryIdx) for m in matches])
    _, _, _, matches_untracked = track_manager.update_tracks(img_1.idx, img_0.idx, matches)
    assert len(matches_untracked) == 0, "No additional tracks should be created by now."

    point_cloud.add_points(track_ids_added, points_3d)

    # Update image data structs w/ new estimates: img_0 is at origin, img_1 is at (R, t)
    img_0.set_pose(np.eye(3), np.zeros((3, 1)))
    img_1.set_pose(R, t)

    print(f"Baseline constructed with {len(points_3d)} 3D points.")


def add_view(img_new: ImageData, img_ref: ImageData, K, dist, track_manager: TrackManager, point_cloud: PointCloud):
    """Adds 3D points from new view using PnP and triangulation.

    img_ref is reference image for which we already have 2D-3D pt correspondence in track_manager
    """
    # Compute KP matches from ref image to new image
    # Matching from new to ref image: Where does ref img tracked KP match to in new img?
    print(f"add_view: Computing matches from {img_ref.idx}:{img_ref.path.name} to {img_new.idx}:{img_new.path.name}")
    matches = compute_matches(img_ref.des, img_new.des, lowe_ratio=0.75)

    # add new img KPs, that are matched to from tracked ref img KPs, to current tracks (3D pts)
    # returns track_ids and (un)tracked KPs in the new image; track_ids used as indices to point cloud
    track_ids_tracked, kp_idx_new_tracked, kp_keys_pending, matches_untracked = track_manager.update_tracks(
        img_new.idx, img_ref.idx, matches
    )

    # Estimate pose of new image
    # Only use object points corresponding to tracked KPs in img_ref for PnP pose estimation
    object_points = point_cloud.get_points_as_array(track_ids_tracked)
    # PnP needs tracked KPs from new image (2D) and matching 3D object pts
    # In other words, new 2D points that observe the same 3D object points as the tracked KPs in the ref image
    img_new_pts_tracked = np.array([img_new.kp[i].pt for i in kp_idx_new_tracked]).astype(np.float32)

    assert len(object_points) == len(img_new_pts_tracked), "Number of 3D points must match number of 2D points"
    assert np.isfinite(object_points).all(), "Object points must be finite"
    assert np.isfinite(img_new_pts_tracked).all(), "Image points must be finite"

    print(f"Estimating pose of {img_new.idx}:{img_new.path.name} with {len(object_points)} 3D-2D correspondences...")
    pnp_ok, rvec, tvec, inliers = cv.solvePnPRansac(
        object_points,
        img_new_pts_tracked,
        K,
        dist,
        flags=cv.SOLVEPNP_EPNP,
    )
    if not pnp_ok:
        raise ValueError("solvePnP failed to estimate pose.")
    print(f"Pose estimation succeeded with {len(inliers)} inliers")
    # Estimated pose is relative to 3D point frame (i.e. the world frame); no pose composition required
    R = cv.Rodrigues(rvec)[0]
    img_new.set_pose(R, tvec)

    # TODO: optionally add reprojection error based KP filtering, followed by PnP re-estimation

    # Triangulate untracked KPs in the new image
    # pts_ref[i] matched to pts_new[i]
    pts_ref = np.array([img_ref.kp[m.queryIdx].pt for m in matches_untracked]).astype(np.float32)
    pts_new = np.array([img_new.kp[m.trainIdx].pt for m in matches_untracked]).astype(np.float32)

    # Projection matrices: from 3D world to each camera 2D image plane
    P_ref = K @ img_ref.pose_matrix
    P_new = K @ img_new.pose_matrix

    # Triangulate the untracked KPs in the new image
    points_4d = cv.triangulatePoints(P_ref, P_new, pts_ref.T, pts_new.T)
    points_3d = (points_4d[:3] / points_4d[3]).T

    track_ids_added = track_manager.add_new_tracks(kp_keys_pending)
    point_cloud.add_points(track_ids_added, points_3d)
    print(f"Added {len(points_3d)} 3D points.")


def pick_best_image_pair(
    edges: list[ViewEdge], store: FeatureStore, R: set[int] | None = None
) -> tuple[ImageData, ImageData, ViewEdge]:
    """Pick best image pair from list of edges.

    Assumption: ImageData.idx matches the node indexes, which it should if the graph was constructed correctly.
    """
    best_edge = max(edges, key=lambda e: e.weight)
    if R:  # if R is not None or not empty
        idx_ref, idx_new = (best_edge.i, best_edge.j) if best_edge.i in R else (best_edge.j, best_edge.i)
        return store[idx_ref], store[idx_new], best_edge
    return store[best_edge.i], store[best_edge.j], best_edge


def process_graph_component(
    K,
    dist,
    edges: list[ViewEdge],
    store: FeatureStore,
    track_manager: TrackManager,
    point_cloud: PointCloud,
) -> tuple[list[ViewEdge], set[int]]:
    # Pick strongest baseline:
    # - The edge of the view graph with greatest weight (ie. # kp matches) determines the two images
    img_0, img_1, best_edge = pick_best_image_pair(edges, store)
    print(
        f"Establishing baseline ({best_edge.weight} matches) from: {img_0.idx}:{img_0.path.name} and {img_1.idx}:{img_1.path.name}"
    )

    # matches -> E -> pose -> triangulation
    compute_baseline_estimate(img_0, img_1, K, track_manager, point_cloud)

    R = set((img_0.idx, img_1.idx))
    U = {node for e in edges for node in (e.i, e.j)}
    U.difference_update(R)
    leftover_edges = edges.copy()
    leftover_edges.remove(best_edge)

    while True:
        # find unregistered images connected to the registered ones
        # "connected" == "sharing matched keypoints"
        candidate_edges = [e for e in edges if (e.i in R and e.j in U) or (e.j in R and e.i in U)]

        if not candidate_edges:
            # U could still be non-empty (disconnected graph)
            print(f"No more candidate edges. Exiting. {len(U)} images left.")
            break

        img_ref, img_new, best_edge = pick_best_image_pair(candidate_edges, store, R)
        print(
            (
                f"\nAdding view {img_new.idx}:{img_new.path.name} w/ ref {img_ref.idx}:{img_ref.path.name}"
                f"(matches: {best_edge.weight})"
            )
        )
        try:
            # matches --> 2D-3D pairs --PnP--> pose -> triangulate untracked
            add_view(img_new, img_ref, K, dist, track_manager, point_cloud)
        except ValueError as e:
            # failed to add new view: indicate the (img_ref, img_new) pair as bad and move on
            # best_edge was the best chance to add img_new (don't consider next best edge w/ img_new)
            U.remove(img_new.idx)
            leftover_edges.remove(best_edge)
            print(
                f"Failed to add view: {img_new.idx}:{img_new.path.name} with ref: {img_ref.idx}:{img_ref.path.name} due to {e}"
            )
            continue

        # move currently processed image/node index from U to R
        R.add(img_new.idx)
        U.remove(img_new.idx)
        leftover_edges.remove(best_edge)

    # No use for edges involving views that were already registered (ie. in R)
    leftover_edges = [e for e in leftover_edges if not (e.i in R or e.j in R)]
    print(f"{R = }\n{U = }")
    print(f"leftover_edges {[(e.i, e.j) for e in leftover_edges]}")

    return leftover_edges, U


def bundle_adjustment(
    images: FeatureStore,
    point_cloud: PointCloud,
    K: NDArray[np.float32],
    track_manager: TrackManager,
    fix_first_camera: bool = True,
):
    """Run bundle adjustment on all cameras and 3D points using pycolmap cost functions."""

    print("Running bundle adjustment...")

    # TODO: Check if this is correct! Plenty of weirdness going on here...

    # Create pycolmap camera model (PINHOLE: fx, fy, cx, cy)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    cam_params = np.array([fx, fy, cx, cy], dtype=np.float64)
    camera_model = pycolmap.CameraModelId.PINHOLE

    # Prepare camera poses (as pycolmap.Rigid3d)
    camera_poses = {}
    for img in images.iter_images_with_pose():
        # Create Rigid3d (cam_from_world transformation)
        # pycolmap.Rotation3d can be constructed directly from rotation matrix
        camera_poses[img.idx] = pycolmap.Rigid3d(rotation=pycolmap.Rotation3d(img.R), translation=img.t.copy())

    # Prepare 3D points
    point_params = {track_id: xyz.copy() for track_id, xyz in point_cloud.items()}

    # Build the optimization problem
    problem = pyceres.Problem()
    loss = pyceres.HuberLoss(1.0)  # Robust loss for outliers

    # Add residual blocks for each observation
    for track_id, kp_keys in track_manager.track_to_kps.items():
        if track_id not in point_params:
            continue

        point_3d = point_params[track_id].astype(np.float64)

        for img_idx, kp_idx in kp_keys:
            if img_idx not in camera_poses:
                continue

            # Get observed 2D point
            observed_pt = np.array(images[img_idx].kp[kp_idx].pt, dtype=np.float64)

            # Create cost function using pycolmap (with built-in Jacobians)
            cost = cost_functions.ReprojErrorCost(camera_model, cam_params, observed_pt)

            # Add residual block
            # Parameter order: [quat, translation, point_3d, camera_params]
            pose = camera_poses[img_idx]
            problem.add_residual_block(
                cost,
                loss,
                [
                    pose.rotation.quat,
                    pose.translation,
                    point_3d,
                    cam_params,
                ],
            )

    # Set quaternion manifold for proper optimization on SO(3)
    for pose in camera_poses.values():
        problem.set_manifold(pose.rotation.quat, pyceres.EigenQuaternionManifold())

    # Fix camera intrinsics
    problem.set_parameter_block_constant(cam_params)

    # Fix the first camera (to avoid gauge freedom)
    if fix_first_camera and camera_poses:
        first_img_idx = min(camera_poses.keys())
        first_pose = camera_poses[first_img_idx]
        problem.set_parameter_block_constant(first_pose.rotation.quat)
        problem.set_parameter_block_constant(first_pose.translation)
        print(f"Fixed camera {first_img_idx} to avoid gauge freedom")

    # Configure solver
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = 100
    options.num_threads = -1

    # Solve
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    print(summary.BriefReport())

    # Update camera poses with optimized values
    for img_idx, pose in camera_poses.items():
        # Convert quaternion back to rotation matrix
        R = pose.rotation.matrix()
        t = pose.translation
        images[img_idx].set_pose(R, t)

    # Update 3D points
    for track_id, point_3d in point_params.items():
        point_cloud.set_point(track_id, point_3d)

    print("Bundle adjustment complete.")


# TODO: nice logging by levels
def main():
    K, dist = calibrate_camera()

    img_dir = Path("data") / "raw" / "statue"
    out_dir = Path("data") / "out" / img_dir.name
    # load all images & extract features
    image_store = FeatureStore(img_dir)
    track_manager = TrackManager()
    point_cloud = PointCloud()

    kp_list, des_list = image_store.get_keypoints(), image_store.get_descriptors()
    view_graph = construct_view_graph(kp_list, des_list, K)

    # Process the first component
    process_graph_component(K, dist, view_graph.edges.copy(), image_store, track_manager, point_cloud)

    # Process all connected components of the view graph
    # Each component will lead to a point cloud with its own reference frame
    # and thus appear disconnected from the others
    # leftover_edges = view_graph.edges.copy()
    # while True:
    #     leftover_edges, U = process_graph_component(K, dist, leftover_edges, image_store, track_manager, point_cloud)
    #     if not U:
    #         break

    bundle_adjustment(image_store, point_cloud, K, track_manager, fix_first_camera=False)

    print(f"Final point cloud size: {point_cloud.size}")
    point_cloud.save_ply(filename=out_dir / f"{img_dir.name}_ba.ply")


if __name__ == "__main__":
    main()
