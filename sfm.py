from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal

import cv2 as cv
import numpy as np
from numpy.typing import NDArray


def calibrate_camera(img_dir: Path = Path("data/calibration")):
    """Compute camera intrinsics given a sample of checkerboard photos."""

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
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  # ty:ignore[no-matching-overload]

    return K, dist


def extract_sift(img_path: Path):
    img = cv.imread(img_path)  # ty:ignore[no-matching-overload]
    # img = cv.resize(img, (800, 600))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des, img


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
    def pose_matrix(self) -> NDArray[np.float32]:
        return np.hstack((self.R, self.t))  # ty:ignore[no-matching-overload]


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
    def size(self):
        return len(self._store)

    def __getitem__(self, img_idx: int):
        return self._store[img_idx]

    def set_pose(self, img_idx: int, R, t):
        self._store[img_idx].R = R
        self._store[img_idx].t = t

    def keypoints(self):
        yield from (item.kp for item in self._store)

    def descriptors(self):
        yield from (item.des for item in self._store)


@dataclass
class ViewEdge:
    i: int
    j: int
    inliers_ij: int  # matches i -> j
    inliers_ji: int  # matches j -> i

    @property
    def weight(self):
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

    def best_edge(self) -> ViewEdge:
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


def construct_view_graph(kp: list, des: list):
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

    def add_points(self, track_ids: list[int], xyz: NDArray[Any]) -> None:
        assert len(track_ids) == xyz.shape[0], "Number of track_ids must match number of 3D points"
        for track_id, pt in zip(track_ids, xyz):
            self._data[track_id] = pt

    def get_point(self, track_id: int) -> Point3D | None:
        return self._data.get(track_id, None)

    def get_points_as_array(self, track_ids: list[int]) -> NDArray[np.floating[Any]]:
        """Returns array of 3D points corresponding to the given track_ids.

        Missing points are returned as np.nan.
        """
        return np.array([self._data.get(track_id, np.nan) for track_id in track_ids])

    def iter_points(self) -> Iterable[Point3D]:
        yield from self._data.values()

    # TODO: .export(), plot_colored(), plot_depth()?


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

    def update_tracks(self, img_idx_new: int, img_idx_ref: int, ref2new_matches):
        """Update tracks with new KP observations from img_new.

        img_ref is the reference image for which we already have 2D-3D pt correspondence in track_manager.
        Some of the KPs in img_ref have tracks, some don't. The KPs in img_new that match the tracked KPs in img_ref
        are added to those tracks. The KPs in img_new that match the untracked KPs in img_ref are added to new tracks.
        """
        # for each match, if ref KP has a track, add new img KP to track; else create new track
        kp_idx_new_tracked, track_ids_tracked = [], []
        track_ids_added, matches_untracked = [], []
        # I wanna add the new KPs (that have matches to tracked ref KPs) to current tracks
        for m, _ in ref2new_matches:
            kp_idx_ref, kp_idx_new = m.queryIdx, m.trainIdx
            # if ref KP has a track, add the matching new KP to the same track
            # this means the same 3D object point is now observed by a new 2D KP
            kp_key_new = (img_idx_new, kp_idx_new)
            kp_key_ref = (img_idx_ref, kp_idx_ref)
            if track_id := self.get_track(kp_key_ref) is not None:
                track_ids_tracked.append(track_id)
                kp_idx_new_tracked.append(kp_idx_new)
                self._register_keypoint_track(kp_key_new, track_id)
            else:  # ref KP is untracked and therefore so is its matching KP from img_new
                matches_untracked.append(m)
                # create new tracks for untracked KPs
                # FIXME: problem: new object points haven't been triangulated yet
                # so I can't yet know their indexes in 3d point cloud, ie. track_ids
                # -> postpone creating new tracks until after triangulation
                # next_track_id will start at point_cloud.size
                # BUT: the self.next_track_id will already be set to point_cloud.size provided TM is used properly
                self._register_keypoint_track(kp_key_new, self.next_track_id)
                track_ids_added.append(self.next_track_id)
                self.next_track_id += 1

        # return tracked KPs in new img and ref2new matches for untracked KPs for triangulation
        return track_ids_tracked, kp_idx_new_tracked, track_ids_added, matches_untracked


def compute_baseline_estimate(img_0: ImageData, img_1: ImageData, K, track_manager, point_cloud):
    """Computes two-view baseline estimate of 3D points and poses

    First image is at the origin.
    """

    # Match key points (via descriptors)
    matches = compute_matches(img_0.des, img_1.des, lowe_ratio=0.75)

    # extract corresponding pixel coordinates
    pts0 = np.ndarray([img_0.kp[m.queryIdx].pt for m in matches]).astype(np.float32)
    pts1 = np.ndarray([img_1.kp[m.trainIdx].pt for m in matches]).astype(np.float32)

    # compute Essential matrix using camera intrinsics
    E, mask = cv.findEssentialMat(pts0, pts1, K, method=cv.RANSAC, prob=0.999, threshold=1.0)

    # estimate camera pose & triangulate 3D points
    retval, R, t, mask, points_4d = cv.recoverPose(
        E=E,
        points1=pts0,
        points2=pts1,
        cameraMatrix=K,
        distanceThresh=50.0,  # mandatory for triangulation
    )

    # mask tells us which pairs of 2D points were successfully triangulated to 3D?
    # TODO: should I filter points_4d using mask?? inliers = mask > 0

    # Homogeneous --> Euclidean
    points_3d = (points_4d[:3] / points_4d[3]).T

    # Create new tracks for the triangulated 3D object points
    # first create tracks for KPs in img_0, then update with KPs in img_1 that match to KPs in img_0
    track_ids_added = track_manager.add_new_tracks([(img_0.idx, m.queryIdx) for m in matches])
    track_manager.update_tracks(img_1.idx, img_0.idx, matches)
    point_cloud.add_points(track_ids_added, points_3d)

    # Update image data structs w/ new estimates: img_0 is at origin, img_1 is at (R, t)
    img_0.set_pose(np.eye(3), np.zeros(3))
    img_1.set_pose(R, t)

    return R, t, points_3d


def add_view(img_new: ImageData, img_ref: ImageData, K, dist, track_manager, point_cloud):
    """Adds 3D points from new view using PnP and triangulation.

    img_ref is reference image for which we already have 2D-3D pt correspondence in track_manager
    """
    # Compute KP matches from ref image to new image
    # Matching from new to ref image: Where does ref img tracked KP match to in new img?
    matches = compute_matches(img_ref.des, img_new.des, lowe_ratio=0.75)

    # add new img KPs, that are matched to from tracked ref img KPs, to current tracks (3D pts)
    # returns track_ids and (un)tracked KPs in the new image; track_ids used as indices to point cloud
    track_ids_tracked, kp_idx_new_tracked, track_ids_added, matches_untracked = track_manager.update_tracks(
        img_new, img_ref, matches
    )

    # Estimate pose of new image
    # Only use object points corresponding to tracked KPs in img_ref for PnP pose estimation
    object_points = point_cloud.get_points_as_array(track_ids_tracked)
    # PnP needs tracked KPs from new image (2D) and matching 3D object pts
    # In other words, new 2D points that observe the same 3D object points as the tracked KPs in the ref image
    # img_new_pts_tracked: NDArray[np.float32] = np.float32([img_new.kp[i].pt for i in kp_idx_new_tracked])
    img_new_pts_tracked = np.ndarray([img_new.kp[i].pt for i in kp_idx_new_tracked]).astype(np.float32)
    pnp_ok, R, t, inliers = cv.solvePnPRansac(object_points, img_new_pts_tracked, K, dist)
    if not pnp_ok:
        raise ValueError("solvePnPRansac failed to estimate pose.")
    img_new.set_pose(R, t)

    # Triangulate untracked KPs in the new image
    # pts_ref[i] matched to pts_new[i]
    pts_ref = np.ndarray([img_ref.kp[m.queryIdx].pt for m in matches_untracked]).astype(np.float32)
    pts_new = np.ndarray([img_new.kp[m.trainIdx].pt for m in matches_untracked]).astype(np.float32)

    # Projection matrices: from 3D world to each camera 2D image plane
    P_ref = K @ img_ref.pose_matrix
    P_new = K @ img_new.pose_matrix

    # Triangulate the untracked KPs in the new image
    points_4d = cv.triangulatePoints(P_ref, P_new, pts_ref.T, pts_new.T)
    points_3d = points_4d[:3] / points_4d[3]

    point_cloud.add_points(track_ids_added, points_3d)

    return R, t, points_3d


def main():
    K, dist = calibrate_camera()

    img_dir = Path("data") / "raw" / "statue"
    # load all images & extract features
    store = FeatureStore(img_dir)
    track_manager = TrackManager()
    point_cloud = PointCloud()

    # TODO: change store to SOA layout? need materialized lists for view graph anyway
    kp_list, des_list = list(store.keypoints()), list(store.descriptors())
    view_graph = construct_view_graph(kp_list, des_list)

    # Pick strongest baseline:
    # - The edge of the view graph with greatest weight (ie. # kp matches) determines the two images
    best_edge = view_graph.best_edge()
    img_0, img_1 = store[best_edge.i], store[best_edge.j]
    # matches -> E -> pose -> triangulation
    _, _, points_3d = compute_baseline_estimate(img_0, img_1, K, track_manager, point_cloud)

    R = set((best_edge.i, best_edge.j))
    U = set(range(len(store.size)))
    U.difference_update(R)

    while True:
        # find unregistered images connected to the registered ones
        # "connected" == "sharing matched keypoints"
        candidate_edges = [e for e in view_graph.edges if (e.i in R and e.j in U) or (e.j in R and e.i in U)]

        if not candidate_edges:
            # TODO: U could still be non-empty (disconnected graph)
            break

        best_edge = max(candidate_edges, key=lambda e: e.weight)
        idx_ref, idx_new = (best_edge.i, best_edge.j) if best_edge.i in R else (best_edge.j, best_edge.i)
        img_ref, img_new = store[idx_ref], store[idx_new]

        # matches --> 2D-3D pairs --PnP--> pose -> triangulate untracked
        _, _, points_3d = add_view(img_new, img_ref, K, dist, track_manager, point_cloud)

        # move currently processed image/node index from U to R
        if best_edge.i in R:
            R.add(best_edge.j)
            U.remove(best_edge.j)
        else:
            R.add(best_edge.i)
            U.remove(best_edge.i)
