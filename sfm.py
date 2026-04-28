from pathlib import Path
from typing import Callable

import cv2 as cv
import joblib
import numpy as np
import tyro
from rich.pretty import pprint

from ba import bundle_adjustment
from config import SfMConfig
from utils import (
    CameraModel,
    CameraType,
    FeatureExtractor,
    FeatureStore,
    FrameLoader,
    NDArrayFloat,
    NDArrayInt,
    PointCloud,
    ReconIO,
    TrackManager,
    ViewData,
    ViewEdge,
    calibrate_camera,
    construct_view_graph,
    make_keypoint_matcher,
)


def bootstrap_from_two_views(
    img_0: ViewData,
    img_1: ViewData,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    match_fn: Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]] | None = None,
    matches: NDArrayInt | None = None,
):
    """Computes two-view baseline estimate of 3D points and camera poses.

    This function initializes the 3D reconstruction pipeline by:
    1. Matching keypoints between two images using the provided matching function
    2. Computing the essential matrix via RANSAC to identify inliers
    3. Recovering camera pose (rotation and translation) for the second image
    4. Triangulating 3D points from the matched keypoint pairs
    5. Creating tracks and adding 3D points to the point cloud
    6. Setting camera poses (first image at origin, second image at computed pose)

    Args:
        img_0: First ImageData object (reference frame at origin)
        img_1: Second ImageData object to match against img_0
        track_manager: TrackManager instance for managing keypoint tracks
        point_cloud: PointCloud instance for storing triangulated 3D points
        match_fn: Callable that takes two ImageData objects and returns a tuple of
                  (descriptors: NDArrayFloat, matches: NDArrayInt) where matches
                  contains pairs of keypoint indices [kp_0_idx, kp_1_idx]
        matches: Matches between keypoints in views in img_0 and img_1.

    Returns:
        None. Modifies in-place: updates track_manager with new tracks, point_cloud
        with 3D points, and camera poses in img_0 and img_1.

    Notes:
        - The first image is set as the reference frame with identity rotation and
          zero translation
        - Only keypoint matches identified as inliers by RANSAC and successfully
          triangulated are included
        - Camera intrinsics are extracted from img_0 and assumed to be identical for img_1
    """
    if matches is None and match_fn is None:
        raise ValueError("One of matches or match_fn must be supplied.")

    # Get camera intrinsics from img_0
    K = img_0.camera_model.get_camera_matrix()

    # Match key points (via descriptors) if not given
    if matches is None:
        print(f"baseline: Computing matches from {img_0.idx}:{img_0.path.name} to {img_1.idx}:{img_1.path.name}")
        _, matches = match_fn(img_0, img_1)

    # extract corresponding pixel coordinates
    pts0, pts1 = img_0.kp[matches[:, 0]], img_1.kp[matches[:, 1]]  # ty:ignore[not-subscriptable]

    # compute Essential matrix using camera intrinsics; mask indicates inliers
    E, mask = cv.findEssentialMat(pts0, pts1, K, method=cv.RANSAC, prob=0.999, threshold=1.0)

    # Estimate camera extrinsics & triangulate 3D points; mask for inliers passing epipolar constraint
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
    matches = matches[inliers]

    # Create new tracks for the triangulated 3D object points
    # first create tracks for KPs in img_0, then add KPs in img_1 that match to KPs in img_0
    kp_key_pairs = [((img_0.idx, m[0]), (img_1.idx, m[1])) for m in matches]
    track_ids_added = track_manager.add_new_tracks(kp_key_pairs)

    point_cloud.add_points(track_ids_added, points_3d)

    # Estimated camera extrinsics, i.e. world-to-camera transform, conventionally named cam_T_world
    # This is NOT the camera's pose in the world frame!
    # img_0 is set to be at the world origin, img_1 is at (R, t)
    img_0.set_extrinsics(np.eye(3), np.zeros((3,)))
    img_1.set_extrinsics(R, t)

    print(f"Baseline constructed with {len(points_3d)} 3D points.")


def _estimate_pose_pnp(world_points: NDArrayFloat, image_points: NDArrayFloat, img: ViewData):
    """Estimate camera pose using PnP given 3D-2D correspondences and camera intrinsics."""

    print(f"Estimating pose of {img.idx}:{img.path.name} with {len(world_points)} 3D-2D correspondences...")

    assert len(world_points) >= 4, "At least 4 3D-2D correspondences are required for PnP"
    assert len(world_points) == len(image_points), "Number of 3D points must match number of 2D points"
    assert np.isfinite(world_points).all(), "Object points must be finite"
    assert np.isfinite(image_points).all(), "Image points must be finite"

    K, dist = img.camera_model.get_camera_matrix(), img.camera_model.dist
    pnp_ok, rvec, tvec, inliers = cv.solvePnPRansac(
        world_points,
        image_points,
        K,
        dist,
        reprojectionError=4.0,  # tighter than default 8.0
        flags=cv.SOLVEPNP_EPNP,
    )
    if not pnp_ok:
        raise ValueError("solvePnP failed to estimate pose.")
    print(
        f"Pose estimation succeeded with {len(inliers)} inliers (Inlier ratio: {len(inliers) / len(world_points):.2f})"
    )

    # Estimated camera extrinsics, i.e. world-to-camera transform, conventionally named cam_T_world
    # This is NOT the camera's pose in the world frame!
    R = cv.Rodrigues(rvec)[0]
    img.set_extrinsics(R, tvec)

    return inliers.ravel()


def _triangulate_new_points(img_ref: ViewData, img_new: ViewData, untracked_matches: NDArrayInt):
    """Triangulate new 3D points from untracked matches between reference and new image.

    Args:
        img_ref: Reference image with known pose.
        img_new: New image with estimated pose.
        untracked_matches: Array of shape (N, 2) containing matches between img_ref and img_new that are not
        associated with any existing track (i.e. new tracks to be added via triangulation).
    """
    assert len(untracked_matches) >= 5, "At least 5 points required for essential matrix estimation"
    # Filter out geometric outliers that don't satisfy the epipolar constraint
    pts_ref, pts_new = img_ref.kp[untracked_matches[:, 0]].T, img_new.kp[untracked_matches[:, 1]].T  # ty:ignore[not-subscriptable]
    K = img_ref.camera_model.get_camera_matrix()  # assume same intrinsics for both images
    _, mask = cv.findEssentialMat(pts_ref.T, pts_new.T, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
    inliers = mask.ravel() > 0
    # Projection matrices: from 3D world to camera 2D image plane
    P_ref, P_new = img_ref.projection_matrix, img_new.projection_matrix
    # Triangulate the untracked KPs in the new image that match to KPs in the ref image, to get new 3D points
    points_4d = cv.triangulatePoints(P_ref, P_new, pts_ref[:, inliers], pts_new[:, inliers])
    points_3d = (points_4d[:3] / points_4d[3]).T
    untracked_matches = untracked_matches[inliers]

    # Depth filter of triangulated points: filter out points that are behind either camera (negative depth)
    imgref_points_3d = img_ref.transform_to_camera_frame(points_3d)
    imgnew_points_3d = img_new.transform_to_camera_frame(points_3d)
    inliers = (imgref_points_3d[:, 2] > 0) & (imgnew_points_3d[:, 2] > 0)
    points_3d = points_3d[inliers]
    untracked_matches = untracked_matches[inliers]
    print(f"Filtered out {np.sum(~inliers)} points that are behind the camera. Remaining points: {len(points_3d)}")

    # Create track for each pair of KPs (ref, new) that were triangulated to a 3D point
    kp_key_pairs = [((img_ref.idx, m[0]), (img_new.idx, m[1])) for m in untracked_matches]

    return points_3d, kp_key_pairs


def add_view(
    img_new: ViewData,
    img_ref: ViewData,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    match_fn: Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]] | None = None,
    matches: NDArrayInt | None = None,
):
    """Adds 3D points from new view using PnP and triangulation.

    img_ref is reference image for which we already have 2D-3D pt correspondence in track_manager

    Args:
        img_new: New image to add.
        img_ref: Reference image with known pose.
        track_manager: Track manager. Required parameter.
        point_cloud: Point cloud. Required parameter.
        match_fn: Keypoint matcher function. Required parameter.
        matches: Matches between keypoints in views in img_0 and img_1.
    """
    if matches is None and match_fn is None:
        raise ValueError("One of matches or match_fn must be supplied.")

    # Compute KP matches from ref image to new image if not supplied
    if matches is None:
        print(
            f"add_view: Computing matches from {img_ref.idx}:{img_ref.path.name} to {img_new.idx}:{img_new.path.name}"
        )
        _, matches = match_fn(img_ref, img_new)

    # add new img KPs, that are matched to from tracked ref img KPs, to current tracks (3D pts)
    # returns track_ids and (un)tracked KPs in the new image; track_ids used as indices to point cloud
    track_ids_seen, tracked_matches, untracked_matches = track_manager.get_track_observations_for_view(
        img_ref.idx, matches
    )
    kp_idx_seen = tracked_matches[:, 1]

    # Estimate pose of new image
    # 3D-to-2D correspondences in new view (via matches w/ ref view) for PnP pose estimation
    world_points = point_cloud.get_points_as_array(track_ids_seen)
    image_points = img_new.kp[kp_idx_seen]  # ty:ignore[not-subscriptable]
    inliers = _estimate_pose_pnp(world_points, image_points, img_new)
    kp_idx_seen, track_ids_seen = kp_idx_seen[inliers], track_ids_seen[inliers]

    # Register the inlier kps to inlier tracks in track manager
    kp_keys_seen = [(img_new.idx, kp_idx) for kp_idx in kp_idx_seen]
    track_manager.add_keypoints_to_tracks(kp_keys_seen, track_ids_seen)

    # Translation vector between the new image and the reference image
    t_ref_new = (img_ref.cam_T_world * img_new.world_T_cam).translation  # ty:ignore[possibly-missing-attribute]
    print(f"DEBUG: Relative translation from ref to new image: {np.linalg.norm(t_ref_new):.2f}")

    points_3d, kp_key_pairs = _triangulate_new_points(img_ref, img_new, untracked_matches)

    track_ids_added = track_manager.add_new_tracks(kp_key_pairs)
    point_cloud.add_points(track_ids_added, points_3d)

    print(f"Added {len(points_3d)} 3D points.")


def pick_best_image_pair(
    edges: list[ViewEdge], store: FeatureStore, R: set[int] | None = None
) -> tuple[ViewData, ViewData, ViewEdge]:
    """Pick best image pair from list of edges.

    Assumption: ImageData.idx matches the node indexes, which it should if the graph was constructed correctly.
    """
    best_edge = max(edges, key=lambda e: e.weight)
    if R:  # if R is not None or not empty
        idx_ref, idx_new = (best_edge.i, best_edge.j) if best_edge.i in R else (best_edge.j, best_edge.i)
        return store[idx_ref], store[idx_new], best_edge
    return store[best_edge.i], store[best_edge.j], best_edge


def process_graph_component(
    edges: list[ViewEdge],
    store: FeatureStore,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    match_fn: Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]],
) -> tuple[list[ViewEdge], set[int]]:
    # Pick strongest baseline:
    # - The edge of the view graph with greatest weight (ie. # kp matches) determines the two images
    img_0, img_1, best_edge = pick_best_image_pair(edges, store)
    print(
        f"Establishing baseline ({best_edge.weight} matches) from: {img_0.idx}:{img_0.path.name} and {img_1.idx}:{img_1.path.name}"
    )
    # matches -> E -> pose -> triangulation
    bootstrap_from_two_views(img_0, img_1, track_manager, point_cloud, match_fn)

    print(f"After compute_baseline_estimate: {track_manager.is_valid()=}")

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
            add_view(img_new, img_ref, track_manager=track_manager, point_cloud=point_cloud, match_fn=match_fn)
            print(f"After add_view: {track_manager.is_valid()=}")

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

    # Filter out any remaining edges that connect registered views/images
    leftover_edges = [e for e in leftover_edges if not (e.i in R and e.j in R)]
    print(f"{R = }\n{U = }")
    print(f"leftover_edges {[(e.i, e.j) for e in leftover_edges]}")

    return leftover_edges, U


# TODO: nice logging by levels
def main(cfg: SfMConfig = SfMConfig()):
    """Run Structure from Motion pipeline with configurable feature extraction and matching.

    Args:
        cfg: Configuration object. Override defaults with --cfg.param_name value
    """

    # Display configuration
    pprint(cfg, expand_all=True)
    print()

    K, dist = calibrate_camera()
    camera_model = CameraModel(model_type=CameraType.PINHOLE, K=K, dist=dist)

    img_dir = Path("data") / "raw" / cfg.dataset
    out_dir = Path("data") / "out" / cfg.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all images & extract features
    print(f"Extracting {cfg.feature_type.upper()} features from {img_dir}...")
    loader = FrameLoader(img_dir, max_size=cfg.max_size, ext="jpg", camera_model=camera_model, undistort=cfg.undistort)
    feature_extractor = FeatureExtractor(cfg, loader)
    image_store = FeatureStore(feature_extractor)
    track_manager = TrackManager()
    point_cloud = PointCloud()
    exporter = ReconIO(point_cloud, image_store, track_manager)

    # Create keypoint matcher with appropriate parameters
    kp_matcher = make_keypoint_matcher(cfg)

    print("Constructing view graph...")
    view_graph = construct_view_graph(image_store, kp_matcher, min_inliers=cfg.min_inliers)

    # Process the first component
    print("Processing graph component...")
    process_graph_component(view_graph.edges.copy(), image_store, track_manager, point_cloud, kp_matcher)

    # Process all connected components of the view graph
    # Each component will lead to a point cloud with its own reference frame
    # and thus appear disconnected from the others
    # leftover_edges = view_graph.edges.copy()
    # while True:
    #     leftover_edges, U = process_graph_component(leftover_edges, image_store, track_manager, point_cloud, kp_matcher)
    #     if not U:
    #         break

    basename = f"{cfg.dataset}_{cfg.feature_type}_{cfg.matcher_type}"
    print(f"Saving initial reconstruction to {out_dir / f'{basename}.ply'}...")
    exporter.save_ply(filename=out_dir / f"{basename}.ply")

    if cfg.dump_sfm_debug:
        sfm_debug_filename = f"{basename}_sfm_debug.joblib"
        joblib.dump(
            (image_store, point_cloud, track_manager),
            out_dir / sfm_debug_filename,
            compress=3,
        )
        print(f"Dumped SFM structs to {out_dir / sfm_debug_filename}")

    if cfg.run_ba:
        print("Running bundle adjustment...")
        bundle_adjustment(image_store, point_cloud, track_manager, fix_first_camera=cfg.fix_first_camera)

        print(f"Final point cloud size: {point_cloud.size}")
        print(f"Saving optimized reconstruction to {out_dir / f'{basename}_ba.ply'}...")
        exporter.save_ply(filename=out_dir / f"{basename}_ba.ply")

    if cfg.save_gsplat:
        print("\nSaving tensors for gsplat...")
        gsplat_file = f"{basename}_ba.pt" if cfg.run_ba else f"{basename}.pt"
        exporter.save_for_gsplat(filename=out_dir / gsplat_file)

    print("\n✓ Done!")


if __name__ == "__main__":
    tyro.cli(main)
