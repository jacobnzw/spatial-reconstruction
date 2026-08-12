from typing import Literal

import cv2 as cv
import numpy as np
import tyro
from loguru import logger
from rich.pretty import pprint

import wandb
from ba import bundle_adjustment_gtsam
from config import FrameLoaderConfig, SfMConfig, frame_loader_preset, write_config_to_json
from utils import (
    FeatureExtractor,
    FeatureStore,
    FrameLoader,
    KeypointMatcher,
    MatcherResult,
    NDArrayFloat,
    NDArrayInt,
    PointCloud,
    ReconIO,
    ReRunLogger,
    TrackManager,
    ViewData,
    ViewGraph,
    log_wandb_artifacts,
)


def bootstrap_from_two_views(
    img_0: ViewData,
    img_1: ViewData,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    matcher_result: MatcherResult,
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
        matcher_result: MatcherResult containing matches between img_0 and img_1

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
    if not matcher_result:
        raise ValueError("Bad matcher result!")

    # extract corresponding pixel coordinates
    matches = matcher_result.matches
    pts0, pts1 = img_0.kp[matches[:, 0]], img_1.kp[matches[:, 1]]  # ty:ignore[not-subscriptable]
    # FIXME: undistorted keypoints cause mess on statue_orbit
    # pts0, pts1 = img_0.get_undistorted_keypoints(), img_1.get_undistorted_keypoints()
    # pts0, pts1 = pts0[matches[:, 0]], pts1[matches[:, 1]]  # ty:ignore[not-subscriptable]

    # compute Essential matrix using camera intrinsics; mask indicates inliers
    K = img_0.camera_model.get_camera_matrix()
    E, mask = cv.findEssentialMat(pts0, pts1, K, method=cv.RANSAC, prob=0.999, threshold=1.0)

    # Estimate camera extrinsics & triangulate 3D points; mask for inliers passing epipolar constraint
    # t known only up to scale -> unit length!
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
    matches = matches[inliers]  # ty:ignore[not-subscriptable]

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

    logger.success(f"Bootstrapped with {len(points_3d)} 3D points.")


def _estimate_pose_pnp(world_points: NDArrayFloat, image_points: NDArrayFloat, img: ViewData):
    """Estimate camera pose using PnP given 3D-2D correspondences and camera intrinsics."""

    logger.info(f"Estimating pose of {img.idx}:{img.path.name} with {len(world_points)} 3D-2D correspondences...")

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

    n_inliers = len(inliers) if inliers is not None else 0
    ratio_inlier = n_inliers / len(world_points)

    if not pnp_ok:
        raise ValueError(f"solvePnP failed to estimate pose: {n_inliers=} {ratio_inlier=}.")

    logger.success(f"Pose estimation succeeded with {n_inliers} inliers (Inlier ratio: {ratio_inlier:.2f})")

    # Estimated camera extrinsics, i.e. world-to-camera transform, conventionally named cam_T_world
    # This is NOT the camera's pose in the world frame!
    R = cv.Rodrigues(rvec)[0]
    img.set_extrinsics(R, tvec)

    return inliers.ravel(), n_inliers, ratio_inlier


def _triangulate_new_points(
    img_ref: ViewData, img_new: ViewData, untracked_matches: NDArrayInt, depth_threshold: float
):
    """Triangulate new 3D points from untracked matches between reference and new image.

    Args:
        img_ref: Reference image with known pose.
        img_new: New image with estimated pose.
        untracked_matches: Array of shape (N, 2) containing matches between img_ref and img_new that are not
        associated with any existing track (i.e. new tracks to be added via triangulation).
    """
    assert len(untracked_matches) >= 5, "At least 5 points required for essential matrix estimation"

    # Filter out geometric outliers that don't satisfy the epipolar constraint
    # alternative: only cv.fisheye.undistortPoints() then find E-mat w/ cameraMatrix=np.eye(3)
    # pts_ref, pts_new = img_ref.get_undistorted_keypoints(), img_new.get_undistorted_keypoints()
    # pts_ref, pts_new = pts_ref[untracked_matches[:, 0]], pts_new[untracked_matches[:, 1]]
    pts_ref, pts_new = img_ref.kp[untracked_matches[:, 0]], img_new.kp[untracked_matches[:, 1]]  # ty:ignore[not-subscriptable]

    # If undistortion applied during image loading, K is the corrected camera matrix for the undistorted image
    K = img_ref.camera_model.get_camera_matrix()  # assume same intrinsics for both images
    # TODO: get this from MatcherResult; don't recompute there; ACTUALLY this is geom. val. on untracked_matches
    _, mask = cv.findEssentialMat(pts_ref, pts_new, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
    inliers = mask.ravel() > 0

    # Projection matrices: from 3D world to camera 2D image plane
    P_ref, P_new = img_ref.projection_matrix, img_new.projection_matrix
    # Triangulate the untracked KPs in the new image that match to KPs in the ref image, to get new 3D points
    points_4d = cv.triangulatePoints(P_ref, P_new, pts_ref[inliers].T, pts_new[inliers].T)
    points_3d = (points_4d[:3] / points_4d[3]).T
    untracked_matches = untracked_matches[inliers]

    # Depth filter of triangulated points: filter out points that are behind either camera (negative depth)
    imgref_points_3d = img_ref.transform_to_camera_frame(points_3d)
    imgnew_points_3d = img_new.transform_to_camera_frame(points_3d)
    inliers = (imgref_points_3d[:, 2] > depth_threshold) & (imgnew_points_3d[:, 2] > depth_threshold)
    points_3d = points_3d[inliers]
    untracked_matches = untracked_matches[inliers]

    ratio_depth_filtered = np.sum(~inliers) / len(inliers)
    logger.debug(f"Depth-filtered points ratio: {ratio_depth_filtered} ({depth_threshold=}).")

    # Create track for each pair of KPs (ref, new) that were triangulated to a 3D point
    kp_key_pairs = [((img_ref.idx, m[0]), (img_new.idx, m[1])) for m in untracked_matches]

    return points_3d, kp_key_pairs, ratio_depth_filtered


def add_view(
    img_new: ViewData,
    img_ref: ViewData,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    matcher_result: MatcherResult,
    depth_threshold: float = 0.0,
) -> tuple[int, float, float]:
    """Adds 3D points from new view using PnP and triangulation.

    img_ref is reference image for which we already have 2D-3D pt correspondence in track_manager

    Args:
        img_new: New image to add.
        img_ref: Reference image with known pose.
        track_manager: Track manager. Required parameter.
        point_cloud: Point cloud. Required parameter.
        matcher_result: Matcher result dataclass.
        depth_threshold: Triangulated points closer than this in either camera (ref or new) are filtered out.
    """
    if not matcher_result:
        raise ValueError("Bad matcher result!")

    # add new img KPs, that are matched to from tracked ref img KPs, to current tracks (3D pts)
    # returns track_ids and (un)tracked KPs in the new image; track_ids used as indices to point cloud
    track_ids_seen, tracked_matches, untracked_matches = track_manager.get_track_observations_for_view(
        img_ref.idx,
        matcher_result.matches,  # ty:ignore[invalid-argument-type]
    )
    kp_idx_seen = tracked_matches[:, 1]

    # Estimate pose of new image
    # 3D-to-2D correspondences in new view (via matches w/ ref view) for PnP pose estimation
    world_points = point_cloud.get_points_as_array(track_ids_seen)
    image_points = img_new.kp[kp_idx_seen]  # ty:ignore[not-subscriptable]
    inliers, n_pnp_inliers, ratio_pnp_inlier = _estimate_pose_pnp(world_points, image_points, img_new)
    kp_idx_seen, track_ids_seen = kp_idx_seen[inliers], track_ids_seen[inliers]

    # Register the inlier kps to inlier tracks in track manager
    kp_keys_seen = [(img_new.idx, kp_idx) for kp_idx in kp_idx_seen]
    track_manager.add_keypoints_to_tracks(kp_keys_seen, track_ids_seen)

    # Translation vector between the new image and the reference image
    t_ref_new = (img_ref.cam_T_world * img_new.world_T_cam).translation  # ty:ignore[unsupported-operator]
    logger.debug(f"Baseline between ref and new image: {np.linalg.norm(t_ref_new):.2f}")

    points_3d, kp_key_pairs, ratio_triang_depth_filtered = _triangulate_new_points(
        img_ref, img_new, untracked_matches, depth_threshold
    )

    track_ids_added = track_manager.add_new_tracks(kp_key_pairs)
    point_cloud.add_points(track_ids_added, points_3d)

    logger.success(f"Added {len(points_3d)} 3D points.")

    return n_pnp_inliers, ratio_pnp_inlier, ratio_triang_depth_filtered


def process_graph_component(
    view_graph: ViewGraph,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    kp_matcher: KeypointMatcher,
    depth_threshold: float,
    rerun_logger: ReRunLogger | None = None,
) -> wandb.Table:

    # Pick strongest baseline:
    # - The edge of the view graph with greatest weight (ie. # kp matches) determines the two images
    view_pair_status = view_graph.find_initial_view_pair()

    if view_pair_status is None:
        raise ValueError("Couldn't find initial view pair!")

    img_0, img_1, matcher_result = view_pair_status

    logger.info(
        f"Initializing reconstruction w/ {len(matcher_result)} matches from: "
        f"{img_0.idx}:{img_0.path.name} and {img_1.idx}:{img_1.path.name}"
    )

    # matches -> E -> pose -> triangulation
    bootstrap_from_two_views(img_0, img_1, track_manager, point_cloud, matcher_result)
    view_graph.mark_edge_registered(img_0.idx, img_1.idx)

    if rerun_logger is not None:
        rerun_logger.log_all(matcher_result, both_cameras=True)  # logs current cloud, cameras, ref-new matches

    log_table = wandb.Table(
        columns=[
            "new_index",
            "ref_index",
            "success",
            "n_matches",
            "n_pnp_inliers",
            "ratio_pnp_inlier",
            "ratio_triang_depth_filtered",
            "baseline",
            "new_file",
            "ref_file",
        ]
    )

    # Process all remaining views one by one
    while True:
        view_pair_status = view_graph.find_next_best_view_pair()

        if view_pair_status is None:
            break

        img_new, img_ref, matcher_result = view_pair_status

        logger.info(
            (
                f"Adding view {img_new.idx}:{img_new.path.name} w/ ref {img_ref.idx}:{img_ref.path.name}"
                f" (matches: {len(matcher_result)})"
            )
        )
        logger.debug(f"{img_ref.idx=} {img_new.idx=} {matcher_result.matches.shape=}.")

        try:
            # matches --> 2D-3D pairs --PnP--> pose -> triangulate untracked
            n_pnp_inliers, ratio_pnp_inlier, ratio_triang_depth_filtered = add_view(
                img_new,
                img_ref,
                track_manager,
                point_cloud,
                matcher_result,
                depth_threshold=depth_threshold,
            )
            add_success = True
            view_graph.mark_edge_registered(img_new.idx, img_ref.idx)

            if rerun_logger:
                rerun_logger.log_all(matcher_result)

        except ValueError as e:
            # failed to add new view: indicate the (img_ref, img_new) pair as bad and move on
            # best_edge was the best chance to add img_new (don't consider next best edge w/ img_new)

            logger.warning(
                f"Failed to add view: {img_new.idx}:{img_new.path.name} w/ ref: {img_ref.idx}:{img_ref.path.name} due to {e}"
            )

            n_pnp_inliers, ratio_pnp_inlier, ratio_triang_depth_filtered = np.nan, np.nan, np.nan
            add_success = False
            view_graph.mark_edge_failed(img_new.idx, img_ref.idx)

        baseline = (
            np.linalg.norm((img_ref.cam_T_world * img_new.world_T_cam).translation) if img_new.has_pose else np.nan
        )
        log_table.add_data(
            img_new.idx,
            img_ref.idx,
            add_success,
            len(matcher_result),  # NOTE: this is the number of matches, not the number of geo valid matches (inliers)
            n_pnp_inliers,
            ratio_pnp_inlier,
            ratio_triang_depth_filtered,
            baseline,
            img_new.path.name,
            img_ref.path.name,
        )

        if not add_success:
            continue

    # TODO: report registered, unregistered views, failed edges, remaining edges

    return log_table


Dataset = Literal["corridor", "statue_orbit"]


def main(cfg: SfMConfig, dataset: Dataset | None = None):
    """Structure from Motion pipeline with configurable feature extraction and matching.

    Args:
        cfg: Configuration object. Override defaults with --cfg.param_name value
        dataset: Dataset preset for convenience: 'corridor' or 'statue_orbit'.
    """

    if dataset is not None:
        cfg.loader = FrameLoaderConfig(**frame_loader_preset(dataset))

    # Display configuration
    pprint(cfg, expand_all=True)

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = cfg.out_basename

    logger.add(out_dir / f"{basename}.log")
    config_dict = write_config_to_json(cfg, out_dir / f"{basename}_config.json")

    wandb_mode = "online" if cfg.log_to_wandb else "disabled"
    run = wandb.init(
        entity="jacobnzw-n-a",
        project="spatial-reconstruction",
        mode=wandb_mode,
        name=cfg.loader.dataset,
        config=config_dict,
        settings=wandb.Settings(console="auto"),  # captures logs written to stdout/stderr
    )

    # Load all images & extract features
    logger.info(f"Extracting {cfg.features.type.upper()} features from {cfg.loader.img_dir}...")
    loader = FrameLoader(cfg.loader)
    feature_extractor = FeatureExtractor(cfg.features, loader)
    image_store = FeatureStore(feature_extractor)
    track_manager = TrackManager()
    point_cloud = PointCloud()
    exporter = ReconIO(point_cloud, image_store, track_manager)

    logger.info("Constructing view graph...")
    kp_matcher = KeypointMatcher(cfg.matcher)
    view_graph = ViewGraph(image_store, kp_matcher, k=5)  # TODO: Add k to config

    logger.info("Processing graph component...")
    log_filepath = out_dir / f"{basename}.rrd"
    rerun_logger = ReRunLogger(log_filepath, point_cloud, image_store, track_manager)
    log_view_table = process_graph_component(
        view_graph, track_manager, point_cloud, kp_matcher, cfg.depth_threshold, rerun_logger
    )

    # TODO: Process all connected components of the view graph. nx.connected_components
    # Each component will lead to a point cloud with its own reference frame and
    # thus appear disconnected from the others

    logger.info(f"Saving initial reconstruction to {out_dir / f'{basename}.ply'}...")
    exporter.save_ply(filename=out_dir / f"{basename}.ply")

    if cfg.dump_sfm_debug:
        filepath = out_dir / f"{basename}_sfm_debug.joblib"
        exporter.dump_sfm_debug(filepath)
        logger.info(f"Dumped SFM structs to {filepath}")

    ba_summary = None
    if cfg.run_ba:
        # IMU data for BA are optional: when None, BA ignores it.
        imu_data_file, imu_calibration = cfg.imu_data, cfg.imu_calibration

        logger.info("Running bundle adjustment...")
        ba_summary = bundle_adjustment_gtsam(
            image_store, point_cloud, track_manager, cfg.fix_first_camera, imu_data_file, imu_calibration
        )

        logger.info(f"Final point cloud size: {point_cloud.size}")
        logger.info(f"Saving optimized reconstruction to {out_dir / f'{basename}_ba.ply'}...")
        exporter.save_ply(out_dir / f"{basename}_ba.ply")

    if cfg.save_gsplat:
        logger.info("\nSaving tensors for gsplat...")
        gsplat_file = f"{basename}_ba.pt" if cfg.run_ba else f"{basename}.pt"
        exporter.save_for_gsplat(out_dir / gsplat_file)

    log_wandb_artifacts(run, cfg, track_manager, log_view_table, ba_summary)
    run.finish()

    logger.success("✓ Done!")


if __name__ == "__main__":
    tyro.cli(main)
