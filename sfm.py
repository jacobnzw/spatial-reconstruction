from functools import partial
from pathlib import Path
from typing import Callable

import cv2 as cv
import numpy as np
import typer

from ba import bundle_adjustment
from utils import (
    KF,
    FeatureStore,
    ImageData,
    NDArrayFloat,
    NDArrayInt,
    PointCloud,
    ReconExporter,
    TrackManager,
    ViewEdge,
    calibrate_camera,
    compute_matches,
    construct_view_graph,
    device,
)

app = typer.Typer()


def compute_baseline_estimate(
    img_0: ImageData,
    img_1: ImageData,
    K: NDArrayFloat,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    match_fn: Callable[[ImageData, ImageData], tuple[NDArrayFloat, NDArrayInt]],
):
    """Computes two-view baseline estimate of 3D points and poses

    First image is at the origin.
    """

    # Match key points (via descriptors)
    print(f"baseline: Computing matches from {img_0.idx}:{img_0.path.name} to {img_1.idx}:{img_1.path.name}")
    _, matches = match_fn(img_0, img_1)

    # extract corresponding pixel coordinates
    pts0 = img_0.kp[matches[:, 0]]
    pts1 = img_1.kp[matches[:, 1]]

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
    matches = matches[inliers]

    # Create new tracks for the triangulated 3D object points
    # first create tracks for KPs in img_0, then add KPs in img_1 that match to KPs in img_0
    kp_key_pairs = [((img_0.idx, m[0]), (img_1.idx, m[1])) for m in matches]
    track_ids_added = track_manager.add_new_tracks(kp_key_pairs)

    point_cloud.add_points(track_ids_added, points_3d)

    # Update image data structs w/ new estimates: img_0 is at origin, img_1 is at (R, t)
    img_0.set_pose(np.eye(3), np.zeros((3, 1)))
    img_1.set_pose(R, t)

    print(f"Baseline constructed with {len(points_3d)} 3D points.")


def add_view(
    img_new: ImageData,
    img_ref: ImageData,
    K: NDArrayFloat,
    dist,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    match_fn: Callable[[ImageData, ImageData], tuple[NDArrayFloat, NDArrayInt]],
):
    """Adds 3D points from new view using PnP and triangulation.

    img_ref is reference image for which we already have 2D-3D pt correspondence in track_manager
    """
    # Compute KP matches from ref image to new image
    # Matching from new to ref image: Where does ref img tracked KP match to in new img?
    print(f"add_view: Computing matches from {img_ref.idx}:{img_ref.path.name} to {img_new.idx}:{img_new.path.name}")
    _, matches = match_fn(img_ref, img_new)

    # add new img KPs, that are matched to from tracked ref img KPs, to current tracks (3D pts)
    # returns track_ids and (un)tracked KPs in the new image; track_ids used as indices to point cloud
    track_ids_tracked, kp_idx_new_tracked, matches_untracked = track_manager.update_tracks(
        img_new.idx, img_ref.idx, matches
    )

    # Estimate pose of new image
    # Only use object points corresponding to tracked KPs in img_ref for PnP pose estimation
    object_points = point_cloud.get_points_as_array(track_ids_tracked)
    # PnP needs tracked KPs from new image (2D) and matching 3D object pts
    # In other words, new 2D points that observe the same 3D object points as the tracked KPs in the ref image
    img_new_pts_tracked = img_new.kp[kp_idx_new_tracked]

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
    pts_ref = img_ref.kp[matches_untracked[:, 0]]  # [:, 0] = queryIdx; [:, 1] = trainIdx
    pts_new = img_new.kp[matches_untracked[:, 1]]

    # Projection matrices: from 3D world to each camera 2D image plane
    P_ref = K @ img_ref.pose_matrix
    P_new = K @ img_new.pose_matrix

    # Triangulate the untracked KPs in the new image
    points_4d = cv.triangulatePoints(P_ref, P_new, pts_ref.T, pts_new.T)
    points_3d = (points_4d[:3] / points_4d[3]).T

    # Create track for each pair of KPs (ref, new) that were triangulated to a 3D point
    kp_key_pairs = [((img_ref.idx, m[0]), (img_new.idx, m[1])) for m in matches_untracked]
    track_ids_added = track_manager.add_new_tracks(kp_key_pairs)
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
    edges: list[ViewEdge],
    store: FeatureStore,
    track_manager: TrackManager,
    point_cloud: PointCloud,
    match_fn: Callable[[ImageData, ImageData], tuple[NDArrayFloat, NDArrayInt]],
) -> tuple[list[ViewEdge], set[int]]:
    # Pick strongest baseline:
    # - The edge of the view graph with greatest weight (ie. # kp matches) determines the two images
    img_0, img_1, best_edge = pick_best_image_pair(edges, store)
    print(
        f"Establishing baseline ({best_edge.weight} matches) from: {img_0.idx}:{img_0.path.name} and {img_1.idx}:{img_1.path.name}"
    )
    K, dist = store.get_intrisics()
    # matches -> E -> pose -> triangulation
    compute_baseline_estimate(img_0, img_1, K, track_manager, point_cloud, match_fn)

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
            add_view(img_new, img_ref, K, dist, track_manager, point_cloud, match_fn)
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
# TODO: use tyro? too many options here
@app.command()
def main(
    feature_type: str = typer.Option(
        "disk",
        "--features",
        "-f",
        help="Feature extraction method: 'sift' or 'disk'",
    ),
    matcher_type: str = typer.Option(
        "lightglue",
        "--matcher",
        "-m",
        help="Keypoint matching method: 'bf' (brute-force) or 'lightglue'",
    ),
    lowe_ratio: float = typer.Option(
        0.75,
        "--lowe-ratio",
        "-l",
        help="Lowe's ratio test threshold for BF matcher (only used when matcher='bf')",
        min=0.0,
        max=1.0,
    ),
    min_dist: float = typer.Option(
        0.0,  # preserve all matches by default
        "--min-dist",
        "-d",
        help="Minimum distance threshold for LightGlue matcher (only used when matcher='lightglue')",
        min=0.0,
        max=1.0,
    ),
    dataset: str = typer.Option(
        "statue",
        "--dataset",
        "-s",
        help="Dataset name (subdirectory in data/raw/)",
    ),
    num_features: int = typer.Option(
        2048,  # default for DISK
        "--num-features",
        "-n",
        help="Maximum number of features to extract per image",
        min=100,
    ),
    min_inliers: int = typer.Option(
        50,
        "--min-inliers",
        "-i",
        help="Minimum number of inliers to consider two views as overlapping",
        min=10,
    ),
    run_ba: bool = typer.Option(
        True,
        "--bundle-adjustment/--no-bundle-adjustment",
        "-b/-nb",
        help="Run bundle adjustment optimization after initial reconstruction",
    ),
):
    """Run Structure from Motion pipeline with configurable feature extraction and matching."""

    # Validate inputs
    if feature_type not in ["sift", "disk"]:
        typer.echo(f"Error: feature_type must be 'sift' or 'disk', got '{feature_type}'", err=True)
        raise typer.Exit(code=1)

    if matcher_type not in ["bf", "lightglue"]:
        typer.echo(f"Error: matcher_type must be 'bf' or 'lightglue', got '{matcher_type}'", err=True)
        raise typer.Exit(code=1)

    # Display configuration
    typer.echo("Configuration:")
    typer.echo(f"  Feature type: {feature_type}")
    typer.echo(f"  Num features: {num_features}")
    typer.echo(f"  Matcher type: {matcher_type}")
    if matcher_type == "bf":
        typer.echo(f"  Lowe ratio: {lowe_ratio}")
    else:
        typer.echo(f"  Min distance: {min_dist}")
    typer.echo(f"  Dataset: {dataset}")
    typer.echo()

    K, dist = calibrate_camera()

    img_dir = Path("data") / "raw" / dataset
    out_dir = Path("data") / "out" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all images & extract features
    typer.echo(f"Extracting {feature_type.upper()} features from {img_dir}...")
    # TODO: max_size in CLI or tyro config; better parameterize as scaling factor < 1, affects SfM accuracy
    image_store = FeatureStore(img_dir, K, dist, method=feature_type, num_features=num_features, max_size=4080)  # type: ignore
    track_manager = TrackManager()
    point_cloud = PointCloud()
    exporter = ReconExporter(point_cloud, image_store)

    # Create keypoint matcher with appropriate parameters
    if matcher_type == "bf":
        kp_matcher = partial(compute_matches, method="bf", lowe_ratio=lowe_ratio)
    else:
        lightglue_matcher = KF.LightGlueMatcher("disk").eval().to(device)
        kp_matcher = partial(
            compute_matches, method="lightglue", min_dist=min_dist, lightglue_matcher=lightglue_matcher
        )

    typer.echo("Constructing view graph...")
    view_graph = construct_view_graph(image_store, kp_matcher, min_inliers=min_inliers)

    # Process the first component
    typer.echo("Processing graph component...")
    process_graph_component(view_graph.edges.copy(), image_store, track_manager, point_cloud, kp_matcher)

    # Process all connected components of the view graph
    # Each component will lead to a point cloud with its own reference frame
    # and thus appear disconnected from the others
    # leftover_edges = view_graph.edges.copy()
    # while True:
    #     leftover_edges, U = process_graph_component(leftover_edges, image_store, track_manager, point_cloud, kp_matcher)
    #     if not U:
    #         break

    basename = f"{dataset}_{feature_type}_{matcher_type}"
    typer.echo(f"Saving initial reconstruction to {out_dir / f'{basename}.ply'}...")
    exporter.save_ply(filename=out_dir / f"{basename}.ply")

    if run_ba:
        typer.echo("Running bundle adjustment...")
        bundle_adjustment(image_store, point_cloud, track_manager, fix_first_camera=False)

        typer.echo(f"Final point cloud size: {point_cloud.size}")
        typer.echo(f"Saving optimized reconstruction to {out_dir / f'{basename}_ba.ply'}...")
        exporter.save_ply(filename=out_dir / f"{basename}_ba.ply")

    typer.echo("✓ Done!")


if __name__ == "__main__":
    app()
