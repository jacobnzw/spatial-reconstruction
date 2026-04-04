from __future__ import print_function
from matplotlib.pylab import e

from collections import deque
from pathlib import Path

import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
import yaml
from gtsam.symbol_shorthand import L, X

import gtsam
from config import SLAMConfig
from gtsam import Point2, Point3, Pose3, Rot3
from sfm import add_view, compute_baseline_estimate
from utils import (
    FeatureStore,
    ImageData,
    PointCloud,
    TrackManager,
    has_overlap,
    make_keypoint_matcher,
    FeatureExtractor,
)


def load_intrinsics(dataset_path, cam_key: str = "cam0"):
    camchain_file = yaml.safe_load((Path(dataset_path) / "dso" / "camchain.yaml").open())
    return np.array(camchain_file[cam_key]["intrinsics"]), np.array(camchain_file[cam_key]["distortion_coeffs"])


def enough_motion_for_keyframe(new_img: ImageData, last_kf_img: ImageData | None, kp_matcher, max_matches=60) -> bool:
    # if last_kf_img is None:  # when processing first image
    #     return True

    pnp_min = 4  # PnP needs at least 4 matches to work
    _, matches = kp_matcher(last_kf_img, new_img)
    if pnp_min <= len(matches) < max_matches:
        return True

    return False


def visual_ISAM2_plot(result):
    """Same plotting function as original"""
    fignum = 0
    fig = plt.figure(fignum)
    if not fig.axes:
        axes = fig.add_subplot(projection="3d")
    else:
        axes = fig.axes[0]
    plt.cla()

    gtsam_plot.plot_3d_points(fignum, result, "rx")

    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        gtsam_plot.plot_pose3(fignum, pose_i, 10)
        i += 1

    axes.set_xlim3d(-10, 10)
    axes.set_ylim3d(-10, 10)
    axes.set_zlim3d(-10, 10)
    plt.pause(0.5)


def gtsam_cam_pose(img: ImageData) -> Pose3:
    return Pose3(Rot3(img.R), Point3(img.t.squeeze()))


def gtsam_landmarks_from_pose(img: ImageData, tm: TrackManager, landmarks: PointCloud) -> list[Point3]:
    """Get landmarks (3D pts) visible from cam pose in img0"""
    tids = tm.get_triangulated_view_tracks(img.idx)
    return tids, [Point3(pt) for pt in landmarks.get_points_as_array(tids)]


def gtsam_keypoints_from_pose(img: ImageData, tm: TrackManager) -> list[Point3]:
    """Get keypoints (2D pts) visible from cam pose in img

    Observations of landmarks (3D pts) in given image.
    Note: These keypoints were actually used in triangulation (not just detected).
    """
    kpkeys = tm.get_triangulated_view_keypoints(img.idx)
    return [Point2(img.kp[pt]) for pt in kpkeys]


def pick_best_reference(
    query_img: ImageData,
    window: deque[ImageData],
    track_manager: TrackManager,
    K: np.array,
    kp_matcher,
    min_inliers: int = 30,
) -> ImageData | None:
    """Pick the reference image that gives the most reliable 3D-2D correspondences."""
    best_ref = None
    best_score = -1

    for ref in window:
        # Use your existing has_overlap (geometric validation + E-matrix inliers)
        is_overlapping, inliers, matches = has_overlap(ref, query_img, K, kp_matcher, min_inliers)
        if not is_overlapping:
            continue

        # Count how many of these inliers are already triangulated tracks
        triangulated_count = 0
        for m in matches[:inliers]:  # only inliers
            kp_key_ref = (ref.idx, m[0])  # queryIdx in ref
            if track_manager.get_track(kp_key_ref) is not None:
                triangulated_count += 1

        if triangulated_count > best_score:
            best_score = triangulated_count
            best_ref = ref

    # Fallback: always prefer the most recent keyframe if it has at least a few points
    if best_ref is None and len(window) > 0:
        best_ref = window[-1]  # most recent is usually the safest geometrically

    return best_ref


def add_initial_factors_and_priors(
    graph: gtsam.NonlinearFactorGraph,
    initial_estimates,
    K: gtsam.Cal3_S2,
    img0: ImageData,
    img1: ImageData,
    track_manager: TrackManager,
    point_cloud: PointCloud,
):
    # Strong prior on first pose (fixes gauge)
    origin_pose = Pose3(Rot3.Ypr(0, 0, 0), Point3(0, 0, 0))
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
    graph.addPriorPose3(X(img0.idx), origin_pose, pose_noise)
    initial_estimates.insert(X(img0.idx), origin_pose)
    # First pose got prior + init value, other poses get only init value
    initial_estimates.insert(X(img1.idx), gtsam_cam_pose(img1))
    # TODO: what's the diff btw: putting a prior on pose vs. using initial value for pose

    # Initials for landmarks from both images (img0 sufficient; no need for  img1)
    # When bootstrapping from first two frames, landmarks observed from img0 same as those from img1,
    # otherwise they couldn't get triangulated.
    x0_lm_tids, x0_landmarks = gtsam_landmarks_from_pose(img0, track_manager, point_cloud)

    # Prior on first landmark (fixes scale)
    lm_noise = gtsam.noiseModel.Isotropic.Sigma(3, 2.0)
    graph.addPriorPoint3(L(0), x0_landmarks[0], lm_noise)

    # Observation noise (pixels)
    obs_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
    # for each landmark observed in this frame
    for lm_tid, lm in zip(x0_lm_tids, x0_landmarks):
        # Add factor for each view observing landmark L(l)
        for img_idx, kp_idx in track_manager.get_keypoints(lm_tid):
            obs = img0.kp[kp_idx] if img_idx == 0 else img1.kp[kp_idx]
            # Landmark indices are track IDs (addresses to the point_cloud)
            graph.add(gtsam.GenericProjectionFactorCal3_S2(obs, obs_noise, X(img_idx), L(lm_tid), K))
        initial_estimates.insert(L(lm_tid), lm)


def add_new_keyframe_factors(
    graph: gtsam.NonlinearFactorGraph,
    initial_estimates: gtsam.Values,
    img: ImageData,
    K: gtsam.Cal3_S2,
    track_manager: TrackManager,
    point_cloud: PointCloud,
):
    tids, lms = gtsam_landmarks_from_pose(img, track_manager, point_cloud)

    # Add factors joining new view (keyframe) pose X with all the visible landmarks L (from pose X)
    obs_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)
    for tid, lm in zip(tids, lms):
        # Get KP observation of the landmark (lm) in the new view (img)
        kp_keys = track_manager.get_keypoints(tid, img.idx)
        assert len(kp_keys) == 1  # geometrically only 1 KP possible; True for symmetric view-to-view KP matches
        obs = img.kp[kp_keys[0][1]][:, None]

        graph.add(gtsam.GenericProjectionFactorCal3_S2(obs, obs_noise, X(img.idx), L(tid), K))
        if not initial_estimates.exists(L(tid)):
            initial_estimates.insert(L(tid), lm)
    initial_estimates.insert(X(img.idx), gtsam_cam_pose(img))


def visual_ISAM2_tumvi_example(cfg: SLAMConfig):
    dataset_path = Path("data/tum") / cfg.dataset
    image_dir = Path(dataset_path) / "dso" / "cam0" / "images"

    # fx, fy, s, cx, cy
    k_vec, dist = load_intrinsics(dataset_path)
    fx, fy, cx, cy = k_vec
    K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    # iSAM2 parameters
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(0.01)
    parameters.relinearizeSkip = 1
    isam = gtsam.ISAM2(parameters)

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    track_manager = TrackManager()
    point_cloud = PointCloud()

    # Load images
    extractor = FeatureExtractor(cfg, image_dir, ext="png")
    image_dir = Path(dataset_path) / "dso" / "cam0" / "images"
    images = FeatureStore(image_dir, K.K(), dist, feature_extractor=extractor, ext="png", max_frames=cfg.max_frames)
    kp_matcher = make_keypoint_matcher(cfg)

    num_recent_keyframes = 10
    keyframe_window = deque(maxlen=num_recent_keyframes)  # keep ~8–15 recent keyframes
    last_keyframe_idx = 0
    keyframe_window.append(images[last_keyframe_idx])

    for img in images:
        if img.idx == 0:
            continue

        # print(f"{last_keyframe_idx=}")
        # if last_kf_img is not None:
        #     print(f"{last_kf_img.idx=}")

        # last_kf_img = None if last_keyframe_idx == 0 else images[last_keyframe_idx]
        last_kf_img = images[last_keyframe_idx]
        if not enough_motion_for_keyframe(img, last_kf_img, kp_matcher, max_matches=150):
            continue  # skip non-keyframes (very important at 20 Hz!)

        # --- Now we have a new keyframe ---
        if last_keyframe_idx == 0:
            print(f"First pair of keyframes: {keyframe_window[0].idx=} and {img.idx=}...")
            compute_baseline_estimate(keyframe_window[0], img, K.K(), track_manager, point_cloud, kp_matcher)
            # now have: first triang 3d points (landmarks) + keypoints for each image (cam pose)
            add_initial_factors_and_priors(
                graph, initial_estimate, K, keyframe_window[0], img, track_manager, point_cloud
            )
            last_keyframe_idx = img.idx
            keyframe_window.append(img)
            continue

        print(f"Processing frame {img.idx}: {img.path}")

        # Normal keyframe: pick most overlapping recent keyframe
        ref = pick_best_reference(img, keyframe_window, track_manager, K.K(), kp_matcher)
        if ref is None:
            print("No good reference found — skipping keyframe")
            continue
        add_view(img, ref, K.K(), dist, track_manager, point_cloud, kp_matcher)

        # --- Now push everything to iSAM2 ---
        add_new_keyframe_factors(graph, initial_estimate, img, K, track_manager, point_cloud)
        isam.update(graph, initial_estimate)
        isam.update()  # optional extra iteration

        # bookkeeping
        last_keyframe_idx = img.idx

        # Show current estimate
        current_estimate = isam.calculateEstimate()
        print(f"Frame {img.idx} poses:")
        for j in range(img.idx + 1):
            if current_estimate.exists(X(j)):
                print(X(j), ":", current_estimate.atPose3(X(j)))
        # visual_ISAM2_plot(current_estimate, ax_lim=10, delay_sec=0.5)

        # Clear for next incremental step
        graph.resize(0)
        initial_estimate.clear()

    # plt.ioff()
    # plt.show()


if __name__ == "__main__":
    cfg = SLAMConfig()
    visual_ISAM2_tumvi_example(cfg)
