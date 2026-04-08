from __future__ import print_function

from collections import deque
from pathlib import Path

import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
import tyro
import yaml
from gtsam.symbol_shorthand import L, X

import gtsam
from config import SLAMConfig
from gtsam import Point2, Point3, Pose3, Rot3
from sfm import add_view, compute_baseline_estimate
from utils import (
    FeatureExtractor,
    FeatureStore,
    ImageData,
    PointCloud,
    TrackManager,
    has_overlap,
    make_keypoint_matcher,
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
    """Get landmarks (3D pts) visible from cam pose in img"""
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

    baseline_pose = gtsam_cam_pose(img1)  # img1 is the second keyframe
    between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]))
    graph.add(gtsam.BetweenFactorPose3(X(img0.idx), X(img1.idx), baseline_pose, between_noise))
    # First pose got prior + init value, other poses get only init value
    initial_estimates.insert(X(img1.idx), gtsam_cam_pose(img1))
    # TODO: what's the diff btw: putting a prior on pose vs. using initial value for pose

    # Initials for landmarks from both images (img0 sufficient; no need for img1)
    # When bootstrapping from first two frames, landmarks observed from img0 same as those from img1,
    # otherwise they couldn't get triangulated.
    x0_lm_tids, x0_landmarks = gtsam_landmarks_from_pose(img0, track_manager, point_cloud)

    # Prior on landmarks (fixes scale; stabilizes optimization in initial phase)
    lm_noise = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)
    # Observation noise (pixels)
    obs_noise = gtsam.noiseModel.Isotropic.Sigma(2, 5.0)
    # for each landmark observed in this frame
    for lm_tid, lm in zip(x0_lm_tids, x0_landmarks):
        graph.addPriorPoint3(L(lm_tid), lm, lm_noise)
        # Add factor for each view observing landmark L(lm_tid)
        for img_idx, kp_idx in track_manager.get_keypoints(lm_tid):
            obs = img0.kp[kp_idx] if img_idx == 0 else img1.kp[kp_idx]
            # Landmark indices are track IDs (addresses to the point_cloud)
            graph.add(gtsam.GenericProjectionFactorCal3_S2(obs, obs_noise, X(img_idx), L(lm_tid), K))
        initial_estimates.insert(L(lm_tid), lm)


def add_new_keyframe_factors(
    graph: gtsam.NonlinearFactorGraph,
    initial_estimates: gtsam.Values,
    isam,
    img_new: ImageData,
    img_ref: ImageData,
    K: gtsam.Cal3_S2,
    track_manager: TrackManager,
    point_cloud: PointCloud,
):
    # Landmark IDs and 3D positions visible from new view (img_new)
    tids, lms = gtsam_landmarks_from_pose(img_new, track_manager, point_cloud)

    # Add factors joining new view (keyframe) pose X with all the visible landmarks L (from pose X)
    obs_noise = gtsam.noiseModel.Isotropic.Sigma(2, 5.0)
    for tid, lm in zip(tids, lms):
        # Get KP observation of the landmark (lm) in the new view (img)
        kp_keys = track_manager.get_keypoints(tid)
        for img_idx, kp_idx in kp_keys:
            if img_idx != img_new.idx and img_idx != img_ref.idx:
                continue  # only add factors for the new keyframe and its reference keyframe (for now)
            # Note: the landmarks observed in new keyframe were triangulated from the reference keyframe,
            # so they must have been observed in the reference keyframe (img_ref).
            img = img_new if img_idx == img_new.idx else img_ref
            obs = img.kp[kp_idx][:, None]  # make (2,) into (2,1) for gtsam
            graph.add(gtsam.GenericProjectionFactorCal3_S2(obs, obs_noise, X(img.idx), L(tid), K))

        # Landmark initials: only add if not already in graph (from previous keyframes)
        # FIXME: how not to re-add L(0) already in graph or isam object? isam.valueExists(L(0))?
        if not isam.valueExists(L(tid)) and not initial_estimates.exists(L(tid)):
            initial_estimates.insert(L(tid), lm)
            # Priors help stabilize in initial phase
            lm_noise = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)
            graph.add(gtsam.PriorFactorPoint3(L(tid), lm, lm_noise))

    # Pose initials: for new pose (possibly ref pose if not already in graph)
    initial_estimates.insert(X(img_new.idx), gtsam_cam_pose(img_new))
    if not isam.valueExists(X(img_ref.idx)) and not initial_estimates.exists(X(img_ref.idx)):
        initial_estimates.insert(X(img_ref.idx), gtsam_cam_pose(img_ref))


def visual_ISAM2_tumvi_example(cfg: SLAMConfig = SLAMConfig()):
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

    keyframe_window = deque(maxlen=cfg.max_window_keyframes)  # keep ~8–15 recent keyframes
    last_keyframe_idx = 0
    keyframe_window.append(images[last_keyframe_idx])

    for img in images:
        if img.idx == 0:
            continue

        last_kf_img = images[last_keyframe_idx]
        if not enough_motion_for_keyframe(img, last_kf_img, kp_matcher, max_matches=cfg.max_motion_matches):
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
        add_new_keyframe_factors(graph, initial_estimate, isam, img, ref, K, track_manager, point_cloud)

        # === DEBUG: inspect the graph before iSAM2 update ===
        print("\n=== GRAPH DEBUG AFTER BOOTSTRAP ===")
        print(f"  Factors in graph: {graph.size()}")
        print(f"  Initial estimates size: {initial_estimate.size()}")

        # Count observations per landmark
        from collections import defaultdict

        obs_per_lm = defaultdict(list)
        for i in range(graph.size()):
            factor = graph.at(i)
            if isinstance(factor, gtsam.GenericProjectionFactorCal3_S2):
                lm_key = factor.keys()[1]  # second key is the landmark
                obs_per_lm[lm_key].append(factor.keys()[0])  # camera that sees it

        bad_lms = []
        for lm_key, cams in obs_per_lm.items():
            if len(cams) < 2:
                bad_lms.append((lm_key, len(cams)))
            # Also check initial depth (points near infinity are deadly)
            if initial_estimate.exists(lm_key):
                pt = initial_estimate.atPoint3(lm_key)
                depth = np.linalg.norm(pt)
                if depth > 100 or depth < 0.1:
                    print(f"  Suspicious landmark {lm_key} depth = {depth:.2f} m")

        print(f"  Landmarks with <2 observations: {len(bad_lms)}")
        if bad_lms:
            print("  →", bad_lms[:10])  # show first 10 bad ones

        # Specifically check the one that failed
        lmark = L(147)  # or whatever track ID corresponds to the bad landmark
        if initial_estimate.exists(lmark):
            pt = initial_estimate.atPoint3(lmark)
            print(f"  Landmark {lmark} initial position: {pt}, norm = {np.linalg.norm(pt):.2f}")

        isam.update(graph, initial_estimate)
        isam.update()  # optional extra iteration

        # bookkeeping
        print(f"DEBUG: keyframe idxs = {[kf.idx for kf in keyframe_window]}, current img idx = {img.idx}")
        last_keyframe_idx = img.idx

        # # Show current estimate
        # current_estimate = isam.calculateEstimate()
        # print(f"Frame {img.idx} poses:")
        # for j in range(img.idx + 1):
        #     if current_estimate.exists(X(j)):
        #         print(X(j), ":", current_estimate.atPose3(X(j)))
        # visual_ISAM2_plot(current_estimate, ax_lim=10, delay_sec=0.5)

        # # Clear for next incremental step
        # if len(keyframe_window) == keyframe_window.maxlen:
        #     print("Clearing graph and initial estimates for next batch...")
        # graph.resize(0)
        initial_estimate.clear()

    # plt.ioff()
    # plt.show()


if __name__ == "__main__":
    tyro.cli(visual_ISAM2_tumvi_example)
