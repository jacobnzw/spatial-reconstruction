from __future__ import print_function

from collections import deque
from pathlib import Path

import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
import tyro
import yaml
from gtsam.symbol_shorthand import L, X
from rich.pretty import pprint

import gtsam
from config import SLAMConfig
from gtsam import (
    Point2,
    Point3,
    Pose3,
    Rot3,
    SmartProjectionParams,
    SmartProjectionPoseFactorCal3_S2,
)
from sfm import add_view, bootstrap_from_two_views
from utils import (
    CameraModel,
    CameraType,
    FeatureExtractor,
    FrameLoader,
    PointCloud,
    TrackManager,
    ViewData,
    has_overlap,
    make_keypoint_matcher,
)


class KeyframeStreamer:
    """Streams keyframes.

    Handles selection of keyframes. By iterating the `FeatureExtractor`, which in turn iterates `FrameLoader`, the class
    obtains new frames and subjects them to keyframe selection criteria. Keyframe is judged by time-based criteria
    as well as keypoint-based motion and track quality criteria.
    The class also provides a method to find the best reference keyframe from the past frames stored in the window.

    Args:
        feature_extractor: An instance of FeatureExtractor to extract keypoints and descriptors.
        matcher: A keypoint matcher function that takes two ImageData objects and returns matches.
        track_manager: TrackManager object.
        max_motion_matches: Maximum number of matches to consider for judging motion (default: 120).
        max_frames: Maximum number of frames to keep in the sliding window for keyframe selection (default: 10).
        fps: Frames per second (Hz) of the read dataset.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        matcher,
        track_manager: TrackManager,
        max_motion_matches=120,
        max_window_keyframes=10,
        fps=20,  # for TUM-VI
    ):
        self.extractor = feature_extractor
        self.matcher = matcher
        self.track_manager = track_manager
        self.max_motion_matches = max_motion_matches
        self.fps = fps
        self._keyframe_window = deque(maxlen=max_window_keyframes)

    def _is_keyframe(self, frame: ViewData) -> bool:
        # TODO: Combine the two because E-mat and PnP are used together in add_view anyway
        pnp_min = 40  # minimum points for reliable pose estimation using PnP
        emat_min = 10  # minimum points for reliable essential matrix estimation (in triangulation)

        last_kf: ViewData = self._keyframe_window[-1]  # pick last keyframe

        # Frame considered only after certain time has passed since last keyframe (eg. 0.5 second)
        if frame.idx - last_kf.idx < self.fps / 2:
            # print(
            #     f"DEBUG: Not enough time elapsed for new keyframe  → {frame.idx=} failed because time gap "
            #     f"{frame.idx - last_kf.idx} < {self.fps / 2} (half FPS)."
            # )
            return False
        max_delta_cond = frame.idx - last_kf.idx > self.fps  # keyframe every second (based on FPS)

        _, matches = self.matcher(last_kf, frame)
        # When searching for second keyframe to bootstrap initial tracks
        n_matches = len(matches)
        if len(self._keyframe_window) < 2 and (emat_min < n_matches < self.max_motion_matches):
            return True

        track_ids, tracked_matches, untracked_matches = self.track_manager.get_track_observations_for_view(
            last_kf.idx, matches
        )
        # TODO: few tracked kps in new frame = n_tracked_matches < n_kp_ref_tracked * threshold ([0, 1])
        # num KPs tracked by ref view may be higher than num tracked matches
        # Replaceable by enough_new_kps?? Nope, enough_new_kps = kps in new frame; not the ref frame
        # n_kp_ref_tracked = len(self.track_manager.get_triangulated_view_keypoints(last_kf.idx))
        n_tracked_matches = len(tracked_matches)
        n_untracked_matches = len(untracked_matches)
        # Need enough matches that correspond to existing tracks (for PnP pose estimation of new keyframe)
        estimation_cond = n_tracked_matches >= pnp_min and n_untracked_matches >= emat_min
        if max_delta_cond and estimation_cond:
            return True  # Force keyframe every second if estimator has enough data

        # Keyframe should have a good number of new keypoints (to be triangulated as new landmarks) to be worth it
        enough_new_kps = n_untracked_matches / n_matches > 0.5  # TODO: tune this threshold

        # Parallax check: ensure sufficient camera motion by checking median displacement of tracked keypoints between last keyframe and new frame
        if n_tracked_matches > 10:  # at least 10 samples for median parallax estimate
            kp_idx_tracked_ref, kp_idx_tracked_new = tracked_matches[:, 0], tracked_matches[:, 1]
            pts_tracked_ref, pts_tracked_new = last_kf.kp[kp_idx_tracked_ref], frame.kp[kp_idx_tracked_new]  # ty:ignore[not-subscriptable]
            displacements = np.linalg.norm(pts_tracked_ref - pts_tracked_new, axis=1)
            median_parallax = np.median(displacements)

            if median_parallax < 15.0:  # Insufficient camera motion
                print(f"DEBUG: Insufficient parallax {median_parallax:.1f} < 15.0")
                return False

        # Enough motion (via KP parallax), new KPs and estimator has enough data
        if estimation_cond and enough_new_kps:
            print(f"New keyframe  → {frame.idx=} passed with {len(matches)} matches to {last_kf.idx=}.")
            print(f"DEBUG: {n_tracked_matches=} (for PnP) and {n_untracked_matches=} (for new landmarks).")
            return True

        print(
            f"DEBUG: {frame.idx=} {estimation_cond=} {enough_new_kps=} {max_delta_cond=} {n_tracked_matches=} {n_untracked_matches=}"
        )
        return False

    def keyframes(self):
        """Yields pairs of consecutive keyframes."""

        # TODO: consider returning struct with matches & stats between KFs, forward to sfm methods
        for idx, frame in enumerate(self.extractor.iter_frames_with_features()):
            # Add 1st frame as keyframe
            if idx == 0:
                self._keyframe_window.append(frame)
                continue

            if self._is_keyframe(frame):
                self._keyframe_window.append(frame)
                yield frame, self._keyframe_window[-2]

    def find_reference_frame(
        self,
        keyframe: ViewData,
        track_manager: TrackManager,
        min_inliers: int = 30,
    ) -> ViewData | None:
        """Prefer the most recent keyframe, only fall back if it has poor overlap."""

        # Always try the most recent keyframe first
        last_kf = self._keyframe_window[-2]
        is_overlapping, inliers, matches = has_overlap(last_kf, keyframe, self.matcher, min_inliers)
        if is_overlapping and inliers >= 45:  # TODO: remove inliers cond; already done in has_overlap
            return last_kf

        best_ref = None
        best_score = -1

        for ref in self._keyframe_window:
            if ref.idx == keyframe.idx:  # don't pick the current keyframe as reference
                continue
            # Use your existing has_overlap (geometric validation + E-matrix inliers)
            is_overlapping, inliers, matches = has_overlap(ref, keyframe, self.matcher, min_inliers)
            if not is_overlapping:
                continue

            # Count how many of these inliers are already triangulated tracks
            triangulated_count = 0
            for m in matches:  # only inliers
                kp_key_ref = (ref.idx, m[0])  # queryIdx in ref
                if track_manager.get_track(kp_key_ref) is not None:
                    triangulated_count += 1

            if triangulated_count > best_score:
                best_score = triangulated_count
                best_ref = ref

        # Fallback: always prefer the most recent keyframe if it has at least a few points
        if best_ref is None and len(self._keyframe_window) > 0:
            best_ref = self._keyframe_window[-2]  # previous keyframe is at -2 since the current keyframe is at -1

        return best_ref


def load_intrinsics(dataset_path, cam_key: str = "cam0"):
    camchain_file = yaml.safe_load((Path(dataset_path) / "dso" / "camchain.yaml").open())
    return np.array(camchain_file[cam_key]["intrinsics"]), np.array(camchain_file[cam_key]["distortion_coeffs"])


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


def gtsam_world_T_cam(img: ViewData) -> Pose3:
    # Pose3 represents cam in world frame: world_T_cam; unlike ViewData.cam_T_world !
    world_T_cam = img.cam_T_world.inv()
    R, t = world_T_cam.rotation.as_matrix(), world_T_cam.translation
    return Pose3(Rot3(R), Point3(t))


def gtsam_landmarks_from_pose(img: ViewData, tm: TrackManager, landmarks: PointCloud) -> list[Point3]:
    """Get landmarks (3D pts) visible from cam pose in img"""
    tids = tm.get_triangulated_view_tracks(img.idx)
    return tids, [Point3(pt) for pt in landmarks.get_points_as_array(tids)]


def gtsam_keypoints_from_pose(img: ViewData, tm: TrackManager) -> list[Point3]:
    """Get keypoints (2D pts) visible from cam pose in img

    Observations of landmarks (3D pts) in given image.
    Note: These keypoints were actually used in triangulation (not just detected).
    """
    kpkeys = tm.get_triangulated_view_keypoints(img.idx)
    return [Point2(img.kp[pt]) for pt in kpkeys]


def add_initial_factors_and_priors(
    graph: gtsam.NonlinearFactorGraph,
    initial_estimates,
    K: gtsam.Cal3_S2,
    img0: ViewData,
    img1: ViewData,
    track_manager: TrackManager,
    point_cloud: PointCloud,
):
    # Strong prior on first pose (fixes gauge)
    origin_pose = Pose3(Rot3.Ypr(0, 0, 0), Point3(0, 0, 0))
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
    graph.addPriorPose3(X(img0.idx), origin_pose, pose_noise)
    initial_estimates.insert(X(img0.idx), origin_pose)

    baseline_pose = gtsam_world_T_cam(img1)  # img1 is the second keyframe
    between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]))
    graph.add(gtsam.BetweenFactorPose3(X(img0.idx), X(img1.idx), baseline_pose, between_noise))
    # First pose got prior + init value, other poses get only init value
    initial_estimates.insert(X(img1.idx), gtsam_world_T_cam(img1))

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
    img_new: ViewData,
    img_ref: ViewData,
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
    initial_estimates.insert(X(img_new.idx), gtsam_world_T_cam(img_new))
    if not isam.valueExists(X(img_ref.idx)) and not initial_estimates.exists(X(img_ref.idx)):
        initial_estimates.insert(X(img_ref.idx), gtsam_world_T_cam(img_ref))


def make_keyframe_streamer(
    image_dir, cfg, camera_model, kp_matcher, track_manager, undistort=False
) -> KeyframeStreamer:
    loader = FrameLoader(
        image_dir,
        max_size=cfg.max_size,
        max_frames=cfg.max_read_frames,
        ext="png",
        camera_model=camera_model,
        undistort=undistort,
    )
    extractor = FeatureExtractor(cfg, loader)
    return KeyframeStreamer(
        extractor,
        kp_matcher,
        track_manager,
        max_motion_matches=cfg.max_motion_matches,
        max_window_keyframes=cfg.max_window_keyframes,
    )


def visual_ISAM2_tumvi_example(cfg: SLAMConfig = SLAMConfig()):
    # Display configuration
    pprint(cfg, expand_all=True)
    print()

    dataset_path = Path("data/tum") / cfg.dataset
    image_dir = Path(dataset_path) / "dso" / "cam0" / "images"

    # Camera intrinsics
    k_vec, dist_vec = load_intrinsics(dataset_path)
    fx, fy, cx, cy = k_vec
    K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    # K = gtsam.Cal3Fisheye(fx, fy, 0.0, cx, cy, *dist_vec)

    # === Smart Projection Setup ===
    smart_params = SmartProjectionParams()
    smart_params.setRankTolerance(1e-5)
    smart_params.setDynamicOutlierRejectionThreshold(True)
    smart_params.setLandmarkDistanceThreshold(True)

    obs_noise = gtsam.noiseModel.Isotropic.Sigma(2, 5.0)

    # One smart factor per track (landmark)
    smart_factors: dict[int, SmartProjectionPoseFactorCal3_S2] = {}
    # smart_factors: dict[int, SmartProjectionPoseFactorCal3Fisheye] = {}

    # iSAM2
    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(0.01)
    parameters.relinearizeSkip = 1
    isam = gtsam.ISAM2(parameters)
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    track_manager = TrackManager()
    point_cloud = PointCloud()

    # Setup keyframe streaming
    camera_model = CameraModel(model_type=CameraType.FISHEYE, K=K.K(), dist=dist_vec)
    kp_matcher = make_keypoint_matcher(cfg)
    streamer = make_keyframe_streamer(image_dir, cfg, camera_model, kp_matcher, track_manager, undistort=cfg.undistort)

    for keyframe, prev_keyframe in streamer.keyframes():
        print(f"Processing keyframe {keyframe.idx}: {keyframe.path.name}")

        # === Bootstrap: first two keyframes ===
        if prev_keyframe.idx == 0:
            print(f"  → Bootstrap with frames {prev_keyframe.idx} and {keyframe.idx}")
            bootstrap_from_two_views(prev_keyframe, keyframe, track_manager, point_cloud, kp_matcher)

            # Add first two poses
            keyframe_pose, prev_keyframe_pose = gtsam_world_T_cam(keyframe), gtsam_world_T_cam(prev_keyframe)
            initial_estimate.insert(X(prev_keyframe.idx), prev_keyframe_pose)
            initial_estimate.insert(X(keyframe.idx), keyframe_pose)

            # Strong prior on first pose
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.6, 0.6, 0.6]))
            graph.add(gtsam.PriorFactorPose3(X(prev_keyframe.idx), prev_keyframe_pose, pose_noise))

            # Between factor for baseline
            between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.8, 0.8, 0.8]))
            graph.add(gtsam.BetweenFactorPose3(X(prev_keyframe.idx), X(keyframe.idx), keyframe_pose, between_noise))

            # Add observations from both bootstrap frames to smart factors
            for track_id in track_manager.track_to_kps.keys():
                for img_idx, kp_idx in track_manager.get_keypoints(track_id):
                    img = prev_keyframe if img_idx == prev_keyframe.idx else keyframe
                    obs = Point2(img.kp[kp_idx])
                    if track_id not in smart_factors:
                        factor = SmartProjectionPoseFactorCal3_S2(obs_noise, K, smart_params)
                        # factor = SmartProjectionPoseFactorCal3Fisheye(obs_noise, K, smart_params)
                        smart_factors[track_id] = factor
                        graph.add(factor)
                    smart_factors[track_id].add(obs, X(img_idx))
            continue

        # === Normal keyframe ===
        # ref = streamer.find_reference_frame(keyframe, track_manager, min_inliers=cfg.min_inliers)
        # if ref is None:
        #     print("  → No good reference found, skipping")
        #     continue
        # Previous keframe as reference: makes sense in sequentially ordered datasets
        ref = prev_keyframe
        add_view(keyframe, ref, track_manager, point_cloud, kp_matcher)

        # === DEPTH FILTER (critical for chirality) ===
        # TODO: remove if depth threshold unified with triangulation
        new_tracks = track_manager.get_triangulated_view_tracks(keyframe.idx)
        for tid in list(new_tracks):
            pt_world = point_cloud.get_point(tid)
            if pt_world is None:
                continue

            # Transform point to camera frame; depth = z-coordinate
            depth_cam = keyframe.transform_to_camera_frame(pt_world)[2]
            if depth_cam < 0.3:  # behind camera or too close
                # Optional: remove the track completely (simple but effective)
                # You can add a remove_track method to TrackManager if you want
                print(f"  → Removed track {tid} (negative depth {depth_cam:.2f})")
                # For now just skip adding it to smart factor
                if tid in smart_factors:
                    del smart_factors[tid]

        keyframe_pose, ref_pose = gtsam_world_T_cam(keyframe), gtsam_world_T_cam(ref)
        # Between factor to reference keyframe (not necessarily the previous keyframe)
        between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.8, 0.8, 0.8]))
        graph.add(gtsam.BetweenFactorPose3(X(ref.idx), X(keyframe.idx), keyframe_pose.between(ref_pose), between_noise))

        # Add new observations to existing smart factors
        for track_id in track_manager.get_triangulated_view_tracks(keyframe.idx):
            kp_keys = track_manager.get_keypoints(track_id, keyframe.idx)
            if not kp_keys:
                continue
            kp_idx = kp_keys[0][1]
            obs = Point2(keyframe.kp[kp_idx])

            if track_id not in smart_factors:
                factor = SmartProjectionPoseFactorCal3_S2(obs_noise, K, smart_params)
                # factor = SmartProjectionPoseFactorCal3Fisheye(obs_noise, K, smart_params)
                smart_factors[track_id] = factor
                graph.add(factor)

            smart_factors[track_id].add(obs, X(keyframe.idx))

        # Add new pose
        initial_estimate.insert(X(keyframe.idx), gtsam_world_T_cam(keyframe))

        # === Update iSAM2 ===
        isam.update(graph, initial_estimate)
        isam.update()  # extra iteration for better convergence

        initial_estimate.clear()  # Only clear temporary initials

        print(f"  → Added keyframe {keyframe.idx}, total tracks: {len(smart_factors)}")

        # Optional visualization every few keyframes
        if keyframe.idx % 5 == 0:
            current_estimate = isam.calculateEstimate()
            visual_ISAM2_plot(current_estimate)

    print("Finished processing all keyframes.")


if __name__ == "__main__":
    tyro.cli(visual_ISAM2_tumvi_example)
