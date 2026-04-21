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
from gtsam import (
    Point2,
    Point3,
    Pose3,
    Rot3,
    SmartProjectionParams,
    SmartProjectionPoseFactorCal3_S2,
)
from sfm import add_view, compute_baseline_estimate
from utils import (
    FeatureExtractor,
    FrameLoader,
    ImageData,
    NDArrayFloat,
    PointCloud,
    TrackManager,
    has_overlap,
    make_keypoint_matcher,
)


class KeyframeStreamer:
    """Streams keyframes from a directory of images.

    This class handles loading images, extracting features, and determining which frames should be
    keyframes based on motion. It also provides a method to find the best reference keyframe for a
    new keyframe based on overlap and triangulated tracks.

    Args:
        image_dir: Directory containing the images.
        feature_extractor: An instance of FeatureExtractor to extract keypoints and descriptors.
        matcher: A keypoint matcher function that takes two ImageData objects and returns matches.
        ext: Image file extension (default: "png").
        max_motion_matches: Maximum number of matches to consider for judging motion (default: 120).
        max_frames: Maximum number of frames to keep in the sliding window for keyframe selection (default: 10).
        max_read_frames: Optional maximum number of frames to read from the directory (default: None, meaning read all).
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        matcher,
        max_motion_matches=120,
        max_window_keyframes=10,
    ):
        self.extractor = feature_extractor
        self.matcher = matcher
        self.max_motion_matches = max_motion_matches
        self._keyframe_window = deque(maxlen=max_window_keyframes)

    def _enough_motion_for_keyframe(self, frame: ImageData) -> bool:
        pnp_min = 40  # PnP needs at least 40 matches to work

        last_kf = self._keyframe_window[-1]  # pick last keyframe
        _, matches = self.matcher(last_kf, frame)

        if pnp_min <= len(matches) < self.max_motion_matches:
            print(f"New keyframe  → {frame.idx=} passed with {len(matches)} matches to {last_kf.idx=}.")
            return True

        return False

    def keyframes(self):
        """Yields pairs of consecutive keyframes."""
        for idx, frame in enumerate(self.extractor.iter_frames_with_features()):
            # Add 1st frame as keyframe
            if idx == 0:
                self._keyframe_window.append(frame)
                continue

            # yield only frames that are sufficiently different from the last keyframe
            if self._enough_motion_for_keyframe(frame):
                self._keyframe_window.append(frame)
                yield frame, self._keyframe_window[-2]

    def find_reference_frame(
        self,
        keyframe: ImageData,
        track_manager: TrackManager,
        K: NDArrayFloat,
        min_inliers: int = 30,
    ) -> ImageData | None:
        """Prefer the most recent keyframe, only fall back if it has poor overlap."""
        # Always try the most recent keyframe first
        last_kf = self._keyframe_window[-2]
        is_overlapping, inliers, matches = has_overlap(last_kf, keyframe, K, self.matcher, min_inliers)
        if is_overlapping and inliers >= 45:
            return last_kf

        best_ref = None
        best_score = -1

        for ref in self._keyframe_window:  # exclude the current frame at the end
            if ref.idx == keyframe.idx:  # don't pick the current keyframe as reference
                continue
            # Use your existing has_overlap (geometric validation + E-matrix inliers)
            is_overlapping, inliers, matches = has_overlap(ref, keyframe, K, self.matcher, min_inliers)
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
    loader = FrameLoader(image_dir, max_size=cfg.max_size, max_frames=cfg.max_read_frames, ext="png")
    extractor = FeatureExtractor(cfg, loader)
    kp_matcher = make_keypoint_matcher(cfg)
    streamer = KeyframeStreamer(
        extractor,
        kp_matcher,
        max_motion_matches=cfg.max_motion_matches,
        max_window_keyframes=cfg.max_window_keyframes,
    )

    for keyframe, prev_keyframe in streamer.keyframes():
        print(f"Processing keyframe {keyframe.idx}: {keyframe.path.name}")

        # === Bootstrap: first two keyframes ===
        if prev_keyframe.idx == 0:
            print(f"  → Bootstrap with frames {prev_keyframe.idx} and {keyframe.idx}")
            compute_baseline_estimate(prev_keyframe, keyframe, K.K(), track_manager, point_cloud, kp_matcher)

            # Add first two poses
            keyframe_pose, prev_keyframe_pose = gtsam_cam_pose(keyframe), gtsam_cam_pose(prev_keyframe)
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
        ref = streamer.find_reference_frame(keyframe, track_manager, K.K())
        if ref is None:
            print("  → No good reference found, skipping")
            continue

        add_view(keyframe, ref, K.K(), dist_vec, track_manager, point_cloud, kp_matcher)

        # === DEPTH FILTER (critical for chirality) ===
        new_tracks = track_manager.get_triangulated_view_tracks(keyframe.idx)
        for tid in list(new_tracks):
            pt = point_cloud.get_point(tid)
            if pt is None:
                continue
            # Transform point to camera frame of the new keyframe
            cam_pose = gtsam_cam_pose(keyframe)
            pt_cam = cam_pose.transformTo(Point3(pt))
            if pt_cam[2] < 0.3:  # behind camera or too close
                # Optional: remove the track completely (simple but effective)
                # You can add a remove_track method to TrackManager if you want
                print(f"  → Removed track {tid} (negative depth {pt_cam[2]:.2f})")
                # For now just skip adding it to smart factor
                if tid in smart_factors:
                    del smart_factors[tid]

        keyframe_pose, ref_pose = gtsam_cam_pose(keyframe), gtsam_cam_pose(ref)
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
        initial_estimate.insert(X(keyframe.idx), gtsam_cam_pose(keyframe))

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
