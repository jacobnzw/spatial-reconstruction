import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    from __future__ import print_function
    import marimo as mo
    import glob
    import os
    from pathlib import Path
    import yaml
    import cv2 as cv

    import gtsam
    import gtsam.utils.plot as gtsam_plot
    import matplotlib.pyplot as plt
    import numpy as np
    from gtsam.examples import SFMdata
    from gtsam.symbol_shorthand import L, X
    from gtsam import Rot3, Pose3, Point3
    from utils import (
        ImageData,
        FeatureStore,
        TrackManager,
        PointCloud,
        make_keypoint_matcher,
        has_overlap,
    )
    from sfm import compute_baseline_estimate, add_view
    from config import SfMConfig
    from collections import deque


@app.cell
def _():
    """
    Modified VisualISAM2Example for TUM-VI corridor4_512_16 (mono left camera)
    """

    dataset_path = Path("data/tum/dataset-corridor4_512_16")
    image_dir = Path(dataset_path) / "dso" / "cam0" / "images"


    def load_intrinsics(dataset_path, cam_key: str = "cam0"):
        camchain_file = yaml.safe_load((Path(dataset_path) / "dso" / "camchain.yaml").open())
        return np.array(camchain_file[cam_key]["intrinsics"]), np.array(
            camchain_file[cam_key]["distortion_coeffs"]
        )


    def enough_motion_for_keyframe(
        new_img: ImageData, last_kf_img: ImageData | None, kp_matcher, max_matches=60
    ) -> bool:
        # if last_kf_img is None:  # when processing first image
        #     return True

        pnp_min = 4  # PnP needs at least 4 matches to work
        _, matches = kp_matcher(last_kf_img, new_img)
        if pnp_min <= len(matches) < max_matches:
            return True

        return False


    # def enough_motion_for_keyframe(
    #     new_img: ImageData,
    #     last_kf_img: ImageData | None,
    #     kp_matcher,
    #     min_matches_for_pnp: int = 40,   # must have at least this many for add_view to succeed
    #     max_matches_for_new_kf: int = 120 # if still > this many, too similar → skip
    # ) -> bool:
    #     if last_kf_img is None:
    #         return True

    #     _, matches = kp_matcher(last_kf_img, new_img)
    #     n = len(matches)

    #     # Too similar (almost no motion) → skip, do not create new keyframe
    #     if n > max_matches_for_new_kf:
    #         return False

    #     # Enough motion + still enough matches for reliable PnP → new keyframe
    #     return n >= min_matches_for_pnp


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

    return dataset_path, enough_motion_for_keyframe, image_dir, load_intrinsics


@app.cell
def _(K, dist, image_dir):
    images = FeatureStore(image_dir, K.K(), dist, ext="png", max_num_images=60)
    return


@app.cell
def _():
    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    type(gtsam.Rot3.Ypr(0, 0, 0))
    type(gtsam.Rot3(np.eye(3)))
    gtsam.Pose3(gtsam.Rot3(np.eye(3)), gtsam.Point3(np.ones(3)))
    Point3(np.ones(3)) == Point3(np.ones(3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Monocular SLAM via iSAM2 on TUM-VI
    """)
    return


@app.cell
def _(Point2, dataset_path, enough_motion_for_keyframe, img0, load_intrinsics):
    def gtsam_cam_pose(img: ImageData) -> Pose3:
        return Pose3(Rot3(img.R), Point3(img.t.squeeze()))


    def gtsam_landmarks_from_pose(
        img: ImageData, tm: TrackManager, landmarks: PointCloud
    ) -> list[Point3]:
        """Get landmarks (3D pts) visible from cam pose in img0"""
        tids = tm.get_triangulated_view_tracks(img.idx)
        return tids, [Point3(pt) for pt in landmarks.get_points_as_array(tids)]


    def gtsam_keypoints_from_pose(img: ImageData, tm: TrackManager) -> list[Point3]:
        """Get keypoints (2D pts) visible from cam pose in img0

        Observations of landmarks (3D pts) in given image.
        Note: These keypoints were actually used in triangulation (not just detected).
        """
        kpkeys = tm.get_triangulated_view_keypoints(img0.idx)
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
                graph.add(
                    gtsam.GenericProjectionFactorCal3_S2(obs, obs_noise, X(img_idx), L(lm_tid), K)
                )
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
            assert (
                len(kp_keys) == 1
            )  # geometrically only 1 KP possible; True for symmetric view-to-view KP matches
            obs = img.kp[kp_keys[0][1]][:, None]

            graph.add(gtsam.GenericProjectionFactorCal3_S2(obs, obs_noise, X(img.idx), L(tid), K))
            if not initial_estimates.exists(L(tid)):
                initial_estimates.insert(L(tid), lm)
        initial_estimates.insert(X(img.idx), gtsam_cam_pose(img))


    def visual_ISAM2_tumvi_example(dataset_path, max_frames: int = 40):
        # plt.ion()

        # fx, fy, s, cx, cy
        k_vec, dist = load_intrinsics(dataset_path)
        fx, fy, cx, cy = k_vec
        K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

        # Observation noise (pixels) — start somewhat loose
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 2.0)

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
        # images = load_tumvi_images(dataset_path, max_frames)
        image_dir = Path(dataset_path) / "dso" / "cam0" / "images"
        images = FeatureStore(image_dir, K.K(), dist, ext="png", max_frames=max_frames)
        kp_matcher = make_keypoint_matcher(SfMConfig())

        num_recent_keyframes = 10
        keyframe_window = deque(maxlen=num_recent_keyframes)  # keep ~8–15 recent keyframes
        last_keyframe_idx = 0
        keyframe_window.append(images[last_keyframe_idx])

        for img in images:
            if img.idx == 0:
                continue
            print(f"Processing frame {img.idx}: {img.path}")

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
                compute_baseline_estimate(
                    keyframe_window[0], img, K.K(), track_manager, point_cloud, kp_matcher
                )
                # now have: first triang 3d points (landmarks) + keypoints for each image (cam pose)
                add_initial_factors_and_priors(
                    graph, initial_estimate, K, keyframe_window[0], img, track_manager, point_cloud
                )
                last_keyframe_idx = img.idx
                keyframe_window.append(img)
                continue

            # Normal keyframe: pick most overlapping recent keyframe
            # TODO: relax the condition, too picky atm!
            ref = pick_best_reference(img, keyframe_window, track_manager, K.K(), kp_matcher)
            if ref is None:
                print("No good reference found — skipping keyframe")
                continue
            # FIXME: fails due to not enough matches
            # _, matches = kp_matcher(img, ref)
            # print(f"{len(matches) = }")
            add_view(img, ref, K.K(), dist, track_manager, point_cloud, kp_matcher)

            # --- Now push everything to iSAM2 ---
            add_new_keyframe_factors(graph, initial_estimate, img, K, track_manager, point_cloud)
            isam.update(graph, initial_estimate)
            isam.update()  # optional extra iteration

            # bookkeeping
            last_keyframe_idx = img.idx

            # # Show current estimate
            # current_estimate = isam.calculateEstimate()
            # print(f"Frame {i} poses:")
            # for j in range(i + 1):
            #     if current_estimate.exists(X(j)):
            #         print(X(j), ":", current_estimate.atPose3(X(j)))
            # visual_ISAM2_plot(current_estimate, ax_lim=10, delay_sec=0.5)

            # Clear for next incremental step
            graph.resize(0)
            initial_estimate.clear()

        # plt.ioff()
        # plt.show()


    visual_ISAM2_tumvi_example(dataset_path, max_frames=120)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## TODO: Undistorting Fisheye
    """)
    return


@app.cell
def _(dataset_path, image_dir, load_intrinsics):
    def undistort_tumvi_image(
        img: np.ndarray, K_np: np.ndarray, D: np.ndarray, balance=0.0
    ) -> np.ndarray:
        """Proper undistortion for TUM-VI equidistant fisheye."""
        h, w = img.shape[:2]

        # Create the undistortion + rectification map once (or cache it)
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_np, D, (w, h), np.eye(3), balance=balance
        )

        map1, map2 = cv.fisheye.initUndistortRectifyMap(K_np, D, np.eye(3), new_K, (w, h), cv.CV_16SC2)

        undist = cv.remap(img, map1, map2, cv.INTER_LINEAR)
        return undist


    image_files = sorted(glob.glob(str(image_dir / "*.png")))
    k_vec, dist = load_intrinsics(dataset_path)
    fx, fy, cx, cy = k_vec
    # K = np.array([
    #     [fx, 0, cx],
    #     [0, fy, cy],
    #     [0, 0, 1],
    # ])
    K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    img = plt.imread(image_files[0])
    # img_undist = cv.fisheye.undistortImage(img, K.K(), dist)
    # img_undist = cv.undistort(img, K.K(), dist)


    fig, ax = plt.subplots(1, 2, figsize=(12, 10))
    ax[0].imshow(img, cmap="gray")
    img_undist = undistort_tumvi_image(img, K.K(), dist, balance=0.0)
    ax[1].imshow(img_undist, cmap="gray")
    plt.show()
    return K, dist


if __name__ == "__main__":
    app.run()
