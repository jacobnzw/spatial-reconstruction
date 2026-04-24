import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from collections import deque

    import cv2 as cv
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from config import SLAMConfig
    from utils import CameraModel, CameraType, FrameLoader, FeatureExtractor, TrackManager, ViewData, has_overlap, make_keypoint_matcher


@app.cell
def _():
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
            pnp_min = 40  # minimum points for reliable pose estimation using PnP
            emat_min = 10  # minimum points for reliable essential matrix estimation

            last_kf: ViewData = self._keyframe_window[-1]  # pick last keyframe
            # Time-based criteria:
            # Ensure keyframe generated every second (based on FPS),
            if frame.idx - last_kf.idx > self.fps:
                print(
                    f"DEBUG: New keyframe  → {frame.idx=} passed because time gap "
                    f"{frame.idx - last_kf.idx} > {self.fps} (FPS)."
                )
                return True
            # Frame considered only after certain time has passed since last keyframe (eg. 0.5 second)
            if frame.idx - last_kf.idx < self.fps / 2:
                print(
                    f"DEBUG: Not enough time elapsed for new keyframe  → {frame.idx=} failed because time gap "
                    f"{frame.idx - last_kf.idx} < {self.fps / 2} (half FPS)."
                )
                return False

            # Keypoint-based criteria:
            _, matches = self.matcher(last_kf, frame)
            track_ids, tracked_matches, untracked_matches = self.track_manager.get_track_observations_for_view(
                last_kf.idx, matches
            )
            n_matches = len(matches)
            n_tracked_matches = len(tracked_matches)
            n_untracked_matches = len(untracked_matches)
            # Need enough matches that correspond to existing tracks (for PnP pose estimation of new keyframe)
            estimation_cond = n_tracked_matches >= pnp_min and n_untracked_matches >= emat_min
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

            if estimation_cond and enough_new_kps:
                print(f"New keyframe  → {frame.idx=} passed with {len(matches)} matches to {last_kf.idx=}.")
                print(f"DEBUG: {n_tracked_matches=} (for PnP) and {n_untracked_matches=} (for new landmarks).")
                return True

            print(f"DEBUG: {estimation_cond=} {enough_new_kps=}")
            return False

        def keyframes(self):
            """Yields pairs of consecutive keyframes."""
            for idx, frame in enumerate(self.extractor.iter_frames_with_features()):
                # Add 1st frame as keyframe
                if idx == 0:
                    self._keyframe_window.append(frame)
                    continue

                if self._is_keyframe(frame):
                    self._keyframe_window.append(frame)
                    yield frame, self._keyframe_window[-2]

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

    return


@app.cell
def _():
    dataset_path = Path("data/tum/dataset-corridor4_512_16/")
    image_dir = Path(dataset_path) / "dso" / "cam0" / "images"
    cfg = SLAMConfig()

    # k_vec, dist_vec = load_intrinsics(dataset_path)
    # fx, fy, cx, cy = k_vec
    # K = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
    # camera_model = CameraModel(model_type=CameraType.FISHEYE, K=K.K(), dist=dist_vec)
    kp_matcher = make_keypoint_matcher(cfg)
    # streamer = make_keyframe_streamer(image_dir, cfg, camera_model, kp_matcher, track_manager, undistort=cfg.undistort)

    # Collect all keyframe pairs
    # keyframe_pairs = list(streamer.keyframes())
    return cfg, dataset_path, image_dir, kp_matcher


@app.cell
def _(keyframe_pairs):
    # Create slider to navigate through keyframes
    frame_slider = mo.ui.slider(
        start=0, stop=len(keyframe_pairs) - 1, step=1, value=0, show_value=True, label="Keyframe pair"
    )
    frame_slider
    return (frame_slider,)


@app.cell
def _(frame_slider, keyframe_pairs):
    current_kf, prev_kf = keyframe_pairs[frame_slider.value]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(current_kf.pixels, cmap="gray")
    ax[0].set_title(f"Current keyframe (idx={current_kf.idx})")
    ax[0].axis("off")

    ax[1].imshow(prev_kf.pixels, cmap="gray")
    ax[1].set_title(f"Previous keyframe (idx={prev_kf.idx})")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(keyframe_pairs, streamer):
    idx_diffs = [curr.idx - prev.idx for curr, prev in keyframe_pairs]
    plt.hist(idx_diffs, bins=100, log=True)
    plt.suptitle(f"{len(keyframe_pairs)} keyframes @ {streamer.max_matches} max_matches")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Visualize Matches: Distorted vs Undistorted
    """)
    return


@app.cell
def _():
    def draw_matches(img0, img1, matcher, K=None):
        kp0 = [cv.KeyPoint(x=p[0], y=p[1], size=1) for p in img0.kp]
        kp1 = [cv.KeyPoint(x=p[0], y=p[1], size=1) for p in img1.kp]

        dist, matches = matcher(img0, img1)
        print(f"Found {len(matches)} matches")
        if K is not None:
            _, mask = cv.findEssentialMat(img0.kp.T, img1.kp.T, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
            matches = matches[mask.ravel() > 0]
        print(f"{len(matches)} matches after geometric filtering")
    
        matches = [
            cv.DMatch(_queryIdx=qm, _trainIdx=tm, _imgIdx=0, _distance=0.0) for d, (qm, tm) in zip(dist, matches)
        ]
        img_matches = cv.drawMatches(
            img0.pixels, kp0, img1.pixels, kp1, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        img_matches = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

        return img_matches

    def load_intrinsics(dataset_path, cam_key: str = "cam0"):
        camchain_file = yaml.safe_load((Path(dataset_path) / "dso" / "camchain.yaml").open())
        return np.array(camchain_file[cam_key]["intrinsics"]), np.array(camchain_file[cam_key]["distortion_coeffs"])

    def undistort_tumvi_image(img: np.ndarray, K: np.ndarray, D: np.ndarray, balance=0.0) -> np.ndarray:
        """Proper undistortion for TUM-VI equidistant fisheye."""
        h, w = img.shape[:2]

        # Create the undistortion + rectification map once (or cache it)
        # new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
        new_K = K
        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv.CV_16SC2)

        undist = cv.remap(img, map1, map2, cv.INTER_LINEAR)
        return undist

    return draw_matches, load_intrinsics, undistort_tumvi_image


@app.cell
def _(
    cfg,
    dataset_path,
    draw_matches,
    image_dir,
    kp_matcher,
    load_intrinsics,
    undistort_tumvi_image,
):
    def load_image_pair(img0_idx, img1_idx, image_dir, cfg, camera_model):
        loader = FrameLoader(
            image_dir,
            max_size=cfg.max_size,
            max_frames=cfg.max_read_frames,
            ext="png",
            camera_model=camera_model,
            undistort=False,
        )
        extractor = FeatureExtractor(cfg, loader)

        img0, img1 = extractor(loader(0)), extractor(loader(1))

        return img0, img1


    # Undistort images: load camera matrix and distortion coefficients
    k_vec, dist_vec = load_intrinsics(dataset_path)
    fx, fy, cx, cy = k_vec
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    )

    # img0, img1 = keyframe_pairs[0]
    idx0, idx1 = 0, 231
    camera_model = CameraModel(CameraType.FISHEYE, K, dist_vec)
    img0, img1 = load_image_pair(idx0, idx1, image_dir, cfg, camera_model)
    img_matches = draw_matches(img0, img1, kp_matcher, K)


    from functools import partial

    undistortion_fn = partial(undistort_tumvi_image, K=K, D=dist_vec, balance=0.0)

    from copy import deepcopy

    img0_undist = deepcopy(img0)
    img0_undist.pixels = undistortion_fn(img0.pixels)
    img1_undist = deepcopy(img1)
    img1_undist.pixels = undistortion_fn(img1.pixels)

    img_matches_undist = draw_matches(img0_undist, img1_undist, kp_matcher, K)

    # Plot vertically stacked
    def plot_matches_compared():
        fig, ax = plt.subplots(2, 1, figsize=(14, 10))
        ax[0].axis("off")
        ax[0].imshow(img_matches)
        ax[1].axis("off")
        ax[1].imshow(img_matches_undist)
        plt.tight_layout()
        plt.show()

    plot_matches_compared()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
