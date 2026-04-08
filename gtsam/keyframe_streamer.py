import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    import marimo as mo
    from utils import FeatureExtractor, make_keypoint_matcher, ImageData, has_overlap
    from config import SLAMConfig
    from collections import deque
    import matplotlib.pyplot as plt
    import cv2 as cv
    import numpy as np
    import yaml


@app.cell
def _(TrackManager, kp_matcher, self):
    class KeyframeStreamer:
        def __init__(
            self,
            image_dir: Path,
            feature_extractor: FeatureExtractor,
            matcher,
            ext="png",
            max_matches=120,
            max_frames=10,
            max_read_frames=None,
        ):
            self.image_dir = sorted(list(Path(image_dir).glob(f"*.{ext}")))
            self.extractor = feature_extractor
            self.matcher = matcher
            self.max_matches = max_matches
            self.max_frames = max_frames
            self.max_read_frames = max_read_frames
            self._frame_window = deque(maxlen=max_frames)

        def _enough_motion_for_keyframe(self, frame: ImageData) -> bool:
            pnp_min = 4  # PnP needs at least 4 matches to work

            last_kf = self._frame_window[-1]  # pick last keyframe
            _, matches = self.matcher(last_kf, frame)

            if pnp_min <= len(matches) < self.max_matches:
                return True

            return False

        def keyframes(self):
            """Yields pairs of consecutive keyframes."""
            for idx, filepath in enumerate(self.image_dir):
                if self.max_read_frames and idx > self.max_read_frames:
                    print(f"Reached {self.max_read_frames=}, stopping further loading.")
                    break

                kp, des, img = self.extractor(filepath)
                frame = ImageData(idx, filepath, img, kp, des)
            
                # Add 1st frame as keyframe
                if idx == 0:
                    self._frame_window.append(frame)
                    continue

                # yield only frames that are sufficiently different from the last keyframe
                if self._enough_motion_for_keyframe(frame):
                    self._frame_window.append(frame)
                    yield frame, self._frame_window[-2]

        def find_reference_frame(
            frame: ImageData,
            track_manager: TrackManager,
            K: np.array,
            min_inliers: int = 30,
        ) -> ImageData | None:
            """Pick the reference image that gives the most reliable 3D-2D correspondences."""
            best_ref = None
            best_score = -1
    
            for ref in self._frame_window:
                # Use your existing has_overlap (geometric validation + E-matrix inliers)
                is_overlapping, inliers, matches = has_overlap(ref, frame, K, kp_matcher, min_inliers)
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
            if best_ref is None and len(self._frame_window) > 0:
                best_ref = self._frame_window[-1]  # most recent is usually the safest geometrically
    
            return best_ref

    return (KeyframeStreamer,)


@app.cell
def _(KeyframeStreamer):
    dataset_path = Path("data/tum/dataset-corridor4_512_16/")
    image_dir = Path(dataset_path) / "dso" / "cam0" / "images"
    cfg = SLAMConfig()

    extractor = FeatureExtractor(cfg, image_dir, ext="png")
    matcher = make_keypoint_matcher(cfg)
    streamer = KeyframeStreamer(image_dir, extractor, matcher, max_read_frames=200)

    # Collect all keyframe pairs
    keyframe_pairs = list(streamer.keyframes())
    return dataset_path, keyframe_pairs, matcher, streamer


@app.cell
def _(keyframe_pairs):
    # Create slider to navigate through keyframes
    frame_slider = mo.ui.slider(
        start=0, 
        stop=len(keyframe_pairs) - 1,
        step=1,
        value=0,
        show_value=True,
        label="Keyframe pair"
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
def _(matcher):
    def draw_matches(img0, img1):
        kp0 = [cv.KeyPoint(x=p[0], y=p[1], size=1) for p in img0.kp]
        kp1 = [cv.KeyPoint(x=p[0], y=p[1], size=1) for p in img1.kp]
    
        dist, matches = matcher(img0, img1)
        matches = [cv.DMatch(_queryIdx=qm, _trainIdx=tm, _imgIdx=0, _distance=0.0) for d, (qm, tm) in zip(dist, matches)]
        img_matches = cv.drawMatches(
            img0.pixels, kp0,
            img1.pixels, kp1,
            matches, None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        img_matches = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)

        return img_matches

    def load_intrinsics(dataset_path, cam_key: str = "cam0"):
        camchain_file = yaml.safe_load((Path(dataset_path) / "dso" / "camchain.yaml").open())
        return np.array(camchain_file[cam_key]["intrinsics"]), np.array(
            camchain_file[cam_key]["distortion_coeffs"]
        )

    def undistort_tumvi_image(
        img: np.ndarray, K: np.ndarray, D: np.ndarray, balance=0.0
    ) -> np.ndarray:
        """Proper undistortion for TUM-VI equidistant fisheye."""
        h, w = img.shape[:2]

        # Create the undistortion + rectification map once (or cache it)
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=balance
        )

        map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv.CV_16SC2)

        undist = cv.remap(img, map1, map2, cv.INTER_LINEAR)
        return undist

    return draw_matches, load_intrinsics, undistort_tumvi_image


@app.cell
def _(
    dataset_path,
    draw_matches,
    keyframe_pairs,
    load_intrinsics,
    undistort_tumvi_image,
):

    img0, img1 = keyframe_pairs[0]
    img_matches = draw_matches(img0, img1)

    # Undistort images: load camera matrix and distortion coefficients
    k_vec, dist_vec = load_intrinsics(dataset_path)
    fx, fy, cx, cy = k_vec
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    from functools import partial
    undistortion_fn = partial(undistort_tumvi_image, K=K, D=dist_vec, balance=0.0)

    from copy import deepcopy
    img0_undist = deepcopy(img0)
    img0_undist.pixels = undistortion_fn(img0.pixels)
    img1_undist = deepcopy(img1)
    img1_undist.pixels = undistortion_fn(img1.pixels)

    img_matches_undist = draw_matches(img0_undist, img1_undist)

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
