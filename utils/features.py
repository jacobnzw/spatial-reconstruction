from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Literal

import cv2 as cv
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray

from .camera import NDArrayFloat, NDArrayInt
from .tracks import KPKey
from .view import FrameLoader, ViewData
from .embedding import ViewEmbedder

# Initialize device for Kornia/PyTorch operations
device = K.core.utils.get_cuda_or_mps_device_if_available()


FeatureType = Literal["sift", "disk"]


@dataclass
class FeatureExtractorConfig:
    type: FeatureType = "sift"
    """Feature extraction method: 'sift' or 'disk'"""

    num: int = 1_000
    """Maximum number of features to extract per image"""


MatcherType = Literal["bf", "lg"]


@dataclass
class MatcherConfig:
    type: MatcherType = "bf"
    """Keypoint matching method: 'bf' (brute-force) or 'lg' (lightglue)"""

    bf_lowe_ratio: float = 0.75
    """Lowe's ratio test threshold for BF matcher"""

    bf_cross_check: bool = True
    """Whether to use cross-checking for BF matcher"""

    lg_min_conf: float = 0.1
    """LightGlue matches with confidence below this threshold are filtered out"""


class FeatureExtractor:
    """Feature extractor class that curries the extraction function based on config."""

    def __init__(self, cfg: FeatureExtractorConfig, loader: FrameLoader, extract_embeddings=False):  # noqa: F821
        self.loader = loader
        self.num_features = cfg.num

        # NOTE: For now embedder dropped here as it is a kind of feature extractor, but it could be separated if needed
        # TODO: FeatureExtractor could accept any type that has .iter_frames() -> Iterable[ViewData] via typing Protocol
        if extract_embeddings:
            self._embedder = ViewEmbedder()  # Initialize the embedding model

        if cfg.type == "sift":
            sift = cv.SIFT_create(nfeatures=cfg.num)  # ty:ignore[unresolved-attribute]
            self._extract_fn = partial(self._extract_sift, sift=sift)
        elif cfg.type == "disk":
            disk_model = KF.DISK.from_pretrained("depth", device=device).eval()
            self._extract_fn = partial(self._extract_disk, disk_model=disk_model)
        else:
            raise ValueError(f"FeatureExtractor: Unknown feature type {cfg.type} in config!")

    def __call__(self, frame: ViewData) -> ViewData:
        """Extract features from a single frame using the curried extraction function."""
        frame = self._extract_fn(frame)

        if hasattr(self, "_embedder"):
            frame = self._embedder(frame)
            return frame

        return frame

    def _extract_sift(self, frame: ViewData, sift: cv.SIFT) -> ViewData:
        """Extract SIFT features from a single image."""
        img_arr = frame.pixels  # (H, W, C)

        # Ensure grayscale is uint8 for SIFT
        if len(img_arr.shape) == 3:  # RGB image, convert to grayscale
            gray = cv.cvtColor(img_arr, cv.COLOR_RGB2GRAY)
        else:
            gray = img_arr

        keypoints, descriptors = sift.detectAndCompute(gray.astype(np.uint8), mask=None)

        # Convert list[cv.KeyPoint] to NDArray (N, 2)
        frame.kp = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        frame.des = descriptors  # ty:ignore[invalid-assignment]

        return frame

    def _extract_disk(self, frame: ViewData, disk_model: torch.nn.Module) -> ViewData:
        """Extract DISK features from a single image."""
        img_arr = frame.pixels  # (H, W, C)

        # Convert to float32 in [0, 1]
        max_val = np.iinfo(img_arr.dtype).max if img_arr.dtype != np.float32 else 1.0
        img_float = img_arr.astype(np.float32) / max_val

        # Convert to tensor and add batch dimension (H, W, C) -> (1, C, H, W)
        img_tensor = K.image.image_to_tensor(img_float, keepdim=False).to(device=device)
        with torch.inference_mode():
            features = disk_model(img_tensor, self.num_features, pad_if_not_divisible=True)[0]

        frame.kp = features.keypoints.cpu().numpy()  # (N, 2)
        frame.des = features.descriptors.cpu().numpy()  # (N, D)

        # Memory management
        # del img_tensor, features
        # torch.cuda.empty_cache()
        return frame

    def iter_frames_with_features(self) -> Iterable[ViewData]:
        """Yields ImageData objects with extracted features."""
        for frame in self.loader.iter_frames():
            yield self(frame)


class FeatureStore:
    """Manages feature extraction and storage for multiple images.

    Handles loading images from a directory, extracting keypoints and descriptors
    using a provided FeatureExtractor, and storing the results. Camera intrinsics
    are stored in each ImageData object.

    Attributes:
        _store: List of ImageData objects containing extracted features for each image.
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        self._store: dict[int, ViewData] = {
            view_data.idx: view_data for view_data in feature_extractor.iter_frames_with_features()
        }
        logger.info(f"Loaded {len(self._store)} frames.")

    def get_pixels(self, kp_keys: list[KPKey]) -> NDArray[np.uint8]:
        """Get pixel color for a given keypoint in an image."""
        pixels = np.zeros((len(kp_keys), 3), dtype=np.uint8)
        for i, (img_idx, kp_idx) in enumerate(kp_keys):
            kp_uv = self._store[img_idx].kp[kp_idx].astype(np.uint16)  # ty:ignore[not-subscriptable]
            pixels[i] = self._store[img_idx].pixels[*np.flip(kp_uv)]  # flip (x, y) to (row, col) for indexing
        return pixels

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def timestamps(self) -> NDArrayInt:
        """Get frame timestamps in consecutive order."""
        return np.array([img.timestamp for img in self._store.values()])

    @property
    def image_indexes(self) -> NDArrayInt:
        """Get frame timestamps in consecutive order."""
        return np.array([idx for idx in self._store.keys()])

    def __getitem__(self, img_idx: int) -> ViewData:
        return self._store[img_idx]

    def get_keypoint(self, kp_key: KPKey) -> NDArrayFloat:
        """Get keypoint for a given keypoint key."""
        img_idx, kp_idx = kp_key
        return self._store[img_idx].kp[kp_idx]  # ty:ignore[not-subscriptable]

    def get_keypoints(self) -> list[NDArrayFloat]:
        """Get keypoints of all images."""
        return [img_data.kp for img_data in self._store.values()]  # ty:ignore[invalid-return-type]

    def get_descriptors(self) -> list[NDArrayFloat]:
        """Get descriptors of all images."""
        return [img_data.des for img_data in self._store.values()]  # ty:ignore[invalid-return-type]

    def iter_images_with_pose(self) -> Iterable[ViewData]:
        """Yield images for which we have a pose estimate."""
        yield from (img_data for img_data in self._store.values() if img_data.has_pose)


def _match_lightglue(
    img_from: ViewData,
    img_to: ViewData,
    lg_matcher: KF.LightGlueMatcher,
    min_conf: float | None = None,
) -> tuple[NDArrayFloat, NDArrayInt]:
    """Match descriptors using LightGlue matcher.

    Args:
        img_from: Query image.
        img_to: Train image.
        lg_matcher: kornia.feature.LightGlueMatcher object.
        min_conf: Minimum confidence threshold for match preservation.

    Returns:
        Tuple of (distances, matches):
            - distances: Array of match distances (N,).
            - matches: Array of match indices (N, 2) where each row is (queryIdx, trainIdx).
    """
    kp_from = torch.from_numpy(img_from.kp).to(device)
    des_from = torch.from_numpy(img_from.des).to(device)
    kp_to = torch.from_numpy(img_to.kp).to(device)
    des_to = torch.from_numpy(img_to.des).to(device)

    lafs_from = KF.laf_from_center_scale_ori(kp_from[None], torch.ones(1, len(kp_from), 1, 1, device=device))
    lafs_to = KF.laf_from_center_scale_ori(kp_to[None], torch.ones(1, len(kp_to), 1, 1, device=device))

    with torch.inference_mode():
        # LG matcher returns confidence scores instead of distances
        confs, idxs = lg_matcher(des_from, des_to, lafs_from, lafs_to)

        if min_conf:  # not None and > 0.0
            if min_conf < 0.0 or min_conf > 1.0:
                raise ValueError(f"min_conf must be in [0, 1], got {min_conf}")
            mask = (confs > min_conf).squeeze()
            confs, idxs = confs[mask], idxs[mask]

        # min_conf=0.0 is valid (retain all matches)
        return confs.detach().cpu().numpy(), idxs.detach().cpu().numpy()


def _match_brute_force(
    img_from: ViewData,
    img_to: ViewData,
    lowe_ratio: float | None = None,
    cross_check: bool = False,
) -> tuple[NDArrayFloat, NDArrayInt]:
    """Match descriptors using brute-force matcher with optional Lowe's ratio test.

    Args:
        img_from: Query image.
        img_to: Train image.
        lowe_ratio: Ratio threshold for Lowe's ratio test. If None, no filtering is applied.
        cross_check: If True, only keep matches that are mutual best matches.

    Returns:
        Tuple of (distances, matches):
            - distances: Array of match distances (N,).
            - matches: Array of match indices (N, 2) where each row is (queryIdx, trainIdx).

    Note:
        Using crossCheck=False means that multiple KPs in img_from can match to the same KP in img_to. Consequently,
        this might result in one KP in img_to triangulating to multiple 3D points.
        Using crossCheck=True would remove this ambiguity, but might also remove valid matches.
    """
    des_from, des_to = img_from.des, img_to.des
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=cross_check)
    if cross_check:
        matches = bf.match(des_from, des_to)  # ty:ignore[no-matching-overload]
    else:
        matches = bf.knnMatch(des_from, des_to, k=2)  # ty:ignore[no-matching-overload]
        if lowe_ratio:
            matches = [m for m, n in matches if m.distance < lowe_ratio * n.distance]
        else:
            matches = [m for m, n in matches]

    dist = np.array([m.distance for m in matches])  # (N,)
    return dist, np.array([(m.queryIdx, m.trainIdx) for m in matches])  # (N, 2)


def make_keypoint_matcher(
    cfg: MatcherConfig,
) -> Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]]:
    """Factory for keypoint matchers."""
    if cfg.type == "bf":
        return partial(_match_brute_force, lowe_ratio=cfg.bf_lowe_ratio, cross_check=cfg.bf_cross_check)
    if cfg.type == "lg":
        lightglue_matcher = KF.LightGlueMatcher("disk").eval().to(device)
        return partial(_match_lightglue, min_conf=cfg.lg_min_conf, lg_matcher=lightglue_matcher)
    else:
        ValueError(f"Unknown matcher type {cfg.type=} in config!")
