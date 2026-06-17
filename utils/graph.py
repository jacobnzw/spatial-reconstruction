from dataclasses import dataclass
from typing import Callable

import cv2 as cv
from loguru import logger
from numpy.typing import NDArray

from .camera import NDArrayFloat, NDArrayInt
from .features import FeatureStore
from .view import ViewData


@dataclass
class ViewEdge:
    i: int
    j: int
    inliers_ij: int  # matches i -> j
    inliers_ji: int  # matches j -> i
    matches_ij: NDArrayInt  # matches i -> j
    matches_ji: NDArrayInt  # matches j -> i

    @property
    def weight(self) -> int:
        # symmetric weight used for ranking
        # FIXME: # matches maximized when images identical, so this won't result in good baseline for triangulation
        # need large-enough relative translation for good baseline + enough plausible matches after geometric verification
        # see has_overlap()
        return min(self.inliers_ij, self.inliers_ji)


class ViewGraph:
    """
    Undirected weighted view graph with asymmetric match support.
    """

    def __init__(self):
        self.edges = []  # list of ViewEdge (global access)

    def add_edge(self, i, j, inliers_ij, inliers_ji, matches_ij, matches_ji):
        """
        Add or update an undirected edge between image i and j.
        """
        if i == j:
            return

        edge = ViewEdge(i, j, inliers_ij, inliers_ji, matches_ij, matches_ji)
        self.edges.append(edge)

    def best_edge(self) -> ViewEdge | None:
        """
        Return the edge with maximum symmetric weight.
        """
        return max(self.edges, key=lambda e: e.weight, default=None)


def construct_view_graph(
    image_store: FeatureStore,  # noqa: F821
    matcher_fn: Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]],
    min_inliers: int = 50,
):
    view_graph = ViewGraph()
    kp, des = image_store.get_keypoints(), image_store.get_descriptors()
    assert len(kp) == len(des)

    N = len(kp)
    for i in range(N):
        for j in range(i + 1, N):
            img_i, img_j = image_store[i], image_store[j]
            # TODO: 1 direction enough, when matches symmetrical (e.g. crossCheck=True)
            ij_overlap, inliers_ij, matches_ij = has_overlap(img_i, img_j, matcher_fn, min_inliers)
            ji_overlap, inliers_ji, matches_ji = has_overlap(img_j, img_i, matcher_fn, min_inliers)

            matches_ij_shape = matches_ij.shape if matches_ij is not None else None
            matches_ji_shape = matches_ji.shape if matches_ji is not None else None
            logger.debug(f"Matcher result for images {i} -> {j}: {ij_overlap=} {matches_ij_shape=} {inliers_ij=}")
            logger.debug(f"Matcher result for images {j} -> {i}: {ji_overlap=} {matches_ji_shape=} {inliers_ji=}")

            # ASK: why the matches should not be preserved ???
            if ij_overlap and ji_overlap:
                view_graph.add_edge(i, j, inliers_ij, inliers_ji, matches_ij, matches_ji)
                logger.debug(f"Added ViewEdge {i}<->{j}: {inliers_ij=} {inliers_ji=}")

    return view_graph


def has_overlap(
    img_from: ViewData,
    img_to: ViewData,
    matcher_fn: Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]] | None = None,
    min_inliers: int = 50,
) -> tuple[bool, int | None, NDArray | None]:
    """Returns True if there is sufficient overlap between two images.

    Matches are validated geometrically by checking for existance of essential matrix.

    Args:
        img_from: Source image.
        img_to: Target image.
        matcher_fn: Keypoint matcher function. Required parameter.
        min_inliers: Minimum number of inliers to consider overlap sufficient.

    Returns:
        Tuple of (flag, n_inliers, matches):
            - flag: Flag set to True if images overlap.
            - n_inliers: Number of inlier matches after geometric validation.
            - matches: Array of match indices (N, 2) where each row is (queryIdx, trainIdx).
    """
    if matcher_fn is None:
        raise ValueError("matcher_fn is required")

    _, matches = matcher_fn(img_from, img_to)
    if len(matches) < min_inliers:
        return False, None, None

    # geometric validation: rejects matches that cannot arise from a rigid 3D scene
    # [:, 0] = queryIdx; [:, 1] = trainIdx
    # TODO: repeated in bootstrap_from_two_views: save E, mask in ViewEdge?
    pts1, pts2 = img_from.kp[matches[:, 0]], img_to.kp[matches[:, 1]]  # ty:ignore[not-subscriptable]

    K = img_from.camera_model.get_camera_matrix()
    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0)

    if E is None:
        return False, None, None

    n_inliers = int((mask > 0).sum())
    if n_inliers < min_inliers:
        return False, None, None

    return True, n_inliers, matches
