from dataclasses import dataclass
from typing import Callable

import cv2 as cv
from numpy.typing import NDArray

from .camera import NDArrayFloat, NDArrayInt
from .view import ViewData


@dataclass
class ViewEdge:
    i: int
    j: int
    inliers_ij: int  # matches i -> j
    inliers_ji: int  # matches j -> i

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

    def add_edge(self, i, j, inliers_ij, inliers_ji):
        """
        Add or update an undirected edge between image i and j.
        """
        if i == j:
            return

        edge = ViewEdge(i, j, inliers_ij, inliers_ji)

        self.edges.append(edge)

    def best_edge(self) -> ViewEdge | None:
        """
        Return the edge with maximum symmetric weight.
        """
        return max(self.edges, key=lambda e: e.weight, default=None)


def has_overlap(
    img_from: ViewData,
    img_to: ViewData,
    matcher_fn: Callable[[ViewData, ViewData], tuple[NDArrayFloat, NDArrayInt]] | None = None,
    min_inliers: int = 50,
) -> tuple[bool, int | None, NDArray | None]:
    """Returns True if there is sufficient overlap between two images.

    Args:
        img_from: Source image.
        img_to: Target image.
        K: Camera intrinsic matrix. If None, uses img_from.camera_model.K.
        matcher_fn: Keypoint matcher function. Required parameter.
        min_inliers: Minimum number of inliers to consider overlap sufficient.
    """
    if matcher_fn is None:
        raise ValueError("matcher_fn is required")

    K = img_from.camera_model.get_camera_matrix()

    _, good = matcher_fn(img_from, img_to)

    if len(good) < min_inliers:
        return False, None, None

    # geometric validation: rejects matches that cannot arise from a rigid 3D scene
    # [:, 0] = queryIdx; [:, 1] = trainIdx
    pts1, pts2 = img_from.kp[good[:, 0]], img_to.kp[good[:, 1]]  # ty:ignore[not-subscriptable]

    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0)

    if E is None:
        return False, None, None

    inliers = int((mask > 0).sum())
    if inliers < min_inliers:
        return False, None, None

    return True, inliers, good


def construct_view_graph(
    image_store: "FeatureStore",  # noqa: F821
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
            ok_ij, inliers_ij, _ = has_overlap(img_i, img_j, matcher_fn, min_inliers)
            ok_ji, inliers_ji, _ = has_overlap(img_j, img_i, matcher_fn, min_inliers)
            # ASK: why the matches should not be preserved ???
            if ok_ij and ok_ji:
                view_graph.add_edge(i, j, inliers_ij, inliers_ji)

    return view_graph
