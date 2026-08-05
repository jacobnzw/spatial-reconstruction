from typing import Callable, Iterable

import cv2 as cv
import faiss
import networkx as nx
import numpy as np
from loguru import logger

from .embedding import ViewEmbedder
from .features import FeatureStore, KPMatches
from .view import ViewData


class ViewGraph:
    """
    Undirected weighted view graph with asymmetric match support.

    TODO: what are the responsiblities now?
    - finds the seed view pair
    - finds the next best view pair
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        matcher_fn: Callable[[ViewData, ViewData], KPMatches],
        min_inliers: int = 50,
        k: int = 5,
    ):
        self._feature_store = feature_store
        self._matcher_fn = matcher_fn
        self._min_inliers = min_inliers

        # Create vector DB index for image embeddings using FAISS
        self._embedder = ViewEmbedder()  # Initialize the embedding model
        self._vector_index = faiss.IndexFlatL2(self._embedder.embedding_dim)
        for view in feature_store.iter_views():
            view.embedding = self._embedder(view).astype(np.float32)
            # self._vector_index.add(embedding.astype(np.float32))
            self._vector_index.add_with_ids(view.embedding, np.array([view.idx], dtype=np.int64))

        # Build up candidate map from vector DB index: view.idx -> list of top K view.idx with similar embeddings
        # Store it in nx.Graph() as node attributes: each view has topk distances, indices
        self._graph = nx.Graph()
        for view in feature_store.iter_views():
            # Query the index for the top K nearest neighbors, k+1 because the first neighbor is the query view itself
            distances, indices = self._vector_index.search(view.embedding, k + 1)

            # Remove the self comparison
            distances = distances.flatten().tolist()[1:]
            indices = indices.flatten().tolist()[1:]

            self._graph.add_node(
                view.idx,
                indices=indices,
                distances=distances,
                registered=False,
            )

            # Edges for each candidate view index; matches will be calculated later
            for v, d in zip(indices, distances):
                self._graph.add_edge(
                    view.idx, v, distance=d, matches=None, n_inliers=None, match_validation_passed=None
                )

    @property
    def unregistered_views(self) -> Iterable[tuple[int, dict]]:
        return ((view_idx, topk) for view_idx, topk in self._graph.nodes.items() if not topk["registered"])

    @property
    def connecting_edges(self) -> Iterable[tuple[float, int, int]]:
        """Returns iterable over edges that connect registered and unregistered views."""
        # TODO: may need additional checks for match_validation_passed: skip anything where matching failed

        def _is_connecting_edge(u, v) -> bool:
            return (self._graph.nodes[u]["registered"] and not self._graph.nodes[v]["registered"]) or (
                not self._graph.nodes[u]["registered"] and self._graph.nodes[v]["registered"]
            )

        return ((d, u, v) for (u, v, d) in self._graph.edges.data("distance") if _is_connecting_edge(u, v))

    # TODO: DRY: merge the two find_* funcs? Keep in case debugging reveals design flaw! Merge when all checks out!
    def find_initial_view_pair(self) -> tuple[ViewData, ViewData, KPMatches] | None:
        most_similar_for_each_view = (
            (topk["distances"][0], view_idx, topk["indices"][0]) for view_idx, topk in self.unregistered_views
        )

        for dist, u, v in sorted(most_similar_for_each_view):
            view_u, view_v = self._feature_store[u], self._feature_store[v]
            validation_passed, n_inliers, matches = self._match_and_validate(view_u, view_v)

            self._graph.add_edge(u, v, matches=matches, n_inliers=n_inliers, match_validation_passed=validation_passed)

            # if match-validate fails, continue w/ next best candidate view pair
            if validation_passed:
                return view_u, view_v, matches

        # In this case, we can't even start the reconstruction
        logger.critical("Failed to find initial view pair!")
        return None

    def find_next_best_view_pair(self) -> tuple[ViewData, ViewData, KPMatches] | None:
        # Iterate from the most similar pair of views: edge connects registered and unregistered view
        for distance, u, v in sorted(self.connecting_edges):
            view_u, view_v = self._feature_store[u], self._feature_store[v]
            validation_passed, n_inliers, matches = self._match_and_validate(view_u, view_v)

            self._graph.add_edge(u, v, matches=matches, n_inliers=n_inliers, match_validation_passed=validation_passed)

            # if match-validate fails, continue w/ next best candidate view pair
            if validation_passed:
                return view_u, view_v, matches

        # TODO: Does this mean no more views available???
        # Either way we need to signal that, keeping in mind disconnected components.
        # Could be benign: we simply registered all images in graph component.
        # Could be failure: no similar images actually pass matching validation checks
        logger.critical("Failed to find next view pair!")
        return None

    def mark_views_registered(self, view_idxs: Iterable[int]):
        for vi in view_idxs:
            self._graph.nodes[vi]["registered"] = True

    def _match_and_validate(self, img_from: ViewData, img_to: ViewData) -> tuple[bool, int | None, KPMatches | None]:
        """Computes keypoint matches and performs geometric validation.

        Matches are validated geometrically by checking for existance of essential matrix.
        Uses supplied matcher_fn passed to ViewGraph.__init__().

        Args:
            img_from: Source image.
            img_to: Target image.

        Returns:
            Tuple of (flag, n_inliers, matches):
                - flag: Flag set to True if images overlap.
                - n_inliers: Number of inlier matches after geometric validation.
                - matches: Array of match indices (N, 2) where each row is (queryIdx, trainIdx).
        """

        _, matches = self._matcher_fn(img_from, img_to)
        if len(matches) < self._min_inliers:
            logger.debug(
                f"Not enough matches for views ({img_from.idx}, {img_to.idx}) {len(matches)} < {self._min_inliers}"
            )
            return False, None, None

        # geometric validation: rejects matches that cannot arise from a rigid 3D scene
        # [:, 0] = queryIdx; [:, 1] = trainIdx
        # TODO: repeated in bootstrap_from_two_views: save E, mask in ViewEdge?
        pts1, pts2 = img_from.kp[matches[:, 0]], img_to.kp[matches[:, 1]]  # ty:ignore[not-subscriptable]

        K = img_from.camera_model.get_camera_matrix()
        # NOTE: RANSAC sensitive to point shuffling, due to its randomness => slightly different inliers
        E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0)

        if E is None:
            logger.debug(f"Failed geometric match validation for views ({img_from.idx}, {img_to.idx})")
            return False, None, None

        n_inliers = int((mask > 0).sum())
        if n_inliers < self._min_inliers:
            return False, None, None

        return True, n_inliers, matches
