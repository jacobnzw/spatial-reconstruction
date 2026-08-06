from typing import Callable, Iterable

import cv2 as cv
import faiss
import networkx as nx
import numpy as np
from loguru import logger

from .embedding import ViewEmbedder
from .features import FeatureStore, KPMatches
from .view import ViewData
from .camera import NDArrayInt


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
        self._feature_store: FeatureStore = feature_store
        self._matcher_fn = matcher_fn
        self._min_inliers = min_inliers
        self._k = max(k, feature_store.size)  # K >= feature_store.size

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
            distances, indices = self._vector_index.search(view.embedding, self._k + 1)

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
                    view.idx,
                    v,
                    distance=d,
                    matches=None,
                    n_inliers=None,
                    matches_ok=None,  # indicates if matches geometrically validated
                    registered=False,  # indicate if triangulation & localization succeeds
                    pnp_ok=None,  # indicate if PnP fails
                )
                # TODO: Merge matches_ok, registered, pnp_ok into RegistrationStatusEnum {None, matches_ok, pnp_ok}

    @property
    def unregistered_views(self) -> Iterable[tuple[int, dict]]:
        return ((view_idx, topk) for view_idx, topk in self._graph.nodes.items() if not topk["registered"])

    @property
    def unregistered_edges(self) -> Iterable[tuple[int, int, dict]]:
        return (
            (u, v, data)
            for (u, v, data) in self._graph.edges.data()
            if not data["registered"] and data["pnp_ok"] is None  # no registration attempt made yet
        )

    @property
    def connecting_edges(self) -> Iterable[tuple[int, int, dict]]:
        """Returns iterable over edges that connect registered and unregistered views."""
        # TODO: may need additional checks for match_validated: skip anything where matching/pnp failed

        def _is_connecting_edge(u, v) -> bool:
            return (self._graph.nodes[u]["registered"] and not self._graph.nodes[v]["registered"]) or (
                not self._graph.nodes[u]["registered"] and self._graph.nodes[v]["registered"]
            )

        return (
            (u, v, data)
            for (u, v, data) in self._graph.edges.data()
            if _is_connecting_edge(u, v) and data["pnp_ok"] is None
        )

    # TODO: DRY: merge the two find_* funcs? Keep in case debugging reveals design flaw! Merge when all checks out!
    # FIXME: Inefficient to re-sort edges on every call
    def find_initial_view_pair(self) -> tuple[ViewData, ViewData, NDArrayInt] | None:
        # Try to validate the most visually similar pair of views
        for u, v, edge_data in sorted(self.unregistered_edges, key=lambda e: e[-1]["distance"]):
            view_u, view_v = self._feature_store[u], self._feature_store[v]
            matches_ok, n_inliers, matches = self._match_and_validate(view_u, view_v)

            # Record matching results to graph edge; writes through to self._graph.edges
            edge_data["matches"] = matches
            edge_data["n_inliers"] = n_inliers
            edge_data["matches_ok"] = matches_ok

            # If match validation fails, continue w/ next best candidate view pair
            if matches_ok:
                return view_u, view_v, matches  # ty:ignore[invalid-return-type]

        # In this case, we can't even start the reconstruction
        logger.critical("Failed to find initial view pair!")
        return None

    def find_next_best_view_pair(self) -> tuple[ViewData, ViewData, NDArrayInt] | None:
        # Iterate from the most similar pair of views: edge connects registered and unregistered view
        for u, v, edge_data in sorted(self.connecting_edges, key=lambda e: e[-1]["distance"]):
            view_u, view_v = self._feature_store[u], self._feature_store[v]
            matches_ok, n_inliers, matches = self._match_and_validate(view_u, view_v)

            # Record matching results to graph edge; writes through to self._graph.edges
            edge_data["matches"] = matches
            edge_data["n_inliers"] = n_inliers
            edge_data["matches_ok"] = matches_ok

            # if match-validate fails, continue w/ next best candidate view pair
            if matches_ok:
                # TODO: Could this be a generator? Would likely not respect updated connecting_edges: => add filter for registered edges in the loop
                # BUT: even if we filter out registered edges, we might skip; with every registered edge more views become registered and .connecting_edges will change
                return view_u, view_v, matches  # ty:ignore[invalid-return-type]

        # Exhausting connecting edges means:
        # (a) all nodes (views) in the graph component are registered (good), or
        # (b) some nodes (views) failed registration (bad: report)
        # there might still be unregistered views (nodes) pertaining to another graph component
        logger.info("No more edges to register!")
        return None

    def mark_edge_registered(self, u, v):
        self._graph.edges[(u, v)]["registered"] = True
        self._graph.nodes[u]["registered"] = True
        self._graph.nodes[v]["registered"] = True

    def _match_and_validate(self, img_from: ViewData, img_to: ViewData) -> tuple[bool, int | None, NDArrayInt | None]:
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
