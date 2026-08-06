from typing import Callable, Iterable

import cv2 as cv
import faiss
import networkx as nx
import numpy as np
from loguru import logger

from .camera import NDArrayInt
from .embedding import ViewEmbedder
from .features import FeatureStore, KPMatches, KeypointMatcher, MatcherResult
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
        kp_matcher: KeypointMatcher,
        k: int = 5,
    ):
        self._feature_store: FeatureStore = feature_store
        self._kp_matcher = kp_matcher
        self._k = min(k, feature_store.size)  # K <= feature_store.size

        # Create vector DB index for image embeddings using FAISS
        self._embedder = ViewEmbedder()  # Initialize the embedding model
        self._vector_index = faiss.IndexIDMap(faiss.IndexFlatL2(self._embedder.embedding_dim))
        for view in feature_store.iter_views():
            # TODO: store emdedding as view graph node data to map view.idx to embedding if faiss won't let me add_with_ids
            # TODO: batch add is faster, but too big a batch eats into RAM
            logger.info(f"Embedding {view.idx = } ...")
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

        return (
            (u, v, data)
            for (u, v, data) in self._graph.edges.data()
            if self._is_connecting_edge(u, v) and data["pnp_ok"] is None
        )

    def _is_connecting_edge(self, u, v) -> bool:
        # ==1 : Either one is True
        return (self._graph.nodes[u]["registered"] + self._graph.nodes[v]["registered"]) == 1

    def _new_ref_order_views(self, u: int, v: int) -> tuple[ViewData, ViewData]:
        assert self._is_connecting_edge(u, v), f"Edge ({u}, {v}) is not a connecting edge!"
        if self._graph.nodes[u]["registered"]:
            (u, v) = (v, u)
        # The registered view is the reference view: comes second
        return self._feature_store[u], self._feature_store[v]

    # TODO: DRY: merge the two find_* funcs? Keep in case debugging reveals design flaw! Merge when all checks out!
    # FIXME: Inefficient to re-sort edges on every call
    def find_initial_view_pair(self) -> tuple[ViewData, ViewData, MatcherResult] | None:
        # Try to validate the most visually similar pair of views
        for u, v, edge_data in sorted(self.unregistered_edges, key=lambda e: e[-1]["distance"]):
            view_u, view_v = self._feature_store[u], self._feature_store[v]
            matcher_result = self._kp_matcher(view_u, view_v)

            # Record matching results to graph edge; writes through to self._graph.edges
            edge_data["matches"] = matcher_result.matches
            edge_data["n_inliers"] = matcher_result.n_inliers
            edge_data["matches_ok"] = matcher_result.success

            # If match validation fails, continue w/ next best candidate view pair
            if matcher_result:
                return view_u, view_v, matcher_result

        # In this case, we can't even start the reconstruction
        logger.critical("Failed to find initial view pair!")
        return None

    def find_next_best_view_pair(self) -> tuple[ViewData, ViewData, MatcherResult] | None:
        # Iterate from the most similar pair of views: edge connects registered and unregistered view
        for u, v, edge_data in sorted(self.connecting_edges, key=lambda e: e[-1]["distance"]):
            view_u, view_v = self._new_ref_order_views(u, v)
            # NOTE: (!) ref -> new is the assumed match direction in add_view()
            matcher_result = self._kp_matcher(view_v, view_u)

            # Record matching results to graph edge; writes through to self._graph.edges
            edge_data["matches"] = matcher_result.matches
            edge_data["n_inliers"] = matcher_result.n_inliers
            edge_data["matches_ok"] = matcher_result.success

            # if match-validate fails, continue w/ next best candidate view pair
            if matcher_result:
                # TODO: Could this be a generator? Would likely not respect updated connecting_edges: => add filter for registered edges in the loop
                # BUT: even if we filter out registered edges, we might skip; with every registered edge more views become registered and .connecting_edges will change
                return view_u, view_v, matcher_result

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

    def mark_edge_failed(self, new_idx, ref_idx):
        self._graph.edges[(new_idx, ref_idx)]["pnp_ok"] = False
        self._graph.edges[(new_idx, ref_idx)]["registered"] = False
        self._graph.nodes[new_idx]["registered"] = False
