import numpy as np

import utils.graph as graph_module
from utils import MatcherResult, ViewGraph


class FakeView:
    def __init__(self, idx: int):
        self.idx = idx
        self.embedding = None


class FakeFeatureStore:
    def __init__(self, views: list[FakeView]):
        self._views = {view.idx: view for view in views}

    @property
    def size(self) -> int:
        return len(self._views)

    def iter_views(self):
        return self._views.values()

    def __getitem__(self, view_idx: int) -> FakeView:
        return self._views[view_idx]


class FakeEmbedder:
    embedding_dim = 2

    def __init__(self, embeddings: dict[int, np.ndarray]):
        self.embeddings = embeddings

    def __call__(self, view: FakeView) -> np.ndarray:
        return self.embeddings[view.idx][None, :].astype(np.float32)


class FakeMatcher:
    """Returns pre-given match results and records call arguments."""

    def __init__(self, results: list[MatcherResult]):
        self.results = iter(results)
        self.calls = []

    def __call__(self, view_from: FakeView, view_to: FakeView) -> MatcherResult:
        self.calls.append((view_from.idx, view_to.idx))
        return next(self.results)


def make_view_graph(monkeypatch, matcher):
    views = [FakeView(idx) for idx in range(3)]
    feature_store = FakeFeatureStore(views)
    embeddings = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([3.0, 0.0]),
    }
    # Use FakeEmbedder as we don't need the real one to test ViewGraph
    monkeypatch.setattr(graph_module, "ViewEmbedder", lambda: FakeEmbedder(embeddings))
    return ViewGraph(feature_store, matcher, k=1), feature_store


def successful_match_result(idx_from: int, idx_to: int) -> MatcherResult:
    return MatcherResult(
        idx_from=idx_from,
        idx_to=idx_to,
        scores=np.array([0.9], dtype=np.float32),
        matches=np.array([[0, 1]], dtype=np.int64),
        inlier_mask=np.array([1], dtype=np.uint8),
    )


def failed_match_result(idx_from: int, idx_to: int) -> MatcherResult:
    return MatcherResult(idx_from=idx_from, idx_to=idx_to)


def test_builds_candidate_graph(monkeypatch):
    view_graph, feature_store = make_view_graph(monkeypatch, FakeMatcher([]))

    assert [view.idx for view in feature_store.iter_views()] == [0, 1, 2]
    assert set(view_graph._graph.nodes) == {0, 1, 2}
    assert set(view_graph._graph.edges) == {(0, 1), (1, 2)}
    assert all(data["registered"] is False for _, data in view_graph._graph.nodes.data())
    assert all(data["matches"] is None for _, _, data in view_graph._graph.edges.data())


def test_unregistered_and_connecting_edges_follow_registration_state(monkeypatch):
    view_graph, _ = make_view_graph(monkeypatch, FakeMatcher([]))

    assert {view_idx for view_idx, _ in view_graph.unregistered_views} == {0, 1, 2}
    assert {(u, v) for u, v, _ in view_graph.unregistered_edges} == {(0, 1), (1, 2)}
    assert list(view_graph.connecting_edges) == []

    view_graph.mark_edge_registered(0, 1)

    assert {view_idx for view_idx, _ in view_graph.unregistered_views} == {2}
    assert {(u, v) for u, v, _ in view_graph.unregistered_edges} == {(1, 2)}
    assert {(u, v) for u, v, _ in view_graph.connecting_edges} == {(1, 2)}


def test_new_ref_order_returns_unregistered_then_registered_view(monkeypatch):
    view_graph, _ = make_view_graph(monkeypatch, FakeMatcher([]))
    view_graph.mark_edge_registered(0, 1)

    new_view, reference_view = view_graph._new_ref_order_views(1, 2)

    assert new_view.idx == 2
    assert reference_view.idx == 1


def test_find_initial_view_pair_skips_failed_match(monkeypatch):
    matcher = FakeMatcher([failed_match_result(0, 1), successful_match_result(1, 2)])
    view_graph, _ = make_view_graph(monkeypatch, matcher)

    result = view_graph.find_initial_view_pair()

    assert result is not None
    view_from, view_to, match_result = result

    assert (view_from.idx, view_to.idx) == (1, 2)
    assert match_result.success
    assert matcher.calls == [(0, 1), (1, 2)], "ViewGraph called KPMatcher w/ wrong inputs."
    assert view_graph._graph.edges[(0, 1)]["matches_ok"] is False
    assert view_graph._graph.edges[(1, 2)]["matches_ok"] is True


def test_find_next_best_view_pair_uses_registered_view_as_reference(monkeypatch):
    matcher = FakeMatcher([successful_match_result(2, 1)])
    view_graph, _ = make_view_graph(monkeypatch, matcher)
    view_graph.mark_edge_registered(0, 1)

    result = view_graph.find_next_best_view_pair()

    assert result is not None
    new_view, reference_view, match_result = result

    assert (new_view.idx, reference_view.idx) == (2, 1)
    assert match_result.success
    assert matcher.calls == [(1, 2)], "ViewGraph called KPMatcher w/ wrong inputs."


def test_mark_edge_failed_excludes_edge_from_connecting_edges(monkeypatch):
    view_graph, _ = make_view_graph(monkeypatch, FakeMatcher([]))
    view_graph.mark_edge_registered(0, 1)
    view_graph.mark_edge_failed(2, 1)

    edge_data = view_graph._graph.edges[(1, 2)]
    assert edge_data["pnp_ok"] is False
    assert edge_data["registered"] is False
    assert list(view_graph.connecting_edges) == []
