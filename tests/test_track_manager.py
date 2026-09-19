import numpy as np

from utils import TrackManager


def test_add_new_tracks():
    track_manager = TrackManager()
    # img_0:kp_0 and img_1:kp_0 are matched; same for img_0:kp_1 and img_1:kp_1
    track_manager.add_new_tracks([((0, 0), (1, 0)), ((0, 1), (1, 1))])

    # We should now have two tracks, each with two KPs
    assert track_manager.get_track((0, 0)) == 0
    assert track_manager.get_track((1, 0)) == 0
    assert track_manager.get_track((0, 1)) == 1
    assert track_manager.get_track((1, 1)) == 1

    assert track_manager.get_keypoints(0) == [(0, 0), (1, 0)]
    assert track_manager.get_keypoints(1) == [(0, 1), (1, 1)]

    assert track_manager.is_valid()


def test_add_keypoints_to_tracks():
    track_manager = TrackManager()
    # Tracks are sequences of (view_idx, kp_idx)
    track_manager.add_new_tracks([((0, 0), (1, 0)), ((0, 1), (1, 1))])

    track_manager.add_keypoints_to_tracks([(2, 4), (2, 7)], [0, 1])

    assert track_manager.get_keypoints(0) == [(0, 0), (1, 0), (2, 4)]
    assert track_manager.get_keypoints(1) == [(0, 1), (1, 1), (2, 7)]
    assert track_manager.is_valid()


def test_get_triangulated_view_kp_keys():
    track_manager = TrackManager()
    track_manager.add_new_tracks([((0, 0), (1, 0)), ((0, 1), (2, 1))])

    assert track_manager.get_triangulated_view_kp_keys(0) == [(0, 0), (0, 1)]
    assert track_manager.get_triangulated_view_kp_keys(1) == [(1, 0)]
    assert track_manager.get_triangulated_view_kp_keys(3) == []


def test_get_triangulated_view_tracks():
    track_manager = TrackManager()
    track_manager.add_new_tracks([((0, 0), (1, 0)), ((0, 1), (2, 1))])

    assert track_manager.get_triangulated_view_tracks(0) == [0, 1]
    assert track_manager.get_triangulated_view_tracks(1) == [0]
    assert track_manager.get_triangulated_view_tracks(3) == []


def test_get_track_observations_for_view():
    track_manager = TrackManager()
    track_manager.add_new_tracks([((0, 0), (1, 0))])

    tracked_ids, tracked_matches, untracked_matches = track_manager.get_track_observations_for_view(
        0,
        np.array([[0, 5], [1, 6], [2, 7]]),
    )

    assert tracked_ids.tolist() == [0]
    assert tracked_matches.tolist() == [[0, 5]]
    assert untracked_matches.tolist() == [[1, 6], [2, 7]]
