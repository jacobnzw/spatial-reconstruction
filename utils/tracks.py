import numpy as np

from .camera import NDArrayInt

# Type alias for keypoint observation (img_id, kp_idx)
KPKey = tuple[int, int]


class TrackManager:
    """
    Manages the mapping between 2D keypoint observations and 3D point tracks.

    A track represents a single 3D point observed across multiple images. Each track
    is identified by a unique track_id and contains a list of keypoint observations
    (KPKey = (img_id, kp_idx)) that correspond to the same 3D point.

    The class maintains a bidirectional mapping:
    - kp_to_track: maps each keypoint observation to its track_id; 1-to-1
    - track_to_kps: maps each track_id to all keypoint observations in that track; 1-to-N

    Note: This assumes symmetric keypoint matches (e.g., BFMatcher with crossCheck=True).
    If crossCheck=False, multiple keypoints in one image could match to the same keypoint
    in another image, which would violate the 1-to-1 constraint and is geometrically invalid.
    """

    def __init__(self):
        self.next_track_id = 0
        self.kp_to_track: dict[KPKey, int] = {}
        self.track_to_kps: dict[int, list[KPKey]] = {}

    def _register_keypoint_track(self, kp_key: KPKey, track_id: int):
        self.kp_to_track[kp_key] = track_id
        self.track_to_kps.setdefault(track_id, []).append(kp_key)

    def get_track(self, kp_key: KPKey) -> int | None:
        return self.kp_to_track.get(kp_key, None)

    def get_keypoints(self, track_id: int, img_idx: int | None = None) -> list[KPKey]:
        kp_keys = self.track_to_kps.get(track_id, [])
        if img_idx is not None:
            kp_keys = [kp_key for kp_key in kp_keys if kp_key[0] == img_idx]
        return kp_keys

    def get_triangulated_view_keypoints(self, image_idx: int) -> list[KPKey]:
        """Get keys of keypoints triangulated from a given view.
        image_idx: int Index of the camera view (image).
        """
        # return filter(lambda kpkey: kpkey[0] == image_idx, self.kp_to_track.keys())
        return [kpkey for kpkey in self.kp_to_track.keys() if kpkey[0] == image_idx]

    def get_triangulated_view_tracks(self, image_idx: int) -> list[int]:
        """Get track_ids of tracks triangulated from a given view."""
        # FIXME: not quite true; some of these were triangulated from other views: more like "tracks_in_view"
        kp_keys = self.get_triangulated_view_keypoints(image_idx)
        return [tid for kp_key in kp_keys if (tid := self.get_track(kp_key)) is not None]

    def add_new_tracks(self, kp_pairs: list[tuple[KPKey, KPKey]]):
        """Create new track for every given pair of KP observations."""
        added_track_ids = []
        for kp_key_pair in kp_pairs:  # kp_key = (img_idx, kp_idx)
            tid = self.next_track_id
            self._register_keypoint_track(kp_key_pair[0], tid)
            self._register_keypoint_track(kp_key_pair[1], tid)
            added_track_ids.append(tid)
            self.next_track_id += 1
        return added_track_ids

    def add_keypoints_to_tracks(self, kp_keys: list[KPKey], track_id: list[int]):
        """Add given KP observations to the given tracks."""
        for kp_key, tid in zip(kp_keys, track_id):
            self._register_keypoint_track(kp_key, tid)

    def get_track_observations_for_view(self, img_idx_ref: int, ref2new_matches: NDArrayInt):
        """Get track observations for a given view.

        Partitions supplied matches in `ref2new_matches` into:
          - tracked, which join KPs in reference view that have tracks to the KPs in the new view, and
          - untracked, which join KPs between the reference and the new view for which there are no tracks yet.

        Used for PnP pose estimation of the new view, where 2D-to-3D correspondences are needed.
        The 3D points are represented by `track_ids`, and the 2D points are represented by `tracked_kp_idxs_new`,
        which can be extracted from the returned `tracked_matches` by

        ```
        tracked_kp_idxs_new = tracked_matches[:, 1]
        ```

        Args:
            img_idx_ref: int Index of the reference image (view).
            ref2new_matches: NDArrayInt of shape (N, 2) containing matches between the reference image and the new image.

        Returns:
            track_ids: NDArray of shape (M,) containing track IDs visible from both the reference and the new view.
            These are the tracks that can be used for PnP pose estimation of the new view.
            tracked_matches: NDArray of shape (M, 2) containing matches of tracked keypoints between the reference
            and the new view. Used for PnP pose estimation of the new view.
            untracked_matches: NDArray of shape (K, 2) containing matches that do not correspond to any existing track
            (i.e., new tracks that can be triangulated from these matches).
        """
        track_ids, tracked_matches, untracked_matches = [], [], []
        for match in ref2new_matches:
            kp_idx_ref, _ = match
            kp_key_ref = (img_idx_ref, kp_idx_ref)
            if (track_id := self.get_track(kp_key_ref)) is not None:
                # tracked match indicates the same 3D world point is now observed by a new 2D KP
                track_ids.append(track_id)
                tracked_matches.append(match)
            else:  # reference view KP is untracked and therefore its matching KP from the new view is also untracked
                untracked_matches.append(match)

        return np.array(track_ids), np.array(tracked_matches), np.array(untracked_matches)

    def is_valid(self) -> bool:
        """Check consistency of the bi-directional map between track_id and KP_key.

        One track_id can map to multiple KPKeys, but one KPKey can only map to one track_id.
        Returns True if the mapping is consistent, False otherwise. Only True if KP matches are symmetrical
        (e.g. BFMatcher cross_check=True).
        """
        for track_id, kp_keys in self.track_to_kps.items():
            for kp_key in kp_keys:
                if self.kp_to_track[kp_key] != track_id:
                    return False
        return True
