from pathlib import Path

import cv2 as cv
import numpy as np
import plotly.graph_objects as go
import rerun as rr
import rerun.blueprint as rrb

import wandb

from .features import FeatureStore, MatcherResult
from .pointcloud import PointCloud

# FIXME: only for type hints: circular import SfMConfig -> utils.__init__ -> .logging;
# from config import SfMConfig
from .tracks import TrackManager


class ReRunLogger:
    """Logs to given log_filepath *.rrd Rerun file for later inspection.

    For viewing use:
        rerun log_filepath
    """

    def __init__(self, log_filepath: str, point_cloud: PointCloud, images: FeatureStore, track_manager: TrackManager):
        self.point_cloud = point_cloud
        self.images = images
        self.track_manager = track_manager

        self.step = 0

        rr.init("structure_from_motion")
        rr.save(log_filepath)  # For streaming to viewer directly: rr.connect_grpc()

        blueprint = rrb.Vertical(
            rrb.Spatial3DView(
                name="3D",
                origin="/",
                line_grid=False,  # There's no clearly defined ground plane.
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(name="Camera", origin="/camera/image"),
                rrb.Spatial2DView(name="KP Matches (ref to new)", origin="/view/matches"),
                rrb.TimeSeriesView(name="Triangulation Pairs", origin="/view/pairs"),
                # rrb.TimeSeriesView()
            ),
            row_shares=[3, 2],
        )
        rr.send_blueprint(blueprint)

        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("camera", rr.ViewCoordinates.RDF)  # [x, y, z]  <==> [Right, Down, Forward]
        rr.log("view/pairs/ref", rr.SeriesPoints(colors=[255, 0, 0], names=["Ref View Index"]))
        rr.log("view/pairs/new", rr.SeriesPoints(colors=[0, 255, 0], names=["New View Index"]))

    def _set_step(self):
        rr.set_time("step", sequence=self.step)
        self.step += 1

    def log_all(self, matcher_result: MatcherResult, both_cameras=False):
        self._set_step()
        if both_cameras:
            self.log_camera(matcher_result.idx_from)
        self.log_camera(matcher_result.idx_to)
        self.log_point_cloud()
        self.log_matches(matcher_result)

    def log_camera(self, view_idx: int):
        # for view in self.images.iter_images_with_pose():
        view = self.images[view_idx]
        rr.log(
            "camera",
            rr.Transform3D(
                translation=view.t,
                rotation=rr.Quaternion(xyzw=view.cam_T_world.rotation.as_quat(scalar_first=False)),
                relation=rr.TransformRelation.ChildFromParent,  # child_T_parent <=> cam_T_world
            ),
        )

        height, width = view.camera_model.get_resolution(rescaled=True)
        rr.log(
            "camera/image",
            rr.Pinhole(
                image_from_camera=view.camera_model.get_camera_matrix(),
                resolution=(width, height),
            ),
        )
        rr.log("camera/image", rr.Image(view.pixels, color_model="RGB"))

        kp_indices = list(map(lambda x: x[1], self.track_manager.get_triangulated_view_kp_keys(view.idx)))
        rr.log("camera/image/keypoints", rr.Points2D(view.kp[kp_indices]))  # ty:ignore[not-subscriptable]

    def log_point_cloud(self):
        points = self.point_cloud.get_points_as_array()
        colors = self._get_point_colors()
        if points.size == 0:
            return

        rr.log("points", rr.Points3D(positions=points, colors=colors))

    def log_matches(self, matcher_result: MatcherResult):
        if matcher_result:
            img_from, img_to = self.images[matcher_result.idx_from], self.images[matcher_result.idx_to]
            img_matches = cv.drawMatches(
                cv.cvtColor(img_from.pixels, cv.COLOR_RGB2BGR),
                img_from.keypoints_as_opencv,
                cv.cvtColor(img_to.pixels, cv.COLOR_RGB2BGR),
                img_to.keypoints_as_opencv,
                matcher_result.matches_as_opencv,
                None,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )  # ty:ignore[no-matching-overload]
            rr.log("view/matches", rr.Image(img_matches, color_model="BGR"))

            # Log the new/ref image index pair on y-axis of the TimeSeriesView
            # x = new view index, y = reference view index.
            # FIXME: ref & new have the same color; distinguish red=ref, green=new
            rr.log("view/pairs/ref", rr.Scalars(matcher_result.idx_from))
            rr.log("view/pairs/new", rr.Scalars(matcher_result.idx_to))

    def _get_point_colors(self):
        """Returns an array of RGB colors for each 3D point."""
        colors = np.zeros((self.point_cloud.size, 3), dtype=np.uint8)
        for track_id, pt in self.point_cloud.items():
            kp_keys = self.track_manager.track_to_kps[track_id]
            # average the colors of all KPs in the track
            colors[track_id] = self.images.get_pixels(kp_keys).mean(axis=0)
        return colors


def build_track_length_histogram(track_lengths: list[int]):
    if not track_lengths:
        track_lengths = [0]

    fig = go.Figure(data=[go.Histogram(x=track_lengths)])
    fig.update_layout(
        title="Track Length Distribution",
        xaxis_title="Track length",
        yaxis_title="Count",
    )
    return fig


def log_wandb_artifacts(run, cfg, track_manager: TrackManager, view_table: wandb.Table, ba_summary: dict | None):
    # log model
    basename = cfg.out_basename
    model_name = cfg.loader.dataset
    model_path = Path(cfg.out_dir) / f"{basename}_ba.ply" if cfg.run_ba else cfg.out_dir / f"{basename}.ply"
    if model_path.exists():
        # TODO: use wandb.Object3D to view in W&B Visual Tab
        # wandb.Object3D.from_numpy
        # wandb.Object3D.from_file() # wandb.Object3D.SUPPORTED_TYPES: {'babylon', 'glb', 'gltf', 'obj', 'pts.json', 'stl'}
        # run.log({"reconstructed_point_cloud": wandb.Object3D(model_path, caption=f"SfM Reconstruction from {model_name}")})

        model_artifact = wandb.Artifact(name=f"{model_name}", type="model")
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)

    # log the status info about how each view was incorporated
    table_artifact = wandb.Artifact(
        name="view_table",
        type="table",
        description="Each row shows how each view was incorporated into the final 3D reconstruction.",
    )
    table_artifact.add(view_table, "view_table")
    run.log_artifact(table_artifact)

    # log camera parameters
    camera_model = cfg.loader.camera_model
    run.config.update(
        {
            "camera": {
                "model": camera_model.type,
                "intrinsics": camera_model.get_intrinsics(rescaled=True),
                "distortion_coeffs": camera_model.dist,
                "scale": camera_model.scale,
            }
        }
    )
    # back up calibration file
    run.save(cfg.loader.calib_file)

    # log track manager stats
    track_lengths = [len(kp_list) for kp_list in track_manager.track_to_kps.values()]
    histogram_fig = build_track_length_histogram(track_lengths)
    run.log({"track_length_histogram": wandb.Plotly(histogram_fig)})

    summary = {
        "number_of_tracks": len(track_manager.track_to_kps),
    }
    if ba_summary is not None:
        summary.update(ba_summary)
    run.summary.update(summary)
