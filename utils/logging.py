from pathlib import Path

import plotly.graph_objects as go

import wandb

# FIXME: only for type hints: circular import SfMConfig -> utils.__init__ -> .logging;
# from config import SfMConfig
from .tracks import TrackManager


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
