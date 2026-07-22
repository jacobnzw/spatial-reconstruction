from pathlib import Path

import plotly.graph_objects as go

import wandb
from config import SfMConfig
from utils import TrackManager


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


def log_wandb_artifacts(run, cfg: SfMConfig, track_manager: TrackManager, ba_summary):

    # TODO: log interesting stats:
    # Reconstruction progress
    # number of registered views
    # number of matches and inlier ratios per image pair
    # Quality metrics
    # reprojection error before and after bundle adjustment
    # baseline length between views

    wandb.Graph.
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

    summary_update = {
        "number_of_tracks": len(track_manager.track_to_kps),
    }
    if ba_summary is not None:
        summary_update.update(
            {
                "initial_cost": ba_summary.initial_cost,
                "final_cost": ba_summary.final_cost,
                "minimizer_type": str(ba_summary.minimizer_type).split(".")[1],
                "termination_type": str(ba_summary.termination_type).split(".")[1],
            }
        )
    run.summary.update(summary_update)
