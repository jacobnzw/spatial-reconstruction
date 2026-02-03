from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from gsplat import rasterization
from gsplat.exporter import export_splats
from kornia.losses import ssim_loss

from utils import ReconIO


def train_gaussians(
    reconstruction_file: Path,
    num_iterations: int = 10_000,
):
    """Train Gaussian splat model from SfM reconstruction.

    Args:
        reconstruction_file: Path to .pt file saved by ReconIO.save_for_gsplat()
        num_iterations: Number of training iterations
    """

    # Load reconstruction data (poses, images, 3D points, colors, intrinsics)
    cam_poses, images_gt, points, point_colors, K, w, h = ReconIO.load_for_gsplat(reconstruction_file)

    print("\nData shapes:")
    print(f"  cam_poses: {cam_poses.shape}")
    print(f"  images_gt: {images_gt.shape}")
    print(f"  points: {points.shape}")
    print(f"  K: {K.shape}")
    print(f"  Image size: {w} x {h}")

    # 1. Initialize Gaussians from the reconstructed 3D points
    means = points.cuda().requires_grad_(True)  # (M, 3)
    colors = point_colors.cuda().requires_grad_(True)  # (M, 3)
    scales = torch.ones((len(means), 3), device="cuda") * 0.01  # Start with small scales
    scales.requires_grad_(True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").repeat(len(means), 1).requires_grad_(True)
    opacities = torch.ones(len(means), device="cuda").requires_grad_(True)  # (M,)

    # Move data to GPU
    # gsplat expects world-to-camera (viewmats)
    # solvePnP returns world-to-camera transformation (object-to-camera)
    # So we can use the poses directly without inversion
    viewmats = cam_poses.cuda()  # (N, 4, 4)
    images_gt = images_gt.cuda()
    K = K.cuda()  # Already tiled to (N, 3, 3)

    print("\nGaussian parameters:")
    print(f"  means: {means.shape}")
    print(f"  quats: {quats.shape}")
    print(f"  scales: {scales.shape}")
    print(f"  opacities: {opacities.shape}")
    print(f"  colors: {colors.shape}")
    print(f"  viewmats: {viewmats.shape}")

    # 2. Optimization loop
    optimizer = torch.optim.Adam([means, colors, scales, quats, opacities], lr=0.001)

    print(f"\nStarting training for {num_iterations} iterations...")
    for iteration in range(num_iterations):
        # Render from camera pose
        render_colors, render_alphas, meta = rasterization(
            means, quats, scales, opacities, colors, viewmats, K, width=w, height=h
        )

        if iteration == 0:
            print("\nFirst iteration render shapes:")
            print(f"  render_colors: {render_colors.shape}")
            print(f"  render_alphas: {render_alphas.shape}")
            print(f"  images_gt: {images_gt.shape}")

        # Compute loss (L1 + SSIM)
        l1_loss = F.l1_loss(render_colors, images_gt)
        ssim_value = ssim_loss(render_colors.permute(0, 3, 1, 2), images_gt.permute(0, 3, 1, 2), window_size=11)
        loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_value)

        # Backprop and update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # so we don't accumulate gradients across iterations

        if iteration % 100 == 0:
            print(
                f"Iteration {iteration}: loss={loss.item():.4f}, l1={l1_loss.item():.4f}, ssim={ssim_value.item():.4f}"
            )

    # 3. Convert RGB colors to Spherical Harmonics format
    # Formula: rgb = 0.5 + SH_C0 * sh_dc
    # So: sh_dc = (rgb - 0.5) / SH_C0
    SH_C0 = 0.28209479177387814  # Constant for 0th order spherical harmonic

    # sh0: DC component (base color), shape (N, 1, 3)
    sh0 = ((colors - 0.5) / SH_C0).unsqueeze(1)  # (M, 3) -> (M, 1, 3)

    # shN: Higher-order SH coefficients, shape (N, 0, 3) for degree 0 (no view-dependent effects)
    shN = torch.zeros((len(colors), 0, 3), device=colors.device)  # (M, 0, 3)

    print("\nExporting Gaussian splats...")
    print(f"  sh0 shape: {sh0.shape}")
    print(f"  shN shape: {shN.shape}")

    # Export trained Gaussians to PLY format
    output_path = reconstruction_file.parent / f"{reconstruction_file.stem}_gsplat.ply"
    export_splats(
        means.detach(),
        scales.detach(),
        quats.detach(),
        opacities.detach(),
        sh0.detach(),
        shN.detach(),
        format="ply",
        save_to=str(output_path),
    )
    print(f"Saved Gaussian splats to: {output_path}")


if __name__ == "__main__":
    tyro.cli(train_gaussians)
