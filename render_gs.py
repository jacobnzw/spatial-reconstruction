from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from gsplat import rasterization
from gsplat.exporter import export_splats
from gsplat.strategy import DefaultStrategy
from kornia.losses import ssim_loss

from utils import ReconIO

device = torch.device("cuda")


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    SH_C0 = 0.28209479177387814
    return (rgb - 0.5) / SH_C0


def get_scale_init(means: torch.Tensor, init_scale: float = 0.3) -> torch.Tensor:  # lowered
    means_cpu = means.detach().cpu()
    dists = torch.cdist(means_cpu[None], means_cpu[None]).squeeze(0)
    dists.fill_diagonal_(float("inf"))
    knn_dists = torch.topk(dists, 4, dim=1, largest=False)[0]
    dist2_avg = (knn_dists[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg + 1e-8)
    return torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3).to(means.device)


def train_gaussians(
    reconstruction_file: Path,
    num_iterations: int = 25_000,  # fewer iters with aggressive pruning
    sh_degree: int = 3,
    init_scale: float = 0.3,
    batch_size: int = 2,
):
    cam_poses, images_gt, points, point_colors, K, w, h = ReconIO.load_for_gsplat(reconstruction_file, device)
    N = len(points)

    means = torch.nn.Parameter(torch.as_tensor(points, dtype=torch.float32, device=device))
    scales = torch.nn.Parameter(get_scale_init(means, init_scale))
    quats = torch.nn.Parameter(F.normalize(torch.rand((N, 4), device=device), dim=-1))
    opacities = torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=device)))

    sh_dim = (sh_degree + 1) ** 2
    shs_init = torch.zeros((N, sh_dim, 3), device=device)
    shs_init[:, 0, :] = rgb_to_sh(torch.as_tensor(point_colors, dtype=torch.float32, device=device))
    sh0 = torch.nn.Parameter(shs_init[:, :1, :].clone())
    shN = torch.nn.Parameter(shs_init[:, 1:, :].clone()) if sh_degree > 0 else torch.zeros((N, 0, 3), device=device)

    params = {"means": means, "scales": scales, "quats": quats, "opacities": opacities, "sh0": sh0}
    if sh_degree > 0:
        params["shN"] = shN

    optimizers = {
        "means": torch.optim.Adam([means], lr=1.6e-4),
        "scales": torch.optim.Adam([scales], lr=5e-3),
        "quats": torch.optim.Adam([quats], lr=1e-3),
        "opacities": torch.optim.Adam([opacities], lr=5e-2),
        "sh0": torch.optim.Adam([sh0], lr=2.5e-3),
    }
    if sh_degree > 0:
        optimizers["shN"] = torch.optim.Adam([shN], lr=2.5e-3 / 20)

    # === Scene scale for better pruning ===
    scene_scale = (means.max(dim=0)[0] - means.min(dim=0)[0]).norm().item()
    print(f"Estimated scene_scale: {scene_scale:.4f}")

    strategy = DefaultStrategy(
        refine_start_iter=1000,  # start later → less early blow-up
        refine_stop_iter=15000,
        refine_every=100,
        reset_every=3000,
        grow_grad2d=0.001,  # higher → much fewer new Gaussians
        prune_opa=0.03,  # prune low-opacity more aggressively
        prune_scale3d=0.015,  # very strict: prune anything > ~0.26 m (scene_scale~17.5)
        grow_scale3d=0.005,  # duplicate only tiny Gaussians
        prune_scale2d=0.08,
        verbose=True,
    )
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)  # ← important!

    viewmats = cam_poses.to(device)
    images_gt = images_gt.to(device)
    K = K.to(device)

    print(f"\nStarting training with DefaultStrategy (packed) — {num_iterations} iters")
    for it in range(num_iterations):
        idx = torch.randint(0, len(viewmats), (batch_size,))
        vm = viewmats[idx]
        Ki = K[idx]
        gt = images_gt[idx]

        scales_r = torch.exp(scales)
        opac_r = torch.sigmoid(opacities)
        shs = torch.cat([sh0, shN], dim=1) if sh_degree > 0 else sh0

        render_colors, _, info = rasterization(
            means, quats, scales_r, opac_r, shs, vm, Ki, width=w, height=h, sh_degree=sh_degree, packed=True
        )

        strategy.step_pre_backward(params, optimizers, strategy_state, it, info)

        l1 = F.l1_loss(render_colors, gt)
        ssim_val = ssim_loss(render_colors.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2), window_size=11)
        loss = 0.8 * l1 + 0.2 * (1 - ssim_val)
        scale_reg = 0.001 * torch.exp(scales).pow(2).mean()  # penalize large scales
        opa_reg = 0.0005 * (1 - torch.sigmoid(opacities)).pow(2).mean()  # encourage opacity to 0 or 1
        center = means.mean(dim=0).detach()
        pos_reg = 0.0002 * torch.norm(means - center, dim=1).mean()
        loss += scale_reg + opa_reg + pos_reg

        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        strategy.step_post_backward(params, optimizers, strategy_state, it, info, packed=True)

        # === Update local references (strategy mutates params in-place) ===
        means = params["means"]
        scales = params["scales"]
        quats = params["quats"]
        opacities = params["opacities"]
        sh0 = params["sh0"]
        shN = params.get("shN", torch.zeros((len(means), 0, 3), device=device))

        if it % 100 == 0 or it == num_iterations - 1:
            print(
                f"Iter {it:6d}  loss={loss.item():.4f}  L1={l1.item():.4f}  "
                f"SSIM={ssim_val.item():.4f}  #GS={len(means)}"
            )

    # === Very strict post-training pruning to kill the halo ===
    with torch.no_grad():
        opa = torch.sigmoid(opacities)
        sca = torch.exp(scales)

        # Core filters:
        # - High opacity (must be visible)
        # - Very small max/avg scale (no large blobs)
        # - Close to the main cluster (remove far outliers)
        # - Tight bounding box around the statue
        center = means.mean(dim=0)
        dist_to_center = torch.norm(means - center, dim=1)

        mask = (
            (opa > 0.08)  # stricter than before
            & (sca.max(dim=1)[0] < 0.15)  # max scale < 15 cm
            & (sca.mean(dim=1) < 0.06)  # average scale small
            & (dist_to_center < 10.0)  # within ~10 m of center
            & (torch.all(means > means.min(dim=0)[0] + 2.0, dim=1))  # avoid extreme outliers
            & (torch.all(means < means.max(dim=0)[0] - 2.0, dim=1))
        )

        print(f"Post-pruning: {len(means)} → {mask.sum()} Gaussians")
        print(f"Removed {len(means) - mask.sum()} halo/outlier Gaussians")

        means = means[mask]
        scales = scales[mask]
        quats = quats[mask]
        opacities = opacities[mask]
        sh0 = sh0[mask]
        shN = shN[mask] if sh_degree > 0 else shN

    # === Export (always pass real tensors) ===
    output_path = reconstruction_file.parent / f"{reconstruction_file.stem}_gsplat.ply"
    shN_export = shN if sh_degree > 0 else torch.zeros((len(means), 0, 3), device=means.device)

    export_splats(
        means.detach(),
        torch.exp(scales.detach()),
        quats.detach(),
        torch.sigmoid(opacities.detach()),
        sh0.detach(),
        shN_export,
        format="ply",
        save_to=str(output_path),
    )
    print(f"Saved {len(means)} Gaussians to {output_path}")


if __name__ == "__main__":
    tyro.cli(train_gaussians)
