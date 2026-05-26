import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import cv2 as cv
    import kornia as K
    import kornia.feature as KF
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from kornia.feature.adalam import AdalamFilter
    from kornia_moons.viz import draw_LAF_matches
    import io
    import requests

    device = K.utils.get_cuda_or_mps_device_if_available()
    print(device)
    return K, KF, cv, device, draw_LAF_matches, io, mo, requests, torch


@app.cell
def _(io, requests):
    def download_image(url: str, filename: str = "") -> str:
        filename = url.split("/")[-1] if len(filename) == 0 else filename
        # Download
        bytesio = io.BytesIO(requests.get(url).content)
        # Save file
        with open(filename, "wb") as outfile:
            outfile.write(bytesio.getbuffer())

        return filename


    url_a = "https://github.com/kornia/data/raw/main/matching/kn_church-2.jpg"
    url_b = "https://github.com/kornia/data/raw/main/matching/kn_church-8.jpg"
    download_image(url_a)
    download_image(url_b)
    return


@app.cell
def _(K, device, torch):
    fname1 = "data/raw/statue/IMG_20260109_174318.jpg"  # "kn_church-2.jpg"
    fname2 = "data/raw/statue/IMG_20260109_174323.jpg"  # "kn_church-8.jpg"

    img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32, device=device)[None, ...]

    hw1 = torch.tensor(img1.shape[2:], device=device)
    hw2 = torch.tensor(img2.shape[2:], device=device)
    return hw1, hw2, img1, img2


@app.cell
def _(KF, device, hw1, hw2, img1, img2, torch):
    lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)
    disk = KF.DISK.from_pretrained("depth").to(device)

    num_features = 2048
    with torch.inference_mode():
        inp = torch.cat([img1, img2], dim=0)
        features1, features2 = disk(inp, num_features, pad_if_not_divisible=True)
        kps1, descs1 = features1.keypoints, features1.descriptors
        kps2, descs2 = features2.keypoints, features2.descriptors
        lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
        lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))
        dists, idxs = lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)


    print(f"{idxs.shape[0]} tentative matches with DISK LightGlue")
    return idxs, kps1, kps2


@app.cell
def _(mo):
    mo.md(r"""
    ### Fundamental Matrix from Matching Keypoints
    """)
    return


@app.cell
def _(cv, idxs, kps1, kps2):
    def get_matching_keypoints(kp1, kp2, idxs):
        mkpts1 = kp1[idxs[:, 0]]
        mkpts2 = kp2[idxs[:, 1]]
        return mkpts1, mkpts2


    mkpts1, mkpts2 = get_matching_keypoints(kps1, kps2, idxs)

    Fm, inliers = cv.findFundamentalMat(
        mkpts1.detach().cpu().numpy(), mkpts2.detach().cpu().numpy(), cv.USAC_MAGSAC, 1.0, 0.999, 100000
    )
    inliers = inliers > 0
    print(f"{inliers.sum()} inliers with DISK")
    return (inliers,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Local Affine Frame (LAF)
    in Kornia is a geometric representation used to describe local image features with a center point, orientation, scale, and affine shape. It is a 2×3 matrix that encodes a local region in an image, enabling robust feature detection and matching under geometric transformations.

    #### Key Features of LAF in Kornia:
    - Structure: A tensor of shape `(B, N, 2, 3)` where: `B` = batch size `N` = number of local features
    2×3 = affine transformation matrix with center, orientation, and scale.

    #### Use Cases:
    - Feature Detection & Description: Used by detectors like DISK, DeDoDe, and LoFTR.
    - Image Matching: Enables geometrically aware matching using models like LightGlue or AdaLAM.
    - Affine Invariance: Supports affine shape estimation via LAFAffineShapeEstimator and orientation estimation via LAFOrienter.

    #### Operations:
    - `make_upright()`: Removes rotation to create upright LAFs.
    - `scale_laf()`: Rescales the local region while preserving center and orientation.
    - `denormalize_laf()`: Converts normalized LAFs to image scale.
    - `laf_to_boundary_points()`: Generates boundary points for visualization.

    #### Related Tools:
    - `ellipse_to_laf()`: Converts ellipse regions (e.g., from Oxford format) into LAF format.
    - `get_laf_pts_to_draw()`: Returns coordinates for plotting LAFs on images.
    - `convert_points_from_homogeneous()`: Used internally to transform points under LAFs.

    Kornia integrates LAFs into end-to-end deep learning pipelines, supporting differentiable image processing and compatibility with models like SAM, LoFTR, and Vision Language Models (VLMs).
    """)
    return


@app.cell
def _(K, KF, draw_LAF_matches, idxs, img1, img2, inliers, kps1, kps2):
    ax = draw_LAF_matches(
        KF.laf_from_center_scale_ori(kps1[None].cpu()),
        KF.laf_from_center_scale_ori(kps2[None].cpu()),
        idxs.cpu(),
        K.tensor_to_image(img1.cpu()),
        K.tensor_to_image(img2.cpu()),
        inliers,
        draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": (1, 1, 0.2, 0.3), "feature_color": None, "vertical": False},
        return_fig_ax=True
    )
    return (ax,)


@app.cell
def _(ax):
    ax
    return


if __name__ == "__main__":
    app.run()
