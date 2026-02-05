import json
import torch
from pathlib import Path
import numpy as np


def convert_to_nerfstudio(pt_file: Path, output_dir: Path):
    """Convert our .pt reconstruction to nerfstudio transforms.json format."""
    data = torch.load(pt_file)

    poses = data["poses"].cpu().numpy()  # (N, 4, 4) camera-to-world
    intrinsics = data["intrinsics"].cpu().numpy()  # (3, 3)

    N, H, W, _ = data["images"].shape

    # Nerfstudio format
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(intrinsics[0, 0]),
        "fl_y": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
        "w": W,
        "h": H,
        "frames": [],
    }

    # Save images and add frame entries
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N):
        # Save image
        img = (data["images"][i].cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image

        Image.fromarray(img).save(img_dir / f"frame_{i:04d}.png")

        # Add frame
        transforms["frames"].append({"file_path": f"images/frame_{i:04d}.png", "transform_matrix": poses[i].tolist()})

    # Save transforms.json
    with open(output_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Saved {N} images and transforms.json to {output_dir}")


if __name__ == "__main__":
    convert_to_nerfstudio(
        Path("data/out/statue_orbit/statue_orbit_disk_lightglue_ba.pt"), Path("data/out/statue_orbit/nerfstudio_format")
    )
