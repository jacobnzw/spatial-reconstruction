from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from .pointcloud import PointCloud
from .features import FeatureStore
from .tracks import TrackManager


# TODO: use the new ViewData convenience funcs to express the calculations
class ReconIO:
    """Handles saving and loading of reconstruction data (PLY files and gsplat tensors)."""

    Vertex = tuple[float, float, float]  # (x, y, z)
    Edge = tuple[int, int]  # (v1, v2)

    def __init__(self, point_cloud: PointCloud, images: FeatureStore, track_manager: TrackManager):  # noqa: F821
        self.point_cloud = point_cloud
        self.images = images
        self.track_manager = track_manager

    @staticmethod
    def _camera_frustum_points(R: NDArray[np.float32], t: NDArray[np.float32], scale: float = 0.1) -> list:
        """
        Returns 5 world-space points for a tiny camera frustum
        """
        # World --> Camera:   Xc = R Xw + t
        # Camera --> World:   Xw = Rᵀ (Xc − t)
        # Camera center in world coordinates: Xc=0 --> Xw = Rᵀ (0 − t) = -Rᵀ t
        C = -R.T @ t.squeeze()  # R.T @ (-t)

        # Camera axes in world frame
        right = R.T @ np.array([1, 0, 0])
        up = R.T @ np.array([0, 1, 0])
        forward = R.T @ np.array([0, 0, 1])

        # Image plane center
        P = C + scale * forward

        # Image plane corners
        s = scale * 0.5
        corners = [
            P + s * (right + up),
            P + s * (right - up),
            P + s * (-right - up),
            P + s * (-right + up),
        ]

        return [C] + corners

    def _add_camera_frustum(
        self,
        vertices: list[Vertex],
        edges: list[Edge],
        R: NDArray[np.float32],
        t: NDArray[np.float32],
        scale: float = 0.1,
    ):
        base_idx = len(vertices)

        pts = self._camera_frustum_points(R, t, scale)
        for p in pts:
            vertices.append((p[0], p[1], p[2]))

        # center → corners
        for i in range(1, 5):
            edges.append((base_idx, base_idx + i))

        # square around image plane
        edges += [
            (base_idx + 1, base_idx + 2),
            (base_idx + 2, base_idx + 3),
            (base_idx + 3, base_idx + 4),
            (base_idx + 4, base_idx + 1),
        ]

    def _get_point_colors(self) -> NDArray[np.uint8]:
        """Returns an array of RGB colors for each 3D point."""
        colors = np.zeros((self.point_cloud.size, 3), dtype=np.uint8)
        for track_id, pt in self.point_cloud.items():
            kp_keys = self.track_manager.track_to_kps[track_id]
            # average the colors of all KPs in the track
            colors[track_id] = self.images.get_pixels(kp_keys).mean(axis=0)
        return colors

    def save_for_gsplat(self, filename: Path):
        """Save SfM reconstruction as tensors for gsplat training.

        Saves:
            - poses: (N, 4, 4) camera-to-world transformation matrices
            - images: (N, H, W, 3) RGB images (float32, range [0, 1])
            - points: (M, 3) 3D point positions
            - colors: (M, 3) RGB colors for 3D points (float32, range [0, 1])
            - intrinsics: (3, 3) camera intrinsic matrix
        """
        filename.parent.mkdir(exist_ok=True, parents=True)

        # Collect camera poses as 4x4 matrices
        poses_list = []
        images_list = []

        for img_data in self.images.iter_images_with_pose():
            # Create 4x4 pose matrix [R | t; 0 0 0 1]
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :3] = img_data.R
            pose_4x4[:3, 3:4] = img_data.t[..., None]
            poses_list.append(pose_4x4)
            images_list.append(img_data.pixels)

        # Stack into tensors
        poses_tensor = torch.from_numpy(np.stack(poses_list, axis=0))  # (N, 4, 4)
        images_tensor = (
            torch.from_numpy(np.stack(images_list, axis=0)).float() / 255.0
        )  # (N, H, W, 3), normalized to [0, 1]

        # Get 3D points and colors (reuse existing method)
        points_3d = self.point_cloud.get_points_as_array()  # (M, 3)
        colors = self._get_point_colors()  # (M, 3) uint8

        points_tensor = torch.from_numpy(points_3d).float()  # (M, 3)
        colors_tensor = torch.from_numpy(colors).float() / 255.0  # (M, 3), normalized to [0, 1]

        # Get camera intrinsics (rescaled) from first image with pose
        K = self.images[0].camera_model.get_camera_matrix()
        intrinsics_tensor = torch.from_numpy(K).float()  # (3, 3)

        # Save as .pt file
        torch.save(
            {
                "poses": poses_tensor,
                "images": images_tensor,
                "points": points_tensor,
                "colors": colors_tensor,
                "intrinsics": intrinsics_tensor,
            },
            filename,
        )

        print("Saved reconstruction for gsplat:")
        print(f"  - {len(poses_list)} camera poses: {poses_tensor.shape}")
        print(f"  - {len(images_list)} images: {images_tensor.shape}")
        print(f"  - {self.point_cloud.size} 3D points: {points_tensor.shape}")
        print(f"  - Colors: {colors_tensor.shape}")
        print(f"  - Intrinsics: {intrinsics_tensor.shape}")
        print(f"  -> {filename}")

    @staticmethod
    def load_for_gsplat(
        filename: Path, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Load SfM reconstruction tensors saved for gsplat training.

        Returns:
            Tuple of (poses, images, points, colors, intrinsics, width, height):
                - poses: (N, 4, 4) camera-to-world transformation matrices
                - images: (N, H, W, 3) RGB images (float32, range [0, 1])
                - points: (M, 3) 3D point positions
                - colors: (M, 3) RGB colors for 3D points (float32, range [0, 1])
                - intrinsics: (N, 3, 3) camera intrinsic matrices (tiled for each camera)
                - width: image width (int)
                - height: image height (int)
        """
        data = torch.load(filename)

        poses = data["poses"]  # (N, 4, 4)
        images = data["images"]  # (N, H, W, 3)
        points = data["points"]  # (M, 3)
        colors = data["colors"]  # (M, 3)
        intrinsics_single = data["intrinsics"]  # (3, 3)

        # Get image dimensions from the images tensor
        N, H, W, _ = images.shape

        # Tile intrinsics to match number of cameras (N, 3, 3)
        intrinsics = intrinsics_single.unsqueeze(0).expand(N, 3, 3)

        print(f"Loaded reconstruction from {filename}:")
        print(f"  - {N} camera poses: {poses.shape}")
        print(f"  - {N} images: {images.shape}")
        print(f"  - {points.shape[0]} 3D points: {points.shape}")
        print(f"  - Colors: {colors.shape}")
        print(f"  - Intrinsics (tiled): {intrinsics.shape}")
        print(f"  - Image size: {W} x {H}")

        return poses.to(device), images.to(device), points.to(device), colors.to(device), intrinsics.to(device), W, H

    def save_ply(self, filename: Path = Path("point_cloud.ply")):
        import pandas as pd

        # Convert to DataFrame
        xyz = self.point_cloud.get_points_as_array()
        df = pd.DataFrame(xyz, columns=["x", "y", "z"])

        # --- OUTLIER REMOVAL ---
        # Calculate the distance from the median to find the "main cluster"
        median = df.median()
        distance = np.sqrt(((df - median) ** 2).sum(axis=1))

        # Keep only points within the 95th percentile of distance
        # This removes the "points at infinity" that squash your visualization
        distance_mask = distance < distance.quantile(0.95)
        df_filtered = df[distance_mask]
        colors_filtered = self._get_point_colors()[distance_mask]

        # Add camera frustums
        vertices = []
        edges = []
        for img in self.images.iter_images_with_pose():
            self._add_camera_frustum(vertices, edges, img.R, img.t)
        # edge indices offset by number of 3D points
        camera_vertex_offset = len(df_filtered)
        edges = [(e[0] + camera_vertex_offset, e[1] + camera_vertex_offset) for e in edges]

        num_vertices = len(df_filtered) + len(vertices)
        num_edges = len(edges)
        # --- DEBUG --- sanity checks
        # TODO: nicer stats
        print()
        print(f"{df.shape = }")
        print(f"# nans: {df.isna().sum().sum()}")
        print(f"# infs: {np.isinf(df.values).sum()}")
        norms = np.linalg.norm(df.values, axis=1)
        print(f"{norms.min() = }\n{norms.max() = }")
        print()
        print(f"{df_filtered.shape = }")

        print(f"Writing {len(df_filtered)} points to {filename}")
        filename.parent.mkdir(exist_ok=True, parents=True)
        with open(filename, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            # camera frustums: colored vertices + edges
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element edge {num_edges}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            # point cloud as white points
            for (x, y, z), (r, g, b) in zip(df_filtered.values, colors_filtered):
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
            # camera frustums: red vertices + edges
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]} 255 0 0\n")
            for e in edges:  # TODO: edges don't work in my viewer
                f.write(f"{e[0]} {e[1]} 255 0 0\n")
