from pathlib import Path

import numpy as np
import pycolmap
from loguru import logger

from .features import FeatureStore
from .pointcloud import PointCloud
from .tracks import TrackManager


# TODO(0): rename ColmapReconstructionAdapter
# TODO(1): consider adding bundle_adjustment calling pycolmap.bundle_adjustment() with options
# TODO: Given (1), this file io.py won't be necessary perhaps? rename colmap_adapter.py? move somewhere else?
class PycolmapReconIO:
    """Builds and exports a pycolmap reconstruction from project data."""

    def __init__(self, point_cloud: PointCloud, images: FeatureStore, track_manager: TrackManager):
        self.point_cloud = point_cloud
        self.images = images
        self.track_manager = track_manager

    def _build_reconstruction(self) -> pycolmap.Reconstruction:
        reconstruction = pycolmap.Reconstruction()
        first_image = next(iter(self.images.iter_images_with_pose()), None)
        if first_image is None:
            raise ValueError("At least one posed image is required to build a pycolmap reconstruction")

        camera_model = first_image.camera_model
        height, width = camera_model.resolution or first_image.pixels.shape[:2]
        fx, fy, cx, cy = camera_model.intrinsics_vector
        distortion = camera_model.distortion
        if len(distortion) < 4:
            distortion = np.pad(distortion, (0, 4 - len(distortion)))

        # ASSUMES: one camera for all photos, for now!
        camera = pycolmap.Camera(
            model="OPENCV",
            width=width,
            height=height,
            params=[fx, fy, cx, cy, *distortion[:4]],
            camera_id=1,
        )
        reconstruction.add_camera(camera)

        rig_id = 1
        rig = pycolmap.Rig(rig_id=rig_id)
        rig.add_ref_sensor(pycolmap.sensor_t(id=camera.camera_id, type=pycolmap.SensorType.CAMERA))
        reconstruction.add_rig(rig)

        for image_data in self.images.iter_images_with_pose():
            # ViewData.R and ViewData.t are already the world-to-camera transform.
            rotation_matrix = np.asarray(image_data.R, dtype=np.float64).reshape(3, 3)
            rotation = pycolmap.Rotation3d(rotation_matrix)
            translation = np.asarray(image_data.t, dtype=np.float64).reshape(3, 1)
            pose = pycolmap.Rigid3d(rotation, translation)

            frame = pycolmap.Frame(rig_id=rig_id, frame_id=image_data.idx, rig_from_world=pose)
            keypoints = image_data.kp
            if keypoints is None:
                raise ValueError(f"Image {image_data.idx} has no extracted keypoints")

            image = pycolmap.Image(
                name=image_data.path.name,
                image_id=image_data.idx,
                camera_id=camera.camera_id,
                frame_id=frame.frame_id,
            )
            # pycolmap.INVALID_POINT3D_ID as temp placeholder until overriden in the next loop
            # All keypoints w/ track_id are deemed triangulated after BA
            image.points2D = pycolmap.Point2DList(
                [pycolmap.Point2D(keypoint, pycolmap.INVALID_POINT3D_ID) for keypoint in keypoints]
            )
            frame.add_data_id(image.data_id)
            reconstruction.add_frame(frame)
            reconstruction.add_image(image)

        for track_id, xyz in self.point_cloud.items():
            track = pycolmap.Track()
            kp_keys = [
                (image_id, keypoint_idx)
                for image_id, keypoint_idx in self.track_manager.get_keypoints(track_id)
                if image_id in reconstruction.images
            ]
            for image_id, keypoint_idx in kp_keys:
                if image_id in reconstruction.images:
                    track.add_element(image_id, keypoint_idx)

            # Average into float, then convert & clip
            average_pixel_color = np.rint(self.images.get_pixels(kp_keys).mean(axis=0)).clip(0, 255).astype(np.uint8)
            reconstruction.add_point3D(np.asarray(xyz, dtype=np.float64).reshape(3, 1), track, average_pixel_color)

        return reconstruction

    def save(self, directory: Path) -> None:
        """Save cameras, images, and 3D points as human-readable text files."""

        logger.info("Building reconstruction...")
        self.reconstruction = self._build_reconstruction()

        directory.mkdir(exist_ok=True, parents=True)

        self.reconstruction.write_text(str(directory))
        self.reconstruction.write_binary(str(directory))
        self.reconstruction.export_PLY(str(directory / f"{directory.name}.ply"))

        logger.success(f"Exported reconstruction: {directory}")
        logger.info("View using: https://colmap.github.io/viewer.html")
