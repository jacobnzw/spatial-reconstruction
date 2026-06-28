from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml
from numpy.typing import NDArray

NDArrayFloat = NDArray[np.floating[Any]]
NDArrayInt = NDArray[np.integer[Any]]


class CameraType(Enum):
    """Enum for different camera models."""

    PINHOLE = "pinhole"
    FISHEYE = "fisheye"


@dataclass
class CameraModel:
    """Camera intrinsic parameters and distortion model.

    Attributes:
        model_type: Type of camera model (pinhole or fisheye).
        K: Camera intrinsic matrix (3x3).
        dist: Distortion coefficients.
    """

    model_type: CameraType = CameraType.PINHOLE

    K: NDArrayFloat = field(default_factory=lambda: np.eye(3))

    dist: NDArrayFloat = field(default_factory=lambda: np.zeros(5))

    scale: float = 1.0  # Scaling factor applied to the image (1.0 means no scaling)

    def get_camera_matrix(self, rescaled: bool = True) -> NDArrayFloat:
        """Get camera matrix K, rescaled if necessary.

        Args:
            rescaled: If True, return the rescaled K based on the current scale factor. If False, return the original K.
        """

        if rescaled and self.scale < 1.0:
            K_rescaled = self.K.copy()
            K_rescaled[0, :] *= self.scale
            K_rescaled[1, :] *= self.scale
            return K_rescaled
        return self.K

    @staticmethod
    def from_calibration(calib_file: str):
        """Loads camera parameters from calibration file."""

        calibration = yaml.safe_load(Path(calib_file).open())

        fx, fy, cx, cy = calibration["intrinsics"]
        return CameraModel(
            model_type=CameraType(calibration["camera_type"]),
            K=np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ]
            ),
            dist=np.array(calibration["distortion_coeffs"]),
        )


def calibrate_camera(camera_params_file: Path, force_recalibrate: bool = False):
    """Compute camera intrinsics given a sample of checkerboard photos.

    Args:
        camera_params_file: Path to save/load calibration parameters (K and dist coefficients).
        force_recalibrate: If True, ignore cached parameters and recalibrate. Defaults to False.

    Returns:
        Tuple of (K, dist) where K is the camera intrinsic matrix (3x3) and dist are the distortion coefficients.
    """

    print("Calibrating camera...")
    # Checkerboard parameters
    CHECKERBOARD = (8, 6)  # inner corners (width, height)
    SQUARE_SIZE = 0.025  # meters (example)

    # Prepare object points (0,0,0), (1,0,0), ...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    # ASSUME: folder containing the camera params file contains calibration images
    img_dir = Path(camera_params_file).parent
    images = list(img_dir.glob("*.jpg"))

    for fname in images:
        img = cv.imread(str(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # ty:ignore[no-matching-overload]

        ret, corners = cv.findChessboardCorners(
            gray,
            CHECKERBOARD,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
        )

        if ret:
            corners_refined = cv.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )

            objpoints.append(objp)
            imgpoints.append(corners_refined)

    # Camera calibration
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  # ty:ignore[no-matching-overload]
    dist = dist.squeeze()

    # Check reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(objpoints)
    print(f"Mean reprojection error: {mean_error:.4f} pixels")
    if mean_error > 0.5:
        print("WARNING: High reprojection error! Calibration may be inaccurate.")

    # Cache the calibration parameters
    k = K.tolist()
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    height, width, _ = img.shape
    calib_data = {
        "camera_type": "pinhole",
        "intrinsics": [fx, fy, cx, cy],
        "distortion_coeffs": dist.tolist(),
        "resolution": [width, height],
    }

    write_mode = "x"  # creates non-existent file, throws if already exists
    if force_recalibrate and camera_params_file.exists():
        print(f"Over-writting calibration file: {camera_params_file}")
        write_mode = "w+"  # overwrites existing file contents
    yaml.safe_dump(calib_data, camera_params_file.open(mode=write_mode), default_flow_style=False)
    print(f"Camera calibration saved to: {camera_params_file}")

    return K, dist
