from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
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

    model_type: CameraType
    K: NDArrayFloat
    dist: NDArrayFloat
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


def calibrate_camera(img_dir: Path = Path("data/calibration"), force_recalibrate: bool = False):
    """Compute camera intrinsics given a sample of checkerboard photos."""
    # Try to load cached calibration parameters
    CALIBRATION_FILENAME = "calibration_params.npz"
    CALIBRATION_PATH = img_dir / CALIBRATION_FILENAME
    if not force_recalibrate and CALIBRATION_PATH.exists():
        print(f"Loading cached calibration parameters from: {CALIBRATION_PATH}")
        with np.load(CALIBRATION_PATH) as data:
            return data["K"], data["dist"]

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
    np.savez(CALIBRATION_PATH, K=K, dist=dist)

    return K, dist
