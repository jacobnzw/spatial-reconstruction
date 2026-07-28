from pathlib import Path

import gtsam
from gtsam.symbol_shorthand import X, P, V, B
import numpy as np
import pyceres
import pycolmap
import pycolmap._core.cost_functions as cost_functions
from loguru import logger

from utils import (
    FeatureStore,
    PointCloud,
    TrackManager,
    ViewData,
)


def bundle_adjustment_gtsam(
    images: FeatureStore,
    point_cloud: PointCloud,
    track_manager: TrackManager,
    fix_first_camera: bool = True,
) -> dict:
    """Run bundle adjustment with GTSAM using reprojection factors.

    Optimizes camera poses and 3D points while keeping camera intrinsics fixed.
    The routine uses the same ``FeatureStore``, ``PointCloud`` and ``TrackManager``
    structures as the other BA implementations in this module.
    """

    if not images.size:
        logger.warning("No images available for GTSAM bundle adjustment; skipping.")
        return {}

    first_img = images[0]
    K = first_img.camera_model.get_camera_matrix(rescaled=True)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    pose_keys: dict[int, int] = {}
    point_keys: dict[int, int] = {}

    for img in images.iter_images_with_pose():
        # pose_key = gtsam.symbol("x", img.idx)
        pose_key = X(img.idx)
        pose_keys[img.idx] = pose_key
        # IMPORTANT(!): world_T_cam expected by GTSAM
        pose = img.world_T_cam
        R, t = gtsam.Rot3(pose.rotation.as_matrix().copy()), pose.translation.copy()
        initial_values.insert(pose_key, gtsam.Pose3(R, t))

    for track_id, xyz in point_cloud.items():
        # point_key = gtsam.symbol("p", track_id)
        point_key = P(track_id)
        point_keys[track_id] = point_key
        initial_values.insert(point_key, gtsam.Point3(*xyz.copy()))

    # Keypoint noise in 2D image plane
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    # NOTE: Huber loss for robustness
    # huber_mest = gtsam.noiseModel.mEstimator.Huber(1.0)
    # noise_model = gtsam.noiseModel.Robust(huber_mest, noise_model)

    # For every triangulated 3D point (landmark) and its corresponding 2D keypoints ...
    for track_id, kp_keys in track_manager.track_to_kps.items():
        point_xyz = point_cloud.get_point(track_id)
        if point_xyz is None:
            continue

        point_key = point_keys.get(track_id)
        if point_key is None:
            continue

        for img_idx, kp_idx in kp_keys:
            if img_idx not in pose_keys:
                continue

            view = images[img_idx]
            if view.kp is None:
                continue

            observed = np.asarray(view.kp[kp_idx], dtype=np.float64)
            if observed.shape != (2,):
                continue

            pose_key = pose_keys[img_idx]
            # TODO: try other suitable factors: SmartProjection...
            # gtsam.SmartProjectionFactorPinholeCameraCal3_S2()
            factor = gtsam.GenericProjectionFactorCal3_S2(
                observed,
                noise_model,
                pose_key,
                point_key,
                calibration,
            )
            graph.add(factor)

    # Add priors on first camera for stability
    if fix_first_camera and pose_keys:
        first_img_idx = min(pose_keys)
        first_pose_key = pose_keys[first_img_idx]

        pose = images[first_img_idx].world_T_cam
        first_pose = gtsam.Pose3(gtsam.Rot3(pose.rotation.as_matrix()), pose.translation)
        graph.add(
            gtsam.PriorFactorPose3(
                first_pose_key,
                first_pose,
                # gtsam.noiseModel.Robust(huber_mest, gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)),
                gtsam.noiseModel.Isotropic.Sigma(6, 1e-4),
            )
        )
        # Weak prior on first point
        first_point_key, first_point = point_keys[0], initial_values.atPoint3(point_keys[0])
        graph.add(
            gtsam.PriorFactorPoint3(
                first_point_key,
                first_point,
                # gtsam.noiseModel.Robust(huber_mest, gtsam.noiseModel.Isotropic.Sigma(3, 1.0)),
                gtsam.noiseModel.Isotropic.Sigma(3, 1.0),
            )
        )

    # NOTE: Getting gtsam::IndeterminantLinearSystemException w/ DoglegOptimizer
    # optimizer = gtsam.DoglegOptimizer(graph, initial_values, gtsam.DoglegParams())
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setVerbosity("TERMINATION")  # SILENT, TERMINATION, ERROR, VALUES, DELTA, LINEAR
    params.setVerbosityLM("TERMINATION")

    # NOTE: Useful for tunning.
    # params.setDiagonalDamping(True)
    # params.setUseFixedLambdaFactor(False)
    # params.setlambdaInitial(1e-4)
    # params.setlambdaFactor(2.0)
    # params.setlambdaUpperBound(1e32)
    # params.setlambdaLowerBound(1e-16)
    # params.setLogFile()
    # params.setLinearSolverType()  # MULTIFRONTAL_SOLVER, MULTIFRONTAL_CHOLESKY, MULTIFRONTAL_QR, SEQUENTIAL_CHOLESKY, SEQUENTIAL_QR
    params.print()

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()

    for img_idx, pose_key in pose_keys.items():
        # IMPORTANT(!): world_T_cam -> cam_T_world expected by set_extrinsics()
        pose = result.atPose3(pose_key).inverse()
        images[img_idx].set_extrinsics(pose.rotation().matrix(), pose.translation())

    for track_id, point_key in point_keys.items():
        point = result.atPoint3(point_key)
        point_cloud.set_point(track_id, point)

    logger.info("GTSAM bundle adjustment complete.")

    wandb_summary = {
        "optimizer": str(optimizer).split()[0],
        "initial_cost": graph.error(initial_values),
        "final_cost": graph.error(result),
        "optimizer_iterations": optimizer.iterations(),
        "max_iterations": params.getMaxIterations(),
        "absolute_error_tol": params.getAbsoluteErrorTol(),
        "relative_error_tol": params.getRelativeErrorTol(),
        "linear_solver_type": params.getLinearSolverType(),
        "ordering_type": params.getOrderingType(),
        "use_fixed_lambda_factor": params.getUseFixedLambdaFactor(),
        "diagonal_damping": params.getDiagonalDamping(),
        "lambda_initial": params.getlambdaInitial(),
        "lambda_factor": params.getlambdaFactor(),
        "lambda_lower_bound": params.getlambdaLowerBound(),
        "lambda_upper_bound": params.getlambdaUpperBound(),
    }
    return wandb_summary


def add_imu_factors(graph, initial, data_file: str, calibration: dict):

    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)

    # Accel, gyro noise covariances
    gyro_sigma = calibration["gyroscope_noise_density"]
    accel_sigma = calibration["accelerometer_noise_density"]
    I_3x3 = np.eye(3)
    pim_params.setGyroscopeCovariance(gyro_sigma**2 * I_3x3)
    pim_params.setAccelerometerCovariance(accel_sigma**2 * I_3x3)
    pim_params.setIntegrationCovariance(1e-7**2 * I_3x3)

    # TODO: Bias values?
    accBias = np.array([-0.3, 0.1, 0.2])
    gyroBias = np.array([0.1, 0.3, -0.1])
    actualBias = gtsam.imuBias.ConstantBias(accBias, gyroBias)

    # Noise models for
    priorNoise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    velNoise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

    graph.push_back(gtsam.PriorFactorPose3(X(0), initial_state.pose(), priorNoise))
    graph.push_back(gtsam.PriorFactorVector(V(0), initial_state.velocity(), velNoise))

    initial.insert(B(0), actualBias)
    initial.insert(X(0), initial_state.pose())
    initial.insert(V(0), initial_state.velocity())

    # IMU Pre-integration object
    pim = gtsam.PreintegratedImuMeasurements(pim_params, actualBias)
    i = 0
    dt = 1 / calibration["update_rate"]
    imu_iter = stream_imu_from_csv(data_file)
    for k, (t_imu, omega, accel) in enumerate(imu_iter):
        ### This is where all the magic happens!
        pim.integrateMeasurement(accel, omega, dt)

        # Create one IMU factor per cam frame (except the initial), integrating IMU measurements between previous and current cam frame
        # TODO: Generalize: creating IMU factor every second => cam frames are 1s apart
        # Need to add IMUFactor when t_imu is close to the next t_cam (frame timestamp), may not be uniformly spaced
        # Create IMU factor every second => uniformly spaced cam frames
        if (k + 1) % int(calibration["update_rate"]) == 0:
            factor = gtsam.ImuFactor(X(i), V(i), X(i + 1), V(i + 1), B(0), pim)
            graph.push_back(factor)

            # We have created the binary constraint, so we clear out the preintegration values.
            pim.resetIntegration()

            i += 1

    return graph


def stream_imu_from_csv(data_file: str):
    import csv

    imu_data = csv.reader(Path(data_file).open())
    next(imu_data)  # skip header
    for row in imu_data:
        timestamp = int(row[0])
        omega = np.fromstring(" ".join(row[1:4]), sep=" ")
        accel = np.fromstring(" ".join(row[4:]), sep=" ")
        yield timestamp, omega, accel


def bundle_adjustment(
    images: FeatureStore,
    point_cloud: PointCloud,
    track_manager: TrackManager,
    fix_first_camera: bool = True,
) -> dict:
    """Run bundle adjustment on all cameras and 3D points.

    Uses pyceres as the optimization backend with pycolmap's ReprojErrorCost
    for computing reprojection errors and analytical Jacobians. Optimizes
    camera poses (rotation + translation) and 3D point positions while keeping
    camera intrinsics fixed.

    Args:
        images: Feature store containing keypoints and camera poses
        point_cloud: 3D point cloud to optimize
        track_manager: Manages correspondences between 2D keypoints and 3D points
        fix_first_camera: If True, fixes the first camera pose to avoid gauge freedom
    """

    # Get camera intrinsics from first image
    first_img = images[0]
    K, dist = first_img.camera_model.get_camera_matrix(), first_img.camera_model.dist
    # Create pycolmap camera model (OPENCV: fx, fy, cx, cy, k1, k2, p1, p2)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    k1, k2, p1, p2 = dist[:4]
    cam_params = np.array([fx, fy, cx, cy, k1, k2, p1, p2], dtype=np.float64)
    camera_model = pycolmap.CameraModelId.OPENCV

    # Prepare camera poses (as pycolmap.Rigid3d)
    camera_poses = {}
    for img in images.iter_images_with_pose():
        # Create Rigid3d (cam_from_world transformation)
        # pycolmap.Rotation3d can be constructed directly from rotation matrix
        camera_poses[img.idx] = pycolmap.Rigid3d(rotation=pycolmap.Rotation3d(img.R.copy()), translation=img.t.copy())

    # Prepare 3D points
    point_params = {track_id: xyz.copy() for track_id, xyz in point_cloud.items()}

    # Build the optimization problem
    problem = pyceres.Problem()
    loss = pyceres.HuberLoss(1.0)  # Robust loss for outliers

    # Add residual blocks for each observation
    for track_id, kp_keys in track_manager.track_to_kps.items():
        if track_id not in point_params:
            continue

        point_3d = point_params[track_id].astype(np.float64)

        for img_idx, kp_idx in kp_keys:
            if img_idx not in camera_poses:
                continue

            # Get observed 2D point
            observed_pt = np.array(images[img_idx].kp[kp_idx], dtype=np.float64)

            # Create cost function using pycolmap (with built-in Jacobians)
            cost = cost_functions.ReprojErrorCost(camera_model, observed_pt)

            # Add residual block
            # Parameter order: [quat, translation, point_3d, camera_params]
            pose = camera_poses[img_idx]
            problem.add_residual_block(
                cost,
                loss,
                [
                    pose.rotation.quat,
                    pose.translation,
                    point_3d,
                    cam_params,
                ],
            )

    # Set quaternion manifold for proper optimization on SO(3)
    for pose in camera_poses.values():
        problem.set_manifold(pose.rotation.quat, pyceres.EigenQuaternionManifold())

    # Fix camera intrinsics
    problem.set_parameter_block_constant(cam_params)

    # Fix the first camera (to avoid gauge freedom)
    if fix_first_camera and camera_poses:
        first_img_idx = min(camera_poses.keys())
        first_pose = camera_poses[first_img_idx]
        problem.set_parameter_block_constant(first_pose.rotation.quat)
        problem.set_parameter_block_constant(first_pose.translation)
        print(f"Fixed camera {first_img_idx} to avoid gauge freedom")

    # Configure solver
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = 100
    options.num_threads = -1

    # Solve
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    logger.info(summary.BriefReport())
    logger.debug(summary.FullReport())

    # Update camera poses with optimized values
    for img_idx, pose in camera_poses.items():
        # Convert quaternion back to rotation matrix
        R = pose.rotation.matrix()
        t = pose.translation
        images[img_idx].set_extrinsics(R, t)

    # Update 3D points
    for track_id, point_3d in point_params.items():
        point_cloud.set_point(track_id, point_3d)

    logger.info("Bundle adjustment complete.")

    wandb_summary = {
        "initial_cost": summary.initial_cost,
        "final_cost": summary.final_cost,
        "linear_solver_type_used": str(summary.linear_solver_type_used).split(".")[1],
        "linear_solver_type_given": str(summary.linear_solver_type_given).split(".")[1],
        "minimizer_type": str(summary.minimizer_type).split(".")[1],
        "termination_type": str(summary.termination_type).split(".")[1],
    }

    return wandb_summary


def bundle_adjustment_pycolmap(
    feature_store: FeatureStore,
    point_cloud: PointCloud,
    track_manager: TrackManager,
    fix_first_camera: bool = True,
):
    """Run bundle adjustment on all cameras and 3D points using pycolmap cost functions."""

    # 1. Initialize the Reconstruction and CameraFirst, set up your camera model.
    # Since you have OpenCV-style calibration (K and dist), the OPENCV model is the best fit.
    # It expects 8 parameters: [fx, fy, cx, cy, k1, k2, p1, p2].
    reconstruction = pycolmap.Reconstruction()

    # Map your OpenCV calibration to COLMAP parameters
    # OpenCV order: [fx, fy, cx, cy, k1, k2, p1, p2]
    first_img: ViewData = feature_store[0]
    K, dist = first_img.camera_model.get_camera_matrix(), first_img.camera_model.dist
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2 = dist[0][:4]  # Assuming standard 4-5 params

    params = [fx, fy, cx, cy, k1, k2, p1, p2]
    camera = pycolmap.Camera(
        model="OPENCV",
        width=3060,
        height=4080,
        params=params,
        camera_id=1,  # You can share one camera across all images
    )
    reconstruction.add_camera(camera)

    # Create a Rig (Even if just one camera)
    rig_id = 1
    rig = pycolmap.Rig(rig_id=rig_id)
    sensor = pycolmap.sensor_t(id=camera.camera_id, type=pycolmap.SensorType.CAMERA)
    rig.add_ref_sensor(sensor)  # Identity transform between rig and camera
    reconstruction.add_rig(rig)

    # 2. Add Images and Observations
    # COLMAP requires Point2D objects that link back to Point3D IDs.
    # Crucial Note: COLMAP uses cam_from_world (World-to-Camera).
    # Your R and t should already be in this format if they came from cv.solvePnP.
    for img_data in feature_store.iter_images_with_pose():
        # Convert your R, t to COLMAP's Rigid3d
        # Note: pycolmap.Rotation3d(R) handles 3x3 matrices directly
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(img_data.R), img_data.t)
        frame = pycolmap.Frame(
            rig_id=rig_id,
            frame_id=img_data.idx,
            rig_from_world=pose,
        )
        # Initialize EVERY point as -1 (Invalid)
        # points2D = [pycolmap.Point2D(kp.pt, pycolmap.INVALID_POINT3D_ID) for kp in img_data.kp]
        image = pycolmap.Image(
            name=img_data.path.name,
            # points2D=pycolmap.Point2DList(points2D),
            image_id=img_data.idx,
            camera_id=camera.camera_id,
            frame_id=frame.frame_id,
        )
        p2d_list = pycolmap.Point2DList()
        for kp in img_data.kp:
            p2d_list.append(pycolmap.Point2D(kp, pycolmap.INVALID_POINT3D_ID))
        image.points2D = p2d_list
        # Link the image to the frame
        # image.frame_id = frame.frame_id
        frame.add_data_id(image.data_id)
        reconstruction.add_frame(frame)
        reconstruction.add_image(image)

    # 3. Add 3D Points and Links: Finally, add the points from your PointCloud.
    # COLMAP needs to know which "image observation" belongs to which 3D point to build the Jacobian.
    for track_id, xyz in point_cloud.items():
        # 1. Create a Track object
        # This tells COLMAP which (image_id, point2D_idx) see this 3D point
        track = pycolmap.Track()
        for img_id, kp_idx in track_manager.track_to_kps[track_id]:
            if img_id in reconstruction.images:  # Only add if image was registered
                # reconstruction.image(img_id).set_point3D_for_point2D(kp_idx, track_id)
                reconstruction.image(img_id).reset_point3D_for_point2D(kp_idx)
                track.add_element(img_id, kp_idx)

        # 2. Add the point to reconstruction
        # add_point3D returns the internal ID (which matches your track_id)
        reconstruction.add_point3D(xyz, track, [128, 128, 128])  # [r, g, b]

    # 4. Run the Bundle Adjustment
    # Now that the graph is linked, you can run the optimization.
    options = pycolmap.BundleAdjustmentOptions()
    # Optional: Fix intrinsics if you trust your OpenCV calibration
    options.refine_focal_length = False
    options.refine_extra_params = False

    # Run it!
    pycolmap.bundle_adjustment(reconstruction, options)

    # Update your local data from the results
    for point3D_id, point3D in reconstruction.points3D.items():
        point_cloud.set_point(point3D_id, point3D.xyz)

    for image_id, image in reconstruction.images.items():
        # Extract optimized R and t
        opt_R = image.cam_from_world().rotation.matrix()
        opt_t = image.cam_from_world().translation
        feature_store[image_id].set_extrinsics(opt_R, opt_t)
