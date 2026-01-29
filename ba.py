import numpy as np
import pyceres
import pycolmap
import pycolmap._core.cost_functions as cost_functions

from utils import (
    FeatureStore,
    PointCloud,
    TrackManager,
)


def bundle_adjustment(
    images: FeatureStore,
    point_cloud: PointCloud,
    track_manager: TrackManager,
    fix_first_camera: bool = True,
):
    """Run bundle adjustment on all cameras and 3D points using pycolmap cost functions."""

    K, dist = images.get_intrisics()
    # Create pycolmap camera model (OPENCV: fx, fy, cx, cy, k1, k2, p1, p2)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    k1, k2, p1, p2 = dist[0][:4]
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
    print(summary.BriefReport())

    # Update camera poses with optimized values
    for img_idx, pose in camera_poses.items():
        # Convert quaternion back to rotation matrix
        R = pose.rotation.matrix()
        t = pose.translation
        images[img_idx].set_pose(R, t)

    # Update 3D points
    for track_id, point_3d in point_params.items():
        point_cloud.set_point(track_id, point_3d)

    print("Bundle adjustment complete.")


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
    K, dist = feature_store.get_intrisics()
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
        feature_store[image_id].set_pose(opt_R, opt_t)
