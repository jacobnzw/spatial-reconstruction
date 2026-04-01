import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import joblib
    import gtsam
    import numpy as np
    from gtsam import (
        Cal3_S2,
        GaussNewtonOptimizer,
        GaussNewtonParams,
        GenericProjectionFactorCal3_S2,
        NonlinearFactorGraph,
        Pose3,
        PriorFactorPoint3,
        PriorFactorPose3,
        Rot3,
        Values,
        symbol_shorthand,
    )

    from utils import (
        FeatureStore,
        PointCloud,
        TrackManager,
        validate_track_manager,
    )

    from collections import defaultdict, Counter


@app.cell
def _():
    load_path = "data/out/statue_orbit/statue_orbit_sift_bf_sfm_structs.joblib"
    images, point_cloud, track_manager = joblib.load(load_path)
    print(f"{validate_track_manager(track_manager)=}")
    return images, point_cloud, track_manager


@app.cell
def _():
    def iter_factor_landmark_keys(factor, lm_symbol='l'):
        yield from (key for key in factor.keys() if gtsam.Symbol(key).chr() == ord(lm_symbol))
    
    def iter_landmark_factors(graph, lm_symbol='l'):
        """Generator yielding factors involving landmark variables. 
        Returns tuples (fidx, factor), where fidx is factor's index in a factor graph.
        """
        yield from ((i, graph.at(i)) for i in range(graph.size()) for key in iter_factor_landmark_keys(graph.at(i)))
    return


@app.cell
def _(images, point_cloud, track_manager):
    # After adding all factors, before optimization:

    def observation_counts():
        # Count observations per landmark
        from collections import defaultdict
        landmark_obs_count = defaultdict(int)

        for i, img in enumerate(images.iter_images_with_pose()):
            for j, measurement in enumerate(img.kp):
                if (landmark_id := track_manager.get_track((img.idx, j))) is not None:
                    landmark_obs_count[landmark_id] += 1

        # Find poorly observed landmarks
        poorly_observed = {lid: count for lid, count in landmark_obs_count.items() if count < 2}
        print(f"Landmarks with <2 observations: {len(poorly_observed)}")
        print(f"Example: {list(poorly_observed.items())[:5]}")

        # Remove them from the problem
        # for landmark_id in poorly_observed:
        #     if initial_estimate.exists(L(landmark_id)):
        #         initial_estimate.erase(L(landmark_id))
        #         print(f"Removed landmark {landmark_id} (only {poorly_observed[landmark_id]} observations)")


        # Before optimization
        print(f"Total cameras: {sum(1 for _ in images.iter_images_with_pose())}")
        print(f"Total landmarks: {point_cloud.size}")
        print(f"Total observations: {sum(landmark_obs_count.values())}")
        print(f"Avg observations per landmark: {sum(landmark_obs_count.values()) / len(landmark_obs_count):.2f}")
    observation_counts()
    return


@app.cell
def _(track_manager):
    # initial_estimate.exists(L(1361))
    tid = 1361
    kpkeys = track_manager.track_to_kps[tid]
    print(f"{tid=} --> {kpkeys=}")
    print(f"{track_manager.track_to_kps[tid]=}")
    print(f"{track_manager.kp_to_track[kpkeys[0]]=}")
    print(f"{track_manager.kp_to_track[kpkeys[1]]=}")

    # TODO: HOW did that happen?
    # assert track_manager.kp_to_track[kpkeys[0]] == track_manager.kp_to_track[kpkeys[1]]
    return


@app.cell
def _(images, point_cloud, track_manager):
    # def bundle_adjustment_gtsam(
    #     images: FeatureStore,
    #     point_cloud: PointCloud,
    #     track_manager: TrackManager,
    #     fix_first_camera: bool = True,
    # ):
    # Recover camera intrinsics
    K, _ = images.get_intrisics()
    fx, fy, s, u0, u1 = K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2]
    K = Cal3_S2(fx, fy, s, u0, u1)

    X = symbol_shorthand.X
    L = symbol_shorthand.L

    # Create a Factor Graph and Values to hold the new data
    graph = NonlinearFactorGraph()
    initial_estimate = Values()

    # Add a prior on pose x0, 0.1 rad on roll,pitch,yaw, and 0.3 m std on x,y,z
    pose_prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
    graph.add(PriorFactorPose3(X(0), Pose3(), pose_prior))

    # Add a prior on landmark l0
    if (point := point_cloud.get_point(0)) is not None:
        print("Adding prior on landmark l0")
        point_prior = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        graph.add(PriorFactorPoint3(L(0), point, point_prior))

    # Add initial guesses to all observed landmarks (3D world points)
    for landmark_id, landmark in point_cloud.items():
        if len(track_manager.track_to_kps[landmark_id]) > 1:
            initial_estimate.insert(L(landmark_id), landmark)

    # Define the camera observation noise model, 1 pixel stddev
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    for i, img in enumerate(images.iter_images_with_pose()):
        # Add factor for each landmark observed by this image
        for j, measurement in enumerate(img.kp):
            # here track == landmark; keypoint == measurement
            # Note: not all image keypoints got triangulated => not all keypoints have a corresponding 3D point (track)
            if (landmark_id := track_manager.get_track((img.idx, j))) is not None:
                # Only add factors for landmarks observed by >1 image keypoint
                if len(track_manager.track_to_kps[landmark_id]) > 1:
                    # keypoint j is a measurement of 3D point landmark_id  (== track_id)
                    factor = GenericProjectionFactorCal3_S2(measurement, measurement_noise, X(i), L(landmark_id), K)
                    graph.add(factor)

        # Add an initial guess for the current pose X(i)
        initial_estimate.insert(X(i), Pose3(Rot3(img.R), img.t))

    # # REMOVE UNDERCONSTRAINED FACTORS
    # landmark_factor_count = defaultdict(int)
    # for fidx, factor in iter_landmark_factors(graph):
    #     for key in factor.keys():
    #         landmark_id = gtsam.symbolIndex(key)
    #         landmark_factor_count[landmark_id] += 1
    # underconstrained = {lid: count for lid, count in landmark_factor_count.items() if count < 2}

    # for fidx, factor in iter_landmark_factors(graph):
    #     for key in iter_factor_landmark_keys(factor):
    #         lid = gtsam.symbolIndex(key)
    #         if lid in underconstrained:
    #             graph.remove(fidx)
    #             initial_estimate.erase(L(lid))
    return L, graph, initial_estimate


@app.cell
def _(graph, initial_estimate):
    # TODO: Configure and run the optimizer
    params = GaussNewtonParams()
    params.setVerbosity("TERMINATION")
    params.setMaxIterations(10000)

    print("Optimizing the factor graph")
    optimizer = GaussNewtonOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print("Optimization complete")
    print(f"initial error={graph.error(initial_estimate)}")
    print(f"final error={graph.error(result)}")

    # Save the factor graph visualization
    save_path = Path("ba_factor_graph.dot")
    try:
        graph.saveGraph(save_path, result)
        print(f"Saved graph to {save_path}")
    except Exception as e:
        print(f"Could not save graph: {e}")
    return


@app.cell
def _(L, graph, images, initial_estimate, track_manager):
    def debug_landmark(problematic_id, graph):
        print(f"\n=== Debugging landmark {problematic_id} ===")
    
        # Check if it exists in initial estimate
        if initial_estimate.exists(L(problematic_id)):
            print(f"Landmark {problematic_id} exists in initial estimate")
            print(f"Position: {initial_estimate.atPoint3(L(problematic_id))}")
        else:
            print(f"WARNING: Landmark {problematic_id} NOT in initial estimate!")
    
        # Find all factors involving this landmark
        print(f"\nFactors involving L({problematic_id}):")
        factor_count = 0
        for i in range(graph.size()):
            factor = graph.at(i)
            keys = factor.keys()
            if L(problematic_id) in keys:
                factor_count += 1
                print(f"  Factor {i}: {[gtsam.Symbol(k).string() for k in keys]}")
                print(f"  Factor {i}: {factor}")
    
        print(f"Total factors for L({problematic_id}): {factor_count}")
    
        # Check observations from track manager
        if problematic_id in track_manager.track_to_kps:
            observations = track_manager.track_to_kps[problematic_id]
            print(f"Observations in track_manager: {observations}")
            print(f"  {[images[im_idx].kp[kp_idx] for im_idx, kp_idx in observations]}")
        else:
            print(f"WARNING: Landmark {problematic_id} has NO observations in track_manager!")
    
        # Count how many factors are involved in each landmark variable
        print("\n=== All underconstrained landmarks ===")
        landmark_factor_count = defaultdict(int)
        for i in range(graph.size()):
            factor = graph.at(i)
            for key in factor.keys():
                if gtsam.Symbol(key).chr() == ord('l'):
                    landmark_id = gtsam.Symbol(key).index()
                    landmark_factor_count[landmark_id] += 1
    
        underconstrained = {lid: count for lid, count in landmark_factor_count.items() if count < 2}
        print(f"Landmarks with <2 factors: {len(underconstrained)}")
        print(f"Examples: {list(underconstrained.items())[:10]}")
    
        return landmark_factor_count
    
    bad_id = 1737
    lm_factor_count = debug_landmark(bad_id, graph)
    lcounts = dict(sorted(Counter(lm_factor_count.values()).items()))
    print(f"{lcounts=}")
    return


@app.cell
def _(L, graph, images, track_manager):
    def compare_tm_actual():
        # Compare track_manager vs actual factors
        tid = 1361
        kpkeys = track_manager.track_to_kps[tid]
        print(f"Track {tid} has {len(kpkeys)} keypoints in track_manager: {kpkeys}")

    
        print(f"Keypoints map to tracks: {[track_manager.get_track(k) for k in kpkeys]}")

        tid = 1370
        kpkeys = track_manager.track_to_kps[tid]
        print(f"Track {tid} has {len(kpkeys)} keypoints in track_manager: {kpkeys}")
    
        # Count actual factors
        factor_count = 0
        for i in range(graph.size()):
            if L(tid) in graph.at(i).keys():
                factor_count += 1
        print(f"But only {factor_count} factors in graph!")
    
        # Check if those keypoints' images have poses
        for img_idx, kp_idx in kpkeys:
            img = images[img_idx]
            print(f"  Image {img_idx}, kp {kp_idx}: has_pose={img.R is not None}")
    compare_tm_actual()
    return


if __name__ == "__main__":
    app.run()
