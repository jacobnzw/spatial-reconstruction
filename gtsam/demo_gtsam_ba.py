import joblib

from ba_gtsam import bundle_adjustment_gtsam

load_path = "data/out/statue_orbit/statue_orbit_sift_bf_sfm_structs.joblib"
image_store, point_cloud, track_manager = joblib.load(load_path)

bundle_adjustment_gtsam(image_store, point_cloud, track_manager)
# print("Creating pose prior...")
# pose_prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))
# print("OK")
