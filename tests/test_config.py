from config import SfMConfig


def test_imu_calib_property_sfm_config():
    cfg = SfMConfig(None, imu_calib_dir="data/calibration/redmi/kalibr/ros")
    calibration = cfg.imu_calibration

    assert calibration["accelerometer_noise_density"] == 0.01
    assert calibration["accelerometer_random_walk"] == 0.0001
    assert calibration["gyroscope_noise_density"] == 0.005
    assert calibration["gyroscope_random_walk"] == 1.0e-05
    assert calibration["update_rate"] == 409.1
    assert calibration["timeshift_cam_imu"] == 0.012794703039743893
