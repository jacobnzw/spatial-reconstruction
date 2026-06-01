
# Spatial Reconstruction Pipeline
Goal of the project is to build a pipeline for spatial reconstruction from a set of images.
So given a bunch of photos of one static object from different angles, we want to reconstruct the 3D model of the object.

Employ classical techniques to learn basic principles from 3D computer vision. Advance to deep learning techniques and Gaussian splatting representation for more accuracy and efficiency.

## 👉 Structure from Motion (SfM) Pipeline
= joint estimation of

- Camera poses (extrinsics)
- Sparse 3D points

From feature correspondences across multiple images
Critically: Each 3D point is typically seen in ≥ 2 images, Often 3–10 images in practice

Two main approaches:
- [Global SfM](https://arxiv.org/pdf/2407.20219v1): simultaneous estimation of all camera poses and 3D points
- Incremental SfM: estimate poses and 3D points incrementally, one image at a time

Approach this incrementally:
- estimate poses and 3D points using only two images
- add view graph construction to work with more images and refine the estimate

Need to construct view graph to represent the relationships between the images. 
Each node in the graph represents an image, and each edge represents the overlap between two images. 
The weight of the edge can represent the amount of overlap or the quality of the match between the two images. 
The view graph can be used to guide the image matching process and to estimate the camera poses.


## Experiments & Results
For feature (keypoint descriptor) extraction, I tested the classical SIFT features, which are still used in research and serve as a good baseline, and the learned neural DISK features (via `kornia` library).
For keypoint descriptor (feature) matching, I employed a classical Brute Force matcher (via the OpenCV's `BFMatcher`) as well as a learned neural LightGlue matcher (via `kornia` library).

### Dataset

I created my own `statue_orbit` dataset (at `data/raw/statue_orbit`) by taking 15 full-resolution (*3060 x 4080*) photos of a kneeling archer miniature statue from the Chinese terracotta army with my Redmi phone.
The statue measures about *12.5 cm* in height, and the base is roughly *4.5 x 4.5 cm*.
The photos have the head region slightly out of focus and photos of the statue's back exhibit, compared to the rest, a marked drop in illumination.
Both of these present a bit of a challenge for the SfM reconstruction pipeline. 
I wanted to code something that will work with real data, not just pristine lab data made with professional rigs.

<figure align=center>
  <img src="assets/statue_orbit.gif" height="400">
</figure>

### Experimental Setups

  - **SIFT+BF**: SIFT features with Brute Force matcher
    - Number of features limited to maximum `num_features=2000`
    - The Brute-Force (BF) matcher computes symmetric matches (via `cross_check=True`)
    - See full parameter config in [`assets/statue_orbit_sift_bf_config.log`](assets/statue_orbit_sift_bf_config.log)
  - **DISK+LG**: DISK features with LightGlue matcher
    - Number of features limited to maximum `num_features=1000`
    - Only LightGlue matches with confidence above `lg_min_conf=0.1` are retained
    - See full parameter config in [`assets/statue_orbit_disk_lg_config.log`](assets/statue_orbit_disk_lg_config.log)

TODO: add links to sift disk/lightglue papers


### Reconstruction from phone photos
Both of the following show a sparse point cloud reconstruction of the kneeling archer statue. 
The estimated statue points are colored by the corresponding pixel value in the images. 
The final estimated camera poses are shown as 5-point red pyramids representing the camera frustums.

![Statue Orbit SIFT BF BA](assets/statue_orbit_sift_bf.jpg)
*Figure: SIFT+BF: Note the pronounced presence of fliers around the head, which is likely due to the out-of-focus head region in the dataset images.*

![Statue Orbit DISK LG](assets/statue_orbit_disk_lg.jpg)
*Figure: DISK+LG: Note the drastically reduced presence of fliers around the head compared to the SIFT+BF setup. The point cloud is also much denser despite the fact that max number of DISK features was limited to half of the max number of SIFT features.*



### Sensitivity to Camera Matrix
The reconstruction is extremely sensitive to small perturbation in the camera intrinsics. 
Below I compare reconstructions with original (calibrated) and perturbed camera intrinsics.

![Statue Orbit SIFT+BF perturbed K](assets/statue_orbit_sift_bf_compare_goodK-green_badK-red.jpg)
*Figure: SIFT+BF setup comparing the orignal reconstruction with camera intrinsics $K$ (green) and reconstruction with perturbed instrinsics $K'=0.999K$ (red). Slight deviation in intrinsics has decisive effect on the quality of the reconstructed point cloud as well as the estimated camera poses.*


### Effect of Bundle Adjustment
In the below figures I compare the resulting point cloud reconstruction pre- and post bundle adjustment for both feature+matcher setups.
The bundle adjustment is done only once at end of the SfM estimation to further refine the object points and the camera poses (the intrisics are fixed during optimization).

![Statue Orbit SIFT BF Compare BA](assets/statue_orbit_sift_bf_compare_ba-green_sfm-red.jpg)
*Figure: SIFT+BF setup: After bundle adjustment the original camera poses out of SfM pipeline (red) are refined (green), while the points are largely unaffected.*

```
Ceres Solver Report: 
Iterations: 47, 
Initial cost: 2.435764e+03, 
Final cost: 7.472441e+02, 
Termination: CONVERGENCE
```
Full solver log in [`assets/statue_orbit_sift_bf_ba.log`](assets/statue_orbit_sift_bf_ba.log)

![Statue Orbit DISK+LG Compare BA](assets/statue_orbit_disk_lg_compare_ba-green_sfm-red.jpg)
*Figure: DISK+LightGlue setup: After bundle adjustment the original camera poses out of SfM pipeline (red) are refined (green), while the points are largely unaffected.*

```
Ceres Solver Report: 
Iterations: 35, 
Initial cost: 6.514872e+03, 
Final cost: 1.699766e+03, 
Termination: CONVERGENCE
```
Full solver log in [`assets/statue_orbit_disk_lg_ba.log`](assets/statue_orbit_disk_lg_ba.log)



### Reconstruction from TUM-VI sequence
Out of curiosity, I wanted to see how my pipeline performs on a real benchmarking dataset. 
<!-- I chose TUM-VI at first because it comes with IMU measurements, which I was hoping to use in related visual-inertial odometry learning project, but that has been put on ice due to time constraints. -->
I picked a sequence of 20 uniformly sampled frames between indices 540 and 640, which, at the frame rate of 20Hz, implies frame sampling frequency of 4Hz (250 ms between frames).
This subsequence contains enough motion so that a sufficient baseline is ensured.
Compared to the phone photos, TUM-VI provides additional challenge because the frames are distorted due to the fisheye cameras used by the recording rig. 
The further difficulty is the presence of many planar surfaces such as walls, which could cause problems for ...

<!-- at first because I wanted to incorporate IMU measurements for the SfM reconstruction -->

<figure align=center>
  <img src="assets/tumvi_corridor4.gif" alt="Alt text" width="400">
</figure>

The effect of undistortion on the reconstruction is compared in the following figures. I used DISK features limited to `num_features=1000` with LightGlue matcher.
With the original distorted images, we get reconstruction that has more points. 
The camera pose estimates are plausible given the frame sequence
<figure align=center>
  <img src="assets/tumvi_corridor4_disk_lg_ba_no-undistort_top.jpg" alt="Alt text" height="600">
  <figcaption>Figure: Reconstruction on select frames of corridor4 TUM-VI sequence on the orignal distorted frames: the corridor is apparent and the estimated camera pose sequence looks plausible.</figcaption>
</figure>

The undistortion procedure has a limiting effect on the field of view of the resulting images, which results in less points in the reconstruction.
The distortion is apparent on the wall reconstruction.
<figure align="center">
  <img src="assets/tumvi_corridor4_disk_lg_ba.jpg" alt="Alt text" height="600">
  <figcaption>Figure: Reconstruction on select frames of corridor4 TUM-VI sequence using the undistorted frames: the corridor is no longer apparent while the estimated camera pose sequence remains plausible.</figcaption>
</figure>


<!-- ## 🚧 Multi-view Stereo (MVS) Pipeline
= dense reconstruction from sparse 3D points and images -->

<!-- ## Reading List
S.D. Prince, Computer Vision: Models, Learning and Inference
SuperPoint
SuperGlue
TODO: list papers to read

## Notes
See [notes/CV_NOTES.md](notes/CV_NOTES.md) -->
