# Research Notes on 3D Computer Vision Fundamentals


## Data Structures
KDTree
Vocabulary-tree matching
Pose graph
View graph
Factor graph

## Mathematical Tooling
There are many tools that constantly recur in CV:
Gauss-Newton
Levenberg-Marquardt
RANSAC
Singular Value Decomposition

#### SE(3)
 - the group of rigid transformations in 3D space, consisting of rotation and translation.
 - 6 degrees of freedom (3 for rotation, 3 for translation)
 - A transformation in SE(3) can be represented as a 4x4 matrix with the following form:
 $$
 \begin{bmatrix}
    R & t \\
    0 & 1
 \end{bmatrix}
 $$, 
 where $R$ is a 3x3 rotation matrix, $t$ is a 3x1 translation vector.

#### Sim(3): 
 - the group of similarity transformations in 3D space, consisting of rotation, translation, and uniform scaling.
 - 7 degrees of freedom (3 for rotation, 3 for translation, 1 for scaling)
 - A transformation in Sim(3) can be represented as a 4x4 matrix with the following form:
 $$
 \begin{bmatrix}
    sR & t \\
    0 & 1
 \end{bmatrix}
 $$, 
 where $R$ is a 3x3 rotation matrix, $t$ is a 3x1 translation vector, and $s$ is a scalar representing uniform scaling.


### Keyframes
Codec keyframes (I-frames) are built into the video and used for decompression. 
Computer vision keyframes are extracted using models or algorithms to capture meaningful moments, which may or may not align with I-frames. 
In practice, vision systems may use codec keyframes as a starting point but refine selection using content-aware models for higher relevance.

Keyframes are crucial in computer vision processing of videos because they reduce data redundancy and computational overhead while preserving essential visual information. Videos typically contain hundreds or thousands of frames per minute, many of which are highly similar or contain minimal new information.  By selecting only the most representative frames—those that capture significant changes in content, motion, or structure—keyframes enable efficient processing for tasks like video summarization, action recognition, and visual localization.

Pinhole Camera Model
Other used camera models: Brown-Conrady model (fisheye), Kannala-Brandt model (fisheye), Dual Quaternion model (SfM), Catadioptric model (mirror+lens), Omnidirectional model (fish-eye+mirror)
homography
Homography Estimation
## Multi-View Geometry: 
[Essential matrix](https://en.wikipedia.org/wiki/Essential_matrix), 
[Fundamental matrix](https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)),
[Epipolar Geometry](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html)

bundle adjustment (gauge ambiguity; Sim(3))
iterative projection matching
reprojection error
photometric error
Sampson error for epipolar geometry
SLAM
Factor graph optimization is a generalization of bundle adjustment?
Umeyama algorithm for rigid transformation estimation with known correspondences


### Perspective n-Point (PnP)
Given 3D points in the world frame, their corresponding 2D projections in the image plane and known camera intrinsics, estimate the camera pose (rotation and translation) that best projects the 3D points to the 2D image points.

**EPnP** estimates camera pose using n 3D-to-2D point correspondences by representing the 3D points as a weighted sum of four virtual control points, enabling linear-time computation (O(n)). Non-iterative, closed-form solution. 
Possibly followed by iterative refinement (Gauss-Newton, Levenberg-Marquardt).


### Keypoint Extraction
- SIFT expensive but robust (invariant to image scaling, rotation, and translation; doesn't work well with changes in lighting or illumination, doesn't work with color; CSIFT addresses this)
- ORB fast but less robust (real-time)
- AKAZE is a faster and more accurate alternative to SIFT, also invariant to illumination changes and point of view.
- DISK is a modern, deep learning-based local feature detector and descriptor designed for robust image matching.  It is particularly effective for outdoor scenes and performs exceptionally well when combined with the LightGlue matcher, as shown in the IMC2021 benchmark.
- LoFTR: A detector-free, transformer-based model that works best for indoor scenes.



### Keypoint Matching
FLANN, Brute Force
In image keypoint matching, inliers are the correct, geometrically consistent correspondences between keypoints in two different images, meaning they belong to the same physical points and follow the same transformation (like rotation, scale, or perspective). They contrast with outliers, which are incorrect matches (e.g., matching a window to a door due to similar appearance) that don't conform to the overall scene transformation, and are typically filtered out using algorithms like RANSAC (Random Sample Consensus). 

## Stereo Principle
(from greek *stereos* = solid, thus relating to solid forms having three dimensions)

The core idea is that a 3D point in the world projects to different locations in images taken from different viewpoints.
This difference in location, known as disparity or parallax, is inversely proportional to its depth. 
By calculating disparity for numerous points and using the known camera geometry (position and orientation), 
the 3D position of those points can be determined through a process called triangulation.


### Structure from motion (SfM)
- given a set of overlapping images of a static scene, estimate camera poses and 3D points from 2D image points
- Goal: Find camera poses (position/orientation) and a sparse 3D structure.
- Input: A set of overlapping images.
- Process: Matches image features (like SIFT) across views to estimate camera movement and a basic 3D point cloud.
- Output: Camera parameters and a sparse point cloud. 

Multi-view Stereo (MVS)
- Goal: Generate a dense, detailed 3D model (point cloud or mesh).
- Input: Images plus the camera poses (usually from SfM) and sparse points.
- Process: Uses depth information and matches features across many images to fill in gaps, creating depth maps for dense reconstruction.
- Output: Dense point cloud, surface mesh, and texture. 

Photogrametry pipeline
- images --> SfM --> MVS
- SfM is used to determine the camera parameters (position, rotation, etc.) and a sparse 3D reconstruction, while MVS takes these known parameters as input to generate a dense 3D model
- App: construct 3D model of buildings from a set of images

### Bundle Adjustment
Non-linear least squares optimization to refine camera poses and 3D point positions to minimize reprojection error.

Inputs: 
 - 3D Points (Point Cloud): Unrefined, sparse 3D coordinates (X, Y, Z) of features identified across multiple images, often from SfM.
 - 2D Image Points (Observations): Corresponding pixel coordinates (u, v) of those 3D points as seen in each image.
 - Camera Poses (Extrinsics): Initial estimates of camera positions (translation) and orientations (rotation) for each image.
 - Camera Intrinsics: Focal length, principal point, and distortion parameters for each camera.
 - Constraints (Optional): Additional data like Ground Control Points (GCPs) or loop closure information for more accuracy. 

Outputs:
 - Refined 3D Points: More accurate 3D coordinates for the sparse point cloud, minimizing overall error.
 - Optimized Camera Poses: Precise camera positions and orientations, creating a more accurate camera trajectory.
 - Calibrated Camera Intrinsics: Improved internal camera parameters (like focal length).
 - Reprojection Error Statistics: Metrics (like Root Mean Square Error - RMSE) quantifying the final accuracy of the adjustment.

 **Gauge freedom**: I could rotate, translate and scale the whole scene (incl. cam poses) and the re-projection error would stay the same. 
    - Fix the first camera to remove the gauge freedom.




### View Graph
Assuming you have multiple images of the same object from different angles, you can create a view graph to represent the relationships between the images. 
Each node in the graph represents an image, and each edge represents the overlap between two images. 
The weight of the edge can represent the amount of overlap or the quality of the match between the two images. 
The view graph can be used to guide the image matching process and to estimate the camera poses.
Overlap = number of matched keypoints between two images.


## SLAM

### Factor Graph
Sliding-window approach: Only the most recent N frames are kept in the graph, and as new frames arrive, the oldest ones 
are removed to maintain a fixed-size window. This approach is common in real-time SLAM systems where memory and 
processing constraints are important.
Can be seen as generalization of bundle adjustment which is a special case for structure-from-motion.

### Relocalization

### IMU Preintegration



## Scene Representation

### Truncated Signed Distance Function (TSDF)
Truncated Signed Distance Fields (TSDF) are a widely used volumetric representation in 3D reconstruction, robotics, 
and computer vision.  They encode, at each voxel in a 3D grid, the signed distance to the nearest observed surface, 
truncated to a fixed range (e.g., ±ϵ) around the surface.  This truncation ensures that only reliable surface 
information is retained, reducing noise and enabling smooth surface reconstruction from noisy depth data like that 
from RGB-D sensors (e.g., Kinect, RealSense). 

Voxel grids serve as the underlying spatial structure for TSDFs.  A voxel grid partitions 3D space into a regular array 
of cubic cells (voxels), where each voxel stores a TSDF value (distance), a weight (confidence), and optionally color 
or normal data. The grid can be dense (uniform resolution) or sparse (using hierarchical structures like octrees or 
hash tables) to manage memory and computational costs. Sparse voxel grids, such as those implemented via voxel hashing 
or octree-based representations, are essential for scalable, real-time mapping in large environments. 


### Neural Radiance Fields (NeRF)

### Gaussian Splatting

Gaussian Splatting does not involve a neural network.  It represents 3D scenes using millions of optimized 
3D Gaussians—each defined by position, covariance (shape and orientation), opacity, and color (often using spherical 
harmonics for view-dependent lighting).  The optimization process uses stochastic gradient descent to refine these 
Gaussian parameters, but it does not rely on neural networks or MLPs.  This explicit, geometric representation allows 
for real-time rendering and is fundamentally different from NeRF, which uses a neural network to implicitly model 
the scene.

Spherical harmonics SH: basis functions for representing functions on a sphere. 
They are the spherical analog of Fourier series and can be used to represent lighting and reflections in a compact and efficient way.
Used to model view-dependent appearance of a surface point.


## 🚧 Links
- real SfM systems (COLMAP, OpenMVG, VisualSfM)
- bundle adjustment libs (ceres, g2o, or pyceres)
- Kornia: A PyTorch-based library that integrates DISK, SIFT, LoFTR, and other feature detection/matching tools.