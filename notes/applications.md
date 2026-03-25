## Applications
### Structure from motion (SfM)
- given a set of overlapping images of a static scene, estimate camera poses and 3D points from 2D image points
- Goal: Find camera poses (position/orientation) and a sparse 3D structure.
- Input: A set of overlapping images.
- Process: Matches image features (like SIFT) across views to estimate camera movement and a basic 3D point cloud.
- Output: Camera parameters and a sparse point cloud. 

### Multi-view Stereo (MVS)
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

 **Gauge Sim(3) ambiguity**: I could rotate, translate and scale the whole scene (incl. cam poses) and the re-projection error would stay the same. 
    - Fix the first camera pose (e.g. at origin) + distance between first two cameras to remove the gauge freedom.


### View Graph
Assuming you have multiple images of the same object from different angles, you can create a view graph to represent the relationships between the images. 
Each node in the graph represents an image, and each edge represents the overlap between two images. 
The weight of the edge can represent the amount of overlap or the quality of the match between the two images. 
The view graph can be used to guide the image matching process and to estimate the camera poses.
Overlap = number of matched keypoints between two images.

### Visual-Inertial Odometry (VIO)

## SLAM

### Factor Graph
A factor graph is a bipartite graph that represents the factorization of a function. 
Two types of nodes: variables and factors (functions). The edges indicate which variables participate in which factors.
The variable nodes are typically drawn as circles and the factor nodes as squares.
In the SLAM context, the variables are the unknown quantities we want to estimate (poses, points, velocities) and 
the factors are the constraints or measurements that provide information about the variables.

In the context of SLAM, the function is the error between the predicted measurements and the actual measurements. 
Factor graphs represent the problem as a set of variables (poses, points, velocities) and factors (constraints, measurements, priors) 
between them. The goal is to find the variable values that best satisfy all the factors, typically by minimizing an objective 
function that measures the error between the factors and the variables.
Factor graph optimization is a generalization of bundle adjustment that can handle a wider range of problems in SLAM and computer vision. 

<!-- TODO: example of factor graph for 3D reconstruction, show how factor graph is reflected in the terms of the objective function -->


Sliding-window approach: Only the most recent N frames are kept in the graph, and as new frames arrive, the oldest ones 
are removed to maintain a fixed-size window. This approach is common in real-time SLAM systems where memory and 
processing constraints are important.

Can be seen as generalization of bundle adjustment which is a special case for structure-from-motion.

### Pose Graph
Pose graph optimization (PGO) is a specific case of factor graph optimization where the variables are poses (rigid transformations) 
and the factors are relative pose constraints (odometry measurements or loop closures). The goal is to find a consistent 
configuration of poses that satisfies all constraints.

A pose graph is a factor graph whose variables are poses and whose measurements are relative measurements between pairs of poses. Optimizing a pose graph means determining the configuration of poses that is maximally consistent with the measurements. PGO is very common in the SLAM community, and several ad-hoc approaches have been proposed. Similar to BA, PGO is highly non-convex, and its solution with ILS requires a reasonably good initial guess.

### Relocalization

### IMU Preintegration
IMU delivers measurements at a orders of magnitude higher rate (100-1000 Hz) than the camera images (10-30 Hz).

IMU does not observe the pose directly, but rather the angular velocity $ \boldsymbol{\omega} $ and linear acceleration $\mathbf{a}$ which need to be integrated over time to obtain the pose.

<!-- TODO: needs verification -->
The continuous-time dynamcs are:
$$
   \begin{align*}
      \dot{\mathbf{q}} &= \frac{1}{2} \mathbf{q} \otimes \boldsymbol{\omega} \\
      \dot{\mathbf{v}} &= \mathbf{a} \\
      \dot{\mathbf{p}} &= \mathbf{v} 
   \end{align*}
$$
<!-- Analytically intractable due to noise and nonlinearity, and computationally expensive if done naively at such high rate. -->


IMU preintegration is a technique to compute the IMU measurements in the local frame of the IMU. 
It is used to compute the relative pose between two keyframes $i$ and $j$