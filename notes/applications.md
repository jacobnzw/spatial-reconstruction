## Applications
### Structure from motion (SfM)
- given a set of overlapping images of a static scene, estimate camera poses and 3D points from 2D image points
- Goal: Estimate camera poses (position/orientation) and a sparse 3D point cloud.
- Input: A set of images (views) of an object that overlap.
  - Each 3D point is typically seen in ≥ 2 images, Often 3–10 images in practice
- Process: Matches image features (like SIFT) across views to estimate camera movement and a basic 3D point cloud.
- Output: Camera parameters and a sparse point cloud. 

Two main approaches:
- [Global SfM](https://arxiv.org/pdf/2407.20219v1): simultaneous estimation of all camera poses and 3D points
- Incremental SfM: estimate poses and 3D points incrementally, one image at a time


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

Factor graphs are a very general framework subsuming a wide range of optimization problems:

- Bundle adjustment
  - Factor: reprojection error (between the 3D points and their corresponding 2D image points)
  - Variables (unknowns): camera poses and 3D points, 
  - Parameters (observations): .

- Pose graph optimization
  - The variables are the robot poses, and the factors are the relative pose measurements between pairs of poses (odometry or loop closures).

- Visual-inertial odometry
  - The variables are the robot poses, velocities, and IMU biases, and the factors are parameterized by the IMU measurements and visual observations.

<!-- TODO: example of factor graph for 3D reconstruction, show how factor graph is reflected in the terms of the objective function -->

Sliding-window approach: Only the most recent N frames are kept in the graph, and as new frames arrive, the oldest ones 
are removed to maintain a fixed-size window. This approach is common in real-time SLAM systems where memory and 
processing constraints are important. 
Structurally isomorphic to fixed-lag smoothing.


### Pose Graph
Pose graph optimization (PGO) is a specific case of factor graph optimization where the variables are poses (rigid transformations) 
and the factors are relative pose constraints (odometry measurements or loop closures). The goal is to find a consistent 
configuration of poses that satisfies all constraints.

A pose graph is a factor graph whose variables are poses and whose measurements are relative measurements between pairs of poses. 
Optimizing a pose graph means determining the configuration of poses that is maximally consistent with the measurements. 
PGO is very common in the SLAM community, and several ad-hoc approaches have been proposed. 
Similar to BA, PGO is highly non-convex, and its solution with ILS requires a reasonably good initial guess.


### [Structure from Motion (SfM) via GTSAM](https://gtsam-jlblanco-docs.readthedocs.io/en/latest/StructureFromMotion.html)
Important note: a very tricky and difficult part of making SFM work is (a) data association, and (b) initialization. 
*GTSAM does neither of these things for you*: it simply provides the “bundle adjustment” optimization. 
In the example, we simply assume the data association is known (it is encoded in the J sets), and we initialize with 
the ground truth, as the intent of the example is simply to show you how to set up the optimization problem.


### Relocalization
Assumption: Map is known.
Solve the kidnapped robot problem: given a map of the environment and a set of sensor measurements, determine the robot's pose in the map.


### Loop Closure
Loop closure is the process of recognizing that the robot has returned to a previously visited location, 
and using this information to correct the accumulated drift in the estimated trajectory. 
Loop closure is an important component of SLAM systems, as it allows for more accurate mapping and localization.


### [IMU Preintegration](https://docs.openvins.com/propagation.html)

See also [OpenIMU docs](https://openimu.readthedocs.io/en/latest/algorithms/STM_Quaternion.html).

NED Coordinate Frame: North-East-Down coordinate frame

IMU delivers measurements at a orders of magnitude higher rate (100-1000 Hz) than the camera images (10-30 Hz).

IMU does not observe the pose directly, but rather the angular velocity $ \boldsymbol{\omega} $ and linear acceleration $\mathbf{a}$ which need to be integrated over time to obtain the pose $\mathbf{p}$.

<!-- TODO: needs fleshing out: formulate model, describe everything, then continue with the estimation algorithm on that model -->

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


$$
\begin{align*}
  \dot{\bar{\mathbf{q}}}(t) &= \frac{1}{2} \bar{\mathbf{q}}(t) \otimes \begin{bmatrix} \mathbf{0} \\ \boldsymbol{\omega}(t) \end{bmatrix} \\
  \dot{\mathbf{p}}(t) &= \mathbf{v}(t) \\
  \dot{\mathbf{v}}(t) &= \mathbf{R}(\bar{\mathbf{q}}(t))^\top \mathbf{a}(t) + \mathbf{g} \\
  \dot{\mathbf{b}}_g(t) &= \mathbf{n}_{wg}(t) \\
  \dot{\mathbf{b}}_a(t) &= \mathbf{n}_{wa}(t) 
\end{align*}
$$

Between keyframes the IMU biases are assumed constant, and the IMU measurements are integrated to compute the relative pose, velocity, and biases between the two keyframes.

Solution to the IMU kinematics is given by the following equations:
$$
\begin{align*}
  \Delta \mathbf{R}_{i+1} &= \Delta \mathbf{R}_i \cdot \exp([\tilde{\boldsymbol{\omega}}_i \Delta t]_\times) \\
  \Delta \mathbf{v}_{i+1} &= \Delta \mathbf{v}_i + \Delta \mathbf{R}_i \cdot \tilde{\mathbf{a}}_i \Delta t \\
  \Delta \mathbf{p}_{i+1} &= \Delta \mathbf{p}_i + \Delta \mathbf{v}_i \Delta t + \frac12 \Delta \mathbf{R}_i \cdot \tilde{\mathbf{a}}_i \Delta t^2 
\end{align*}
$$

These need to be fed with estimates of the true angular velocity and linear acceleration, which are obtained from bias estimates:
$$
\begin{align*}
  \tilde{\boldsymbol{\omega}}_i &= \boldsymbol{\omega}_{m,i} - \hat{\mathbf{b}}_g \\
  \tilde{\mathbf{a}}_i &= \mathbf{a}_{m,i} - \hat{\mathbf{b}}_a
\end{align*}
$$
The IMU bias estimates are held constant for the duration of the preintegration between keyframes.
However, the IMU bias estimates are actively updated using EKF (or other filter) running independently.