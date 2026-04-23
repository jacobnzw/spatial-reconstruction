# Computer Vision Notes
Goal is not explain everything formally consistenly from scratch, but rather record my own realizations and intuitions.
Many tutorials, countless papers and books already exist discussing these topics in great detail and visuals.

Prince in his book basically sets up the mathematical model of pinhole camera in 3D space, derives the projection matrix and its components (intrinsic, extrinsic). That's the basic geometry. 
Then one can see many estimations problems arising from that setup, estimation of:
- camera intrinsic matrix from 2D-3D point correspondences (camera calibration)
- camera pose (extrinsic matrix) from 2D-3D point correspondences
- 3D point positions from 2D-3D point correspondences

All of these problems are formulated probabilistically as maximum likelihood estimation and boil down to non-linear 
least squares optimization problems.


### Projective Pinhole Camera Model

In homogeneous coordinates, a point in 3D space is represented as a 4D vector $(x, y, z, w)$, where $w$ is a scaling factor. 
The point in Euclidean coordinates is then $(x/w, y/w, z/w)$. When $w=1$, the homogeneous point corresponds to a point in Euclidean space.

Pinhole camera model in homogeneous coordinates:
$$
\lambda
\begin{bmatrix}
   x \\
   y \\
   1
\end{bmatrix} = 
\begin{bmatrix}
   \phi_x & \gamma & \delta_x & 0 \\
   0 & \phi_y & \delta_y & 0 \\
   0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
   u \\
   v \\
   w \\
   1
\end{bmatrix}
$$
where $\phi_x$ and $\phi_y$ are the focal lengths in the x and y directions, $\gamma$ is the skewness between the x and y axes, 
and $(\delta_x, \delta_y)$ is the principal point (optical center) of the camera in pixel coordinates. 
The scaling factor $\lambda$ is due to the fact that the homogeneous coordinates are not unique, and can be scaled by any non-zero factor.
Effectively scaling by $\lambda$ slides the point into the image plane along the ray from the camera center to the point in 3D space.

To complete the model, we add extrinsic parameters (rotation $\mathbf{R}$ and translation $\mathbf{t}$) that define 
the position and orientation of the camera in 3D space, and thus relate the world coordinate system (frame) to 
the camera coordinate system (frame).
$$
\lambda
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix} = 
\begin{bmatrix}
   \phi_x & \gamma & \delta_x & 0 \\
   0 & \phi_y & \delta_y & 0 \\
   0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
   r_{11} & r_{12} & r_{13} & t_x \\
   r_{21} & r_{22} & r_{23} & t_y \\
   r_{31} & r_{32} & r_{33} & t \\
   0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
u \\
v \\
w \\
1
\end{bmatrix}
$$
In matrix form,
$$
\lambda
\tilde{\mathbf{x}} = 
\begin{bmatrix}
   \mathbf{K} & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
   \mathbf{R} & \mathbf{t} \\
   \mathbf{0}^T & 1
\end{bmatrix}
\tilde{\mathbf{w}} = 
\mathbf{K}
\begin{bmatrix}
   \mathbf{R} & \mathbf{t}
\end{bmatrix}
\tilde{\mathbf{w}}
$$
where $\tilde{\mathbf{x}}$ is a projection of the world point $\tilde{\mathbf{w}}$ onto the image plane expressed 
in homogeneous coordinates, and $\mathbf{K}$ is the intrinsic matrix and rotation matrix $\mathbf{R}$ 
and translation vector $\mathbf{t}$ are the extrinsic parameters that determine the camera pose in the world frame.

*In the SE3 variable naming convention the camera pose in the world frame is a transformation 
from the world frame to the camera frame: `camera_SE3_world`. 
See [this great summary](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RigidTransform.html#scipy.spatial.transform.RigidTransform).*

**Caveats:** the rays go through the optical center of the camera and form the same ray bundle geometry, 
just reflected through the optical center, on the back of the camera. For example, routines for solving a PnP problem
have to account for solutions "behind the camera" which are valid geometrically but not physically realizable. 

#### Distortion models
Camera lenses introduce radial and tangential distortion that deviate projections from being linear. 
**Brown-Conrady model** is a widely used polynomial distortion model in computer vision for correcting lens distortions, including barrel, pincushion, and mustache distortions. 
While it is commonly applied to standard and wide-angle lenses, it is less accurate for fisheye lenses with very large fields of view (FOV > 180°), where models like **Kannala-Brandt** or fisheye-specific models are preferred.


### [Homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
Pinhole camera views a plane in the world. 
There is a unique homography matrix H that maps points in the world plane to the image plane.
It's therefore a family of 2D-to-2D *projective transformations*.
Special cases: rotation, translation, similarity, uniform scaling, shearing.
Lines that were parallel are not constrained to remain parallel in the image plane.
Linear in homogeneous coordinates, nonlinear in Cartesian coordinates.
$$
\begin{align*}
\lambda
\begin{bmatrix}
   x \\
   y \\
   1
\end{bmatrix} 
&= 
\begin{bmatrix}
   \phi_x & \gamma & \delta_x \\
   0 & \phi_y & \delta_y \\
   0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
   r_{11} & r_{12} & r_{13} & t_x \\
   r_{21} & r_{22} & r_{23} & t_y \\
   r_{31} & r_{32} & r_{33} & t_z
\end{bmatrix}
\begin{bmatrix}
   u \\
   v \\
   0 \\
   1
\end{bmatrix} \\
&= 
\begin{bmatrix}
   \phi_x & \gamma & \delta_x \\
   0 & \phi_y & \delta_y \\
   0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
   r_{11} & r_{12} & t_x \\
   r_{21} & r_{22} & t_y \\
   r_{31} & r_{32} & t_z
\end{bmatrix}
\begin{bmatrix}
   u \\
   v \\
   1
\end{bmatrix} \\
\tilde{\mathbf{x}} &= \mathbf{H} \tilde{\mathbf{w}}
\end{align*}
$$

#### Homography Estimation
Because homography is a $3\times 3$ matrix with $8$ degrees of freedom, we need at least $4$ point correspondences 
(pairs of points) to estimate it.
It's a nonlinear optimization problem, no closed-form solution, thus need to rely on gradient based iterative methods,
that need a good starting estimate. This is often obtained by *direct linear transform (DLT)* method, which uses 
homogeneous coordinate formulation of the problem to arrive at a closed-form solution which serves as a good initialization.



## Multi-View Geometry
When multiple cameras observe the same scene, the geometry of the problem changes. 
The relative positions and orientations of the cameras become important and give rise to new geometric entities and constraints.
For example, a ray through the optical center of camera A and a point in the image plane of camera A, where the 3D point X from the world projects to, defines a line in 3D space. When the same 3D world point X is viewed in the image plane of camera B, ray from camera A becomes a line across the image plane of camera B, called an **epipolar line**.
Same holds the other way around: basically, points in one camera's image plane become lines in the other camera's image plane.

**The epipolar constraint** states that the 3D point must lie on this epipolar line.
This reduces the search for the 3D point from a 3D space to a 1D line, significantly simplifying the problem.

[Epipolar Geometry](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html)

### [Essential matrix](https://en.wikipedia.org/wiki/Essential_matrix)
Mathematical constraint between two views of the same scene captured by normalized cameras. 
$$
   \tilde{\mathbf{x}}_1^T \mathbf{E} \tilde{\mathbf{x}}_2 = 0
$$
where $\tilde{\mathbf{x}}_1, \tilde{\mathbf{x}}_2 \in \mathbb{R}^3$ are the homogeneous coordinates of the 
corresponding 2D points in the two image planes, and 
$$ 
   \mathbf{E} = \mathbf{t}_\times \mathbf{R} 
$$ 
is the essential matrix, where $\mathbf{t}_\times$ 
is the skew-symmetric matrix representation of the cross-product operation
$$
   \mathbf{t}_\times = 
   \begin{bmatrix}
      0 & -t_z & t_y \\
      t_z & 0 & -t_x \\
      -t_y & t_x & 0
   \end{bmatrix}
$$
and $\mathbf{R}$ is the rotation matrix.
The rotation and translation are a relative pose of camera B wrt. camera A, which is assumed to be at the 
origin with no rotation (identity rotation matrix and zero translation vector).

#### Properties
- Rank 2: $\text{rank}(\mathbf{E}) = 2$
- Singular values: $\sigma_1 = \sigma_2 > 0, \sigma_3 = 0$
- Determinant: $\det(\mathbf{E}) = 0$
- Operates on homogeneous coordinates: $\tilde{\mathbf{x}} \in \mathbb{R}^3$
- 7 DOF: 3 for rotation, 3 for translation, 1 for scale
- Scale ambiguity: Multiplying all entries by any constant does not change its properties.

### [Fundamental matrix](https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision))
Fundamental matrix plays role of the essential matrix for cameras with arbitrary intrinsics $\mathbf{K}_1, \mathbf{K}_2$.
The realationship to essential matrix is
$$
   \mathbf{E} = \mathbf{K}_2^T \mathbf{F} \mathbf{K}_1 \ \Leftrightarrow\  F = \mathbf{K}_2^{-1} \mathbf{E} \mathbf{K}_1^{-1}
$$

*Eight-point algorithm* is used to estimate the fundamental matrix from a set of 8 pairs of points (correspondences).


iterative projection matching
reprojection error
photometric error
Sampson error for epipolar geometry
Umeyama algorithm for rigid transformation estimation with known correspondences



## Stereo Principle
(from greek *stereos* = solid, thus relating to solid (3D) forms)

The core idea is that a 3D point in the world projects to different locations in images taken from different viewpoints.
This difference in location, known as disparity or parallax, is inversely proportional to its depth. 
By calculating disparity for numerous points and using the known camera geometry (position and orientation), 
the 3D position of those points can be determined through a process called triangulation.


### Perspective n-Point (PnP)
Given 3D points in the world frame, their corresponding 2D projections in the image plane and known camera intrinsics, 
estimate the camera pose (rotation and translation) that best projects the 3D points to the 2D image points.

**EPnP** estimates camera pose using n 3D-to-2D point correspondences by representing the 3D points as a weighted sum 
of four virtual control points, enabling linear-time computation (O(n)). Non-iterative, closed-form solution. 
Possibly followed by iterative refinement (Gauss-Newton, Levenberg-Marquardt).

The minimum number of points required for solvePnP depends on the specific algorithm method used and the geometric configuration of the points:

- General Case (Non-Planar): Most methods, including EPnP, DLS, UPnP, and SQPnP, require a minimum of 4 points. 
- P3P Methods (SOLVEPNP_P3P, SOLVEPNP_AP3P): These require exactly 4 points.  The first three are used to solve the Perspective-Three-Point problem (which can yield up to four solutions), and the fourth point is used to select the best solution by minimizing reprojection error.
- Iterative Method (SOLVEPNP_ITERATIVE):
  - With an initial guess (useExtrinsicGuess=true), a minimum of 3 points is sufficient.
  - Without an initial guess, it requires at least 4 points for planar objects (using homography decomposition) or 6 points for non-planar objects (using DLT initialization). 
- Planar-Specific Methods:
  - IPPE and IPPE_SQUARE: Require a minimum of 4 points, and the object points must be coplanar.
  - SQPnP: Requires 3 or more points. 

Important Note: Using points that are collinear (lying on a single line) or have insufficient variation in 3D space can 
lead to inaccurate results. It is recommended to use points with good variation across at least two axes.


#### Keyframes
Codec keyframes (I-frames) are built into the video and used for decompression. 
Computer vision keyframes are extracted using models or algorithms to capture meaningful moments, 
which may or may not align with I-frames. 
In practice, vision systems may use codec keyframes as a starting point but refine selection using content-aware 
models for higher relevance.

Keyframes are crucial in computer vision processing of videos because they reduce data redundancy and computational 
overhead while preserving essential visual information. Videos typically contain hundreds or thousands of frames per 
minute, many of which are highly similar or contain minimal new information. By selecting only the most representative 
frames—those that capture significant changes in content, motion, or structure—keyframes enable efficient processing 
for tasks like video summarization, action recognition, and visual localization.

### Keypoints 
#### Extraction
- SIFT expensive but robust (invariant to image scaling, rotation, and translation; doesn't work well with changes in 
lighting or illumination, doesn't work with color; CSIFT addresses this)
- ORB fast but less robust (real-time)
- AKAZE is a faster and more accurate alternative to SIFT, also invariant to illumination changes and point of view.
- DISK is a modern, deep learning-based local feature detector and descriptor designed for robust image matching.  
It is particularly effective for outdoor scenes and performs exceptionally well when combined with the LightGlue 
matcher, as shown in the IMC2021 benchmark.
- LoFTR: A detector-free, transformer-based model that works best for indoor scenes.

#### Matching
FLANN, Brute Force
In image keypoint matching, inliers are the correct, geometrically consistent correspondences between keypoints in two 
different images, meaning they belong to the same physical points and follow the same transformation (like rotation, 
scale, or perspective). They contrast with outliers, which are incorrect matches (e.g., matching a window to a door 
due to similar appearance) that don't conform to the overall scene transformation, and are typically filtered out using 
algorithms like RANSAC (Random Sample Consensus).

Lessons from debugging:
- Query set vs. train set
- TODO: lessons from using matchers:
  - several queries to one train, directionality matters,  
  - BFMatcher crossCheck=True vs. False; without cross-check, multiple queries can match to the same train, with 
  cross-check, only one-to-one matches are allowed. 
  In case of many-to-one, we might have one KP triangulated to multiple 3D points, which is geometrically impossible.


