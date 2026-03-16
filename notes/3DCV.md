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


### Pinhole Camera Model
Other used camera models: Brown-Conrady model (fisheye), Kannala-Brandt model (fisheye), Dual Quaternion model (SfM), Catadioptric model (mirror+lens), Omnidirectional model (fish-eye+mirror)
Projection matrix, camera matrix, intrinsic matrix, extrinsic matrix

Field of view, aspect ratio, principal point, focal length, skew, distortion

**Caveats:** the rays go through the optical center of the camera and form the same ray bundle geometry, 
just reflected through the optical center, on the back of the camera. For example, routines for solving a PnP problem
have to account for solutions "behind the camera" which are valid geometrically but not physically realizable. 

## Multi-View Geometry
When multiple cameras observe the same scene, the geometry of the problem changes. 
The relative positions and orientations of the cameras become important and give rise to new geometric entities and constraints.
For example, a ray through the optical center of camera A and a point in the image plane of camera A, where the 3D point X from the world projects to, defines a line in 3D space. When the same 3D world point X is viewed in the image plane of camera B, ray from camera A becomes a line across the image plane of camera B, called an **epipolar line**.
Same holds the other way around: basically, points in one camera's image plane become lines in the other camera's image plane.

**The epipolar constraint** states that the 3D point must lie on this epipolar line.
This reduces the search for the 3D point from a 3D space to a 1D line, significantly simplifying the problem.

[Epipolar Geometry](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html)

[Essential matrix](https://en.wikipedia.org/wiki/Essential_matrix)
what does it tell me? basic properties? decomposition into R,t?

[Fundamental matrix](https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision))
what does it tell me? basic properties? relation to Essential matrix?
Enhanced version of the essential matrix that accounts for camera intrinsics.

[Homography](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
Homography Estimation


iterative projection matching
reprojection error
photometric error
Sampson error for epipolar geometry
Umeyama algorithm for rigid transformation estimation with known correspondences



#### Keyframes
Codec keyframes (I-frames) are built into the video and used for decompression. 
Computer vision keyframes are extracted using models or algorithms to capture meaningful moments, 
which may or may not align with I-frames. 
In practice, vision systems may use codec keyframes as a starting point but refine selection using content-aware 
models for higher relevance.

Keyframes are crucial in computer vision processing of videos because they reduce data redundancy and computational overhead while preserving essential visual information. Videos typically contain hundreds or thousands of frames per minute, many of which are highly similar or contain minimal new information. By selecting only the most representative frames—those that capture significant changes in content, motion, or structure—keyframes enable efficient processing for tasks like video summarization, action recognition, and visual localization.


## Stereo Principle
(from greek *stereos* = solid, thus relating to solid (3D) forms)

The core idea is that a 3D point in the world projects to different locations in images taken from different viewpoints.
This difference in location, known as disparity or parallax, is inversely proportional to its depth. 
By calculating disparity for numerous points and using the known camera geometry (position and orientation), 
the 3D position of those points can be determined through a process called triangulation.


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


