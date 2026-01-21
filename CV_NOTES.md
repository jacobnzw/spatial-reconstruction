# Research Notes on 3D Computer Vision Fundamentals


KDTree
Vocabulary-tree matching

Pinhole Camera Model
homography
Homography Estimation
## Multi-View Geometry: 
[Essential matrix](https://en.wikipedia.org/wiki/Essential_matrix), 
[Fundamental matrix](https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)),
[Epipolar Geometry](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html)
singular value decomposition
bundle adjustment
reprojection error
photometric error

### Perspective n-Point (PnP)
Given 3D points in the world frame, their corresponding 2D projections in the image plane and known camera intrinsics, estimate the camera pose (rotation and translation) that best projects the 3D points to the 2D image points.

### Keypoint Extraction
- SIFT expensive but robust (invariant to image scaling, rotation, and translation)
- ORB fast but less robust (real-time)

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


### View Graph
Assuming you have multiple images of the same object from different angles, you can create a view graph to represent the relationships between the images. 
Each node in the graph represents an image, and each edge represents the overlap between two images. 
The weight of the edge can represent the amount of overlap or the quality of the match between the two images. 
The view graph can be used to guide the image matching process and to estimate the camera poses.