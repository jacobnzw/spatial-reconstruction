## Data Structures

**KDTree** is a space-partitioning data structure for organizing points in a k-dimensional space. It is used to partition a space into cells, which can be searched efficiently. It is used for fast nearest neighbor search, which is a common operation in computer vision.

**Vocabulary-tree** is a tree data structure that is used to cluster visual words. 
A visual word is a descriptor that has been clustered into a vocabulary tree. 
It is a representative of a group of similar descriptors. 
It is used to compress and index large-scale image collections for fast image retrieval, 
which is a common operation in computer vision. 


## Mathematical Tooling
Nonlinear least squares problems are notoriously recurring in formulations of many 3D CV problems, 
such as bundle adjustment, structure-from-motion, SLAM, etc.

### Singular Value Decomposition
The most common matrix decomposition in existence with countless applications.

### RANSAC
RAndom Sample Consensus is an algorithm for robustly fitting a model to a set of data in the presence of outliers.
   1. Select a random subset of the data.
   2. Fit the model to the subset.
   3. Count the number of inliers.
   4. Repeat steps 1-3 for a number of iterations.
   5. Keep the model with the most inliers.

### Iterative Optimization
An iterative methods of solving non-linear least squares problems.
<!-- TODO: formulas are sus, check extensively -->
#### Gauss-Newton
The iterates are given by
$$
 \mathbf{x}_{k+1} = \mathbf{x}_k + \left( \mathbf{J}^T \mathbf{J} \right)^{-1} \mathbf{J}^T \mathbf{r}
$$
#### Levenberg-Marquardt
The iterates are given by
$$
 \mathbf{x}_{k+1} = \mathbf{x}_k + \left( \mathbf{J}^T \mathbf{J} + \lambda \mathbf{I} \right)^{-1} \mathbf{J}^T \mathbf{r}
$$




## Representations of Rigid Transformations

#### Euler Angles
A rotation can be represented by three angles (roll, pitch, yaw) 
that specify the rotation around each of the three (x,y,z) axes.

#### SO(3)
 - the group of rotations in 3D space.
 - 3 degrees of freedom
 - A rotation in SO(3) can be represented as a 3x3 rotation matrix $R$ with the following properties:
    - $R^T R = I$
    - $\det(R) = 1$
    - $R \in \mathbb{R}^{3\times 3}$

##### Trace of a rotation matrix
defined as the sum of its main diagonal entries, is invariant under coordinate system changes and equals the sum of its eigenvalues. This property allows the rotation angle to be directly extracted from the matrix elements without needing to identify the rotation axis first. 

2D Rotation Matrices
For a 2×2 rotation matrix representing a rotation by angle $\theta$, the trace is: 
Tr(R)=2cosθ This formula implies that the angle of rotation can be calculated as 
$\theta = \arccos\left(\frac{\text{tr}(R)}{2}\right)$. 

3D Rotation Matrices
For a 3×3 rotation matrix representing a rotation by angle $\theta$ around any axis, the trace is: 
$$
\begin{equation}
   \text{Tr}(R) = 1 + 2\cos\theta
\end{equation}
$$
Consequently, the rotation angle is determined by: 
$$
\begin{equation}
   \cos\theta = 2\text{Tr}(R) − 1 \ \Rightarrow\  \theta = \arccos(2\text{Tr}(R) − 1)
\end{equation}
$$
The eigenvalues for a 3D rotation are $1, e^{i\theta}, e^{-i\theta}$, and their sum $1 + 2\cos\theta$ confirms the trace formula. 


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

#### Sim(3)
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

#### Axis-angle
A rotation can be represented by an axis of rotation and an angle of rotation around that axis.

#### Quaternion
A rotation can be represented by a unit quaternion, which is a 4-dimensional vector 
$(w, x, y, z)$ with the following properties:

   - $w^2 + x^2 + y^2 + z^2 = 1$
   - $w, x, y, z \in \mathbb{R}$

**Advantages in 3D Applications**: Compared to rotation matrices and Euler angles, 
quaternions are more compact (4 numbers vs. 9 or 3), computationally efficient, avoid gimbal lock, 
and allow smooth interpolation (e.g., spherical linear interpolation, or slerp).

Preferable for interpolation, because it can be easily re-normalized if it gets perturbed by many sucessive transformations. 
Avoids gimbal lock.

#### Dual Quaternions
<!-- TODO: verify -->
8-dimensional numbers formed by combining two quaternions using a dual unit $ \varepsilon $ (where $ \varepsilon^2 = 0 $). They are written as $ \hat{q} = q_r + \varepsilon q_d $, where:
- $ q_r $: represents rotation (like a normal quaternion),
- $ q_d $: encodes translation (related to position).

Can represent a full 6DOF rigid transformation (rotation + translation) with no gimbal lock and easy interpolation.





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
- opencv has tutorials on basic 3D CV tasks