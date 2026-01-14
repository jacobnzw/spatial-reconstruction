
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



## 🚧 Multi-view Stereo (MVS) Pipeline
= dense reconstruction from sparse 3D points and images

## 🚧 Links
- real SfM systems (COLMAP, OpenMVG, VisualSfM)
- bundle adjustment libs (ceres, g2o, or pyceres)

## Reading List
TODO: list papers to read