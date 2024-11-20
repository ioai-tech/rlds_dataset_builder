# IO-ULTRA-EMBODIMENT-DATASET

## Camera and Depth Image Specifications

The image dimensions and formats for the primary, left, right, depth, and fisheye cameras are as follows:

- **Primary Camera (image)**
  - Image dimensions: 1920 x 1080 x 3

- **Left Camera (image_left)**
  - Image dimensions: 1920 x 1080 x 3

- **Right Camera (image_right)**
  - Image dimensions: 1920 x 1080 x 3

- **Depth Camera (depth)**
  - Image dimensions: 640 x 400 x 3

- **Fisheye Camera (image_fisheye)**
  - Image dimensions: 1920 x 1080 x 3

## Camera Intrinsic Parameters
Parameters for each camera include focal lengths (fx, fy) and principal points (cx, cy).

## Camera Extrinsic Parameters
Transformation matrices (4x4) that describe the spatial relationship between various cameras.

## Joint States
1x177 tensor representing the joint states.

## Fingers Haptics
10x96 tensor capturing the haptic feedback from the gloves fingers.

## End-Effector Poses

Each frame in the action sequence contains the position and orientation of the left and right end effectors relative to the primary camera link. The data is structured as follows:
- 3x left end effector (EEF) position (x, y, z)
- 4x left EEF orientation quaternions (w, x, y, z)
- 3x right EEF position (x, y, z)
- 4x right EEF orientation quaternions (w, x, y, z)
