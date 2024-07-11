# IO-ULTRA-EMBODIMENT-DATASET

## Camera and Depth Image Specifications

The image dimensions and formats for the primary, left, right, depth, and fisheye cameras are as follows:

- **Primary Camera (cam_rgb)**
  - Image dimensions: 1920 x 1080 pixels
  - Image format: JPEG

- **Left Camera (cam_left)**
  - Image dimensions: 1920 x 1080 pixels
  - Image format: JPEG

- **Right Camera (cam_right)**
  - Image dimensions: 1920 x 1080 pixels
  - Image format: JPEG

- **Depth Camera (cam_depth)**
  - Image dimensions: 1280 x 800 pixels
  - Image format: PNG

- **Fisheye Camera (cam_fisheye)**
  - Image dimensions: 1280 x 720 pixels
  - Image format: JPEG

The intrinsic matrix for the primary, depth, left, right, and fisheye cameras is of size 3 x 3.

## Action Frame Specification

Each frame in the action sequence contains the position and orientation of the left and right end effectors relative to the primary camera link. The data is structured as follows:
- 3x left end effector (EEF) position (x, y, z)
- 4x left EEF orientation quaternions (w, x, y, z)
- 3x right EEF position (x, y, z)
- 4x right EEF orientation quaternions (w, x, y, z)
