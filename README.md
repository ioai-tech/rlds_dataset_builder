# RLDS Dataset Conversion

## Overview

This repo demonstrates how to convert **IO Ultra Embodiment Dataset** into **RLDS** format for X-embodiment experiment integration.

**IO Ultra Embodiment Dataset** provides comprehensive multimodal data for robotics and Embodied AI research and related applications. It includes multiple high-resolution RGB and depth images with camera intrinsic and extrinsic parameters, joint state data, dual end-effectors poses, fingers haptics and language-based instructions. 

## Dataset Structure

### RGB Camera Images:

- image: RGB observation from the main camera (shape: 1080x1920x3).
- image_left: RGB observation from the left camera (shape: 1080x1920x3).
- image_right: RGB observation from the right camera (shape: 1080x1920x3).
- image_fisheye: RGB observation from the fisheye camera (shape: 1080x1920x3).

### Depth Images:

- depth: Depth observation from the main camera (shape: 400x640x1).

### Camera Intrinsic Parameters
Parameters for each camera include focal lengths (fx, fy) and principal points (cx, cy).

### Camera Extrinsic Parameters
Transformation matrices (4x4) that describe the spatial relationship between various cameras.


### Joint States
177-dimensional tensor representing the joint states.

### Fingers Haptics
10x96 tensor capturing the haptic feedback from the robot's fingers.

### End-Effector Poses
14-dimensional tensor describing the end-effector poses for the robot's arms, including positions and orientations.

### language_instruction
Text instructions corresponding to tasks.


## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds_env
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `tensorflow_hub`, `apache_beam`, `matplotlib`, `plotly` and `wandb`.


## Run Example RLDS Dataset Creation

Before modifying the code to convert your own dataset, run the provided example dataset creation script to ensure
everything is installed correctly. Run the following lines to create some dummy data and convert it to RLDS.
```
cd example_dataset
python3 create_example_data.py
tfds build
```

This should create a new dataset in `~/tensorflow_datasets/example_dataset`. Please verify that the example
conversion worked before moving on.


## Converting **IO Ultra Embodiment Dataset** to RLDS

Now we can modify the provided example to convert your own data. Follow the steps below:

1. **Modify Dataset Splits**: The function `_split_generator()` determines the splits of the generated dataset (e.g. training, validation etc.).
If your dataset defines a train vs validation split, please provide the corresponding information to `_generate_examples()`, e.g. 
by pointing to the corresponding folders (like in the example) or file IDs etc. If your dataset does not define splits,
remove the `val` split and only include the `train` split. You can then remove all arguments to `_generate_examples()`.

2. **Run Dataset Conversion Code**:
```
cd io_ultra_embodiment_dataset
tfds build --overwrite
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/io_ultra_embodiment_dataset`.


### Parallelizing Data Processing
By default, dataset conversion is single-threaded. If you are parsing a large dataset, you can use parallel processing.
For this, replace the last two lines of `_generate_examples()` with the commented-out `beam` commands. This will use 
Apache Beam to parallelize data processing. Before starting the processing, you need to install your dataset package 
by filling in the name of your dataset into `setup.py` and running `pip install -e .`

Then, make sure that no GPUs are used during data processing (`export CUDA_VISIBLE_DEVICES=`) and run:
```
tfds build --overwrite --beam_pipeline_options="direct_running_mode=multi_processing,direct_num_workers=10"
```
You can specify the desired number of workers with the `direct_num_workers` argument.

## Visualize Converted Dataset
To verify that the data is converted correctly, please run the data visualization script from the base directory:
```
python3 visualize_dataset.py io_ultra_embodiment_dataset
``` 
This will display a few random episodes from the dataset with language commands and visualize action and state histograms per dimension.
Note, if you are running on a headless server you can modify `WANDB_ENTITY` at the top of `visualize_dataset.py` and 
add your own WandB entity -- then the script will log all visualizations to WandB. 


