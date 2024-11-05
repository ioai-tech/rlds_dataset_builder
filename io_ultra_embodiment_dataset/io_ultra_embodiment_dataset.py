from typing import Iterator, Tuple, Any

import os
import json
import numpy as np
from pathlib import Path
import importlib
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class IoUltraEmbodimentDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for IO-ULTRA-EMBODIMENT-DATASET."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="IO-ULTRA-EMBODIMENT-DATASET: An egocentric, real-world dataset of embodied intelligence manipulation.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(1080, 1920, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "depth": tfds.features.Image(
                                        shape=(400, 640, 1),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Main camera depth observation.",
                                    ),
                                    "image_left": tfds.features.Image(
                                        shape=(1080, 1920, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Left camera RGB observation.",
                                    ),
                                    "image_right": tfds.features.Image(
                                        shape=(1080, 1920, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Right camera RGB observation.",
                                    ),
                                    "image_fisheye": tfds.features.Image(
                                        shape=(1080, 1920, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Fisheye camera observation.",
                                    ),
                                    "main_rgb_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Main RGB camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "main_depth_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Main depth camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "left_camera_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Left camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "right_camera_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Right camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "fisheye_camera_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Fisheye camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "fingers_haptics": tfds.features.Tensor(
                                        shape=(10, 96),
                                        dtype=np.float32,
                                        doc="The haptic data of ten fingers (resolution of 12 * 8).",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Robot end effector pose based on main RGB camera link, consists of [3x left EEF position, 4x left EEF orientation quaternions,"
                                "3x right EEF position, 4x right EEF orientation quaternions].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(path="data/train"),
            "val": self._generate_examples(path="data/val"),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator for each split."""
        pd = importlib.import_module("pandas")
        Image = importlib.import_module("PIL.Image")

        def _read_images(image_root):
            if not image_root.exists():
                return []
            image_files = sorted(
                os.listdir(image_root),
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            images = []

            # Define how to load a single image
            def load_image(file):
                return np.array(Image.open(os.path.join(image_root, file)))

            # Use ThreadPoolExecutor to load images in parallel
            # Adjust max_workers based on your hardware and requirements
            with ThreadPoolExecutor(max_workers=None) as executor:
                images = list(executor.map(load_image, image_files))

            return images

        def _read_csv_data(csv_path):
            if not csv_path.exists():
                return pd.DataFrame()
            return pd.read_csv(csv_path)

        def _read_json_data(json_path):
            if not json_path.exists():
                return ""
            with open(json_path, "r") as f:
                data = json.load(f)
            return data["description"]

        def _parse_example(episode_path: str):
            episode_path = Path(episode_path)

            # Define paths for different image types and data files
            cam_rgb_path = episode_path / "images" / "rgb"
            cam_depth_path = episode_path / "images" / "depth"
            cam_left_path = episode_path / "images" / "usb_cam_left"
            cam_right_path = episode_path / "images" / "usb_cam_right"
            cam_fisheye_path = episode_path / "images" / "usb_cam_fisheye"
            haptics_path = episode_path / "haptic"
            ee_path = episode_path / "ee_pose.csv"
            json_path = episode_path / "annotation.json"

            # Read data
            rgb_images = _read_images(cam_rgb_path)
            depth_images = _read_images(cam_depth_path)
            cam_left_images = _read_images(cam_left_path)
            cam_right_images = _read_images(cam_right_path)
            cam_fisheye_images = _read_images(cam_fisheye_path)
            ee_data = _read_csv_data(ee_path)
            haptic_data = []
            for haptic_file in sorted(haptics_path.glob("*.csv")):
                haptic_df = _read_csv_data(haptic_file)
                haptic_data.append(
                    haptic_df.iloc[:, 2:98].to_numpy(dtype=np.float32)
                )  # Read from the 3rd column onward

            haptic_data = np.array(
                haptic_data
            )  # Shape: (10, N, 96) if N is the number of rows in each CSV

            language_instruction = _read_json_data(json_path)

            # compute Kona language embedding
            language_embedding = self._embed([language_instruction])[0].numpy()

            # Placeholder for missing images
            placeholder_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
            placeholder_depth = np.zeros((400, 640, 1), dtype=np.uint16)
            placeholder_fisheye = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            data_length = len(haptic_data)
            action_data = ee_data[
                [
                    "left_ee_pos_x",
                    "left_ee_pos_y",
                    "left_ee_pos_z",
                    "left_ee_quat_w",
                    "left_ee_quat_x",
                    "left_ee_quat_y",
                    "left_ee_quat_z",
                    "right_ee_pos_x",
                    "right_ee_pos_y",
                    "right_ee_pos_z",
                    "right_ee_quat_w",
                    "right_ee_quat_x",
                    "right_ee_quat_y",
                    "right_ee_quat_z",
                ]
            ].to_numpy(dtype=np.float32)
            for i in range(data_length):
                episode.append(
                    {
                        "observation": {
                            "image": (
                                rgb_images[i]
                                if i < len(rgb_images)
                                else placeholder_rgb
                            ),
                            "depth": (
                                (depth_images[i].astype(np.uint16)).reshape(400, 640, 1)
                                if i < len(depth_images)
                                else placeholder_depth
                            ),
                            "image_left": (
                                cam_left_images[i]
                                if i < len(cam_left_images)
                                else placeholder_rgb
                            ),
                            "image_right": (
                                cam_right_images[i]
                                if i < len(cam_right_images)
                                else placeholder_rgb
                            ),
                            "image_fisheye": (
                                cam_fisheye_images[i]
                                if i < len(cam_fisheye_images)
                                else placeholder_fisheye
                            ),
                            "main_rgb_intrinsic": np.array(
                                [
                                    [1035.43652, 0.0, 966.34289],
                                    [0.0, 1035.67895, 539.37719],
                                    [0.0, 0.0, 1.0],
                                ]
                            ).astype(np.float32),
                            "main_depth_intrinsic": np.array(
                                [
                                    [307.24807, 0.0, 317.95391],
                                    [0.0, 307.24807, 201.31274],
                                    [0.0, 0.0, 1.0],
                                ]
                            ).astype(np.float32),
                            "left_camera_intrinsic": np.array(
                                [
                                    [789.59206, 0.0, 992.78203],
                                    [0.0, 790.79838, 593.08542],
                                    [0.0, 0.0, 1.0],
                                ]
                            ).astype(np.float32),
                            "right_camera_intrinsic": np.array(
                                [
                                    [789.89023, 0.0, 994.41009],
                                    [0.0, 791.36530, 593.94354],
                                    [0.0, 0.0, 1.0],
                                ]
                            ).astype(np.float32),
                            "fisheye_camera_intrinsic": np.array(
                                [
                                    [607.84390, 0.0, 1033.62034],
                                    [0.0, 608.56325, 546.09438],
                                    [0.0, 0.0, 1.0],
                                ]
                            ).astype(np.float32),
                            "fingers_haptics": haptic_data[:, i],
                        },
                        "action": action_data[i],
                        "discount": 1.0,
                        "reward": float(i == (data_length - 1)),
                        "is_first": i == 0,
                        "is_last": i == (data_length - 1),
                        "is_terminal": i == (data_length - 1),
                        "language_instruction": language_instruction,
                        "language_embedding": language_embedding,
                    }
                )

            # create output data sample
            episode_path_str = str(episode_path)
            sample = {
                "steps": episode,
                "episode_metadata": {"file_path": episode_path_str},
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path_str, sample

        # create list of all examples
        episode_paths = [
            str(episode_path)
            for category_path in Path(path).iterdir()
            if category_path.is_dir()
            for episode_path in category_path.iterdir()
            if episode_path.is_dir()
        ]

        # for smallish datasets, use single-thread parsing
        # for sample in episode_paths:
        #     yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        beam = tfds.core.lazy_imports.apache_beam
        return beam.Create(episode_paths) | beam.Map(_parse_example)
