from typing import List, Iterator, Tuple, Any

import json

import numpy as np
from pathlib import Path
import importlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import yaml

# import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class IoUltraEmbodimentDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for IO-ULTRA-EMBODIMENT-DATASET."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "2.0.0": "Convert mcap to RLDS.",
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
                                        doc="Main camera RGB observation.",
                                    ),
                                    "depth": tfds.features.Image(
                                        shape=(400, 640, 1),
                                        dtype=np.uint16,
                                        doc="Main camera depth observation.",
                                    ),
                                    "image_left": tfds.features.Image(
                                        shape=(1080, 1920, 3),
                                        dtype=np.uint8,
                                        doc="Left camera RGB observation.",
                                    ),
                                    "image_right": tfds.features.Image(
                                        shape=(1080, 1920, 3),
                                        dtype=np.uint8,
                                        doc="Right camera RGB observation.",
                                    ),
                                    "image_fisheye": tfds.features.Image(
                                        shape=(1080, 1920, 3),
                                        dtype=np.uint8,
                                        doc="Fisheye camera observation.",
                                    ),
                                    "main_rgb_intrinsic": tfds.features.Tensor(
                                        shape=(1, 4),
                                        dtype=np.float32,
                                        doc="Main RGB camera intrinsic parameters: fx, fy, cx, cy",
                                    ),
                                    "main_depth_intrinsic": tfds.features.Tensor(
                                        shape=(1, 4),
                                        dtype=np.float32,
                                        doc="Main depth camera intrinsic parameters: fx, fy, cx, cy",
                                    ),
                                    "left_camera_intrinsic": tfds.features.Tensor(
                                        shape=(1, 4),
                                        dtype=np.float32,
                                        doc="Left camera intrinsic parameters: fx, fy, cx, cy",
                                    ),
                                    "right_camera_intrinsic": tfds.features.Tensor(
                                        shape=(1, 4),
                                        dtype=np.float32,
                                        doc="Right camera intrinsic parameters: fx, fy, cx, cy",
                                    ),
                                    "fisheye_camera_intrinsic": tfds.features.Tensor(
                                        shape=(1, 4),
                                        dtype=np.float32,
                                        doc="Fisheye camera intrinsic parameters: fx, fy, cx, cy",
                                    ),
                                    "main_depth_to_main_rgb_extrinsic": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Extrinsic parameters of main depth camera relative to main RGB camera.",
                                    ),
                                    "left_camera_to_main_rgb_extrinsic": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Extrinsic parameters of left camera relative to main RGB camera.",
                                    ),
                                    "right_to_main_rgb_extrinsic": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Extrinsic parameters of right camera relative to main RGB camera.",
                                    ),
                                    "fisheye_camera_to_main_rgb_extrinsic": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Extrinsic parameters of fisheye camera relative to main RGB camera.",
                                    ),
                                    "joint_states": tfds.features.Tensor(
                                        shape=(1, 177),
                                        dtype=np.float32,
                                        doc="The joint states of 177 joints.",
                                    ),
                                    "fingers_haptics": tfds.features.Tensor(
                                        shape=(10, 96),
                                        dtype=np.int32,
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
            "train": self._generate_examples(
                path="/home/io003/data/io_data/202411_google_mcap2rlds"
            ),
            # "val": self._generate_examples(path="data/val"),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator for each split."""
        Image = importlib.import_module("PIL.Image")
        mcap_reader = importlib.import_module("mcap.reader")
        make_reader = mcap_reader.make_reader

        mcap_ros1_decoder = importlib.import_module("mcap_ros1.decoder")
        DecoderFactory = mcap_ros1_decoder.DecoderFactory

        def _read_images_from_mcap(mcap_path: str, topic: str) -> List[np.ndarray]:
            """
            Read images from a specific topic in an MCAP file.

            Args:
                mcap_path (str): Path to the MCAP file
                topic (str): ROS topic to extract images from

            Returns:
                List of NumPy image arrays
            """
            images = []

            # Open the MCAP file with ROS2 decoder
            reader = make_reader(
                open(mcap_path, "rb"), decoder_factories=[DecoderFactory()]
            )

            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                if channel.topic != topic:
                    continue

                try:
                    # Handle different image message types
                    if schema.name == "sensor_msgs/Image":
                        # For uncompressed images
                        data = ros_msg.data

                        height, width = 400, 640

                        dtype = np.dtype("uint16").newbyteorder("<")

                        img_array = np.frombuffer(data, dtype=dtype)
                        image = img_array.reshape(height, width, 1)
                        images.append(image)

                    elif schema.name == "sensor_msgs/CompressedImage":
                        image = Image.open(BytesIO(ros_msg.data))
                        images.append(np.array(image))

                except Exception as e:
                    print(f"Error processing message from topic {topic}: {str(e)}")

            return images

        def _read_images(
            mcap_path: str, image_topics: dict
        ) -> Tuple[List[np.ndarray], ...]:
            """
            Read images from multiple topics in an MCAP file.

            Args:
                mcap_path (str): Path to the MCAP file
                image_topics (dict): Dictionary mapping image type to ROS topic

            Returns:
                Tuple of image lists for different camera views
            """
            rgb_images = _read_images_from_mcap(
                mcap_path, image_topics.get("rgb", "/rgbd/color/image_raw/compressed")
            )

            depth_images = _read_images_from_mcap(
                mcap_path, image_topics.get("depth", "/rgbd/depth/image_raw")
            )

            cam_left_images = _read_images_from_mcap(
                mcap_path,
                image_topics.get("left", "/usb_cam_left/mjpeg_raw/compressed"),
            )

            cam_right_images = _read_images_from_mcap(
                mcap_path,
                image_topics.get("right", "/usb_cam_right/mjpeg_raw/compressed"),
            )

            cam_fisheye_images = _read_images_from_mcap(
                mcap_path,
                image_topics.get("fisheye", "/usb_cam_fisheye/mjpeg_raw/compressed"),
            )

            return (
                rgb_images,
                depth_images,
                cam_left_images,
                cam_right_images,
                cam_fisheye_images,
            )

        def _read_json_data(json_path):
            if not json_path.exists():
                return ""
            with open(json_path, "r") as f:
                data = json.load(f)
            return data["description"]

        def _extract_joint_states(
            mcap_path: str, topic: str = "/joint_states"
        ) -> np.ndarray:
            """
            Extract joint positions from MCAP file.

            Args:
                mcap_path (str): Path to MCAP file
                topic (str): ROS topic for joint states

            Returns:
                np.ndarray: Joint positions with shape (1, 177)
            """
            reader = make_reader(
                open(mcap_path, "rb"), decoder_factories=[DecoderFactory()]
            )

            joint_positions = []
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                if channel.topic == topic and schema.name == "sensor_msgs/JointState":
                    joint_positions.append(
                        np.array(ros_msg.position, dtype=np.float32).reshape(1, 177)
                    )

            return joint_positions

        def _extract_touch_data(
            mcap_path: str, topic: str = "/mocap/touch_data"
        ) -> np.ndarray:
            """
            Extract touch data from MCAP file.

            Args:
                mcap_path (str): Path to MCAP file
                topic (str): ROS topic for touch data

            Returns:
                np.ndarray: Touch data with shape (10, 96)
            """
            reader = make_reader(
                open(mcap_path, "rb"), decoder_factories=[DecoderFactory()]
            )

            touch_data = []
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                if channel.topic == topic and schema.name == "io_msgs/squashed_touch":
                    # Assuming each touch message contains multiple touch entries
                    touch_entry = []
                    for finger in ros_msg.data:
                        # Extract first 96 elements
                        touch_entry.append(np.array(finger.data)[:96])
                    touch_array = np.array(touch_entry, dtype=np.int32)
                    touch_data.append(touch_array)

            return touch_data

        def _extract_ee_poses(mcap_path: str) -> np.ndarray:
            """
            Extract end-effector poses from MCAP file.

            Args:
                mcap_path (str): Path to MCAP file

            Returns:
                np.ndarray: End-effector poses with shape (N, 14)
            """
            reader = make_reader(
                open(mcap_path, "rb"), decoder_factories=[DecoderFactory()]
            )

            left_poses = []
            right_poses = []

            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                if (
                    channel.topic == "/left_ee_pose"
                    and schema.name == "geometry_msgs/PoseStamped"
                ):
                    pose = ros_msg.pose
                    left_poses.append(
                        [
                            pose.position.x,
                            pose.position.y,
                            pose.position.z,
                            pose.orientation.w,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                        ]
                    )

                if (
                    channel.topic == "/right_ee_pose"
                    and schema.name == "geometry_msgs/PoseStamped"
                ):
                    pose = ros_msg.pose
                    right_poses.append(
                        [
                            pose.position.x,
                            pose.position.y,
                            pose.position.z,
                            pose.orientation.w,
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                        ]
                    )

            # Combine left and right poses
            if left_poses and right_poses:
                # Ensure same length by taking minimum
                min_length = min(len(left_poses), len(right_poses))
                action_data = np.hstack(
                    [
                        np.array(left_poses[:min_length], dtype=np.float32),
                        np.array(right_poses[:min_length], dtype=np.float32),
                    ]
                )
                return action_data

        def _read_camera_params(config_dir: Path):
            """
            Read camera parameters from configuration files.

            Args:
                config_dir (Path): Path to configuration directory

            Returns:
                dict: Dictionary containing camera parameters
            """
            # Read YAML files
            with open(config_dir / "camera_info.yml", "r") as f:
                camera_info = yaml.safe_load(f)
            with open(config_dir / "depth_to_rgb.yml", "r") as f:
                depth_to_rgb = yaml.safe_load(f)
            with open(config_dir / "orbbec_depth.yml", "r") as f:
                orbbec_depth = yaml.safe_load(f)
            with open(config_dir / "orbbec_rgb.yml", "r") as f:
                orbbec_rgb = yaml.safe_load(f)

            # Extract intrinsic parameters
            params = {}

            # Main RGB intrinsics (from orbbec_rgb.yml)
            cam_matrix = orbbec_rgb["camera"]["cam_matrix"]
            params["main_rgb_intrinsic"] = np.array(
                [[cam_matrix[0], cam_matrix[4], cam_matrix[2], cam_matrix[5]]],
                dtype=np.float32,
            )

            # Main depth intrinsics (from orbbec_depth.yml)
            cam_matrix = orbbec_depth["camera"]["cam_matrix"]
            params["main_depth_intrinsic"] = np.array(
                [[cam_matrix[0], cam_matrix[4], cam_matrix[2], cam_matrix[5]]],
                dtype=np.float32,
            )

            # Left camera (cam1)
            params["left_camera_intrinsic"] = np.array(
                [camera_info["cam1"]["intrinsics"]], dtype=np.float32
            )

            # Right camera (cam2)
            params["right_camera_intrinsic"] = np.array(
                [camera_info["cam2"]["intrinsics"]], dtype=np.float32
            )

            # Fisheye camera (cam3)
            params["fisheye_camera_intrinsic"] = np.array(
                [camera_info["cam3"]["intrinsics"]], dtype=np.float32
            )

            # Extract extrinsic parameters
            # Main depth to RGB extrinsic (from depth_to_rgb.yml)
            rot = np.array(depth_to_rgb["extrinsic"]["rot"]).reshape(3, 3)
            trans = np.array(depth_to_rgb["extrinsic"]["trans"]).reshape(3, 1)
            params["main_depth_to_main_rgb_extrinsic"] = np.vstack(
                [np.hstack([rot, trans]), np.array([0, 0, 0, 1], dtype=np.float32)]
            )

            # Left camera to RGB extrinsic (cam1)
            params["left_camera_to_main_rgb_extrinsic"] = np.array(
                camera_info["cam1"]["T_cn_cnm1"], dtype=np.float32
            )

            # Right camera to RGB extrinsic (cam2)
            params["right_to_main_rgb_extrinsic"] = np.array(
                camera_info["cam2"]["T_cn_cnm1"], dtype=np.float32
            )

            # Fisheye camera to RGB extrinsic (cam3)
            params["fisheye_camera_to_main_rgb_extrinsic"] = np.array(
                camera_info["cam3"]["T_cn_cnm1"], dtype=np.float32
            )

            return params

        def _parse_example(episode_path: str):
            episode_path = Path(episode_path)

            # Define paths for different image types and data files
            json_path = episode_path / "annotation.json"

            # Assuming mcap_path is derived from episode_path
            mcap_path = episode_path / "data.mcap"

            # Read images from MCAP
            (
                rgb_images,
                depth_images,
                cam_left_images,
                cam_right_images,
                cam_fisheye_images,
            ) = _read_images(
                mcap_path,
                image_topics={
                    "rgb": "/rgbd/color/image_raw/compressed",
                    "depth": "/rgbd/depth/image_raw",
                    "left": "/usb_cam_left/mjpeg_raw/compressed",
                    "right": "/usb_cam_right/mjpeg_raw/compressed",
                    "fisheye": "/usb_cam_fisheye/mjpeg_raw/compressed",
                },
            )

            joint_states = _extract_joint_states(mcap_path)
            haptic_data = _extract_touch_data(mcap_path)
            action_data = _extract_ee_poses(mcap_path)

            language_instruction = _read_json_data(json_path)

            # Read camera parameters
            config_dir = episode_path / "config"
            camera_params = _read_camera_params(config_dir)

            main_rgb_intrinsic = camera_params["main_rgb_intrinsic"]
            main_depth_intrinsic = camera_params["main_depth_intrinsic"]
            left_camera_intrinsic = camera_params["left_camera_intrinsic"]
            right_camera_intrinsic = camera_params["right_camera_intrinsic"]
            fisheye_camera_intrinsic = camera_params["fisheye_camera_intrinsic"]
            main_depth_to_main_rgb_extrinsic = camera_params[
                "main_depth_to_main_rgb_extrinsic"
            ].astype(np.float32)
            left_camera_to_main_rgb_extrinsic = camera_params[
                "left_camera_to_main_rgb_extrinsic"
            ]
            right_to_main_rgb_extrinsic = camera_params["right_to_main_rgb_extrinsic"]
            fisheye_camera_to_main_rgb_extrinsic = camera_params[
                "fisheye_camera_to_main_rgb_extrinsic"
            ]

            # compute Kona language embedding
            language_embedding = self._embed([language_instruction])[0].numpy()

            # Placeholder for missing images
            placeholder_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
            placeholder_depth = np.zeros((400, 640, 1), dtype=np.uint16)
            placeholder_fisheye = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # assemble episode
            episode = []
            data_length = len(rgb_images)

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
                                depth_images[i]
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
                            "main_rgb_intrinsic": main_rgb_intrinsic,
                            "main_depth_intrinsic": main_depth_intrinsic,
                            "left_camera_intrinsic": left_camera_intrinsic,
                            "right_camera_intrinsic": right_camera_intrinsic,
                            "fisheye_camera_intrinsic": fisheye_camera_intrinsic,
                            "main_depth_to_main_rgb_extrinsic": main_depth_to_main_rgb_extrinsic,
                            "left_camera_to_main_rgb_extrinsic": left_camera_to_main_rgb_extrinsic,
                            "right_to_main_rgb_extrinsic": right_to_main_rgb_extrinsic,
                            "fisheye_camera_to_main_rgb_extrinsic": fisheye_camera_to_main_rgb_extrinsic,
                            "joint_states": joint_states[i],
                            "fingers_haptics": haptic_data[i],
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
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return beam.Create(episode_paths) | beam.Map(_parse_example)
