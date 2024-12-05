import tensorflow_datasets as tfds
import tensorflow_hub as hub

from dataclasses import dataclass
from typing import List, Iterator, Tuple, Any, Dict, Optional
from pathlib import Path
import logging
import json
import yaml
import numpy as np

# Import utilities for image and data processing
from io import BytesIO
from contextlib import contextmanager
from PIL import Image
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class McapProcessor:
    """Class for processing MCAP (ROS message capture) files"""

    def __init__(self, mcap_path: Path):
        """
        Initialize McapProcessor with path to MCAP file
        Args:
            mcap_path: Path to the MCAP file
        """
        self.mcap_path = mcap_path
        self.messages = []
        self._process_mcap()

    @contextmanager
    def _open_mcap(self):
        """
        Context manager for safely opening and handling MCAP files
        Yields:
            MCAP reader object with ROS message decoder
        """
        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            yield reader

    def _process_mcap(self):
        """Process all messages in the MCAP file and store them for later use"""
        with self._open_mcap() as reader:
            for schema, channel, message, ros_msg in reader.iter_decoded_messages():
                self.messages.append((schema, channel, ros_msg))

    def get_messages_by_topic(
        self, topic: str, schema_name: Optional[str] = None
    ) -> List[Any]:
        """
        Retrieve messages for a specific ROS topic
        Args:
            topic: ROS topic name
            schema_name: Optional schema name filter
        Returns:
            List of messages matching the topic and schema
        """
        return [
            ros_msg
            for schema, channel, ros_msg in self.messages
            if channel.topic == topic
            and (schema_name is None or schema.name == schema_name)
        ]

    def get_images(self, topic: str, compressed: bool = True) -> List[np.ndarray]:
        """
        Extract image data from ROS messages
        Args:
            topic: Image topic name
            compressed: Whether the image is compressed
        Returns:
            List of numpy arrays containing image data
        """
        schema_name = (
            "sensor_msgs/CompressedImage" if compressed else "sensor_msgs/Image"
        )
        images = []

        for ros_msg in self.get_messages_by_topic(topic, schema_name):
            try:
                image = Image.open(BytesIO(ros_msg.data))
                images.append(np.array(image))
            except Exception as e:
                logger.error(f"Error processing image from topic {topic}: {e}")
                continue

        return images

    def get_joint_states(self) -> List[np.ndarray]:
        """
        Extract robot joint state data
        Returns:
            List of numpy arrays containing joint positions (16 joints)
        """
        joint_msgs = self.get_messages_by_topic(
            "io_teleop/joint_states", "sensor_msgs/JointState"
        )
        return [np.array(msg.position, dtype=np.float32) for msg in joint_msgs]

    def get_joint_commands(self) -> List[np.ndarray]:
        """
        Extract joint command data
        Returns:
            List of numpy arrays containing joint commands (12 joints)
        """
        joint_cmd_msgs = self.get_messages_by_topic(
            "io_teleop/joint_cmd", "sensor_msgs/JointState"
        )
        return [np.array(msg.position, dtype=np.float32) for msg in joint_cmd_msgs]

    def get_gripper_status(self) -> List[np.ndarray]:
        """
        Extract gripper status data
        Returns:
            List of numpy arrays containing gripper status (2 grippers)
        """
        gripper_msgs = self.get_messages_by_topic(
            "io_teleop/target_gripper_status", "sensor_msgs/JointState"
        )
        return [np.array(msg.position, dtype=np.float32) for msg in gripper_msgs]

    def get_ee_poses(self) -> np.ndarray:
        """
        Extract end effector poses for both arms
        Returns:
            Numpy array containing concatenated left and right end effector poses
        """
        left_poses = []
        right_poses = []

        # Process left end effector poses
        for pose_msg in self.get_messages_by_topic(
            "io_teleop/target_ee_poses", "geometry_msgs/PoseArray"
        ):
            pose = pose_msg.poses
            left_poses.append(
                [
                    pose[0].position.x,
                    pose[0].position.y,
                    pose[0].position.z,
                    pose[0].orientation.w,
                    pose[0].orientation.x,
                    pose[0].orientation.y,
                    pose[0].orientation.z,
                ]
            )
            right_poses.append(
                [
                    pose[1].position.x,
                    pose[1].position.y,
                    pose[1].position.z,
                    pose[1].orientation.w,
                    pose[1].orientation.x,
                    pose[1].orientation.y,
                    pose[1].orientation.z,
                ]
            )

        # Combine poses ensuring equal length
        return np.hstack(
            [
                np.array(left_poses, dtype=np.float32),
                np.array(right_poses, dtype=np.float32),
            ]
        )


class IoUltraEmbodimentDataset(tfds.core.GeneratorBasedBuilder):
    """
    TensorFlow Dataset Builder for IO-ULTRA-EMBODIMENT-DATASET.
    This dataset contains egocentric, real-world data of embodied intelligence manipulation.
    """

    VERSION = tfds.core.Version("2.2.0")
    RELEASE_NOTES = {
        "2.2.0": "Data from teleop to RLDS.",
    }

    def __init__(self, *args, **kwargs):
        """
        Initialize dataset builder with Universal Sentence Encoder for language embedding
        """
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """
        Define dataset metadata and feature specifications
        Returns:
            DatasetInfo object containing dataset specifications
        """
        return tfds.core.DatasetInfo(
            builder=self,
            description="IO-TELEOP-DATASET: A real-world dataset of robotic manipulation based on human-in-the-loop solutions.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    # Camera images from different views
                                    "image01": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        doc="Camera 01 RGB observation.",
                                    ),
                                    "image02": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        doc="Camera 02 RGB observation.",
                                    ),
                                    "image03": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        doc="Camera 03 RGB observation.",
                                    ),
                                    "image04": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        doc="Camera 04 RGB observation.",
                                    ),
                                    # Robot state data
                                    "joint_states": tfds.features.Tensor(
                                        shape=(16,),
                                        dtype=np.float32,
                                        doc="16 joint positions of the robot.",
                                    ),
                                }
                            ),
                            # Teleop command data
                            "action": tfds.features.FeaturesDict(
                                {
                                    "joint_commands": tfds.features.Tensor(
                                        shape=(12,),
                                        dtype=np.float32,
                                        doc="Commands for the 12 joint positions.",
                                    ),
                                    "gripper_status": tfds.features.Tensor(
                                        shape=(2,),
                                        dtype=np.float32,
                                        doc="2x gripper status. The first is right, the second is left. The range of values is [0, 1], where 0 means that the gripper is fully open and 1 means that the gripper is fully closed.",
                                    ),
                                    "ee_poses": tfds.features.Tensor(
                                        shape=(14,),
                                        dtype=np.float32,
                                        doc="Target end effector pose based on VR frame, consists of [3x right EEF position, 4x right EEF orientation quaternions,"
                                        "3x left EEF position, 4x left EEF orientation quaternions]. Not sent directly to the robot.",
                                    ),
                                }
                            ),
                            # Episode information
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
                            # Language instruction and embedding
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding using Universal Sentence Encoder.",
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
        """
        Define data splits (currently only training split is implemented)
        """
        return {
            "train": self._generate_examples(path="data/train"),
            "val": self._generate_examples(path="data/val"),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """
        Generate examples for each split
        Args:
            path: Path to data directory
        Yields:
            Tuple of example ID and example data
        """

        def _read_json_data(json_path):
            """Read language instruction from JSON annotation file"""
            if not json_path.exists():
                return ""
            with open(json_path, "r") as f:
                data = json.load(f)
            return data["description"]

        def _parse_example(episode_path: str):
            """
            Parse a single episode's data
            Args:
                episode_path: Path to episode directory
            Returns:
                Tuple of episode ID and processed data
            """
            # Setup paths and process basic data
            episode_path = Path(episode_path)
            mcap_path = episode_path / "data.mcap"
            json_path = episode_path / "annotation.json"

            # Load configuration and language instruction
            language_instruction = _read_json_data(json_path)
            language_embedding = self._embed([language_instruction])[0].numpy()

            # Process MCAP data
            mcap_processor = McapProcessor(mcap_path)

            # Extract all required data
            image01 = mcap_processor.get_images(
                "/ob_camera_01/color/image_raw/compressed"
            )
            image02 = mcap_processor.get_images(
                "/ob_camera_02/color/image_raw/compressed"
            )
            image03 = mcap_processor.get_images(
                "/ob_camera_03/color/image_raw/compressed"
            )
            image04 = mcap_processor.get_images(
                "/ob_camera_04/color/image_raw/compressed"
            )
            joint_states = mcap_processor.get_joint_states()
            joint_commands = mcap_processor.get_joint_commands()
            gripper_status = mcap_processor.get_gripper_status()
            ee_poses = mcap_processor.get_ee_poses()

            # Verify data consistency
            data_lengths = [
                len(x)
                for x in [
                    image01,
                    image02,
                    image03,
                    image04,
                    joint_states,
                    joint_commands,
                    gripper_status,
                    ee_poses,
                ]
            ]
            data_length = min(data_lengths)

            if len(set(data_lengths)) > 1:
                logger.warning(
                    f"Inconsistent data lengths in {episode_path}: {data_lengths}"
                )

            # Create episode data structure
            episode = []
            for i in range(data_length):
                episode.append(
                    {
                        "observation": {
                            # Images from all cameras
                            "image01": (
                                image01[i] if i < len(image01) else image01[i - 1]
                            ),
                            "image02": (
                                image02[i] if i < len(image02) else image02[i - 1]
                            ),
                            "image03": (
                                image03[i] if i < len(image03) else image03[i - 1]
                            ),
                            "image04": (
                                image04[i] if i < len(image04) else image04[i - 1]
                            ),
                            # Robot state data
                            "joint_states": joint_states[i],
                        },
                        "action": {
                            "joint_commands": (
                                joint_commands[i]
                                if i < len(joint_commands)
                                else joint_commands[i - 1]
                            ),
                            "gripper_status": (
                                gripper_status[i]
                                if i < len(gripper_status)
                                else gripper_status[i - 1]
                            ),
                            "ee_poses": (
                                ee_poses[i] if i < len(ee_poses) else ee_poses[i - 1]
                            ),
                        },
                        "discount": 1.0,
                        "reward": float(i == (data_length - 1)),
                        "is_first": i == 0,
                        "is_last": i == (data_length - 1),
                        "is_terminal": i == (data_length - 1),
                        "language_instruction": language_instruction,
                        "language_embedding": language_embedding,
                    }
                )

            # Create final sample with metadata
            episode_path_str = str(episode_path)
            sample = {
                "steps": episode,
                "episode_metadata": {"file_path": episode_path_str},
            }

            return episode_path_str, sample

        # Get list of all episode paths
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
