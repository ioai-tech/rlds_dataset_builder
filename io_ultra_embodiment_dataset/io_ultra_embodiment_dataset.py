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
from mcap_ros1.decoder import DecoderFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CameraParams:
    """Data class for storing camera parameters"""

    intrinsic: np.ndarray  # Camera intrinsic matrix
    extrinsic: Optional[np.ndarray] = None  # Optional camera extrinsic matrix


class ConfigManager:
    """Class for managing camera configuration and parameters"""

    def __init__(self, config_dir: Path):
        """
        Initialize ConfigManager with configuration directory
        Args:
            config_dir: Path to configuration files directory
        """
        self.config_dir = config_dir
        self.camera_params = self._load_camera_params()

    def _load_yaml(self, file_path: Path) -> dict:
        """
        Safely load a YAML file with error handling
        Args:
            file_path: Path to YAML file
        Returns:
            dict: Parsed YAML content
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML file is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file {file_path}: {e}")

    def _load_camera_params(self) -> Dict[str, CameraParams]:
        """
        Load and process all camera parameters from configuration files
        Returns:
            Dict[str, CameraParams]: Dictionary mapping camera names to their parameters
        """
        # Load all required configuration files
        camera_info = self._load_yaml(self.config_dir / "camera_info.yml")
        depth_to_rgb = self._load_yaml(self.config_dir / "depth_to_rgb.yml")
        orbbec_depth = self._load_yaml(self.config_dir / "orbbec_depth.yml")
        orbbec_rgb = self._load_yaml(self.config_dir / "orbbec_rgb.yml")

        params = {}

        # Process main RGB camera parameters
        rgb_matrix = orbbec_rgb["camera"]["cam_matrix"]
        params["main_rgb"] = CameraParams(
            intrinsic=np.array(
                [[rgb_matrix[0], rgb_matrix[4], rgb_matrix[2], rgb_matrix[5]]],
                dtype=np.float32,
            )
        )

        # Process main depth camera parameters and its extrinsic transformation
        depth_matrix = orbbec_depth["camera"]["cam_matrix"]
        rot = np.array(depth_to_rgb["extrinsic"]["rot"]).reshape(3, 3)
        trans = np.array(depth_to_rgb["extrinsic"]["trans"]).reshape(3, 1)
        depth_extrinsic = np.vstack(
            [np.hstack([rot, trans]), np.array([0, 0, 0, 1], dtype=np.float32)]
        ).astype(np.float32)

        params["main_depth"] = CameraParams(
            intrinsic=np.array(
                [[depth_matrix[0], depth_matrix[4], depth_matrix[2], depth_matrix[5]]],
                dtype=np.float32,
            ),
            extrinsic=depth_extrinsic,
        )

        # Process additional cameras (left, right, fisheye)
        for cam_id in ["cam1", "cam2", "cam3"]:
            cam_key = {"cam1": "left", "cam2": "right", "cam3": "fisheye"}[cam_id]
            params[cam_key] = CameraParams(
                intrinsic=np.array(
                    [camera_info[cam_id]["intrinsics"]], dtype=np.float32
                ),
                extrinsic=np.array(camera_info[cam_id]["T_cn_cnm1"], dtype=np.float32),
            )

        return params


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
                if compressed:
                    image = Image.open(BytesIO(ros_msg.data))
                    images.append(np.array(image))
                else:
                    height, width = 400, 640  # Default dimensions for depth images
                    dtype = np.dtype("uint16").newbyteorder("<")
                    img_array = np.frombuffer(ros_msg.data, dtype=dtype)
                    images.append(img_array.reshape(height, width, 1))
            except Exception as e:
                logger.error(f"Error processing image from topic {topic}: {e}")
                continue

        return images

    def get_joint_states(self) -> List[np.ndarray]:
        """
        Extract robot joint state data
        Returns:
            List of numpy arrays containing joint positions (177 joints)
        """
        joint_msgs = self.get_messages_by_topic(
            "/joint_states", "sensor_msgs/JointState"
        )
        return [
            np.array(msg.position, dtype=np.float32).reshape(1, 177)
            for msg in joint_msgs
        ]

    def get_touch_data(self) -> List[np.ndarray]:
        """
        Extract tactile sensor data from all fingers
        Returns:
            List of numpy arrays containing touch sensor readings
        """
        touch_msgs = self.get_messages_by_topic(
            "/mocap/touch_data", "io_msgs/squashed_touch"
        )
        return [
            np.array(
                [np.array(finger.data)[:96] for finger in msg.data], dtype=np.int32
            )
            for msg in touch_msgs
        ]

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
            "/left_ee_pose", "geometry_msgs/PoseStamped"
        ):
            pose = pose_msg.pose
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

        # Process right end effector poses
        for pose_msg in self.get_messages_by_topic(
            "/right_ee_pose", "geometry_msgs/PoseStamped"
        ):
            pose = pose_msg.pose
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

        # Combine poses ensuring equal length
        min_length = min(len(left_poses), len(right_poses))
        return np.hstack(
            [
                np.array(left_poses[:min_length], dtype=np.float32),
                np.array(right_poses[:min_length], dtype=np.float32),
            ]
        )


class IoUltraEmbodimentDataset(tfds.core.GeneratorBasedBuilder):
    """
    TensorFlow Dataset Builder for IO-ULTRA-EMBODIMENT-DATASET.
    This dataset contains egocentric, real-world data of embodied intelligence manipulation.
    """

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "2.0.0": "Convert mcap to RLDS.",
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
            description="IO-ULTRA-EMBODIMENT-DATASET: An egocentric, real-world dataset of embodied intelligence manipulation.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    # Camera images from different views
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
                                    # Camera intrinsic parameters
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
                                    # Camera extrinsic parameters (transformations)
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
                                    # Robot state data
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
                            # Robot action data (end effector poses)
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Robot end effector pose based on main RGB camera link, consists of [3x left EEF position, 4x left EEF orientation quaternions,"
                                "3x right EEF position, 4x right EEF orientation quaternions].",
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
            config_dir = episode_path / "config"

            # Load configuration and language instruction
            config_manager = ConfigManager(config_dir)
            language_instruction = _read_json_data(json_path)
            language_embedding = self._embed([language_instruction])[0].numpy()

            # Process MCAP data
            mcap_processor = McapProcessor(mcap_path)

            # Extract all required data
            rgb_images = mcap_processor.get_images(
                "/rgbd/color/image_raw/compressed", compressed=True
            )
            depth_images = mcap_processor.get_images(
                "/rgbd/depth/image_raw", compressed=False
            )
            left_images = mcap_processor.get_images(
                "/usb_cam_left/mjpeg_raw/compressed"
            )
            right_images = mcap_processor.get_images(
                "/usb_cam_right/mjpeg_raw/compressed"
            )
            fisheye_images = mcap_processor.get_images(
                "/usb_cam_fisheye/mjpeg_raw/compressed"
            )
            joint_states = mcap_processor.get_joint_states()
            haptic_data = mcap_processor.get_touch_data()
            action_data = mcap_processor.get_ee_poses()

            # Verify data consistency
            data_lengths = [
                len(x)
                for x in [
                    rgb_images,
                    depth_images,
                    left_images,
                    right_images,
                    fisheye_images,
                    joint_states,
                    haptic_data,
                ]
            ]
            data_length = min(data_lengths)

            if len(set(data_lengths)) > 1:
                logger.warning(
                    f"Inconsistent data lengths in {episode_path}: {data_lengths}"
                )

            # Get camera parameters
            main_rgb_intrinsic = config_manager.camera_params["main_rgb"].intrinsic
            main_depth_intrinsic = config_manager.camera_params["main_depth"].intrinsic
            main_depth_to_main_rgb_extrinsic = config_manager.camera_params[
                "main_depth"
            ].extrinsic
            left_camera_intrinsic = config_manager.camera_params["left"].intrinsic
            left_camera_to_main_rgb_extrinsic = config_manager.camera_params[
                "left"
            ].extrinsic
            right_camera_intrinsic = config_manager.camera_params["right"].intrinsic
            right_to_main_rgb_extrinsic = config_manager.camera_params[
                "right"
            ].extrinsic
            fisheye_camera_intrinsic = config_manager.camera_params["fisheye"].intrinsic
            fisheye_camera_to_main_rgb_extrinsic = config_manager.camera_params[
                "fisheye"
            ].extrinsic

            # Create episode data structure
            episode = []
            for i in range(data_length):
                episode.append(
                    {
                        "observation": {
                            # Images from all cameras
                            "image": (
                                rgb_images[i]
                                if i < len(rgb_images)
                                else rgb_images[i - 1]
                            ),
                            "depth": (
                                depth_images[i]
                                if i < len(depth_images)
                                else depth_images[i - 1]
                            ),
                            "image_left": (
                                left_images[i]
                                if i < len(left_images)
                                else left_images[i - 1]
                            ),
                            "image_right": (
                                right_images[i]
                                if i < len(right_images)
                                else right_images[i - 1]
                            ),
                            "image_fisheye": (
                                fisheye_images[i]
                                if i < len(fisheye_images)
                                else fisheye_images[i - 1]
                            ),
                            # Camera parameters
                            "main_rgb_intrinsic": main_rgb_intrinsic,
                            "main_depth_intrinsic": main_depth_intrinsic,
                            "left_camera_intrinsic": left_camera_intrinsic,
                            "right_camera_intrinsic": right_camera_intrinsic,
                            "fisheye_camera_intrinsic": fisheye_camera_intrinsic,
                            "main_depth_to_main_rgb_extrinsic": main_depth_to_main_rgb_extrinsic,
                            "left_camera_to_main_rgb_extrinsic": left_camera_to_main_rgb_extrinsic,
                            "right_to_main_rgb_extrinsic": right_to_main_rgb_extrinsic,
                            "fisheye_camera_to_main_rgb_extrinsic": fisheye_camera_to_main_rgb_extrinsic,
                            # Robot state data
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
