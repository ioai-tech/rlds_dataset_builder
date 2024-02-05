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


class IoPickAndPlaceDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(720, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'depth': tfds.features.Image(
                            shape=(720, 1280, 1),
                            dtype=np.uint16,
                            encoding_format='png',
                            doc='Main camera depth observation.',
                        ),
                        'image_left': tfds.features.Image(
                            shape=(720, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Another camera RGB observation from left.',
                        ),
                        'image_right': tfds.features.Image(
                            shape=(720, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Another camera RGB observation from right.',
                        ),
                        'image_fisheye': tfds.features.Image(
                            shape=(1024, 1280, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Fisheye camera RGB observation.',
                        ),
                        'end_effector_pose': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot end effector pose, consists of [3x EEF position, '
                                '4x EEF orientation w/x/y/z].',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x EEF relative position, 3x EEF relative orientation yaw/pitch/roll, '
                            '1x gripper motion variation].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/home/io003/data/io_data/io_rlds_test/input/train'),
            'val': self._generate_examples(path='/home/io003/data/io_data/io_rlds_test/input/val'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        pd = importlib.import_module('pandas')
        Image = importlib.import_module('PIL.Image')

        def _read_images(image_root):
            image_files = sorted(os.listdir(image_root),
                                 key=lambda x: int(x.split('_')[-2]))
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
            return pd.read_csv(csv_path)

        def _read_json_data(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data['natural_language_description']

        def _parse_example(episode_path: str):
            episode_path = Path(episode_path)

            # Define paths for different image types and data files
            rgb_path = episode_path / 'rgb'
            depth_path = episode_path / 'depth'
            cam_01_path = episode_path / 'cam_01'
            cam_02_path = episode_path / 'cam_02'
            cam_fisheye_path = episode_path / 'cam_fisheye'
            csv_path = episode_path / 'result.csv'
            json_path = episode_path / 'info.json'

            # Read data
            rgb_images = _read_images(rgb_path) if rgb_path.exists() else []
            depth_images = _read_images(
                depth_path) if depth_path.exists() else []
            cam_01_images = _read_images(
                cam_01_path) if cam_01_path.exists() else []
            cam_02_images = _read_images(
                cam_02_path) if cam_02_path.exists() else []
            cam_fisheye_images = _read_images(
                cam_fisheye_path) if cam_fisheye_path.exists() else []
            csv_data = _read_csv_data(
                csv_path) if csv_path.exists() else pd.DataFrame()
            language_instruction = _read_json_data(
                json_path) if json_path.exists() else ""
            # compute Kona language embedding
            language_embedding = self._embed([language_instruction])[0].numpy()

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            data_length = len(csv_data)
            pose_data = csv_data[['camera_link_position_x', 'camera_link_position_y', 'camera_link_position_z', 'camera_link_orientation_x',
                                  'camera_link_orientation_y', 'camera_link_orientation_z', 'camera_link_orientation_w']].to_numpy(dtype=np.float32)
            action_data = csv_data[['gripper_closed', 'ee_command_position_x', 'ee_command_position_y', 'ee_command_position_z',
                                    'ee_command_rotation_x', 'ee_command_rotation_y', 'ee_command_rotation_z']].to_numpy(dtype=np.float32)
            for i in range(data_length):
                episode.append({
                    'observation': {
                        'image': rgb_images[i] if i < len(rgb_images) else None,
                        'depth': (depth_images[i].astype(np.uint16)).reshape(720, 1280, 1) if i < len(depth_images) else None,
                        'image_left': cam_01_images[i] if i < len(cam_01_images) else None,
                        'image_right': cam_02_images[i] if i < len(cam_02_images) else None,
                        'image_fisheye': cam_fisheye_images[i] if i < len(cam_fisheye_images) else None,
                        'end_effector_pose': pose_data[i]
                    },
                    'action': action_data[i],
                    'discount': 1.0,
                    'reward': float(i == (data_length - 1)),
                    'is_first': i == 0,
                    'is_last': i == (data_length - 1),
                    'is_terminal': i == (data_length - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            episode_path_str = str(episode_path)
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path_str
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path_str, sample

        # create list of all examples
        episode_paths = [str(p) for p in Path(path).iterdir() if p.is_dir()]

        # for smallish datasets, use single-thread parsing
        # for sample in episode_paths.iterdir():
        #     if sample.is_dir():
        #         yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        beam = tfds.core.lazy_imports.apache_beam
        return (
            beam.Create(episode_paths)
            | beam.Map(_parse_example)
        )
