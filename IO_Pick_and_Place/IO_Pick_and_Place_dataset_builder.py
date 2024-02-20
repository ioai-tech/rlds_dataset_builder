"""IO_Pick_and_Place dataset."""

from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_example(episode_path):
        # load raw data --> this should change for your dataset
        data = np.load(
            episode_path, allow_pickle=True
        )  # this is a list of dicts in our case

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        buff_path = ""
        for i, step in enumerate(data):
            # compute Kona language embedding
            if i == 0:
                # only run language embedding once since instruction is constant -- otherwise very slow
                sentence = step["language_instruction"].replace("_", " ")
                language_embedding = _embed([sentence])[0].numpy()

                # read path once
                buff_path = step["path"]

            episode.append(
                {
                    "observation": {
                        "image": step["image"],
                        "depth": step["depth"],
                        "image_left_side": step["image_01"],
                        "image_right_side": step["image_02"],
                        "image_fisheye": step["image_fisheye"],
                        "main_camera_intrinsic": np.array(
                            [
                                [453.52313232, 0.0, 327.23739624],
                                [0.0, 453.5272522, 185.1178894],
                                [0.0, 0.0, 1.0],
                            ]
                        ).astype(np.float32),
                        "left_camera_intrinsic": np.array(
                            [
                                [346.168705, 0.0, 311.412655],
                                [0.0, 344.76852, 179.28344],
                                [0.0, 0.0, 1.0],
                            ]
                        ).astype(np.float32),
                        "left_camera_extrinsic": np.array(
                            [
                                [0.99670459, -0.06338258, 0.05062223, -0.10706285],
                                [0.08021159, 0.86308497, -0.49864862, 0.14927003],
                                [-0.01208565, 0.50106585, 0.86532476, 0.08247642],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        ).astype(np.float32),
                        "right_camera_intrinsic": np.array(
                            [
                                [353.318005, 0.0, 316.83905],
                                [0.0, 353.27634, 187.08699],
                                [0.0, 0.0, 0.5],
                            ]
                        ).astype(np.float32),
                        "right_camera_extrinsic": np.array(
                            [
                                [0.98091352, 0.02216007, 0.19317765, 0.1456508],
                                [0.08330038, 0.84980977, -0.52046556, 0.20813367],
                                [-0.17569781, 0.52662347, 0.83174395, 0.06388365],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        ).astype(np.float32),
                        "fisheye_camera_intrinsic": np.array(
                            [
                                [435.09697, 1.3845, 640.69483],
                                [0.0, 433.14429, 510.82893],
                                [0.0, 0.0, 1.0],
                            ]
                        ).astype(np.float32),
                        "fisheye_camera_extrinsic": np.array(
                            [
                                [0.997521, 0.00225683, 0.07033323, 0.05337087],
                                [-0.02101277, 0.96343829, 0.26710509, 0.11895332],
                                [-0.06715892, -0.26792083, 0.96109734, 0.19188521],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        ).astype(np.float32),
                        "state": step["state"],
                    },
                    "action": step["action"],
                    "discount": 1.0,
                    "reward": float(i == (len(data) - 1)),
                    "is_first": i == 0,
                    "is_last": i == (len(data) - 1),
                    "is_terminal": i == (len(data) - 1),
                    "language_instruction": sentence,
                    "language_embedding": language_embedding,
                }
            )

        # create output data sample
        sample = {"steps": episode, "episode_metadata": {"file_path": buff_path}}

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)


class IOPickAndPlace(MultiThreadedDatasetBuilder):
    """DatasetBuilder for IO_Pick_and_Place dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.Google open-x format revised",
    }

    N_WORKERS = 20  # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = (
        400  # number of paths converted & stored in memory before writing to disk
    )
    # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
    # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = (
        _generate_examples  # handle to parse function from file paths to RLDS episodes
    )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(360, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "depth": tfds.features.Image(
                                        shape=(720, 1280, 1),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Main camera depth observation.",
                                    ),
                                    "image_left_side": tfds.features.Image(
                                        shape=(360, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Left camera RGB observation.",
                                    ),
                                    "image_right_side": tfds.features.Image(
                                        shape=(360, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Right camera RGB observation.",
                                    ),
                                    "image_fisheye": tfds.features.Image(
                                        shape=(640, 800, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera depth observation.",
                                    ),
                                    "main_camera_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Main camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "left_camera_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Left camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "left_camera_extrinsic": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Left camera extrinsic matrix in OpenCV convention.",
                                    ),
                                    "right_camera_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Right camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "right_camera_extrinsic": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Right camera extrinsic matrix in OpenCV convention.",
                                    ),
                                    "fisheye_camera_intrinsic": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Fisheye camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "fisheye_camera_extrinsic": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Fisheye camera extrinsic matrix in OpenCV convention.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc="Robot end effector pose, consists of [3x EEF position, "
                                        "4x EEF orientation w/x/y/z]."
                                        "1x relative gripper action(+1 for closing, -1 for opening)",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [3x EEF relative position, "
                                "3x EEF relative orientation yaw/pitch/roll, "
                                "1x relative gripper action(+1 for closing, -1 for opening).",
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
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        print(self.info)
        return {
            "train": glob.glob("data/train/episode_*.npy"),
            # 'val': glob.glob('data/val/episode_*.npy')
        }
