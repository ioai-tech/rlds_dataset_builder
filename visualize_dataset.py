import wandb
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import argparse
import tqdm
import importlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress debug warning messages


WANDB_ENTITY = None
WANDB_PROJECT = "vis_rlds"


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="name of the dataset to visualize")
args = parser.parse_args()

if WANDB_ENTITY is not None:
    render_wandb = True
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT)
else:
    render_wandb = False


# create TF dataset
dataset_name = args.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, split="train")
ds = ds.shuffle(100)

# visualize episodes
for i, episode in enumerate(ds.take(5)):
    images01 = []
    images02 = []
    images03 = []
    images04 = []
    for step in episode["steps"]:
        images01.append(step["observation"]["image01"].numpy())
        images02.append(step["observation"]["image02"].numpy())
        images03.append(step["observation"]["image03"].numpy())
        images04.append(step["observation"]["image04"].numpy())
    image_strip_01 = np.concatenate(images01[::4], axis=1)
    image_strip_02 = np.concatenate(images02[::4], axis=1)
    image_strip_03 = np.concatenate(images03[::4], axis=1)
    image_strip_04 = np.concatenate(images04[::4], axis=1)
    caption = step["language_instruction"].numpy().decode() + " (temp. downsampled 4x)"

    if render_wandb:
        wandb.log({f"image01_{i}": wandb.Image(image_strip_01, caption=caption)})
        wandb.log({f"image02_{i}": wandb.Image(image_strip_02, caption=caption)})
        wandb.log({f"image03_{i}": wandb.Image(image_strip_03, caption=caption)})
        wandb.log({f"image04_{i}": wandb.Image(image_strip_04, caption=caption)})
    else:
        plt.figure()
        plt.imshow(image_strip_01)
        plt.imshow(image_strip_02)
        plt.imshow(image_strip_03)
        plt.imshow(image_strip_04)
        plt.title(caption)

# visualize action and state statistics
joint_states, joint_commands, gripper_status, ee_poses = [], [], [], []
for episode in tqdm.tqdm(ds.take(500)):
    for step in episode["steps"]:
        joint_states.append(step["observation"]["joint_states"].numpy())
        joint_commands.append(step["action"]["joint_commands"].numpy())
        gripper_status.append(step["action"]["gripper_status"].numpy())
        ee_poses.append(step["action"]["ee_poses"].numpy())
joint_states = np.array(joint_states)
joint_commands = np.array(joint_commands)
gripper_status = np.array(gripper_status)
ee_poses = np.array(ee_poses)
joint_states_mean = joint_states.mean(0)
joint_commands_mean = joint_commands.mean(0)
gripper_status_mean = gripper_status.mean(0)
ee_poses_mean = ee_poses.mean(0)


def vis_stats(vector, vector_mean, tag):
    assert len(vector.shape) == 2
    assert len(vector_mean.shape) == 1
    assert vector.shape[1] == vector_mean.shape[0]

    n_elems = vector.shape[1]
    fig = plt.figure(tag, figsize=(5 * n_elems, 5))
    for elem in range(n_elems):
        plt.subplot(1, n_elems, elem + 1)
        plt.hist(vector[:, elem], bins=20)
        plt.title(vector_mean[elem])

    if render_wandb:
        wandb.log({tag: wandb.Image(fig)})


vis_stats(joint_states, joint_states_mean, "joint_states_stats")
vis_stats(joint_commands, joint_commands_mean, "joint_commands_stats")
vis_stats(gripper_status, gripper_status_mean, "gripper_status_stats")
vis_stats(ee_poses, ee_poses_mean, "ee_poses_stats")

if not render_wandb:
    plt.show()
