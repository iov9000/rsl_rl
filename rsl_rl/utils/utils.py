#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import git
import os
import pathlib
import torch
import numpy as np
import pickle
from torch.nn.utils import spectral_norm, weight_norm


def ortho_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def linlayer(in_dim, out_dim, bias=True, wnorm=False, snorm=False):
    # return layer_init(nn.Linear(in_dim, out_dim, bias=bias))
    if wnorm:
        return weight_norm(torch.nn.Linear(in_dim, out_dim, bias=bias), "weight")
    elif snorm:
        return spectral_norm(torch.nn.Linear(in_dim, out_dim, bias=bias), "weight")
    else:
        return torch.nn.Linear(in_dim, out_dim, bias=bias)


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat(
        (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0])
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(
        tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list
    )
    # add at least one full length trajectory
    trajectories = trajectories + (
        torch.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),
    )
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(
        0, tensor.shape[0], device=tensor.device
    ).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def demos_gen_dict_isaac(data, batch_size, shuffle=False):
    """Yields batch of specified size"""
    if batch_size <= 0:
        return

    # flatten the data
    obs = data["obs"].reshape(-1, data["obs"].shape[-1])
    acs = data["acs"].reshape(-1, data["acs"].shape[-1])
    rew = data["rew"].reshape(-1, 1)
    term = data["term"].reshape(-1, 1)
    trunc = data["trunc"].reshape(-1, 1)

    b_inds = np.arange(len(obs))
    if shuffle:
        np.random.shuffle(b_inds)

    for i in range(batch_size, len(obs) - 1, batch_size):
        mb_inds = b_inds[i - batch_size : i]

        yield {
            "obs": obs[mb_inds],
            "acs": acs[mb_inds],
            "rew": rew[mb_inds],
            "term": term[mb_inds],
            "trunc": trunc[mb_inds],
        }


def load_il_demos(folder, env_name, subsample, n_demos, load_support=False):
    if folder is None:
        folder = "demos"
    expert_demos = {}
    fname = f"demo_il_{env_name}_{n_demos}.pkl"
    try:
        # expert_demos['all'] = np.load(os.path.join(folder, f"demo_hf_{env_name}_{n_demos}.npy"))
        expert_demos = pickle.load(open(os.path.join(folder, fname), "rb"))

        # overwrite with subsampled after assigning to support items
        expert_demos["obs"] = expert_demos["obs"][::subsample]
        expert_demos["acs"] = expert_demos["acs"][::subsample]
        expert_demos["rew"] = expert_demos["rew"][::subsample]
        expert_demos["term"] = expert_demos["term"][::subsample]
        expert_demos["trunc"] = expert_demos["trunc"][::subsample]
    except Exception as e:
        print(e, "Generate demos using models trained in IsaacLab first")
        assert False

    return expert_demos


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            print(content[1100:1200])
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths
