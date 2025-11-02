# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import git
import importlib
import os
import pathlib
import torch
import numpy as np
import pickle
from torch.nn.utils import spectral_norm, weight_norm
import warnings
from tensordict import TensorDict
from typing import Callable


def ortho_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, torch.nn.Linear):
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


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolve the activation function from the name.

    Args:
        act_name: Name of the activation function.

    Returns:
        The activation function.

    Raises:
        ValueError: If the activation function is not found.
    """
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """Resolve the optimizer from the name.

    Args:
        optimizer_name: Name of the optimizer.

    Returns:
        The optimizer.

    Raises:
        ValueError: If the optimizer is not found.
    """
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")


def split_and_pad_trajectories(
    tensor: torch.Tensor | TensorDict, dones: torch.Tensor
) -> tuple[torch.Tensor | TensorDict, torch.Tensor]:
    """Split trajectories at done indices.

    Split trajectories, concatenate them and pad with zeros up to the length of the longest trajectory. Return masks
    corresponding to valid parts of the trajectories.

    Example (transposed for readability):
        Input: [[a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]]

        Output:[[a1, a2, a3, a4], | [[True, True, True, True],
                [a5, a6, 0, 0],   |  [True, True, False, False],
                [b1, b2, 0, 0],   |  [True, True, False, False],
                [b3, b4, b5, 0],  |  [True, True, True, False],
                [b6, 0, 0, 0]]    |  [True, False, False, False]]

    Assumes that the input has the following order of dimensions: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have the order (num_envs, num_transitions_per_env, ...) for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories

    if isinstance(tensor, TensorDict):
        padded_trajectories = {}
        for k, v in tensor.items():
            # Split the tensor into trajectories
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            # Add at least one full length trajectory
            trajectories = (*trajectories, torch.zeros(v.shape[0], *v.shape[2:], device=v.device))
            # Pad the trajectories to the length of the longest trajectory
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)
            # Remove the added trajectory
            padded_trajectories[k] = padded_trajectories[k][:, :-1]
        padded_trajectories = TensorDict(
            padded_trajectories, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)]
        )
    else:
        # Split the tensor into trajectories
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        # Add at least one full length trajectory
        trajectories = (*trajectories, torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device))
        # Pad the trajectories to the length of the longest trajectory
        padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
        # Remove the added trajectory
        padded_trajectories = padded_trajectories[:, :-1]
    # Create masks for the valid parts of the trajectories
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def demos_gen_dict_isaac(data: dict, batch_size: int, shuffle: bool = False):
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


def load_il_demos(
    folder: str, env_name: str, subsample: int, n_demos: int = None, n_steps: int = None, load_support: bool = False
):
    if folder is None:
        folder = "demos"
    expert_demos = {}
    if n_demos is not None:
        fname = f"demo_il_{env_name}_{n_demos}.pkl"
    if n_steps is not None:
        fname = f"{env_name}_{n_steps}_demo_data.pkl"
    try:
        # expert_demos['all'] = np.load(os.path.join(folder, f"demo_hf_{env_name}_{n_demos}.npy"))
        expert_demos = pickle.load(open(os.path.join(folder, fname), "rb"))

        # overwrite with subsampled after assigning to support items
        expert_demos["obs"] = expert_demos["obs"][::subsample]
        if "next_obs" in expert_demos.keys():
            expert_demos["next_obs"] = expert_demos["next_obs"][::subsample]
        else:
            expert_demos["next_obs"] = expert_demos["obs"][::subsample]
        expert_demos["acs"] = expert_demos["acs"][::subsample]
        if "rews" in expert_demos.keys():
            expert_demos["rew"] = expert_demos["rews"][::subsample]
        elif "rew" in expert_demos.keys():
            expert_demos["rew"] = expert_demos["rew"][::subsample]

        if "term" in expert_demos.keys():
            expert_demos["term"] = expert_demos["term"][::subsample]
        if "trunc" in expert_demos.keys():
            expert_demos["trunc"] = expert_demos["trunc"][::subsample]
        else:
            expert_demos["term"] = expert_demos["dones"][::subsample]
            expert_demos["trunc"] = expert_demos["dones"][::subsample]

    except Exception as e:
        print(e, "Generate demos using models trained in IsaacLab first")
        assert False

    return expert_demos


def unpad_trajectories(trajectories: torch.Tensor | TensorDict, masks: torch.Tensor) -> torch.Tensor | TensorDict:
    """Do the inverse operation of `split_and_pad_trajectories()`."""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir: str, repositories: list[str]) -> list[str]:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            t = repo.head.commit.tree
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # Skip if not a git repository
            continue
        # Get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # Check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # Write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            print(content[1100:1200])
            f.write(content)
        # Add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths


def string_to_callable(name: str) -> Callable:
    """Resolve the module and function names to return the function.

    Args:
        name: Function name. The format should be 'module:attribute_name'.

    Returns:
        The function loaded from the module.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When unable to resolve the attribute.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        # Check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as err:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}'."
        )
        raise ValueError(msg) from err


def resolve_obs_groups(
    obs: TensorDict, obs_groups: dict[str, list[str]], default_sets: list[str]
) -> dict[str, list[str]]:
    """Validate the observation configuration and defaults missing observation sets.

    The input is an observation dictionary `obs` containing observation groups and a configuration dictionary
    `obs_groups` where the keys are the observation sets and the values are lists of observation groups.

    The configuration dictionary could for example look like:
        {
            "policy": ["group_1", "group_2"],
            "critic": ["group_1", "group_3"]
        }

    This means that the 'policy' observation set will contain the observations "group_1" and "group_2" and the 'critic'
    observation set will contain the observations "group_1" and "group_3". This function will check that all the
    observations in the 'policy' and 'critic' observation sets are present in the observation dictionary from the
    environment.

    Additionally, if one of the `default_sets`, e.g. "critic", is not present in the configuration dictionary, this
    function will:

    1. Check if a group with the same name exists in the observations and assign this group to the observation set.
    2. If 1. fails, it will assign the observations from the 'policy' observation set to the default observation set.

    Args:
        obs: Observations from the environment in the form of a dictionary.
        obs_groups: Observation sets configuration.
        default_sets: Reserved observation set names used by the algorithm (besides 'policy'). If not provided in
            'obs_groups', a default behavior gets triggered.

    Returns:
        The resolved observation groups.

    Raises:
        ValueError: If any observation set is an empty list.
        ValueError: If any observation set contains an observation term that is not present in the observations.
    """
    # Check if policy observation set exists
    if "policy" not in obs_groups:
        if "policy" in obs:
            obs_groups["policy"] = ["policy"]
            warnings.warn(
                "The observation configuration dictionary 'obs_groups' must contain the 'policy' key."
                " As an observation group with the name 'policy' was found, this is assumed to be the observation set."
                " Consider adding the 'policy' key to the 'obs_groups' dictionary for clarity."
                " This behavior will be removed in a future version."
            )
        else:
            raise ValueError(
                "The observation configuration dictionary 'obs_groups' must contain the 'policy' key."
                f" Found keys: {list(obs_groups.keys())}"
            )

    # Check all observation sets for valid observation groups
    for set_name, groups in obs_groups.items():
        # Check if the list is empty
        if len(groups) == 0:
            msg = f"The '{set_name}' key in the 'obs_groups' dictionary can not be an empty list."
            if set_name in default_sets:
                if set_name not in obs:
                    msg += " Consider removing the key to default to the observations used for the 'policy' set."
                else:
                    msg += (
                        f" Consider removing the key to default to the observation '{set_name}' from the environment."
                    )
            raise ValueError(msg)
        # Check groups exist inside the observations from the environment
        for group in groups:
            if group not in obs:
                raise ValueError(
                    f"Observation '{group}' in observation set '{set_name}' not found in the observations from the"
                    f" environment. Available observations from the environment: {list(obs.keys())}"
                )

    # Fill missing observation sets
    for default_set_name in default_sets:
        if default_set_name not in obs_groups:
            if default_set_name in obs:
                obs_groups[default_set_name] = [default_set_name]
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' must contain the '{default_set_name}' key."
                    f" As an observation group with the name '{default_set_name}' was found, this is assumed to be the"
                    f" observation set. Consider adding the '{default_set_name}' key to the 'obs_groups' dictionary for"
                    " clarity. This behavior will be removed in a future version."
                )
            else:
                obs_groups[default_set_name] = obs_groups["policy"].copy()
                warnings.warn(
                    f"The observation configuration dictionary 'obs_groups' must contain the '{default_set_name}' key."
                    f" As the configuration for '{default_set_name}' is missing, the observations from the 'policy' set"
                    f" are used. Consider adding the '{default_set_name}' key to the 'obs_groups' dictionary for"
                    " clarity. This behavior will be removed in a future version."
                )

    # Print the final parsed observation sets
    print("-" * 80)
    print("Resolved observation sets: ")
    for set_name, groups in obs_groups.items():
        print("\t", set_name, ": ", groups)
    print("-" * 80)

    return obs_groups
