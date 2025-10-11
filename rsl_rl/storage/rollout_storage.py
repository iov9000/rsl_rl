# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.critic_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, device=device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.actions_log_prob = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.mu = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )
            self.sigma = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=self.device
            )
            self.returns = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )
            self.advantages = torch.zeros(
                num_transitions_per_env, num_envs, 1, device=self.device
            )

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow! You should call clear() before adding new transitions."
            )

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(
                transition.actions_log_prob.view(-1, 1)
            )
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = (
            hidden_states[0]
            if isinstance(hidden_states[0], tuple)
            else (hidden_states[0],)
        )
        hid_c = (
            hidden_states[1]
            if isinstance(hidden_states[1], tuple)
            else (hidden_states[1],)
        )
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(
                    self.observations.shape[0], *hid_c[i].shape, device=self.device
                )
                for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(
        self, last_values, gamma, lam, normalize_advantage: bool = True
    ):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def get_random_batch(self, batch_size, shuffle=True, flatten=True):
        if flatten:
            observations = self.observations.flatten(0, 1)
            next_observations = self.next_observations.flatten(0, 1)

            if self.privileged_observations is not None:
                critic_observations = self.privileged_observations.flatten(0, 1)
            else:
                critic_observations = observations

            actions = self.actions.flatten(0, 1)
            dones = self.dones.flatten(0, 1)
            values = self.values.flatten(0, 1)
            returns = self.returns.flatten(0, 1)
            old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
            advantages = self.advantages.flatten(0, 1)
            old_mu = self.mu.flatten(0, 1)
            old_sigma = self.sigma.flatten(0, 1)
        else:
            observations = self.observations
            next_observations = self.next_observations

            if self.privileged_observations is not None:
                critic_observations = self.privileged_observations
            else:
                critic_observations = observations
            actions = self.actions
            dones = self.dones
            values = self.values
            returns = self.returns
            old_actions_log_prob = self.actions_log_prob
            advantages = self.advantages
            old_mu = self.mu
            old_sigma = self.sigma

        if shuffle:
            indices = torch.randint(
                0, len(observations), (batch_size,), device=self.device
            )
        else:
            idx = torch.randint(0, len(observations) - batch_size, (1,)).item()
            indices = torch.arange(idx, idx + batch_size, device=self.device)

        return {
            "observations": observations[indices],
            "critic_observations": critic_observations[indices],
            "next_observations": next_observations[indices],
            "actions": actions[indices],
            "values": values[indices],
            "advantages": advantages[indices],
            "returns": returns[indices],
            "old_actions_log_prob": old_actions_log_prob[indices],
            "mu": old_mu[indices],
            "sigma": old_sigma[indices],
            "dones": dones[indices],
        }

    def mini_batch_generator(self, num_mini_batches, num_epochs=8, flatten=True):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        if flatten:
            observations = self.observations.flatten(0, 1)
            next_observations = self.next_observations.flatten(0, 1)
            if self.privileged_observations is not None:
                critic_observations = self.privileged_observations.flatten(0, 1)
            else:
                critic_observations = observations

            actions = self.actions.flatten(0, 1)
            values = self.values.flatten(0, 1)
            returns = self.returns.flatten(0, 1)
            old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
            advantages = self.advantages.flatten(0, 1)
            old_mu = self.mu.flatten(0, 1)
            old_sigma = self.sigma.flatten(0, 1)
            dones = self.dones.flatten(0, 1)
        else:
            observations = self.observations
            next_observations = self.next_observations

            if self.privileged_observations is not None:
                critic_observations = self.privileged_observations
            else:
                critic_observations = observations
            actions = self.actions
            values = self.values
            returns = self.returns
            old_actions_log_prob = self.actions_log_prob
            advantages = self.advantages
            old_mu = self.mu
            old_sigma = self.sigma
            dones = self.dones

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                dones_batch = dones[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield {
                    "observations": obs_batch,
                    "critic_observations": critic_observations_batch,
                    "next_observations": next_obs_batch,
                    "actions": actions_batch,
                    "values": target_values_batch,
                    "advantages": advantages_batch,
                    "returns": returns_batch,
                    "old_actions_log_prob": old_actions_log_prob_batch,
                    "mu": old_mu_batch,
                    "sigma": old_sigma_batch,
                    "dones": dones_batch,
                    "hidden_states": (None, None),
                    "masks": None,
                }

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError(
                "This function is only available for reinforcement learning training."
            )
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.privileged_observations, self.dones
            )
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[
                    :, first_traj:last_traj
                ]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                next_obs_batch = self.next_observations[:, start:stop]
                dones_batch = self.dones[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                        first_traj:last_traj
                    ]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][
                        first_traj:last_traj
                    ]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield {
                    "observations": obs_batch,
                    "critic_observations": critic_obs_batch,
                    "next_observations": next_obs_batch,
                    "actions": actions_batch,
                    "values": values_batch,
                    "advantages": advantages_batch,
                    "returns": returns_batch,
                    "old_actions_log_prob": old_actions_log_prob_batch,
                    "mu": old_mu_batch,
                    "sigma": old_sigma_batch,
                    "hidden_states": (hid_a_batch, hid_c_batch),
                    "masks": masks_batch,
                    "dones": dones_batch,
                }

                first_traj = last_traj


class DemoBuffer:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device

        self.obs_shape = obs_shape
        self.actions_shape = actions_shape
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def load_demos(self, demos):
        self.num_demo_envs = demos["obs"].shape[1]

        self.observations = demos["obs"].to(self.device)
        self.next_observations = torch.cat(
            (demos["obs"][1:], torch.unsqueeze(demos["obs"][-1], 0)), dim=0
        )
        self.actions = demos["acs"].to(self.device)
        self.rewards = demos["rew"].to(self.device)
        self.dones = torch.logical_or(demos["term"], demos["trunc"]).to(self.device)

        # self.observations = (
        #     demos["obs"][:, 0, :]
        #     .view(-1, 1, self.obs_shape)
        #     .expand(-1, self.num_envs, self.obs_shape)
        #     .contiguous()
        #     .view(-1, self.obs_shape)
        # ).to(self.device)

    def add_transitions(self, transition: Transition):
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def get_num_minibatches(self, batch_size):
        buffer_size = len(self.observations)

        return buffer_size // batch_size

    def get_random_batch(self, batch_size, shuffle=True, flatten=True):
        if flatten:
            observations = self.observations.flatten(0, 1)
            next_observations = self.next_observations.flatten(0, 1)
            actions = self.actions.flatten(0, 1)
            dones = self.dones.flatten(0, 1)

        else:
            observations = self.observations
            next_observations = self.next_observations
            actions = self.actions
            dones = self.dones

        # here, take len(actions) it is smaller than len(observations)
        if shuffle:
            indices = torch.randint(0, len(actions), (batch_size,), device=self.device)
        else:
            idx = torch.randint(0, len(actions) - batch_size, (1,)).item()
            indices = torch.arange(idx, idx + batch_size, device=self.device)

        return {
            "observations": observations[indices],
            "next_observations": next_observations[indices],
            "actions": actions[indices],
            "dones": dones[indices],
        }

    def mini_batch_generator(self, batch_size, shuffle=False, flatten=True):
        if flatten:
            observations = self.observations.flatten(0, 1)
            next_observations = self.next_observations.flatten(0, 1)
            actions = self.actions.flatten(0, 1)
            rewards = self.rewards.flatten(0, 1)
            dones = self.dones.flatten(0, 1)
        else:
            observations = self.observations
            next_observations = self.next_observations
            actions = self.actions
            rewards = self.rewards
            dones = self.dones

        buffer_size = len(self.observations)

        if shuffle:
            indices = torch.randperm(buffer_size - 1, device=self.device)
        else:
            indices = torch.arange(buffer_size - 1, device=self.device)

        num_batches = buffer_size // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_idx = indices[start:end]

            if flatten:
                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                actions_batch = actions[batch_idx]
                rewards_batch = rewards[batch_idx]
                dones_batch = dones[batch_idx]
            else:
                obs_batch = observations[batch_idx, 0, :]
                next_obs_batch = next_observations[batch_idx, 0, :]
                actions_batch = actions[batch_idx, 0, :]
                rewards_batch = rewards[batch_idx, 0, :]
                dones_batch = dones[batch_idx, 0, :]

            yield {
                "observations": obs_batch,
                "next_observations": next_obs_batch,
                "actions": actions_batch,
                "rewards": rewards_batch,
                "dones": dones_batch,
            }
