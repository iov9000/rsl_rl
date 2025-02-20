#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import Discriminator
from rsl_rl.storage import DemoBuffer
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.utils import ortho_layer_init

from collections import deque


class SWIL(PPO):
    discriminator: Discriminator

    def __init__(self, actor_critic, discriminator, il_opt, device, **kwargs):
        # init swil
        super().__init__(
            actor_critic,
            device=device,
            **kwargs,
        )

        self.il_lr = il_opt.learning_rate
        self.use_actions = il_opt.use_actions
        self.use_dones = il_opt.use_dones
        self.use_next_obs = il_opt.use_next_obs
        self.use_weight_norm = il_opt.use_weight_norm
        self.use_spectral_norm = il_opt.use_spectral_norm
        self.l2_coeff = il_opt.l2_coeff
        self.num_irl_epochs = il_opt.num_irl_epochs
        self.irl_batch_size = il_opt.irl_batch_size
        self.shuffle_atom_batches = il_opt.shuffle_atom_batches
        self.max_q_len = il_opt.max_q_len
        self.repl_loss_type = il_opt.repl_loss_type

        # SWIL specific arguments
        self.n_proj = il_opt.n_proj
        self.use_linear_proj = il_opt.use_linear_proj

        # SWIL components
        self.discriminator = discriminator
        self.discriminator.to(self.device)

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.il_lr, weight_decay=self.l2_coeff
        )
        self.pi_atoms_sorted = deque(maxlen=self.max_q_len)
        self.pi_atoms_sorted_idx = deque(maxlen=self.max_q_len)
        self.exp_atoms_sorted = deque(maxlen=self.max_q_len)
        self.exp_atoms_sorted_idx = deque(maxlen=self.max_q_len)

    def init_storage_from_demos(
        self,
        demos,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        action_shape,
    ):
        self.demos_storage = DemoBuffer(
            num_envs,
            num_transitions_per_env,
            obs_shape[-1],
            action_shape[-1],
            self.device,
        )
        self.demos_storage.load_demos(demos)

    def test_mode(self):
        self.actor_critic.test()
        self.discriminator.test()

    def train_mode(self):
        self.actor_critic.train()
        self.discriminator.train()

    def update_ac(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        for batch in generator:
            obs_batch = batch["observations"]
            critic_obs_batch = batch["critic_observations"]
            actions_batch = batch["actions"]
            old_actions_log_prob_batch = batch["old_actions_log_prob"]
            returns_batch = batch["returns"]
            advantages_batch = batch["advantages"]
            masks_batch = batch["masks"]
            hid_states_batch = batch["hidden_states"]
            target_values_batch = batch["values"]
            old_sigma_batch = batch["sigma"]
            old_mu_batch = batch["mu"]

            self.actor_critic.act(
                obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def proj(self, ob, ac, nob=None, d=None, noise=False, keep_proj=False):
        # actually use random projections here
        if self.use_linear_proj:
            # sample a number of sphere directions
            if not keep_proj:
                self.rnd = torch.randn(self.n_proj, self.layer_dims[0])
            rnd = self.rnd
            norm_rnd = rnd / torch.norm(rnd, dim=-1, keepdim=True)
            input_ = self.concatenate_inputs(ob, ac, nob, d)
            rew = torch.matmul(input_, norm_rnd.T)
            return rew

        # random NN projections
        elif self.n_proj > 1 and not self.use_linear_proj:
            with torch.no_grad():
                self.discriminator.apply(ortho_layer_init)

        return self.discriminator(self.concatenate_inputs(ob, ac, nob, d))

    def gsw_dist_nn(
        self,
        obs_pi,
        acs_pi,
        nobs_pi,
        d_pi,
        obs_exp,
        acs_exp,
        nobs_exp,
        d_exp,
        random=False,
    ):
        """
        Calculates GSW between two empirical state-action distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        """
        if random:
            with torch.no_grad():
                self.discriminator.apply(ortho_layer_init)

        # project slices
        pi_slices = self.proj(obs_pi, acs_pi, nobs_pi, d_pi)
        exp_slices = self.proj(obs_exp, acs_exp, nobs_exp, d_exp)

        # sort slices
        pi_slices_sorted, pi_slices_sorted_idx = torch.sort(
            pi_slices, dim=0, stable=True
        )
        exp_slices_sorted, exp_slices_sorted_idx = torch.sort(
            exp_slices, dim=0, stable=True
        )

        self.pi_atoms_sorted.append(pi_slices_sorted)
        self.exp_atoms_sorted.append(exp_slices_sorted)
        self.pi_atoms_sorted_idx.append(pi_slices_sorted_idx)
        self.exp_atoms_sorted_idx.append(exp_slices_sorted_idx)

        # TODO: figure out shuffling with torch.randperm to keep everything on GPU

        return torch.sqrt(torch.sum((pi_slices_sorted - exp_slices_sorted) ** 2))

    def compute_loss(
        self,
        exp_obs,
        exp_acs,
        pi_obs,
        pi_acs,
        exp_nobs=None,
        exp_dones=None,
        pi_nobs=None,
        pi_dones=None,
    ):
        # compute sliced Wasserstein distance here
        self.policy_obs = copy.deepcopy(pi_obs)
        self.policy_acs = copy.deepcopy(pi_acs)

        self.buffer_empty_cnt = 0

        d_loss = -self.gsw_dist_nn(
            pi_obs,
            pi_acs,
            pi_nobs,
            pi_dones,
            exp_obs,
            exp_acs,
            exp_nobs,
            exp_dones,
        )

        return d_loss

    def concatenate_inputs(self, ob, ac, nob, d):
        input_ = [ob]
        if self.use_actions:
            input_.append(ac)
        if self.use_next_obs and nob is not None:
            input_.append(nob)
        if self.use_dones and d is not None:
            input_.append(d)

        return torch.cat(input_, axis=-1)

    def forward(self, ob, ac, nob=None, d=None):
        d_out = self.discriminator(self.concatenate_inputs(ob, ac, nob, d))

        if self.divergence_type == "fkl":
            d_out_div = torch.exp(d_out)  # (N*T,) p/q TODO: clip
        elif self.divergence_type == "rkl":
            d_out_div = d_out  # (N*T,) log (p/q)
        elif (
            self.divergence_type == "js"
        ):  # https://pytorch.org/docs/master/generated/torch.nn.Softplus.html
            d_out_div = torch.nn.functional.softplus(d_out)  # (N*T,) log (1 + p/q)

        # XXX: log D vs log(1-D)!!!!
        return d_out_div

    def get_reward(self, ob, ac, nob=None, d=None):
        with torch.no_grad():
            num_envs = ob.shape[0]
            obs_t_slice = self.discriminator(self.concatenate_inputs(ob, ac, nob, d))
            rew = torch.zeros(num_envs, device=self.device)

            for p, (sorted_proj, sorted_proj_tgt) in enumerate(
                zip(self.pi_atoms_sorted, self.exp_atoms_sorted)
            ):
                n = len(sorted_proj)
                if n > 0:
                    # idx = torch.searchsorted(sorted_proj.T.contiguous(), obs_t_slice.T.contiguous())#, right=True)
                    # determine slice index in previously sorted atoms used for SWD computation
                    idx = torch.searchsorted(
                        sorted_proj.T, obs_t_slice.T
                    )  # , right=True)

                    idx[idx == n] -= 1
                    # shift extreme indices
                    w = 1

                    # TODO: what if target CDF is left or mixed with source CDF?
                    for j, i in enumerate(idx):
                        rew_j = 0

                        # calculate diff when replacing atom
                        a_prev = (sorted_proj_tgt[i, j] - sorted_proj[i, j]) ** 2
                        a_prev_2 = (
                            sorted_proj_tgt[i - 1, j] - sorted_proj[i - 1, j]
                        ) ** 2

                        a_new = (sorted_proj_tgt[i, j] - obs_t_slice[0, j]) ** 2
                        a_new_2 = (sorted_proj_tgt[i - 1, j] - obs_t_slice[0, j]) ** 2

                        a_i = (sorted_proj_tgt[i, j] - sorted_proj[i, j]) ** 2
                        a_h = (sorted_proj_tgt[i - 1, j] - sorted_proj[i - 1, j]) ** 2
                        a_new_i = (sorted_proj_tgt[i, j] - obs_t_slice[0, j]) ** 2
                        a_new_h = (sorted_proj_tgt[i - 1, j] - obs_t_slice[0, j]) ** 2

                        rew_incr = torch.abs(sorted_proj[i, j] - obs_t_slice[0, j])
                        rew_decr = torch.abs(sorted_proj[i - 1, j] - obs_t_slice[0, j])

                        if self.repl_loss_type == "diff":
                            rew_j = a_new - a_prev
                            rew += w * (rew_j)
                        elif self.repl_loss_type == "diffmax0":
                            rew_j = a_new - a_prev
                            if rew_j > 0:  # > 0 bc we flip it later
                                rew_j = 0
                            rew += w * (rew_j)
                        elif self.repl_loss_type == "diff2":
                            diff_h = a_new_h - a_h
                            diff_i = a_new_i - a_i
                            rew_j = torch.where(rew_incr > rew_decr, diff_h, diff_i)
                            rew += w * (rew_j)
                        elif self.repl_loss_type == "diff2max0":
                            if rew_incr > rew_decr:
                                rew_j = a_new_h - a_h
                            else:
                                rew_j = a_new_i - a_i
                            if rew_j > 0:  # > 0 bc we flip it later
                                rew_j = 0
                            # XXX: sign problems!!!???
                            rew += w * (rew_j)

                        elif self.repl_loss_type == "diff3":
                            rew_j = a_new - a_prev
                            if sorted_proj[i, j] > sorted_proj_tgt[i, j]:
                                rew += w * rew_j
                            else:
                                rew -= w * rew_j
                        else:
                            if rew_incr > rew_decr:
                                rew += w * (a_new_h)
                            else:
                                rew += w * (a_new_i)
                else:
                    # TODO: hierarchy of multiple batches?
                    self.buffer_empty_cnt += 1
                    print("Atom buffer empty", self.buffer_empty_cnt)
                    self.pi_atoms_sorted = copy.deepcopy(self.pi_atoms_sorted_bkp)
                    self.exp_atoms_sorted = copy.deepcopy(self.exp_atoms_sorted_bkp)

        return rew

    def update_discriminator(self):
        demos_generator = self.demos_storage.mini_batch_generator(
            self.irl_batch_size, shuffle=True, flatten=True
        )
        # num_mini_batches = self.demos_storage.get_num_minibatches(self.irl_batch_size)
        # generator = self.storage.mini_batch_generator(
        #     num_mini_batches, self.num_irl_epochs, flatten=False
        # )
        # TODO: make sure all demo transitions are used!!!
        d_loss_avg = 0
        update_cnt = 0

        for epoch in range(self.num_irl_epochs):
            for (
                exp_obs_batch,
                exp_actions_batch,
                exp_next_obs_batch,
                exp_dones_batch,
            ) in demos_generator:
                rollout_buffer_batch = self.storage.get_random_batch(len(exp_obs_batch))
                obs_batch = rollout_buffer_batch["observations"]
                actions_batch = rollout_buffer_batch["actions"]
                next_obs_batch = rollout_buffer_batch["next_observations"]
                dones_batch = rollout_buffer_batch["dones"]

                d_loss = self.compute_loss(
                    exp_obs=exp_obs_batch,
                    exp_acs=exp_actions_batch,
                    pi_obs=obs_batch,
                    pi_acs=actions_batch,
                    exp_nobs=exp_next_obs_batch,
                    exp_dones=exp_dones_batch,
                    pi_nobs=next_obs_batch,
                    pi_dones=dones_batch,
                )
                d_loss_avg += d_loss.item()
                update_cnt += 1
                self.optimizer_d.zero_grad()
                d_loss.backward()
                self.optimizer_d.step()

        return d_loss_avg / update_cnt
