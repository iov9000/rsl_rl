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
        self.exp_atoms_sorted = deque(maxlen=self.max_q_len)

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
                self.rnd = torch.randn(self.n_proj, self.discriminator.input_dim).to(
                    self.device
                )
            rnd = self.rnd
            norm_rnd = rnd / torch.norm(rnd, dim=-1, keepdim=True)
            input_ = self.concatenate_inputs(ob, ac, nob, d)
            prj = torch.matmul(input_, norm_rnd.T)
            return prj

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
        exp_slices = self.proj(obs_exp, acs_exp, nobs_exp, d_exp).unsqueeze(1)

        # sort slices
        pi_slices_sorted, pi_slices_sorted_idx = torch.sort(
            pi_slices, dim=0, stable=True
        )
        exp_slices_sorted, exp_slices_sorted_idx = torch.sort(
            exp_slices, dim=0, stable=True
        )

        self.pi_atoms_sorted.append(pi_slices_sorted)
        self.exp_atoms_sorted.append(exp_slices_sorted)

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
        return self.discriminator(self.concatenate_inputs(ob, ac, nob, d))

    def get_reward(self, ob, ac, nob=None, d=None):
        with torch.no_grad():
            num_envs = ob.shape[0]
            obs_t_slice = self.discriminator(self.concatenate_inputs(ob, ac, nob, d))
            rew = torch.zeros(num_envs, device=self.device)

            if len(self.pi_atoms_sorted) > 0:
                # sample random sorted atom batch from queues
                idx_ = torch.randint(0, high=len(self.pi_atoms_sorted), size=())
                idx__ = torch.randint(0, high=len(self.exp_atoms_sorted), size=())

                sorted_proj = self.pi_atoms_sorted[idx_]
                sorted_proj_tgt = self.exp_atoms_sorted[idx__]

                # TODO: figure out rewards based on multiple batches
                # -> some sort of hierarchy? OT approaches?
                # for _, (sorted_proj, sorted_proj_tgt) in enumerate(
                #     zip(pi_atoms_sorted, exp_atoms_sorted)
                # ):
                n = len(sorted_proj)
                # determine slice index in previously sorted atoms used for SWD computation
                idx = torch.searchsorted(
                    torch.transpose(sorted_proj, 0, -1), obs_t_slice.unsqueeze(0)
                ).squeeze()  # , right=True)

                # shift extreme indices
                idx[idx == n] -= 1
                idx[torch.where(idx == -1)] += 1

                if self.repl_loss_type == "diff":
                    a_prev = (
                        sorted_proj_tgt[idx.long()] - sorted_proj[idx.long()]
                    ) ** 2
                    a_new = (sorted_proj_tgt[idx.long()] - obs_t_slice) ** 2
                    rew_j = a_new - a_prev
                elif self.repl_loss_type == "diffmax0":
                    a_prev = (
                        sorted_proj_tgt[idx.long()] - sorted_proj[idx.long()]
                    ) ** 2
                    a_new = (sorted_proj_tgt[idx.long()] - obs_t_slice) ** 2
                    a_new[a_new > 0] = 0
                    rew_j = a_new - a_prev
                    if rew_j > 0:
                        rew_j = 0
                elif self.repl_loss_type == "diff2":
                    idx_i = idx.long()
                    idx_h = (idx.long() - 1).clamp_(0, None)

                    sorted_proj_tgt_i = sorted_proj_tgt[idx_i].squeeze()
                    sorted_proj_tgt_h = sorted_proj_tgt[idx_h].squeeze()
                    sorted_proj_i = torch.gather(
                        sorted_proj, 0, idx_i.view(1, sorted_proj.shape[1], 1)
                    ).squeeze()
                    sorted_proj_h = torch.gather(
                        sorted_proj,
                        0,
                        idx_h.view(1, sorted_proj.shape[1], 1),
                    ).squeeze()

                    obs_t_slice = obs_t_slice.squeeze()

                    a_i = (sorted_proj_tgt_i - sorted_proj_i) ** 2
                    a_h = (sorted_proj_tgt_h - sorted_proj_h) ** 2
                    a_new_i = (sorted_proj_tgt_i - obs_t_slice) ** 2
                    a_new_h = (sorted_proj_tgt_h - obs_t_slice) ** 2
                    diff_h = a_new_h - a_h
                    diff_i = a_new_i - a_i
                    rew = torch.where(diff_i > diff_h, diff_h, diff_i)

        return rew

    def update_discriminator(self):
        demos_generator = self.demos_storage.mini_batch_generator(
            self.irl_batch_size, shuffle=True, flatten=False
        )
        # num_mini_batches = self.demos_storage.get_num_minibatches(self.irl_batch_size)
        # generator = self.storage.mini_batch_generator(
        #     num_mini_batches, self.num_irl_epochs, flatten=False
        # )
        # TODO: make sure all demo transitions are used!!!
        d_loss_avg = 0
        update_cnt = 0

        for epoch in range(self.num_irl_epochs):
            for demo_buffer_batch in demos_generator:
                exp_obs_batch = demo_buffer_batch["observations"]
                exp_actions_batch = demo_buffer_batch["actions"]
                exp_next_obs_batch = demo_buffer_batch["next_observations"]
                exp_dones_batch = demo_buffer_batch["dones"]

                # since we have much more rollout data, sample a random batch
                # TODO: think of ways to be more efficient here -> e.g. some smart sampling strategy
                rollout_buffer_batch = self.storage.get_random_batch(
                    len(exp_obs_batch), flatten=False
                )

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
                if not self.use_linear_proj:
                    self.optimizer_d.zero_grad()
                    d_loss.backward()
                    self.optimizer_d.step()

        update_dict = {"d_loss": d_loss_avg / update_cnt}

        return update_dict


class NaSWIL(SWIL):
    """Non-Adversarial Sliced Wasserstein Imitation Learning"""

    def compute_loss(
        self,
        obs_pi,
        acs_pi,
        nobs_pi,
        d_pi,
        obs_pi_2,
        acs_pi_2,
        nobs_pi_2,
        d_pi_2,
    ):
        # project atoms with same random projections
        pi_slices = self.proj(obs_pi, acs_pi, nobs_pi, d_pi)
        pi_slices_2 = self.proj(obs_pi_2, acs_pi_2, nobs_pi_2, d_pi_2, keep_proj=True)

        pred_diffs = self.forward(obs_pi_2, acs_pi_2, nobs_pi_2, d_pi_2)

        with torch.no_grad():
            # sort transposed slices
            pi_slices_sorted, _ = torch.sort(pi_slices, dim=0, stable=True)
            # exp_slices_sorted, exp_slices_sorted_idx = torch.sort(exp_slices, dim=0, stable=True)

            # sort and insert using torch.searchsorted
            idx_j = torch.searchsorted(
                pi_slices_sorted.contiguous(), pi_slices_2.contiguous()
            )

            # if 0: idx = 0, if len(pi_slices_sorted): idx = -1 else:
            # (by default, searchsorted output idx replaces next largest item in sorted array)
            # min(b2[k]-b1_s[i], b2[k]-b1_s[j])

            idx_j[torch.where(idx_j == idx_j.shape[-1])] -= 1
            idx_i = idx_j - 1
            assert torch.sum(idx_j - idx_i) > 0
            idx_i[torch.where(idx_i == -1)] += 1

            # get first batch at indices
            b1_s_i = torch.take_along_dim(pi_slices_sorted, idx_i, dim=1)
            b1_s_j = torch.take_along_dim(pi_slices_sorted, idx_j, dim=1)

            # compute distances for all indices
            if self.repl_loss_type == "diff2":
                diffs = -torch.minimum(b1_s_i - pi_slices_2, b1_s_j - pi_slices_2)
            elif self.repl_loss_type == "diff2max0":
                diffs = torch.minimum(
                    (pi_slices_2 - b1_s_i).clamp_(0, None),
                    (pi_slices_2 - b1_s_j).clamp_(0, None),
                )

        # sum up loss and return it
        l2_pred_diff_loss = torch.sum(torch.nn.functional.mse_loss(pred_diffs, diffs))

        return l2_pred_diff_loss, diffs

    def get_reward(self, ob, ac, nob=None, d=None):
        reward = torch.squeeze(self.forward(ob, ac, nob, d))
        return reward

    def update_discriminator(self):
        d_loss_avg = 0
        update_cnt = 0
        for epoch in range(self.num_irl_epochs):
            rollout_buffer_batch = self.storage.get_random_batch(
                self.irl_batch_size, flatten=True
            )
            obs_batch = rollout_buffer_batch["observations"]
            actions_batch = rollout_buffer_batch["actions"]
            next_obs_batch = rollout_buffer_batch["next_observations"]
            dones_batch = rollout_buffer_batch["dones"]

            rollout_buffer_batch_2 = self.storage.get_random_batch(
                self.irl_batch_size, flatten=True
            )
            obs_batch_2 = rollout_buffer_batch_2["observations"]
            actions_batch_2 = rollout_buffer_batch_2["actions"]
            next_obs_batch_2 = rollout_buffer_batch_2["next_observations"]
            dones_batch_2 = rollout_buffer_batch_2["dones"]

            d_loss, diffs = self.compute_loss(
                obs_batch,
                actions_batch,
                next_obs_batch,
                dones_batch,
                obs_batch_2,
                actions_batch_2,
                next_obs_batch_2,
                dones_batch_2,
            )

            d_loss_avg += d_loss.item()
            update_cnt += 1

            self.optimizer_d.zero_grad()
            d_loss.backward()
            self.optimizer_d.step()

        update_dict = {"d_loss": d_loss_avg / update_cnt, "diffs": diffs}

        return update_dict
