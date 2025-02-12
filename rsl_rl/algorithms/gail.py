#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import Discriminator
from rsl_rl.storage import DemoBuffer
from rsl_rl.algorithms.ppo import PPO


class GAIL(PPO):
    discriminator: Discriminator

    def __init__(self, actor_critic, discriminator, il_opt, device, **kwargs):
        # init gail
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
        self.divergence_type = il_opt.divergence_type
        self.num_irl_epochs = il_opt.num_irl_epochs
        self.irl_batch_size = il_opt.irl_batch_size
        self.loss_type = il_opt.loss_type

        # GAIL components
        self.discriminator = discriminator
        self.discriminator.to(self.device)

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.il_lr, weight_decay=self.l2_coeff
        )

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
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
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

    def compute_loss(
        self,
        exp_obs,
        exp_acs,
        policy_obs,
        policy_acs,
        exp_obs_next=None,
        exp_dones=None,
    ):
        policy_out = self.forward(policy_obs, policy_acs)
        expert_out = self.forward(exp_obs, exp_acs)

        d_out = torch.cat([expert_out, policy_out])

        labels = torch.cat(
            [torch.zeros(expert_out.size()), torch.ones(policy_out.size())]
        ).to(self.device)
        if self.loss_type == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(d_out, labels)
        elif self.loss_type == "ls":
            loss = torch.sum((expert_out - 1) ** 2 + (policy_out + 1) ** 2)

        return loss

    def concatenate_inputs(self, ob, ac, nob, d):
        input_ = [ob]
        if self.use_actions:
            input_.append(ac)
        if self.use_next_obs:
            input_.append(nob)
        if self.use_dones:
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
        if self.loss_type == "bce":
            self.reward = -torch.squeeze(torch.log(torch.sigmoid(d_out_div) + 1e-8))
        elif self.loss_type == "ls":
            self.reward = torch.squeeze(
                torch.maximum(
                    torch.zeros_like(d_out_div), 1 - 0.25 * (d_out_div - 1) ** 2
                )
            )

        return self.reward

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
                obs_batch = rollout_buffer_batch[0]
                actions_batch = rollout_buffer_batch[2]

                d_loss = self.compute_loss(
                    exp_obs_batch, exp_actions_batch, obs_batch, actions_batch
                )
                d_loss_avg += d_loss.item()
                update_cnt += 1
                self.optimizer_d.zero_grad()
                d_loss.backward()
                self.optimizer_d.step()

        return d_loss_avg / update_cnt
