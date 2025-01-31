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

    def __init__(self, actor_critic, discriminator, il_opt, **kwargs):
        # init gail
        super().__init__(
            actor_critic,
            **kwargs,
        )

        self.il_lr = il_opt.irl.disc_lr
        self.lr = il_opt.irl.disc_lr
        self.use_actions = il_opt.irl.use_actions
        self.use_dones = il_opt.irl.use_dones
        self.use_next_obs = il_opt.irl.use_next_obs
        self.use_weight_norm = il_opt.irl.use_weight_norm
        self.use_ll_weight_norm = il_opt.irl.use_ll_weight_norm
        self.use_spectral_norm = il_opt.irl.use_spectral_norm
        self.l2_coeff = il_opt.irl.l2_coeff

        # GAIL components
        self.discriminator = discriminator
        self.discriminator.to(self.device)

        self.storage = None  # initialized later
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.il_lr, weight_decay=self.l2_coeff
        )

    def init_storage_from_demos(
        self,
        demos,
        num_envs,
        obs_shape,
        action_shape,
    ):
        self.demos_storage = DemoBuffer(
            num_envs,
            obs_shape,
            action_shape,
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

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss

    def compute_batch_loss(self, update_dict):
        # Discriminator loss
        d_loss = self.discriminator.compute_loss(update_dict)
        # Policy loss
        policy_loss = self.compute_policy_loss(update_dict)

        return {
            "d_loss": d_loss,
            "policy_loss": policy_loss,
        }

    def update_discriminator(self):
        demos_generator = self.demos_storage.mini_batch_generator(
            self.num_mini_batches, self.num_irl_epochs
        )

        # TODO: zip with storage generator?
        for obs_batch, actions_batch, next_obs_batch, dones_batch in demos_generator:
            update_dict = prepare_batch_update_irl_isaac(
                env, cfg, d, obs_, actions_, dones_
            )

            loss_dict = self.alg_il.compute_loss(update_dict)

            loss = loss_dict["d_loss"]
            self.optimizer_d.zero_grad()
            loss.backward()
            self.optimizer_d.step()
