#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import gymnasium as gym
from rsl_rl.utils import linlayer


class Discriminator(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_obs,
        hidden_dims=[256, 256, 256],
        activation="elu",
        use_spectral_norm=False,
        use_weight_norm=False,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim = num_obs
        # Policy
        disc_layers = []
        disc_layers.append(
            linlayer(
                mlp_input_dim,
                hidden_dims[0],
                wnorm=use_weight_norm,
                snorm=use_spectral_norm,
            )
        )
        disc_layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                disc_layers.append(
                    linlayer(
                        hidden_dims[layer_index],
                        1,
                        wnorm=use_weight_norm,
                        snorm=use_spectral_norm,
                    )
                )
            else:
                disc_layers.append(
                    nn.Linear(
                        linlayer(
                            hidden_dims[layer_index],
                            hidden_dims[layer_index + 1],
                            wnorm=use_weight_norm,
                            snorm=use_spectral_norm,
                        )
                    )
                )
                disc_layers.append(activation)
        self.discriminator_network = nn.Sequential(*disc_layers)

        print(f"Discriminator MLP: {self.discriminator_network}")

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self, ob, ac, nob, d):
        if isinstance(ob, dict):
            ob = ob["policy"]
        if len(ob.shape) != len(ac.shape):
            ac = torch.unsqueeze(ac, -1)
        if d is not None:
            if len(ob.shape) != len(d.shape):
                d = torch.unsqueeze(d, -1)

        input_ = [ob]
        if self.use_actions:
            input_.append(ac)
        if self.use_next_obs:
            input_.append(nob)
        if self.use_dones:
            input_.append(d)

        net_input = torch.cat(input_, axis=-1)

        return self.discriminator_network(net_input)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
