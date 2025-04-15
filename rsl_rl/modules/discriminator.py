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
        use_last_layer_weight_norm=False,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim = num_obs
        self.input_dim = mlp_input_dim
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
                        wnorm=use_last_layer_weight_norm,
                        snorm=use_spectral_norm,
                    )
                )
            else:
                disc_layers.append(
                    linlayer(
                        hidden_dims[layer_index],
                        hidden_dims[layer_index + 1],
                        wnorm=use_weight_norm,
                        snorm=use_spectral_norm,
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

    def forward(self, x):
        return self.discriminator_network(x)

    def reset(self, dones=None):
        pass


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
