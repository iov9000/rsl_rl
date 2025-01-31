#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import gymnasium as gym


class Discriminator(torch.nn.Module):
    def __init__(self, env, opt, device):
        super(Discriminator, self).__init__()

        self.opt = opt
        self.layer_dims = opt.irl.d_layer_dims
        self.lr = opt.irl.disc_lr
        self.use_actions = opt.irl.use_actions
        self.use_dones = opt.irl.use_dones
        self.use_next_obs = opt.irl.use_next_obs
        self.use_cnn_base = opt.irl.use_cnn_base
        self.bias = opt.irl.use_disc_bias
        self.use_weight_norm = opt.irl.use_weight_norm
        self.use_ll_weight_norm = opt.irl.use_ll_weight_norm
        self.use_spectral_norm = opt.irl.use_spectral_norm
        self.l2_coeff = opt.irl.l2_coeff

        # specify nonlinearity
        if opt.irl.disc_nonlin == "relu":
            nonlin = torch.nn.ReLU()
        elif opt.irl.disc_nonlin == "leakyrelu":
            nonlin = torch.nn.LeakyReLU()
        elif opt.irl.disc_nonlin == "prelu":
            nonlin = torch.nn.PReLU()
        elif opt.irl.disc_nonlin == "tanh":
            nonlin = torch.nn.Tanh()
        elif opt.irl.disc_nonlin == "id":
            nonlin = torch.nn.Identity()
        else:
            nonlin = torch.nn.PReLU()
        self.nonlin = nonlin

        # specify the layer dimensions
        if isinstance(env.observation_space, gym.spaces.Dict):
            # if we're using Isaac envs
            self.ob_shapes = list(env.observation_space["policy"].shape)
        else:
            self.ob_shapes = list(env.observation_space.shape)
        # ob_shapes = list(env.observation_space.shape)
        self.ac_shapes = list(env.action_space.shape)
        if not self.ac_shapes:
            self.ac_shapes = [1]

        dim0 = self.ob_shapes[-1]
        if self.use_actions:
            dim0 = dim0 + self.ac_shapes[-1]
        if opt.irl.use_dones:
            dim0 = dim0 + 1
        if opt.irl.use_next_obs:
            dim0 = dim0 + self.ob_shapes[-1]

        self.layer_dims = [dim0] + self.layer_dims

    def base_fwd(self, base_fn, ob, ac, nob=None, d=None):
        #  match tensor sizes
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

        if self.use_cnn_base or self.is_atari:
            base_out = base_fn(*input_)
        else:
            base_out = base_fn(torch.cat(input_, axis=-1))

        return base_out

    def forward(self, ob, ac, nob=None, d=None):
        return NotImplementedError

    def get_reward(self, ob, ac, nob=None, d=None):
        return NotImplementedError

    def compute_loss(self):
        return NotImplementedError

    def update(self):
        return NotImplementedError


class Discriminator(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], num_actions)
                )
            else:
                actor_layers.append(
                    nn.Linear(
                        actor_hidden_dims[layer_index],
                        actor_hidden_dims[layer_index + 1],
                    )
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

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

    def forward(self):
        raise NotImplementedError

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
