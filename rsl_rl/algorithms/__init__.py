#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .gail import GAIL
from .swil import SWIL, NaSWIL

__all__ = ["PPO", "GAIL", "SWIL", "NaSWIL"]
