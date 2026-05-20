# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the Franka RJ45 insertion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def plug_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
) -> torch.Tensor:
    """Terminate when the plug falls below a minimum environment-frame height.

    Args:
        env: The environment instance.
        minimum_height: Minimum allowed plug root height [m].
        plug_cfg: The plug rigid object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    plug_height = plug.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return plug_height < minimum_height
