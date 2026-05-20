# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Franka RJ45 insertion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _finite(tensor: torch.Tensor) -> torch.Tensor:
    """Return a finite tensor by replacing invalid values with zeros."""
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return normalized finite quaternions, falling back to identity."""
    quat = _finite(quat)
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    quat = quat / torch.clamp(norm, min=1e-6)
    identity = torch.zeros_like(quat)
    identity[..., 3] = 1.0
    return torch.where(norm > 1e-6, quat, identity)


def joint_pos_rel_finite(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return finite relative joint positions for policy input.

    Args:
        env: The environment instance.
        asset_cfg: The articulation entity.

    Returns:
        Relative joint positions [rad or m, depending on joint type].
    """
    return _finite(base_mdp.joint_pos_rel(env, asset_cfg))


def joint_vel_rel_finite(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return finite relative joint velocities for policy input.

    Args:
        env: The environment instance.
        asset_cfg: The articulation entity.

    Returns:
        Relative joint velocities [rad/s or m/s, depending on joint type].
    """
    return _finite(base_mdp.joint_vel_rel(env, asset_cfg))


def last_action_finite(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """Return finite previous actions for policy input.

    Args:
        env: The environment instance.
        action_name: Optional action term name.

    Returns:
        Previous action tensor.
    """
    return _finite(base_mdp.last_action(env, action_name))


def plug_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
) -> torch.Tensor:
    """Return the plug root position in the robot root frame.

    Args:
        env: The environment instance.
        robot_cfg: The robot entity that provides the root frame.
        plug_cfg: The plug rigid object entity.

    Returns:
        Plug position [m] with shape ``(num_envs, 3)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    plug: RigidObject = env.scene[plug_cfg.name]
    plug_pos_b, _ = subtract_frame_transforms(
        _finite(robot.data.root_pos_w.torch),
        _finite_quat(robot.data.root_quat_w.torch),
        _finite(plug.data.root_pos_w.torch),
    )
    return _finite(plug_pos_b)


def plug_quat_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
) -> torch.Tensor:
    """Return the plug root orientation in the robot root frame.

    Args:
        env: The environment instance.
        robot_cfg: The robot entity that provides the root frame.
        plug_cfg: The plug rigid object entity.

    Returns:
        Plug orientation quaternion ``(x, y, z, w)`` with shape ``(num_envs, 4)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    plug: RigidObject = env.scene[plug_cfg.name]
    _, plug_quat_b = subtract_frame_transforms(
        _finite(robot.data.root_pos_w.torch),
        _finite_quat(robot.data.root_quat_w.torch),
        _finite(plug.data.root_pos_w.torch),
        _finite_quat(plug.data.root_quat_w.torch),
    )
    return _finite_quat(plug_quat_b)


def plug_socket_offset(
    env: ManagerBasedRLEnv,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
) -> torch.Tensor:
    """Return the socket-to-plug translational error in the environment frame.

    Args:
        env: The environment instance.
        plug_cfg: The plug rigid object entity.
        socket_cfg: The socket rigid object entity.

    Returns:
        Socket position minus plug position [m] with shape ``(num_envs, 3)``.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    socket: RigidObject = env.scene[socket_cfg.name]
    return _finite(socket.data.root_pos_w.torch - plug.data.root_pos_w.torch)
