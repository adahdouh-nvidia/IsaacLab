# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the Franka RJ45 insertion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


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


def _resolve_joint_ids(robot: Articulation, asset_cfg: SceneEntityCfg) -> list[int] | slice:
    """Resolve joint ids from a scene entity config."""
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_names = asset_cfg.joint_names if asset_cfg.joint_names is not None else [".*"]
        joint_ids, _ = robot.find_joints(joint_names)
    return joint_ids


def ee_to_plug_distance(
    env: ManagerBasedRLEnv,
    std: float,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward keeping the Franka TCP near the plug.

    Args:
        env: The environment instance.
        std: Tanh kernel standard deviation [m].
        plug_cfg: The plug rigid object entity.
        ee_frame_cfg: The end-effector frame sensor entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = _finite(ee_frame.data.target_pos_w.torch[..., 0, :])
    distance = torch.linalg.norm(_finite(plug.data.root_pos_w.torch) - ee_pos_w, dim=1)
    return _finite(1.0 - torch.tanh(distance / std))


def plug_to_socket_distance(
    env: ManagerBasedRLEnv,
    std: float,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
) -> torch.Tensor:
    """Reward moving the plug root toward the socket root.

    Args:
        env: The environment instance.
        std: Tanh kernel standard deviation [m].
        plug_cfg: The plug rigid object entity.
        socket_cfg: The socket rigid object entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    socket: RigidObject = env.scene[socket_cfg.name]
    distance = torch.linalg.norm(_finite(socket.data.root_pos_w.torch) - _finite(plug.data.root_pos_w.torch), dim=1)
    return _finite(1.0 - torch.tanh(distance / std))


def plug_grasped(
    env: ManagerBasedRLEnv,
    grasp_offset: tuple[float, float, float],
    hold_radius: float,
    std: float,
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward keeping the plug root close to the gripper TCP.

    Args:
        env: The environment instance.
        grasp_offset: Plug-frame point that should stay in the gripper [m].
        hold_radius: Radius around the TCP that counts as held [m].
        std: Tanh kernel standard deviation outside the held radius [m].
        plug_cfg: The plug rigid object entity.
        ee_frame_cfg: The end-effector frame sensor entity.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = _finite(ee_frame.data.target_pos_w.torch[..., 0, :])
    offset_b = torch.tensor(grasp_offset, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    grasp_pos_w = _finite(plug.data.root_pos_w.torch) + math_utils.quat_apply(
        _finite_quat(plug.data.root_quat_w.torch), offset_b
    )
    distance = torch.linalg.norm(grasp_pos_w - ee_pos_w, dim=1)
    slip_distance = torch.clamp(distance - hold_radius, min=0.0)
    return _finite(1.0 - torch.tanh(slip_distance / std))


def gripper_closed(
    env: ManagerBasedRLEnv,
    target_position: float,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
) -> torch.Tensor:
    """Reward keeping the Franka fingers at or below the grasp target.

    Args:
        env: The environment instance.
        target_position: Maximum desired finger joint position [m].
        std: Tanh kernel standard deviation for opening beyond the target [m].
        robot_cfg: Robot entity with finger joints selected.

    Returns:
        Reward tensor with shape ``(num_envs,)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids = _resolve_joint_ids(robot, robot_cfg)
    finger_pos = _finite(robot.data.joint_pos.torch[:, joint_ids])
    opening_error = torch.clamp(finger_pos - target_position, min=0.0).mean(dim=1)
    return _finite(1.0 - torch.tanh(opening_error / std))


class insertion_bonus(ManagerTermBase):
    """Reward socket engagement while logging sticky insertion success."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Log success rate for completed episodes and clear the sticky bit."""
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
        self._succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        engaged_threshold: float,
        success_threshold: float,
        plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
        socket_cfg: SceneEntityCfg = SceneEntityCfg("socket"),
    ) -> torch.Tensor:
        """Return an engagement bonus and update sticky success.

        Args:
            env: The environment instance.
            engaged_threshold: Distance for the coarse engaged zone [m].
            success_threshold: Distance for an insertion success [m].
            plug_cfg: The plug rigid object entity.
            socket_cfg: The socket rigid object entity.

        Returns:
            Engagement bonus with shape ``(num_envs,)``.
        """
        plug: RigidObject = env.scene[plug_cfg.name]
        socket: RigidObject = env.scene[socket_cfg.name]
        plug_pos_w = plug.data.root_pos_w.torch
        socket_pos_w = socket.data.root_pos_w.torch
        valid = torch.isfinite(plug_pos_w).all(dim=1) & torch.isfinite(socket_pos_w).all(dim=1)
        distance = torch.linalg.norm(_finite(socket_pos_w) - _finite(plug_pos_w), dim=1)
        engaged = valid & (distance < engaged_threshold)
        self._succeeded |= valid & (distance < success_threshold)
        return engaged.float()


def plug_upright(
    env: ManagerBasedRLEnv,
    plug_up_axis: tuple[float, float, float] = (0.0, 0.0, -1.0),
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
) -> torch.Tensor:
    """Penalize plug tilt away from the nominal upright latch pose.

    The RJ45 reset quaternion is a 180-degree rotation about local X, so the
    local ``-Z`` axis is treated as the visual up direction for this posture
    term.

    Args:
        env: The environment instance.
        plug_up_axis: Local plug axis that should align with world ``+Z``.
        plug_cfg: The plug rigid object entity.

    Returns:
        Tilt penalty with shape ``(num_envs,)``.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    axis_b = torch.tensor(plug_up_axis, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    axis_w = math_utils.quat_apply(_finite_quat(plug.data.root_quat_w.torch), axis_b)
    return _finite(1.0 - torch.clamp(axis_w[:, 2], min=-1.0, max=1.0))


def action_rate_l2_finite(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a finite action-rate penalty.

    Args:
        env: The environment instance.

    Returns:
        Action-rate penalty with shape ``(num_envs,)``.
    """
    return _finite(base_mdp.action_rate_l2(env))


def joint_vel_l2_finite(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a finite joint-velocity penalty.

    Args:
        env: The environment instance.
        asset_cfg: Articulation entity whose joint velocities are penalized.

    Returns:
        Joint-velocity penalty with shape ``(num_envs,)``.
    """
    return _finite(base_mdp.joint_vel_l2(env, asset_cfg))
