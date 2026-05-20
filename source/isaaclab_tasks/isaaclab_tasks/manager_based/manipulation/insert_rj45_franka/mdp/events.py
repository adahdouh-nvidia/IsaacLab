# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the Franka RJ45 insertion task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def place_plug_in_gripper(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    plug_pos_in_env: tuple[float, float, float] = (0.485, 0.0, 0.39),
    plug_quat_xyzw: tuple[float, float, float, float] = (0.70710678, 0.70710678, 0.0, 0.0),
    plug_cfg: SceneEntityCfg = SceneEntityCfg("plug"),
) -> None:
    """Place the plug at a fixed gripper-aligned pose during reset.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        plug_pos_in_env: Plug root position in each environment frame [m].
        plug_quat_xyzw: Plug root orientation ``(x, y, z, w)``.
        plug_cfg: The plug rigid object entity.
    """
    plug: RigidObject = env.scene[plug_cfg.name]
    num_envs = len(env_ids)
    plug_pos_w = torch.tensor(plug_pos_in_env, device=env.device, dtype=torch.float32).repeat(num_envs, 1)
    plug_pos_w += env.scene.env_origins[env_ids]
    plug_quat_w = torch.tensor(plug_quat_xyzw, device=env.device, dtype=torch.float32).repeat(num_envs, 1)
    plug_pose_w = torch.cat([plug_pos_w, plug_quat_w], dim=-1)
    plug_vel_w = torch.zeros(num_envs, 6, device=env.device)

    plug.write_root_pose_to_sim_index(root_pose=plug_pose_w, env_ids=env_ids)
    plug.write_root_velocity_to_sim_index(root_velocity=plug_vel_w, env_ids=env_ids)


def close_gripper_on_plug(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    closed_position: float = 0.005,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
) -> None:
    """Close the Franka finger joints around the reset plug pose.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        closed_position: Target finger joint position [m].
        robot_cfg: Robot entity with finger joints selected.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids = robot_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_names = robot_cfg.joint_names if robot_cfg.joint_names is not None else ["panda_finger.*"]
        joint_ids, _ = robot.find_joints(joint_names)
    joint_pos = torch.full((len(env_ids), len(joint_ids)), closed_position, device=env.device)
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_position_target_index(target=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
