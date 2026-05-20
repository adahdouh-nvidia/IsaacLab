# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive-backed Franka RJ45 insertion environment for smoke testing."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.configclass import configclass

from .franka_rj45_env_cfg import (
    HELD_PLUG_POS,
    HELD_PLUG_QUAT_XYZW,
    SOCKET_POS,
    FrankaRJ45InsertEnvCfg,
    FrankaRJ45InsertEnvCfg_PLAY,
)


@configclass
class FrankaRJ45InsertStubEnvCfg(FrankaRJ45InsertEnvCfg):
    """RJ45 insertion environment using simple cuboids instead of USD assets."""

    def __post_init__(self) -> None:
        """Replace the USD RJ45 parts with primitive assets."""
        super().__post_init__()

        self.scene.plug = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plug",
            spawn=sim_utils.CuboidCfg(
                # Single primitive proxy for the plug body plus clip.
                size=(0.018, 0.040, 0.016),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.006),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.78, 0.05)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=HELD_PLUG_POS, rot=HELD_PLUG_QUAT_XYZW),
        )

        self.scene.socket = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Socket",
            spawn=sim_utils.CuboidCfg(
                size=(0.024, 0.020, 0.018),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.22, 0.85)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=SOCKET_POS, rot=HELD_PLUG_QUAT_XYZW),
        )


@configclass
class FrankaRJ45InsertStubEnvCfg_PLAY(FrankaRJ45InsertEnvCfg_PLAY, FrankaRJ45InsertStubEnvCfg):
    """Play variant of the primitive-backed RJ45 insertion environment."""

    def __post_init__(self) -> None:
        """Post-initialize the stub play scene."""
        super().__post_init__()
