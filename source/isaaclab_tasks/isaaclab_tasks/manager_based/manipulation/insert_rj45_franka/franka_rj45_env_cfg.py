# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka RJ45 insertion environment."""

from __future__ import annotations

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .mdp import cable as cable_mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


ASSET_DIR = os.path.expanduser("~/isaaclab_assets/rj45")
"""Default directory containing the split RJ45 USD assets."""

HELD_PLUG_POS = (0.485, 0.0, 0.39)
"""Plug root position for the default Franka grasp pose [m]."""

HELD_PLUG_QUAT_XYZW = (0.70710678, 0.70710678, 0.0, 0.0)
"""Plug root orientation for the default Franka grasp pose.

This keeps the RJ45 upright flip and adds a 90-degree yaw so the plug points
outward from the gripper instead of sideways into the socket.
"""

SOCKET_POS = (0.54, HELD_PLUG_POS[1], HELD_PLUG_POS[2])
"""Socket root position for the near-insertion reset [m]."""

PLUG_GRASP_OFFSET = (0.0, -0.025, 0.0)
"""Plug-frame point held by the gripper, relative to the exposed plug root [m]."""

GRIPPER_GRASP_POS = 0.005
"""Finger joint target for holding the RJ45 plug [m]."""


@configclass
class RJ45SplitCableSolverCfg(MJWarpSolverCfg):
    """MJWarp rigid scene with a task-local VBD sidecar for the RJ45 cable.

    The main Newton model contains only MJWarp-compatible rigid bodies: Franka,
    plug, socket, table, ground, and a non-colliding cable proxy for rendering.
    The true ``JointType.CABLE`` rod is built in a separate cable-only Newton
    model that is stepped by :class:`NewtonRJ45SplitCableManager`.
    """

    class_type: type[NewtonManager] | str = "{DIR}.franka_rj45_env_cfg:NewtonRJ45SplitCableManager"
    """Manager class that keeps rigid scene stepping and VBD cable stepping split."""

    solver_type: str = "rj45_mjwarp_vbd_cable"
    """Task-local solver tag for MJWarp rigid bodies plus a VBD cable sidecar."""


class NewtonRJ45SplitCableManager(NewtonMJWarpManager):
    """MJWarp manager with an extra per-substep VBD solve for the cable only."""

    @classmethod
    def _run_solver_substeps(cls, contacts) -> None:
        """Step rigid MJWarp physics, then the cable-only VBD sidecar."""
        for _ in range(cls._num_substeps):
            cable_mdp.sync_split_vbd_cable_from_rigid_state(cls._state_0)
            if cls._needs_collision_pipeline:
                cls._collision_pipeline.collide(cls._state_0, cls._contacts)
            cls._step_solver(cls._state_0, cls._state_0, cls._control, contacts, cls._solver_dt)
            cable_mdp.step_split_vbd_cable(cls._state_0, cls._solver_dt)
            cls._state_0.clear_forces()

    @classmethod
    def _simulate_full(cls) -> None:
        """Run actuators and MJWarp rigid substeps with cable-only VBD substeps."""
        physics_dt = cls._solver_dt * cls._num_substeps
        contacts = cls._contacts if cls._needs_collision_pipeline else None

        for _ in range(cls._decimation):
            if cls._adapter is not None:
                cls._adapter.step(cls._state_0, cls._control, physics_dt)
            for callback in cls._post_actuator_callbacks:
                callback()
            cls._run_solver_substeps(contacts)

        cls._update_sensors(contacts)

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Run MJWarp rigid physics plus the cable-only VBD sidecar."""
        contacts = cls._contacts if cls._needs_collision_pipeline else None
        cls._run_solver_substeps(contacts)
        cls._update_sensors(contacts)


# Compatibility aliases for local code that imported the earlier RJ45 names.
RJ45VBDSolverCfg = RJ45SplitCableSolverCfg
NewtonRJ45VBDManager = NewtonRJ45SplitCableManager


@configclass
class RJ45SimCfg(PresetCfg):
    """Simulation presets for PhysX and the two supported Newton RJ45 paths."""

    physx: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physics=PhysxCfg(
            bounce_threshold_velocity=0.2,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )

    newton_mjwarp: SimulationCfg = SimulationCfg(
        dt=1.0 / 600.0,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=256,
                nconmax=160,
                cone="pyramidal",
                integrator="implicitfast",
                impratio=1,
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    """MJWarp rigid-body scene with the RJ45 cable represented as a D6 capsule-chain proxy."""

    newton_vbd: SimulationCfg = SimulationCfg(
        dt=1.0 / 600.0,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=RJ45SplitCableSolverCfg(
                njmax=256,
                nconmax=160,
                cone="pyramidal",
                integrator="implicitfast",
                impratio=1,
            ),
            num_substeps=1,
            debug_mode=False,
            # The sidecar VBD cable swaps state buffers in Python, so keep the
            # faithful cable path eager until that loop is graph-captured too.
            use_cuda_graph=False,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    """MJWarp rigid scene plus a true Newton VBD rod cable sidecar."""

    default = physx


@configclass
class FrankaRJ45SceneCfg(InteractiveSceneCfg):
    """Rigid scene with a Franka, split RJ45 parts, table, ground, and lights.

    The flexible cable is not spawned here. It is added by the Newton builder
    hooks in :mod:`.mdp.cable` so the backend can choose either the true VBD rod
    path or the MJWarp-compatible rigid proxy.
    """

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # The splitter bakes the RJ45 clip/latch mesh into this same rigid body.
    # Keeping it as one object avoids an unjointed clip falling away from the gripper.
    plug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plug",
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_DIR}/rj45_plug_body.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.006),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=HELD_PLUG_POS, rot=HELD_PLUG_QUAT_XYZW),
    )

    socket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_DIR}/rj45_socket.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SOCKET_POS, rot=HELD_PLUG_QUAT_XYZW),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.707, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    def __post_init__(self) -> None:
        """Apply backend-safe Franka settings."""
        if self.robot.spawn.rigid_props is None:
            self.robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
        self.robot.spawn.rigid_props.disable_gravity = True
        self.robot.actuators["panda_hand"].effort_limit_sim = 1000.0
        self.robot.actuators["panda_hand"].stiffness = 2500.0
        self.robot.actuators["panda_hand"].damping = 200.0


@configclass
class ActionsCfg:
    """Action specifications for Franka joint-position control."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
        clip={
            "panda_joint1": (-0.35, 0.35),
            "panda_joint2": (-0.85, -0.30),
            "panda_joint3": (-0.35, 0.35),
            "panda_joint4": (-3.05, -2.55),
            "panda_joint5": (-0.35, 0.35),
            "panda_joint6": (2.75, 3.30),
            "panda_joint7": (0.45, 1.05),
        },
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        # The task starts with the plug already pinched; keep the fingers at
        # that grasp width so early random policy actions do not drop it.
        open_command_expr={"panda_finger_.*": GRIPPER_GRASP_POS},
        close_command_expr={"panda_finger_.*": GRIPPER_GRASP_POS},
    )


@configclass
class ObservationsCfg:
    """Observation groups for the RJ45 insertion policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Low-dimensional policy observations."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel_finite, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel_finite, noise=Unoise(n_min=-0.01, n_max=0.01))
        plug_position = ObsTerm(
            func=mdp.plug_position_in_robot_root_frame,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        plug_quat = ObsTerm(func=mdp.plug_quat_in_robot_root_frame)
        plug_socket_offset = ObsTerm(
            func=mdp.plug_socket_offset,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        actions = ObsTerm(func=mdp.last_action_finite)

        def __post_init__(self) -> None:
            """Enable lightweight observation noise during training."""
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events for starting each episode with the plug pinched."""

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Do not read the TCP from FrameTransformer at reset: body poses are stale until the
    # next sim step, so the plug is expressed directly in the environment frame.
    place_plug_in_gripper = EventTerm(
        func=mdp.place_plug_in_gripper,
        mode="reset",
        params={
            "plug_pos_in_env": HELD_PLUG_POS,
            "plug_quat_xyzw": HELD_PLUG_QUAT_XYZW,
            "plug_cfg": SceneEntityCfg("plug"),
        },
    )

    close_gripper_on_plug = EventTerm(
        func=mdp.close_gripper_on_plug,
        mode="reset",
        params={
            "closed_position": GRIPPER_GRASP_POS,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
        },
    )


@configclass
class RJ45EventCfg(PresetCfg):
    """Event presets for backend selection.

    The current reset terms are backend-neutral, but the preset keeps CLI usage
    symmetric with ``env.sim=newton_mjwarp``.
    """

    default: EventCfg = EventCfg()
    physx: EventCfg = EventCfg()
    newton_mjwarp: EventCfg = EventCfg()
    newton_vbd: EventCfg = EventCfg()


@configclass
class RewardsCfg:
    """Factory-style staged rewards for RJ45 insertion."""

    ee_to_plug_distance = RewTerm(func=mdp.ee_to_plug_distance, params={"std": 0.05}, weight=0.5)

    plug_grasped = RewTerm(
        func=mdp.plug_grasped,
        params={"grasp_offset": PLUG_GRASP_OFFSET, "hold_radius": 0.025, "std": 0.01},
        weight=4.0,
    )

    gripper_closed = RewTerm(
        func=mdp.gripper_closed,
        params={
            "target_position": GRIPPER_GRASP_POS,
            "std": 0.002,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
        },
        weight=2.0,
    )

    plug_to_socket_distance = RewTerm(func=mdp.plug_to_socket_distance, params={"std": 0.25}, weight=5.0)

    insertion_bonus = RewTerm(
        func=mdp.insertion_bonus,
        params={"engaged_threshold": 0.02, "success_threshold": 0.005},
        weight=20.0,
    )

    plug_upright = RewTerm(func=mdp.plug_upright, weight=-0.5)

    action_rate = RewTerm(func=mdp.action_rate_l2_finite, weight=-1e-2)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2_finite,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the RJ45 insertion task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    plug_dropped = DoneTerm(
        func=mdp.plug_dropped,
        params={"minimum_height": -0.05, "plug_cfg": SceneEntityCfg("plug")},
    )


@configclass
class FrankaRJ45InsertEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based Franka task for inserting an RJ45 plug into a socket."""

    sim: RJ45SimCfg = RJ45SimCfg()
    scene: FrankaRJ45SceneCfg = FrankaRJ45SceneCfg(
        num_envs=128,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: RJ45EventCfg = RJ45EventCfg()

    def __post_init__(self) -> None:
        """Post-initialize task timing, viewer, Franka settings, and cable hooks."""
        self.decimation = 2
        self.episode_length_s = 10.0

        # Newton currently falls back to invalid inertias for unauthored Franka MassAPI
        # links; disabling arm gravity avoids NaNs while keeping contact dynamics active.
        if self.scene.robot.spawn.rigid_props is None:
            self.scene.robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
        self.scene.robot.spawn.rigid_props.disable_gravity = True

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.0, -1.0, 0.65)
        self.viewer.lookat = (0.49, 0.0, 0.39)

        # Register only the cable extension. Robot, plug, and socket remain
        # standard rigid assets from the scene config above.
        cable_mdp.register_cable_callbacks(
            self,
            source_usd_path=os.path.join(os.environ.get("ISAACLAB_RJ45_ASSET_DIR", ASSET_DIR), "rj45_plug.usd"),
        )


@configclass
class FrankaRJ45InsertEnvCfg_PLAY(FrankaRJ45InsertEnvCfg):
    """Play variant with fewer environments and deterministic observations."""

    def __post_init__(self) -> None:
        """Post-initialize the play scene."""
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
