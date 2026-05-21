# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-builder cable hooks for the Franka RJ45 insertion task.

This module follows the per-world builder-hook pattern used by
``isaaclab_contrib.deformable`` instead of adding the rods in a late flat
``MODEL_INIT`` callback. The hook is invoked while each environment world is
open in the Newton :class:`ModelBuilder`, so each Isaac Lab environment gets
one independent cable rod without reopening or guessing world scopes.

Newton's upstream RJ45 example uses :meth:`ModelBuilder.add_rod`, which creates
``JointType.CABLE`` joints. MJWarp does not support those joints in this Isaac
Lab checkout, so the ``newton_vbd`` preset keeps MJWarp as the rigid scene
solver and builds the true rod in a separate cable-only VBD sidecar model. The
``newton_mjwarp`` preset remains a rigid D6 capsule-chain approximation only.

Implemented against Isaac Lab develop commit
``53bc3e02d7b1b798355cbbf2e155999bbff7a543``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import warp as wp
from pxr import Gf, Usd, UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


logger = logging.getLogger(__name__)

CABLE_RADIUS: float = 0.00325
"""Cable capsule radius [m]."""

CABLE_BEND_STIFFNESS: float = 0.1
"""Reference cable bend stiffness [N*m/rad]."""

CABLE_BEND_DAMPING: float = 0.1
"""Reference cable bend damping."""

CABLE_ROD_BEND_STIFFNESS: float = CABLE_BEND_STIFFNESS
"""VBD rod bend stiffness [N*m/rad]."""

CABLE_ROD_STRETCH_STIFFNESS: float = 1.0e9
"""VBD rod stretch stiffness [N/m]."""

CABLE_ROD_STRETCH_DAMPING: float = 0.1
"""VBD rod stretch damping."""

CABLE_CONTACT_KE: float = 1.0e8
"""Cable contact stiffness [N/m]."""

CABLE_CONTACT_KD: float = 1.0e-3
"""Cable contact damping."""

CABLE_ROD_CONTACT_KE: float = CABLE_CONTACT_KE
"""VBD rod contact stiffness [N/m]."""

CABLE_ROD_CONTACT_KD: float = CABLE_CONTACT_KD
"""VBD rod contact damping."""

CABLE_FRICTION: float = 2.0
"""Cable friction coefficient."""

CABLE_RENDER_COLOR: wp.vec3 = wp.vec3(0.0, 0.65, 0.12)
"""Cable capsule render color."""

CABLE_D6_BEND_STIFFNESS_SCALE: float = 2.0e-3
"""Scale applied to keep the MJWarp D6 fallback's bend spring visually similar."""

CABLE_D6_BEND_DAMPING: float = 0.02
"""D6 proxy bend damping used by the MJWarp-compatible fallback."""

CABLE_D6_CONTACT_KE: float = 2.0e4
"""D6 proxy contact stiffness [N/m] used by the MJWarp-compatible fallback."""

CABLE_D6_CONTACT_KD: float = 0.0
"""D6 proxy contact damping used by the MJWarp-compatible fallback."""

CABLE_D6_FRICTION: float = 1.0
"""D6 proxy friction coefficient used by the MJWarp-compatible fallback."""

CABLE_VISUAL_PROXY_MASS: float = 1.0e-4
"""Tiny positive proxy body mass [kg] required by MuJoCo conversion."""

CABLE_VISUAL_PROXY_INERTIA: float = 1.0e-8
"""Tiny positive proxy body inertia [kg*m^2] required by MuJoCo conversion."""

CABLE_KINEMATIC_COUNT: int = 4
"""First rod bodies that are teleported with the plug every solver step."""

KINEMATIC_PREFIX_COUNT: int = CABLE_KINEMATIC_COUNT
"""Backward-compatible alias for the cable kinematic prefix count."""

LOCK_FAR_END: bool = True
"""Whether the far cable end is pinned in world space for the VBD rod path."""

SOURCE_CABLE_CURVE_PATH = "/World/CableCurve"
SOURCE_PLUG_PATH = "/World/Plug"
CABLE_PLUG_ANCHOR_POS: wp.vec3 = wp.vec3(0.0, -0.0415, -0.00065)
"""Cable centerline start point in the split plug frame [m]."""


@dataclass
class _CableState:
    """Bookkeeping for the currently registered RJ45 cable hooks."""

    source_usd_path: Path
    plug_pos: tuple[float, float, float]
    plug_quat_xyzw: tuple[float, float, float, float]
    source_points: tuple[wp.vec3, ...] | None = None
    plug_body_ids: list[int] = field(default_factory=list)
    cable_body_ids_per_env: list[list[int]] = field(default_factory=list)
    cable_joint_ids: list[int] = field(default_factory=list)
    anchor_body_ids: list[int] = field(default_factory=list)
    anchor_offsets: list[wp.vec3] = field(default_factory=list)
    anchor_rotations: list[wp.quat] = field(default_factory=list)
    align_body_ids: list[int] = field(default_factory=list)
    align_next_body_ids: list[int] = field(default_factory=list)
    physics_ready_handle: Any | None = None
    solver_type: str = ""
    uses_vbd_rods: bool = False
    uses_split_vbd_sidecar: bool = False
    sidecar_builder: Any | None = None
    sidecar_model: Any | None = None
    sidecar_state_0: Any | None = None
    sidecar_state_1: Any | None = None
    sidecar_control: Any | None = None
    sidecar_solver: Any | None = None
    sidecar_body_ids_per_env: list[list[int]] = field(default_factory=list)
    sidecar_joint_ids: list[int] = field(default_factory=list)
    sidecar_anchor_body_ids: list[int] = field(default_factory=list)
    sidecar_anchor_offsets: list[wp.vec3] = field(default_factory=list)
    sidecar_anchor_rotations: list[wp.quat] = field(default_factory=list)
    sidecar_align_body_ids: list[int] = field(default_factory=list)
    sidecar_align_next_body_ids: list[int] = field(default_factory=list)
    anchor_body_ids_wp: wp.array | None = None
    plug_body_ids_wp: wp.array | None = None
    anchor_offsets_wp: wp.array | None = None
    anchor_rotations_wp: wp.array | None = None
    align_body_ids_wp: wp.array | None = None
    align_next_body_ids_wp: wp.array | None = None
    sidecar_anchor_body_ids_wp: wp.array | None = None
    sidecar_anchor_offsets_wp: wp.array | None = None
    sidecar_anchor_rotations_wp: wp.array | None = None
    sidecar_align_body_ids_wp: wp.array | None = None
    sidecar_align_next_body_ids_wp: wp.array | None = None
    sidecar_body_ids_wp: wp.array | None = None
    proxy_body_ids_wp: wp.array | None = None


_STATE: _CableState | None = None
_WARNED_D6_FAR_END_SKIP = False
_WARNED_D6_FALLBACK = False


@wp.kernel
def _sync_cable_anchors_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    plug_body_ids: wp.array(dtype=wp.int32),
    anchor_body_ids: wp.array(dtype=wp.int32),
    anchor_offsets: wp.array(dtype=wp.vec3),
    anchor_rotations: wp.array(dtype=wp.quat),
    prefix_count: int,
) -> None:
    """Teleport kinematic cable prefix bodies to the current plug pose."""
    tid = wp.tid()
    # Flat index math: tid spans all envs x prefix bodies. Dividing by
    # prefix_count recovers env_id; the same flat tid indexes that env's prefix
    # body id and its rest offset/rotation.
    env_id = tid // prefix_count
    plug_idx = plug_body_ids[env_id]
    anchor_idx = anchor_body_ids[tid]

    plug_tf = body_q[plug_idx]
    plug_pos = wp.transform_get_translation(plug_tf)
    plug_rot = wp.transform_get_rotation(plug_tf)

    anchor_world = plug_pos + wp.quat_rotate(plug_rot, anchor_offsets[tid])
    anchor_rot = wp.normalize(wp.mul(plug_rot, anchor_rotations[tid]))
    body_q[anchor_idx] = wp.transform(anchor_world, anchor_rot)
    body_qd[anchor_idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _sync_split_cable_anchors_kernel(
    rigid_body_q: wp.array(dtype=wp.transform),
    cable_body_q: wp.array(dtype=wp.transform),
    cable_body_qd: wp.array(dtype=wp.spatial_vector),
    plug_body_ids: wp.array(dtype=wp.int32),
    anchor_body_ids: wp.array(dtype=wp.int32),
    anchor_offsets: wp.array(dtype=wp.vec3),
    anchor_rotations: wp.array(dtype=wp.quat),
    prefix_count: int,
) -> None:
    """Teleport sidecar VBD prefix bodies to the current rigid plug pose."""
    tid = wp.tid()
    env_id = tid // prefix_count
    plug_idx = plug_body_ids[env_id]
    anchor_idx = anchor_body_ids[tid]

    plug_tf = rigid_body_q[plug_idx]
    plug_pos = wp.transform_get_translation(plug_tf)
    plug_rot = wp.transform_get_rotation(plug_tf)

    anchor_world = plug_pos + wp.quat_rotate(plug_rot, anchor_offsets[tid])
    anchor_rot = wp.normalize(wp.mul(plug_rot, anchor_rotations[tid]))
    cable_body_q[anchor_idx] = wp.transform(anchor_world, anchor_rot)
    cable_body_qd[anchor_idx] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _copy_split_cable_to_proxy_kernel(
    cable_body_q: wp.array(dtype=wp.transform),
    cable_body_qd: wp.array(dtype=wp.spatial_vector),
    proxy_body_q: wp.array(dtype=wp.transform),
    proxy_body_qd: wp.array(dtype=wp.spatial_vector),
    cable_body_ids: wp.array(dtype=wp.int32),
    proxy_body_ids: wp.array(dtype=wp.int32),
) -> None:
    """Copy sidecar VBD cable transforms into the MJWarp-compatible proxy."""
    tid = wp.tid()
    cable_body_id = cable_body_ids[tid]
    proxy_body_id = proxy_body_ids[tid]
    proxy_body_q[proxy_body_id] = cable_body_q[cable_body_id]
    proxy_body_qd[proxy_body_id] = cable_body_qd[cable_body_id]


@wp.kernel
def _align_cable_orientations_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_ids: wp.array(dtype=wp.int32),
    next_body_ids: wp.array(dtype=wp.int32),
) -> None:
    """Swing dynamic cable capsules so their local +Z follows the centerline."""
    tid = wp.tid()
    body_idx = body_ids[tid]
    next_body_idx = next_body_ids[tid]

    body_tf = body_q[body_idx]
    next_tf = body_q[next_body_idx]
    body_pos = wp.transform_get_translation(body_tf)
    next_body_pos = wp.transform_get_translation(next_tf)

    target_axis = next_body_pos - body_pos
    target_len = wp.length(target_axis)
    if target_len <= 1.0e-8:
        return
    target_axis = target_axis / target_len

    body_rot = wp.transform_get_rotation(body_tf)
    current_axis = wp.quat_rotate(body_rot, wp.vec3(0.0, 0.0, 1.0))
    dot_axis = wp.clamp(wp.dot(current_axis, target_axis), -1.0, 1.0)
    swing_axis = wp.cross(current_axis, target_axis)
    swing_axis_len = wp.length(swing_axis)

    if swing_axis_len > 1.0e-6:
        swing_axis = swing_axis / swing_axis_len
        swing_angle = wp.atan2(swing_axis_len, dot_axis)
        swing_rot = wp.quat_from_axis_angle(swing_axis, swing_angle)
        body_q[body_idx] = wp.transform(body_pos, wp.normalize(wp.mul(swing_rot, body_rot)))
    elif dot_axis < -0.999:
        fallback_axis = wp.cross(current_axis, wp.vec3(1.0, 0.0, 0.0))
        fallback_len = wp.length(fallback_axis)
        if fallback_len <= 1.0e-6:
            fallback_axis = wp.cross(current_axis, wp.vec3(0.0, 1.0, 0.0))
            fallback_len = wp.length(fallback_axis)
        fallback_axis = fallback_axis / fallback_len
        swing_rot = wp.quat_from_axis_angle(fallback_axis, 3.141592653589793)
        body_q[body_idx] = wp.transform(body_pos, wp.normalize(wp.mul(swing_rot, body_rot)))


def register_cable_callbacks(
    env_cfg: ManagerBasedRLEnvCfg,
    source_usd_path: str | os.PathLike[str] | None = None,
    source_points: list[wp.vec3] | tuple[wp.vec3, ...] | None = None,
) -> None:
    """Register Newton builder and step callbacks for the RJ45 cable.

    Args:
        env_cfg: The USD-backed RJ45 insertion environment config.
        source_usd_path: Optional combined RJ45 USD path containing
            ``/World/CableCurve``. If omitted, ``ISAACLAB_RJ45_ASSET_DIR`` is
            used, falling back to ``~/isaaclab_assets/rj45``.
        source_points: Optional cable centerline points in the plug frame [m].
            Used when a USD ``BasisCurves`` source is not available.
    """
    if _is_stub_env_cfg(env_cfg):
        clear_cable_callbacks()
        return

    if source_usd_path is None:
        asset_dir = Path(os.environ.get("ISAACLAB_RJ45_ASSET_DIR", "~/isaaclab_assets/rj45")).expanduser()
        source_usd_path = asset_dir / "rj45_plug.usd"

    clear_cable_callbacks()

    global _STATE
    _STATE = _CableState(
        source_usd_path=Path(source_usd_path).expanduser(),
        plug_pos=tuple(float(x) for x in env_cfg.scene.plug.init_state.pos),
        plug_quat_xyzw=tuple(float(x) for x in env_cfg.scene.plug.init_state.rot),
        source_points=None if source_points is None else tuple(source_points),
        solver_type=_infer_solver_type_from_env_cfg(env_cfg),
    )

    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    if not hasattr(NewtonManager, "_per_world_builder_hooks"):
        NewtonManager._per_world_builder_hooks = []
    if not hasattr(NewtonManager, "_post_replicate_hooks"):
        NewtonManager._post_replicate_hooks = []
    NewtonManager._per_world_builder_hooks.append(_add_cable_to_builder_world)
    if _color_cable_rods not in NewtonManager._post_replicate_hooks:
        NewtonManager._post_replicate_hooks.append(_color_cable_rods)
    _STATE.physics_ready_handle = NewtonManager.register_callback(
        _on_physics_ready,
        PhysicsEvent.PHYSICS_READY,
        order=50,
        name="rj45_cable_physics_ready",
        wrap_weak_ref=False,
    )


def clear_cable_callbacks() -> None:
    """Remove RJ45 cable hooks registered by :func:`register_cable_callbacks`."""
    global _STATE
    from isaaclab_newton.physics import NewtonManager

    if hasattr(NewtonManager, "_per_world_builder_hooks"):
        NewtonManager._per_world_builder_hooks = [
            hook for hook in NewtonManager._per_world_builder_hooks if hook is not _add_cable_to_builder_world
        ]
    if hasattr(NewtonManager, "_post_replicate_hooks"):
        NewtonManager._post_replicate_hooks = [
            hook for hook in NewtonManager._post_replicate_hooks if hook is not _color_cable_rods
        ]
    if hasattr(NewtonManager, "_post_actuator_callbacks"):
        NewtonManager._post_actuator_callbacks = [
            callback for callback in NewtonManager._post_actuator_callbacks if callback is not _sync_cable_anchors
        ]
    if _STATE is not None and _STATE.physics_ready_handle is not None:
        _STATE.physics_ready_handle.deregister()
    _STATE = None


def _is_stub_env_cfg(env_cfg: ManagerBasedRLEnvCfg) -> bool:
    """Return whether the config is one of the cable-free stub variants."""
    return any("Stub" in cls.__name__ for cls in type(env_cfg).__mro__)


def _infer_solver_type_from_env_cfg(env_cfg: ManagerBasedRLEnvCfg) -> str:
    """Infer the configured Newton solver type from the environment config."""
    sim_cfg = getattr(env_cfg, "sim", None)
    physics_cfg = getattr(sim_cfg, "physics", None)
    solver_cfg = getattr(physics_cfg, "solver_cfg", None)
    return str(getattr(solver_cfg, "solver_type", "")).lower()


def _active_solver_type() -> str:
    """Return the active Newton solver type, falling back to registration-time config."""
    try:
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        physics_cfg = getattr(sim, "_physics", None) if sim is not None else None
        solver_cfg = getattr(physics_cfg, "solver_cfg", None)
        solver_type = str(getattr(solver_cfg, "solver_type", "")).lower()
        if solver_type:
            return solver_type
    except Exception as exc:
        logger.debug("Unable to query active Newton solver type for RJ45 cable: %s", exc)

    try:
        from isaaclab.physics import PhysicsManager

        physics_cfg = PhysicsManager._cfg
        solver_cfg = getattr(physics_cfg, "solver_cfg", None)
        solver_type = str(getattr(solver_cfg, "solver_type", "")).lower()
        if solver_type:
            return solver_type
    except Exception as exc:
        logger.debug("Unable to query PhysicsManager solver type for RJ45 cable: %s", exc)

    return "" if _STATE is None else _STATE.solver_type


def _resolve_cable_builder_mode() -> str:
    """Return the RJ45 cable representation supported by the active Newton solver."""
    solver_type = _active_solver_type()
    if solver_type == "rj45_mjwarp_vbd_cable":
        return "split_vbd_proxy"
    if solver_type == "mujoco_warp":
        return "mjwarp_d6"
    if solver_type == "vbd":
        raise RuntimeError(
            "RJ45 env.sim=newton_vbd no longer routes the whole scene through SolverVBD. "
            "Use the task preset's RJ45SplitCableSolverCfg so MJWarp owns robot/plug/socket "
            "and VBD owns only the cable sidecar."
        )
    raise RuntimeError(
        "RJ45 cable supports only env.sim=newton_vbd (rigid scene + VBD rod cable) "
        "or env.sim=newton_mjwarp (MJWarp rigid scene + rigid D6 cable proxy). "
        f"Active Newton solver_type={solver_type!r}; refusing to build a cable representation implicitly."
    )


def _color_cable_rods(builder) -> None:
    """Color a main-model VBD rod builder if a legacy caller registered one."""
    if _STATE is not None and _STATE.uses_vbd_rods:
        builder.color()


def _add_cable_to_builder_world(
    builder,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Add one plug-anchored RJ45 cable to the currently open builder world."""
    if _STATE is None:
        return

    import newton

    cable_builder_mode = _resolve_cable_builder_mode()
    if env_idx == 0:
        _STATE.plug_body_ids.clear()
        _STATE.cable_body_ids_per_env.clear()
        _STATE.cable_joint_ids.clear()
        _STATE.anchor_body_ids.clear()
        _STATE.anchor_offsets.clear()
        _STATE.anchor_rotations.clear()
        _STATE.align_body_ids.clear()
        _STATE.align_next_body_ids.clear()
        _STATE.uses_vbd_rods = False
        _STATE.uses_split_vbd_sidecar = cable_builder_mode == "split_vbd_proxy"
        _STATE.sidecar_builder = newton.ModelBuilder() if _STATE.uses_split_vbd_sidecar else None
        _STATE.sidecar_model = None
        _STATE.sidecar_state_0 = None
        _STATE.sidecar_state_1 = None
        _STATE.sidecar_control = None
        _STATE.sidecar_solver = None
        _STATE.sidecar_body_ids_per_env.clear()
        _STATE.sidecar_joint_ids.clear()
        _STATE.sidecar_anchor_body_ids.clear()
        _STATE.sidecar_anchor_offsets.clear()
        _STATE.sidecar_anchor_rotations.clear()
        _STATE.sidecar_align_body_ids.clear()
        _STATE.sidecar_align_next_body_ids.clear()

    plug_body_id = _find_plug_body_id(builder, env_idx)
    filtered_shape_ids = set(builder.body_shapes[plug_body_id])
    socket_body_id = _find_named_body_id(builder, env_idx, "Socket", required=False)
    if socket_body_id is not None:
        filtered_shape_ids.update(builder.body_shapes[socket_body_id])

    cable_points = _make_world_cable_points(_STATE, env_position, env_rotation)

    cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
    plug_pos_w, plug_rot_w = _make_plug_world_pose(_STATE, env_position, env_rotation)
    plug_rot_w_inv = wp.quat_inverse(plug_rot_w)
    root_anchor_offset = wp.quat_rotate(plug_rot_w_inv, cable_points[0] - plug_pos_w)
    root_anchor_rot = wp.normalize(wp.mul(plug_rot_w_inv, cable_quats[0]))

    if cable_builder_mode == "split_vbd_proxy":
        if _STATE.sidecar_builder is None:
            raise RuntimeError("RJ45 VBD cable sidecar builder was not initialized.")
        _STATE.sidecar_builder.begin_world()
        sidecar_cfg = dataclasses.replace(
            _STATE.sidecar_builder.default_shape_cfg,
            ke=CABLE_ROD_CONTACT_KE,
            kd=CABLE_ROD_CONTACT_KD,
            mu=CABLE_FRICTION,
        )
        sidecar_bodies, sidecar_joints = _add_vbd_rod_cable(
            builder=_STATE.sidecar_builder,
            positions=cable_points,
            quaternions=cable_quats,
            cfg=sidecar_cfg,
            label=f"rj45_vbd_cable_{env_idx}",
        )
        _STATE.sidecar_builder.end_world()
        _STATE.sidecar_body_ids_per_env.append(sidecar_bodies)
        _STATE.sidecar_joint_ids.extend(int(joint_id) for joint_id in sidecar_joints)

        # MJWarp cannot own JointType.CABLE. The main model gets a non-colliding
        # tiny-mass capsule chain solely so the Newton viewer has cable bodies
        # to display; its transforms are overwritten from the VBD sidecar.
        proxy_cfg = dataclasses.replace(
            builder.default_shape_cfg,
            has_shape_collision=False,
            has_particle_collision=False,
            is_solid=False,
            ke=0.0,
            kd=0.0,
            mu=0.0,
        )
        rod_bodies, _ = _add_d6_capsule_chain(
            builder=builder,
            positions=cable_points,
            quaternions=cable_quats,
            cfg=proxy_cfg,
            label=f"rj45_cable_proxy_{env_idx}",
            root_parent_body_id=plug_body_id,
            root_parent_xform=wp.transform(root_anchor_offset, root_anchor_rot),
            warn_far_end_skip=False,
        )
        for body_id in rod_bodies:
            _set_body_visual_proxy(builder, body_id)
    elif cable_builder_mode == "mjwarp_d6":
        global _WARNED_D6_FALLBACK
        if not _WARNED_D6_FALLBACK:
            logger.warning(
                "RJ45 cable is using the MJWarp D6 fallback, not Newton's JointType.CABLE rod path. "
                "Launch with env.sim=newton_vbd env.events=newton_vbd for the Cosserat-style VBD cable."
            )
            _WARNED_D6_FALLBACK = True
        cable_cfg = dataclasses.replace(
            builder.default_shape_cfg,
            ke=CABLE_D6_CONTACT_KE,
            kd=CABLE_D6_CONTACT_KD,
            mu=CABLE_D6_FRICTION,
        )
        rod_bodies, _ = _add_d6_capsule_chain(
            builder=builder,
            positions=cable_points,
            quaternions=cable_quats,
            cfg=cable_cfg,
            label=f"rj45_cable_{env_idx}",
            root_parent_body_id=plug_body_id,
            root_parent_xform=wp.transform(root_anchor_offset, root_anchor_rot),
        )
    else:
        raise AssertionError(f"Unhandled RJ45 cable builder mode: {cable_builder_mode}")

    # Prefix capsules overlap the plug at rest. Filtering these pairs prevents
    # the contact solver from fighting the kinematic teleport every substep.
    for body_id in rod_bodies[:CABLE_KINEMATIC_COUNT]:
        for cable_shape_id in builder.body_shapes[body_id]:
            for filtered_shape_id in filtered_shape_ids:
                builder.add_shape_collision_filter_pair(cable_shape_id, filtered_shape_id)

    _STATE.plug_body_ids.append(plug_body_id)
    _STATE.cable_body_ids_per_env.append(rod_bodies)
    if _STATE.uses_split_vbd_sidecar:
        # Match Newton's RJ45 example: include the last kinematic body so the
        # rendered boot segment points at the first dynamic body after bending.
        align_start = max(CABLE_KINEMATIC_COUNT - 1, 0)
        align_bodies = sidecar_bodies
        align_body_ids = _STATE.sidecar_align_body_ids
        align_next_body_ids = _STATE.sidecar_align_next_body_ids
        for segment_id in range(align_start, len(align_bodies) - 1):
            align_body_ids.append(align_bodies[segment_id])
            align_next_body_ids.append(align_bodies[segment_id + 1])

    for prefix_id in range(CABLE_KINEMATIC_COUNT):
        anchor_pos_w = cable_points[prefix_id]
        anchor_rot_w = cable_quats[prefix_id]
        anchor_offset = wp.quat_rotate(plug_rot_w_inv, anchor_pos_w - plug_pos_w)
        anchor_rotation = wp.normalize(wp.mul(plug_rot_w_inv, anchor_rot_w))
        if _STATE.uses_split_vbd_sidecar:
            _STATE.sidecar_anchor_body_ids.append(sidecar_bodies[prefix_id])
            _STATE.sidecar_anchor_offsets.append(anchor_offset)
            _STATE.sidecar_anchor_rotations.append(anchor_rotation)
        else:
            _STATE.anchor_body_ids.append(rod_bodies[prefix_id])
            _STATE.anchor_offsets.append(anchor_offset)
            _STATE.anchor_rotations.append(anchor_rotation)


def _on_physics_ready(_payload: Any) -> None:
    """Create device arrays and install cable stepping state."""
    if _STATE is None:
        return

    from isaaclab_newton.physics import NewtonManager

    device = NewtonManager.get_device()
    _STATE.plug_body_ids_wp = wp.array(_STATE.plug_body_ids, dtype=wp.int32, device=device)
    if _STATE.uses_split_vbd_sidecar:
        _initialize_split_vbd_sidecar(device)
        sync_split_vbd_cable_from_rigid_state(NewtonManager.get_state_0())
        _copy_split_vbd_cable_to_proxy(NewtonManager.get_state_0())
        logger.info("Registered RJ45 split VBD cable sidecar for %d environments.", len(_STATE.plug_body_ids))
        return

    if not _STATE.anchor_body_ids:
        return

    _STATE.anchor_body_ids_wp = wp.array(_STATE.anchor_body_ids, dtype=wp.int32, device=device)
    _STATE.anchor_offsets_wp = wp.array(_STATE.anchor_offsets, dtype=wp.vec3, device=device)
    _STATE.anchor_rotations_wp = wp.array(_STATE.anchor_rotations, dtype=wp.quat, device=device)
    if _STATE.align_body_ids:
        _STATE.align_body_ids_wp = wp.array(_STATE.align_body_ids, dtype=wp.int32, device=device)
        _STATE.align_next_body_ids_wp = wp.array(_STATE.align_next_body_ids, dtype=wp.int32, device=device)

    if not _STATE.uses_vbd_rods:
        NewtonManager._post_actuator_callbacks = [
            callback for callback in NewtonManager._post_actuator_callbacks if callback is not _sync_cable_anchors
        ]
        NewtonManager.register_post_actuator_callback(_sync_cable_anchors)
    logger.info("Registered RJ45 cable sync for %d environments.", len(_STATE.plug_body_ids))


def _initialize_split_vbd_sidecar(device: str) -> None:
    """Finalize the cable-only VBD model and allocate sidecar device arrays."""
    if _STATE is None or _STATE.sidecar_builder is None:
        raise RuntimeError("RJ45 VBD cable sidecar was requested before its builder was created.")
    if not _STATE.sidecar_anchor_body_ids:
        raise RuntimeError("RJ45 VBD cable sidecar has no kinematic prefix bodies to sync.")

    import newton
    from newton.solvers import SolverVBD

    _STATE.sidecar_builder.color()
    model = _STATE.sidecar_builder.finalize(device=device)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0, None)
    state_1.assign(state_0)

    solver = SolverVBD(
        model,
        iterations=12,
        friction_epsilon=0.1,
        rigid_contact_hard=False,
        rigid_contact_k_start=1.0e5,
        rigid_body_contact_buffer_size=256,
    )
    configure_vbd_cable_solver(solver, model, joint_ids=_STATE.sidecar_joint_ids)

    _STATE.sidecar_model = model
    _STATE.sidecar_state_0 = state_0
    _STATE.sidecar_state_1 = state_1
    _STATE.sidecar_control = control
    _STATE.sidecar_solver = solver

    sidecar_body_ids = [body_id for body_ids in _STATE.sidecar_body_ids_per_env for body_id in body_ids]
    proxy_body_ids = [body_id for body_ids in _STATE.cable_body_ids_per_env for body_id in body_ids]
    if len(sidecar_body_ids) != len(proxy_body_ids):
        raise RuntimeError(
            "RJ45 sidecar cable and MJWarp proxy body counts differ: "
            f"{len(sidecar_body_ids)} vs {len(proxy_body_ids)}."
        )

    _STATE.sidecar_anchor_body_ids_wp = wp.array(_STATE.sidecar_anchor_body_ids, dtype=wp.int32, device=device)
    _STATE.sidecar_anchor_offsets_wp = wp.array(_STATE.sidecar_anchor_offsets, dtype=wp.vec3, device=device)
    _STATE.sidecar_anchor_rotations_wp = wp.array(_STATE.sidecar_anchor_rotations, dtype=wp.quat, device=device)
    _STATE.sidecar_body_ids_wp = wp.array(sidecar_body_ids, dtype=wp.int32, device=device)
    _STATE.proxy_body_ids_wp = wp.array(proxy_body_ids, dtype=wp.int32, device=device)
    if _STATE.sidecar_align_body_ids:
        _STATE.sidecar_align_body_ids_wp = wp.array(_STATE.sidecar_align_body_ids, dtype=wp.int32, device=device)
        _STATE.sidecar_align_next_body_ids_wp = wp.array(
            _STATE.sidecar_align_next_body_ids, dtype=wp.int32, device=device
        )


def _sync_cable_anchors() -> None:
    """Launch the Warp kernel that keeps kinematic cable prefix bodies on the plug."""
    if (
        _STATE is None
        or _STATE.plug_body_ids_wp is None
        or _STATE.anchor_body_ids_wp is None
        or _STATE.anchor_offsets_wp is None
        or _STATE.anchor_rotations_wp is None
    ):
        return

    from isaaclab_newton.physics import NewtonManager

    state = NewtonManager.get_state_0()
    wp.launch(
        kernel=_sync_cable_anchors_kernel,
        dim=len(_STATE.anchor_body_ids),
        inputs=[
            state.body_q,
            state.body_qd,
            _STATE.plug_body_ids_wp,
            _STATE.anchor_body_ids_wp,
            _STATE.anchor_offsets_wp,
            _STATE.anchor_rotations_wp,
            CABLE_KINEMATIC_COUNT,
        ],
        device=NewtonManager.get_device(),
    )


def sync_split_vbd_cable_from_rigid_state(rigid_state: Any) -> None:
    """Sync the sidecar cable prefix from the current rigid plug poses."""
    if (
        _STATE is None
        or not _STATE.uses_split_vbd_sidecar
        or _STATE.sidecar_state_0 is None
        or _STATE.plug_body_ids_wp is None
        or _STATE.sidecar_anchor_body_ids_wp is None
        or _STATE.sidecar_anchor_offsets_wp is None
        or _STATE.sidecar_anchor_rotations_wp is None
    ):
        return

    from isaaclab_newton.physics import NewtonManager

    wp.launch(
        kernel=_sync_split_cable_anchors_kernel,
        dim=len(_STATE.sidecar_anchor_body_ids),
        inputs=[
            rigid_state.body_q,
            _STATE.sidecar_state_0.body_q,
            _STATE.sidecar_state_0.body_qd,
            _STATE.plug_body_ids_wp,
            _STATE.sidecar_anchor_body_ids_wp,
            _STATE.sidecar_anchor_offsets_wp,
            _STATE.sidecar_anchor_rotations_wp,
            CABLE_KINEMATIC_COUNT,
        ],
        device=NewtonManager.get_device(),
    )


def step_split_vbd_cable(rigid_state: Any, dt: float) -> None:
    """Step the cable-only VBD model and copy it into the rigid-scene proxy."""
    if (
        _STATE is None
        or not _STATE.uses_split_vbd_sidecar
        or _STATE.sidecar_solver is None
        or _STATE.sidecar_state_0 is None
        or _STATE.sidecar_state_1 is None
        or _STATE.sidecar_control is None
    ):
        return

    # Sync immediately before the VBD solve so the rod root follows the plug
    # pose produced by the just-completed MJWarp rigid substep.
    sync_split_vbd_cable_from_rigid_state(rigid_state)
    _STATE.sidecar_solver.step(
        _STATE.sidecar_state_0,
        _STATE.sidecar_state_1,
        _STATE.sidecar_control,
        None,
        dt,
    )
    _align_split_cable_orientations(_STATE.sidecar_state_1)
    _STATE.sidecar_state_0, _STATE.sidecar_state_1 = _STATE.sidecar_state_1, _STATE.sidecar_state_0
    _copy_split_vbd_cable_to_proxy(rigid_state)
    _STATE.sidecar_state_0.clear_forces()


def _copy_split_vbd_cable_to_proxy(rigid_state: Any) -> None:
    """Copy solved VBD cable poses into the main MJWarp proxy bodies."""
    if (
        _STATE is None
        or _STATE.sidecar_state_0 is None
        or _STATE.sidecar_body_ids_wp is None
        or _STATE.proxy_body_ids_wp is None
    ):
        return

    from isaaclab_newton.physics import NewtonManager

    wp.launch(
        kernel=_copy_split_cable_to_proxy_kernel,
        dim=_STATE.sidecar_body_ids_wp.shape[0],
        inputs=[
            _STATE.sidecar_state_0.body_q,
            _STATE.sidecar_state_0.body_qd,
            rigid_state.body_q,
            rigid_state.body_qd,
            _STATE.sidecar_body_ids_wp,
            _STATE.proxy_body_ids_wp,
        ],
        device=NewtonManager.get_device(),
    )


def _align_cable_orientations(state: Any) -> None:
    """Launch the optional visual pass that aligns cable capsule rotations."""
    if (
        _STATE is None
        or _STATE.align_body_ids_wp is None
        or _STATE.align_next_body_ids_wp is None
        or not _STATE.uses_vbd_rods
    ):
        return

    from isaaclab_newton.physics import NewtonManager

    wp.launch(
        kernel=_align_cable_orientations_kernel,
        dim=len(_STATE.align_body_ids),
        inputs=[
            state.body_q,
            _STATE.align_body_ids_wp,
            _STATE.align_next_body_ids_wp,
        ],
        device=NewtonManager.get_device(),
    )


def _align_split_cable_orientations(state: Any) -> None:
    """Align sidecar cable capsules before copying them into the visual proxy."""
    if (
        _STATE is None
        or _STATE.sidecar_align_body_ids_wp is None
        or _STATE.sidecar_align_next_body_ids_wp is None
    ):
        return

    from isaaclab_newton.physics import NewtonManager

    wp.launch(
        kernel=_align_cable_orientations_kernel,
        dim=len(_STATE.sidecar_align_body_ids),
        inputs=[
            state.body_q,
            _STATE.sidecar_align_body_ids_wp,
            _STATE.sidecar_align_next_body_ids_wp,
        ],
        device=NewtonManager.get_device(),
    )


def run_vbd_cable_solver_substeps(manager_cls: type) -> None:
    """Legacy helper for the removed main-model VBD rod experiment.

    Current RJ45 presets do not call this path. ``env.sim=newton_vbd`` uses
    :func:`step_split_vbd_cable` so MJWarp owns the rigid scene and VBD owns
    only the sidecar cable model.

    Args:
        manager_cls: Legacy Newton VBD manager class.
    """
    if _STATE is None or not _STATE.uses_vbd_rods:
        contacts = manager_cls._contacts if manager_cls._needs_collision_pipeline else None
        if manager_cls._needs_collision_pipeline:
            manager_cls._collision_pipeline.collide(manager_cls._state_0, manager_cls._contacts)
        manager_cls._run_solver_substeps(contacts)
        return

    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics import NewtonManager

    contacts = manager_cls._contacts if manager_cls._needs_collision_pipeline else None
    if manager_cls._use_single_state:
        for _ in range(manager_cls._num_substeps):
            _sync_cable_anchors()
            if manager_cls._needs_collision_pipeline:
                manager_cls._collision_pipeline.collide(manager_cls._state_0, manager_cls._contacts)
            manager_cls._step_solver(
                manager_cls._state_0,
                manager_cls._state_0,
                manager_cls._control,
                contacts,
                manager_cls._solver_dt,
            )
            _align_cable_orientations(manager_cls._state_0)
            manager_cls._state_0.clear_forces()
        return

    cfg = PhysicsManager._cfg
    need_copy_on_last = (
        cfg is not None and cfg.use_cuda_graph and manager_cls._num_substeps % 2 == 1  # type: ignore[union-attr]
    )
    for substep_id in range(manager_cls._num_substeps):
        _sync_cable_anchors()
        if manager_cls._needs_collision_pipeline:
            manager_cls._collision_pipeline.collide(manager_cls._state_0, manager_cls._contacts)
        manager_cls._step_solver(
            manager_cls._state_0,
            manager_cls._state_1,
            manager_cls._control,
            contacts,
            manager_cls._solver_dt,
        )
        _align_cable_orientations(manager_cls._state_1)
        if need_copy_on_last and substep_id == manager_cls._num_substeps - 1:
            manager_cls._state_0.assign(manager_cls._state_1)
        else:
            NewtonManager._state_0, NewtonManager._state_1 = manager_cls._state_1, manager_cls._state_0
        manager_cls._state_0.clear_forces()


def configure_vbd_cable_solver(solver: Any, model: Any | None = None, joint_ids: list[int] | None = None) -> None:
    """Configure VBD cable joints to use compliant penalty constraints.

    Args:
        solver: Active :class:`newton.solvers.SolverVBD` instance.
        model: Optional finalized Newton model, used only as a label fallback.
        joint_ids: Optional explicit cable joint ids. When omitted, the ids
            recorded for the active RJ45 cable representation are used.
    """
    if _STATE is None or not hasattr(solver, "set_joint_constraint_mode"):
        return

    if joint_ids is None:
        if _STATE.uses_vbd_rods:
            joint_ids = list(_STATE.cable_joint_ids)
        elif _STATE.uses_split_vbd_sidecar:
            joint_ids = list(_STATE.sidecar_joint_ids)
        else:
            return
    else:
        joint_ids = list(joint_ids)
    if not joint_ids and model is not None:
        joint_labels = getattr(model, "joint_label", None) or getattr(model, "joint_key", None) or []
        joint_ids = [joint_id for joint_id, label in enumerate(joint_labels) if "rj45" in str(label)]

    for joint_id in joint_ids:
        # SolverVBD defaults structural joints to hard constraints. Cable bend
        # stiffness only affects motion when the rod joints run in compliant
        # penalty mode, matching Newton's reference RJ45 example.
        solver.set_joint_constraint_mode(int(joint_id), False)

    logger.info("Configured %d RJ45 cable VBD joints as compliant constraints.", len(joint_ids))


def _find_plug_body_id(builder, env_idx: int) -> int:
    """Resolve the plug body index for one builder world."""
    plug_body_id = _find_named_body_id(builder, env_idx, "Plug", required=True)
    if plug_body_id is None:
        raise RuntimeError(f"Unable to find RJ45 plug body in Newton builder world {env_idx}.")
    return plug_body_id


def _find_named_body_id(builder, env_idx: int, body_name: str, required: bool = True) -> int | None:
    """Resolve a named body index for one builder world."""
    body_world = getattr(builder, "body_world", [])
    candidates = []
    for body_id, label in enumerate(getattr(builder, "body_label", [])):
        if len(body_world) > body_id and int(body_world[body_id]) != env_idx:
            continue
        label_str = str(label)
        if label_str.endswith(f"/{body_name}") or label_str.endswith(body_name):
            candidates.append(body_id)

    if not candidates:
        shape_world = getattr(builder, "shape_world", [])
        for shape_id, label in enumerate(getattr(builder, "shape_label", [])):
            if len(shape_world) > shape_id and int(shape_world[shape_id]) != env_idx:
                continue
            if f"/{body_name}" in str(label):
                candidates.append(int(builder.shape_body[shape_id]))

    if not candidates:
        if not required:
            return None
        labels = [str(label) for label in getattr(builder, "body_label", [])[-20:]]
        raise RuntimeError(
            f"Unable to find RJ45 {body_name} body in Newton builder world {env_idx}. Recent bodies: {labels}"
        )
    return candidates[-1]


def _set_body_kinematic(builder, body_id: int) -> None:
    """Set a Newton builder body to zero mass and inertia."""
    builder.body_mass[body_id] = 0.0
    builder.body_inv_mass[body_id] = 0.0
    builder.body_inertia[body_id] = wp.mat33(0.0)
    builder.body_inv_inertia[body_id] = wp.mat33(0.0)


def _set_body_visual_proxy(builder, body_id: int) -> None:
    """Set a tiny positive mass so MJWarp accepts a visual-only proxy body."""
    builder.body_mass[body_id] = CABLE_VISUAL_PROXY_MASS
    builder.body_inv_mass[body_id] = 1.0 / CABLE_VISUAL_PROXY_MASS
    builder.body_inertia[body_id] = wp.mat33(
        CABLE_VISUAL_PROXY_INERTIA,
        0.0,
        0.0,
        0.0,
        CABLE_VISUAL_PROXY_INERTIA,
        0.0,
        0.0,
        0.0,
        CABLE_VISUAL_PROXY_INERTIA,
    )
    builder.body_inv_inertia[body_id] = wp.mat33(
        1.0 / CABLE_VISUAL_PROXY_INERTIA,
        0.0,
        0.0,
        0.0,
        1.0 / CABLE_VISUAL_PROXY_INERTIA,
        0.0,
        0.0,
        0.0,
        1.0 / CABLE_VISUAL_PROXY_INERTIA,
    )


def _add_vbd_rod_cable(
    builder,
    positions: list[wp.vec3],
    quaternions: list[wp.quat],
    cfg,
    label: str,
):
    """Add the faithful Newton cable rod used by the VBD solver.

    Args:
        builder: Newton model builder for the currently open environment world.
        positions: Cable centerline node positions [m] in world frame.
        quaternions: Per-segment orientations in world frame.
        cfg: Shape collision configuration for each capsule.
        label: Prefix for generated body, shape, joint, and articulation labels.

    Returns:
        Tuple of created body indices and joint indices.
    """
    rod_bodies, rod_joints = builder.add_rod(
        positions=positions,
        quaternions=quaternions,
        radius=CABLE_RADIUS,
        cfg=cfg,
        bend_stiffness=CABLE_ROD_BEND_STIFFNESS,
        bend_damping=CABLE_BEND_DAMPING,
        stretch_stiffness=CABLE_ROD_STRETCH_STIFFNESS,
        stretch_damping=CABLE_ROD_STRETCH_DAMPING,
        label=label,
    )

    # The first four capsules are embedded in the boot of the plug in the
    # reference demo. Massless bodies are cheaper and more stable than adding
    # explicit fixed joints because the sync kernel overwrites their poses each
    # substep before collision detection.
    kinematic_body_ids = list(rod_bodies[:CABLE_KINEMATIC_COUNT])
    if LOCK_FAR_END:
        # The far end is a world-space anchor, matching the Newton RJ45 example.
        kinematic_body_ids.append(rod_bodies[-1])

    for body_id in kinematic_body_ids:
        _set_body_kinematic(builder, body_id)

    for body_id in rod_bodies:
        for shape_id in builder.body_shapes[body_id]:
            builder.shape_color[shape_id] = CABLE_RENDER_COLOR

    return rod_bodies, rod_joints


def _add_d6_capsule_chain(
    builder,
    positions: list[wp.vec3],
    quaternions: list[wp.quat],
    cfg,
    label: str,
    root_parent_body_id: int,
    root_parent_xform: wp.transform,
    warn_far_end_skip: bool = True,
):
    """Add an MJWarp-compatible bend-stiff capsule chain.

    Args:
        builder: Newton model builder for the currently open environment world.
        positions: Cable centerline node positions [m] in world frame.
        quaternions: Per-segment orientations in world frame.
        cfg: Shape collision configuration for each capsule.
        label: Prefix for generated body, shape, joint, and articulation labels.
        root_parent_body_id: Plug body index to attach the cable root to.
        root_parent_xform: Cable root transform in the plug frame.
        warn_far_end_skip: Whether to warn that the D6 approximation skips the
            far-end VBD anchor.

    Returns:
        Tuple of created body indices and joint indices.
    """
    if len(quaternions) != len(positions) - 1:
        raise ValueError(
            f"Expected {len(positions) - 1} cable segment orientations, got {len(quaternions)}."
        )

    bodies: list[int] = []
    joints: list[int] = []
    segment_lengths: list[float] = []

    for segment_id, (p0, p1, segment_quat) in enumerate(zip(positions[:-1], positions[1:], quaternions)):
        segment_length = float(wp.length(p1 - p0))
        if segment_length <= 1.0e-9:
            raise ValueError(f"RJ45 cable segment {segment_id} is too short: {segment_length:.3e} m.")

        half_height = 0.5 * segment_length
        body_id = builder.add_link(
            xform=wp.transform(p0, segment_quat),
            com=wp.vec3(0.0, 0.0, half_height),
            label=f"{label}_edge_body_{segment_id}",
        )
        builder.add_shape_capsule(
            body_id,
            xform=wp.transform(wp.vec3(0.0, 0.0, half_height), wp.quat_identity()),
            radius=CABLE_RADIUS,
            half_height=half_height,
            cfg=cfg,
            color=CABLE_RENDER_COLOR,
            label=f"{label}_edge_capsule_{segment_id}",
        )
        bodies.append(body_id)
        segment_lengths.append(segment_length)

    # Root the articulation on the plug, not the world. A world-rooted fixed
    # joint makes the cable look static when the plug moves or falls.
    joints.append(
        builder.add_joint_fixed(
            parent=root_parent_body_id,
            child=bodies[0],
            parent_xform=root_parent_xform,
            child_xform=wp.transform_identity(),
            label=f"{label}_plug_root",
            collision_filter_parent=False,
        )
    )

    for segment_id in range(len(bodies) - 1):
        parent_quat = quaternions[segment_id]
        child_quat = quaternions[segment_id + 1]
        # Align the child anchor's rest frame to the parent anchor frame so the
        # D6 angular spring targets the source cable's curved rest shape.
        child_anchor_rot = wp.normalize(wp.mul(wp.quat_inverse(child_quat), parent_quat))
        joints.append(
            builder.add_joint_d6(
                parent=bodies[segment_id],
                child=bodies[segment_id + 1],
                linear_axes=[],
                angular_axes=_make_bend_axes(builder),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, segment_lengths[segment_id]), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), child_anchor_rot),
                label=f"{label}_bend_{segment_id}",
                collision_filter_parent=True,
            )
        )

    if LOCK_FAR_END and warn_far_end_skip:
        global _WARNED_D6_FAR_END_SKIP
        if not _WARNED_D6_FAR_END_SKIP:
            logger.warning(
                "RJ45 cable far-end lock is skipped for the MJWarp D6 fallback: "
                "world-loop welds can make articulated assets disappear in the Newton viewer."
            )
            _WARNED_D6_FAR_END_SKIP = True

    builder.add_articulation(joints, label=f"{label}_articulation")
    return bodies, joints


def _make_bend_axes(builder) -> list:
    """Create angular D6 axes with the Newton example's bend spring settings."""
    # ``JointType.CABLE`` is a VBD penalty; MJWarp receives these D6 axes as
    # hinge actuators with physical torque gains. The scale keeps the fallback
    # visibly flexible without changing the reference constants above.
    bend_stiffness = CABLE_BEND_STIFFNESS * CABLE_D6_BEND_STIFFNESS_SCALE
    return [
        builder.JointDofConfig(
            axis=axis,
            limit_ke=0.0,
            limit_kd=0.0,
            target_ke=bend_stiffness,
            target_kd=CABLE_D6_BEND_DAMPING,
            effort_limit=1.0e6,
        )
        for axis in (wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0))
    ]


def _make_world_cable_points(
    state: _CableState,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> list[wp.vec3]:
    """Transform source cable points from plug frame into one environment world."""
    plug_pos_w, plug_rot_w = _make_plug_world_pose(state, env_position, env_rotation)
    source_points = state.source_points if state.source_points is not None else _load_cable_centerline(state.source_usd_path)
    return [plug_pos_w + wp.quat_rotate(plug_rot_w, point) for point in source_points]


def _make_plug_world_pose(
    state: _CableState,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> tuple[wp.vec3, wp.quat]:
    """Return the reset plug pose for one environment world."""
    env_pos = _vec3(env_position)
    env_rot = _quat(env_rotation)
    plug_pos = _vec3(state.plug_pos)
    plug_rot = _quat(state.plug_quat_xyzw)
    return env_pos + wp.quat_rotate(env_rot, plug_pos), wp.normalize(wp.mul(env_rot, plug_rot))


@lru_cache(maxsize=8)
def _load_cable_centerline(source_usd_path: Path) -> tuple[wp.vec3, ...]:
    """Read the source ``CableCurve`` once and return points in plug frame."""
    stage = Usd.Stage.Open(str(source_usd_path))
    if stage is None:
        raise FileNotFoundError(f"Unable to open RJ45 source USD: {source_usd_path}")

    curve_prim = stage.GetPrimAtPath(SOURCE_CABLE_CURVE_PATH)
    if not curve_prim.IsValid():
        raise ValueError(f"Source USD '{source_usd_path}' does not contain {SOURCE_CABLE_CURVE_PATH}.")
    plug_prim = stage.GetPrimAtPath(SOURCE_PLUG_PATH)
    if not plug_prim.IsValid():
        raise ValueError(f"Source USD '{source_usd_path}' does not contain {SOURCE_PLUG_PATH}.")

    points = UsdGeom.BasisCurves(curve_prim).GetPointsAttr().Get()
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    curve_world = xform_cache.GetLocalToWorldTransform(curve_prim)
    plug_world = xform_cache.GetLocalToWorldTransform(plug_prim)
    plug_world_inv = plug_world.GetInverse()

    out = []
    for point in points:
        point_w = curve_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        point_plug = plug_world_inv.Transform(point_w)
        out.append(
            wp.vec3(
                float(point_plug[0]),
                float(point_plug[1]),
                float(point_plug[2]),
            )
        )
    if len(out) <= CABLE_KINEMATIC_COUNT:
        raise ValueError(
            f"RJ45 cable curve needs more than {CABLE_KINEMATIC_COUNT} points; "
            f"found {len(out)} in {source_usd_path}."
        )

    # The source curve is authored against the combined USD plug, while the
    # task spawns a split plug body whose rear boot sits at about -44 mm in its
    # local y-axis. Snapping the first cable point to that rear boot makes the
    # cable visibly emerge from the rendered plug while preserving the curve's
    # Newton-example rest shape.
    anchor_delta = CABLE_PLUG_ANCHOR_POS - out[0]
    anchored_points = [point + anchor_delta for point in out]

    # The source curve is a U-shaped path that doubles back toward the plug.
    # Keeping the return branch creates a second open green end near the plug,
    # which reads as a disconnected cable in the viewer. Use the outward branch
    # ending at the farthest point from the plug anchor for this insertion task.
    tail_index = max(range(len(anchored_points)), key=lambda index: _distance_sq(anchored_points[index], anchored_points[0]))
    return tuple(anchored_points[: tail_index + 1])


def _distance_sq(lhs: wp.vec3, rhs: wp.vec3) -> float:
    """Return squared distance between two vectors [m^2]."""
    dx = float(lhs[0] - rhs[0])
    dy = float(lhs[1] - rhs[1])
    dz = float(lhs[2] - rhs[2])
    return dx * dx + dy * dy + dz * dz


def _vec3(values: list[float] | tuple[float, float, float]) -> wp.vec3:
    """Create a Warp vector from a three-value sequence."""
    return wp.vec3(float(values[0]), float(values[1]), float(values[2]))


def _quat(values: list[float] | tuple[float, float, float, float]) -> wp.quat:
    """Create a normalized Warp quaternion from an ``(x, y, z, w)`` sequence."""
    return wp.normalize(wp.quat(float(values[0]), float(values[1]), float(values[2]), float(values[3])))
