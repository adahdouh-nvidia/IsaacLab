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
``JointType.CABLE`` joints. The task uses that faithful rod path when the
``newton_vbd`` preset is active. MJWarp does not support ``JointType.CABLE`` in
this Isaac Lab checkout, so the ``newton_mjwarp`` preset keeps a visually
similar D6 capsule-chain fallback.

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

CABLE_BEND_STIFFNESS: float = 1.0
"""Cable bend stiffness [N*m/rad]."""

CABLE_BEND_DAMPING: float = 0.02
"""Cable bend damping."""

CABLE_ROD_BEND_STIFFNESS: float = 0.01
"""VBD rod bend stiffness [N*m/rad]."""

CABLE_ROD_STRETCH_STIFFNESS: float = 2.0e4
"""VBD rod stretch stiffness [N/m]."""

CABLE_ROD_STRETCH_DAMPING: float = 0.0
"""VBD rod stretch damping."""

CABLE_CONTACT_KE: float = 2.0e4
"""Cable contact stiffness [N/m]."""

CABLE_CONTACT_KD: float = 0.0
"""Cable contact damping."""

CABLE_ROD_CONTACT_KE: float = CABLE_CONTACT_KE
"""VBD rod contact stiffness [N/m]."""

CABLE_ROD_CONTACT_KD: float = CABLE_CONTACT_KD
"""VBD rod contact damping."""

CABLE_FRICTION: float = 1.0
"""Cable friction coefficient."""

CABLE_RENDER_COLOR: wp.vec3 = wp.vec3(0.0, 0.65, 0.12)
"""Cable capsule render color."""

CABLE_D6_BEND_STIFFNESS_SCALE: float = 2.0e-4
"""Scale applied to the reference bend stiffness for the MJWarp D6 fallback."""

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
    anchor_body_ids_wp: wp.array | None = None
    plug_body_ids_wp: wp.array | None = None
    anchor_offsets_wp: wp.array | None = None
    anchor_rotations_wp: wp.array | None = None
    align_body_ids_wp: wp.array | None = None
    align_next_body_ids_wp: wp.array | None = None


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
    return "" if _STATE is None else _STATE.solver_type


def _should_build_vbd_rods() -> bool:
    """Return whether the current backend supports Newton cable joints."""
    return _active_solver_type() == "vbd"


def _color_cable_rods(builder) -> None:
    """Color the Newton builder when real VBD rods were registered."""
    if _STATE is not None and _STATE.uses_vbd_rods:
        builder.color()


def _add_cable_to_builder_world(
    builder,
    env_idx: int,
    env_position: list[float],
    env_rotation: list[float] | tuple[float, float, float, float],
) -> None:
    """Add one plug-anchored Newton rod cable to the currently open builder world."""
    if _STATE is None:
        return

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

    plug_body_id = _find_plug_body_id(builder, env_idx)
    filtered_shape_ids = set(builder.body_shapes[plug_body_id])
    socket_body_id = _find_named_body_id(builder, env_idx, "Socket", required=False)
    if socket_body_id is not None:
        filtered_shape_ids.update(builder.body_shapes[socket_body_id])

    cable_points = _make_world_cable_points(_STATE, env_position, env_rotation)

    import newton

    cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
    plug_pos_w, plug_rot_w = _make_plug_world_pose(_STATE, env_position, env_rotation)
    plug_rot_w_inv = wp.quat_inverse(plug_rot_w)
    root_anchor_offset = wp.quat_rotate(plug_rot_w_inv, cable_points[0] - plug_pos_w)
    root_anchor_rot = wp.normalize(wp.mul(plug_rot_w_inv, cable_quats[0]))

    if _should_build_vbd_rods():
        _STATE.uses_vbd_rods = True
        cable_cfg = dataclasses.replace(
            builder.default_shape_cfg,
            ke=CABLE_ROD_CONTACT_KE,
            kd=CABLE_ROD_CONTACT_KD,
            mu=CABLE_FRICTION,
        )
        rod_bodies, rod_joints = _add_vbd_rod_cable(
            builder=builder,
            positions=cable_points,
            quaternions=cable_quats,
            cfg=cable_cfg,
            label=f"rj45_cable_{env_idx}",
        )
        _STATE.cable_joint_ids.extend(int(joint_id) for joint_id in rod_joints)
    else:
        global _WARNED_D6_FALLBACK
        if not _WARNED_D6_FALLBACK:
            logger.warning(
                "RJ45 cable is using the MJWarp D6 fallback, not Newton's JointType.CABLE rod path. "
                "Launch with env.sim=newton_vbd env.events=newton_vbd for the Cosserat-style VBD cable."
            )
            _WARNED_D6_FALLBACK = True
        cable_cfg = dataclasses.replace(
            builder.default_shape_cfg,
            ke=CABLE_CONTACT_KE,
            kd=CABLE_CONTACT_KD,
            mu=CABLE_FRICTION,
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

    # Prefix capsules overlap the plug at rest. Filtering these pairs prevents
    # the contact solver from fighting the kinematic teleport every substep.
    for body_id in rod_bodies[:CABLE_KINEMATIC_COUNT]:
        for cable_shape_id in builder.body_shapes[body_id]:
            for filtered_shape_id in filtered_shape_ids:
                builder.add_shape_collision_filter_pair(cable_shape_id, filtered_shape_id)

    _STATE.plug_body_ids.append(plug_body_id)
    _STATE.cable_body_ids_per_env.append(rod_bodies)
    if _STATE.uses_vbd_rods:
        # Match Newton's RJ45 example: include the last kinematic body so the
        # rendered boot segment points at the first dynamic body after bending.
        align_start = max(CABLE_KINEMATIC_COUNT - 1, 0)
        for segment_id in range(align_start, len(rod_bodies) - 1):
            _STATE.align_body_ids.append(rod_bodies[segment_id])
            _STATE.align_next_body_ids.append(rod_bodies[segment_id + 1])

    for prefix_id in range(CABLE_KINEMATIC_COUNT):
        anchor_body_id = rod_bodies[prefix_id]
        anchor_pos_w = cable_points[prefix_id]
        anchor_rot_w = cable_quats[prefix_id]
        _STATE.anchor_body_ids.append(anchor_body_id)
        _STATE.anchor_offsets.append(wp.quat_rotate(plug_rot_w_inv, anchor_pos_w - plug_pos_w))
        _STATE.anchor_rotations.append(wp.normalize(wp.mul(plug_rot_w_inv, anchor_rot_w)))


def _on_physics_ready(_payload: Any) -> None:
    """Create device arrays and install the per-step anchor sync callback."""
    if _STATE is None or not _STATE.anchor_body_ids:
        return

    from isaaclab_newton.physics import NewtonManager

    device = NewtonManager.get_device()
    _STATE.plug_body_ids_wp = wp.array(_STATE.plug_body_ids, dtype=wp.int32, device=device)
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


def _sync_cable_anchors() -> None:
    """Launch the Warp kernel that keeps kinematic cable bodies on the plug."""
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


def run_vbd_cable_solver_substeps(manager_cls: type) -> None:
    """Run VBD substeps with plug-anchor sync before collision.

    Args:
        manager_cls: Active Newton VBD manager class for the task.
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


def configure_vbd_cable_solver(solver: Any, model: Any | None = None) -> None:
    """Configure VBD cable joints to use compliant penalty constraints.

    Args:
        solver: Active :class:`newton.solvers.SolverVBD` instance.
        model: Optional finalized Newton model, used only as a label fallback.
    """
    if _STATE is None or not _STATE.uses_vbd_rods or not hasattr(solver, "set_joint_constraint_mode"):
        return

    joint_ids = list(_STATE.cable_joint_ids)
    if not joint_ids and model is not None:
        joint_labels = getattr(model, "joint_label", None) or getattr(model, "joint_key", None) or []
        joint_ids = [joint_id for joint_id, label in enumerate(joint_labels) if "rj45_cable_" in str(label)]

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

    if LOCK_FAR_END:
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
            target_kd=CABLE_BEND_DAMPING,
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
