# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split the Newton RJ45 combined USD into Isaac Lab rigid-object USDs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


@dataclass(frozen=True)
class PartSpec:
    """Description of a single RJ45 part to export."""

    source_paths: tuple[str, ...]
    output_name: str
    root_name: str
    mesh_names: tuple[str, ...]
    approximation: str
    color: tuple[float, float, float]
    mass: float | None = None
    kinematic: bool = False


PART_SPECS = {
    "plug": PartSpec(
        source_paths=("/World/Plug", "/World/Latch"),
        output_name="rj45_plug_body.usd",
        root_name="Plug",
        mesh_names=("PlugMesh", "LatchMesh"),
        approximation="convexDecomposition",
        color=(0.95, 0.78, 0.05),
        mass=0.006,
    ),
    "socket": PartSpec(
        source_paths=("/World/Socket",),
        output_name="rj45_socket.usd",
        root_name="Socket",
        mesh_names=("SocketMesh",),
        approximation="convexHull",
        color=(0.05, 0.22, 0.85),
        kinematic=True,
    ),
    "latch": PartSpec(
        source_paths=("/World/Latch",),
        output_name="rj45_latch.usd",
        root_name="Latch",
        mesh_names=("LatchMesh",),
        approximation="convexDecomposition",
        color=(0.95, 0.78, 0.05),
        mass=0.001,
    ),
    "cable": PartSpec(
        source_paths=("/World/Cable",),
        output_name="rj45_cable.usd",
        root_name="Cable",
        mesh_names=("CableMesh",),
        approximation="convexHull",
        color=(0.02, 0.02, 0.025),
        mass=0.002,
    ),
}


def _first_mesh_under(stage: Usd.Stage, prim_path: str) -> Usd.Prim:
    """Find the first mesh under a source part prim."""
    source_prim = stage.GetPrimAtPath(prim_path)
    if not source_prim.IsValid():
        raise ValueError(f"Source prim '{prim_path}' does not exist in {stage.GetRootLayer().realPath}.")
    if UsdGeom.Mesh(source_prim):
        return source_prim
    for prim in Usd.PrimRange(source_prim):
        if prim != source_prim and UsdGeom.Mesh(prim):
            return prim
    raise ValueError(f"Source prim '{prim_path}' has no mesh child.")


def _set_xform(dst_prim: Usd.Prim, transform: Gf.Matrix4d) -> None:
    """Set a prim transform as one matrix transform op."""
    dst_xform = UsdGeom.Xformable(dst_prim)
    dst_xform.ClearXformOpOrder()
    dst_xform.AddTransformOp().Set(transform)


def _copy_mesh(src_mesh_prim: Usd.Prim, dst_mesh: UsdGeom.Mesh, transform: Gf.Matrix4d) -> None:
    """Copy mesh topology and authored geometry attributes."""
    src_mesh = UsdGeom.Mesh(src_mesh_prim)
    dst_mesh.CreatePointsAttr(src_mesh.GetPointsAttr().Get())
    dst_mesh.CreateFaceVertexCountsAttr(src_mesh.GetFaceVertexCountsAttr().Get())
    dst_mesh.CreateFaceVertexIndicesAttr(src_mesh.GetFaceVertexIndicesAttr().Get())

    normals = src_mesh.GetNormalsAttr().Get()
    if normals is not None:
        dst_mesh.CreateNormalsAttr(normals)
        dst_mesh.SetNormalsInterpolation(src_mesh.GetNormalsInterpolation())

    extent = src_mesh.GetExtentAttr().Get()
    if extent is not None:
        dst_mesh.CreateExtentAttr(extent)

    _set_xform(dst_mesh.GetPrim(), transform)


def _bind_preview_surface(
    stage: Usd.Stage,
    mesh_prim: Usd.Prim,
    root_name: str,
    color: tuple[float, float, float],
) -> None:
    """Bind a minimal UsdPreviewSurface material to the mesh."""
    material = UsdShade.Material.Define(stage, f"/Materials/{root_name}Material")
    shader = UsdShade.Shader.Define(stage, f"/Materials/{root_name}Material/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface_output)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)


def _author_physics(root_prim: Usd.Prim, mesh_prims: list[Usd.Prim], spec: PartSpec) -> None:
    """Author rigid body, collision, mesh approximation, and mass schemas."""
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    rigid_body_api.CreateKinematicEnabledAttr(spec.kinematic)

    for mesh_prim in mesh_prims:
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.CreateApproximationAttr(spec.approximation)

    if spec.mass is not None:
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.CreateMassAttr(spec.mass)


def _export_part(src_stage: Usd.Stage, out_dir: Path, spec: PartSpec) -> Path:
    """Export one RJ45 part to a standalone USD."""
    out_path = out_dir / spec.output_name
    dst_stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(dst_stage, UsdGeom.GetStageUpAxis(src_stage))
    UsdGeom.SetStageMetersPerUnit(dst_stage, UsdGeom.GetStageMetersPerUnit(src_stage))

    root = UsdGeom.Xform.Define(dst_stage, f"/{spec.root_name}")
    dst_stage.SetDefaultPrim(root.GetPrim())
    source_root_prim = src_stage.GetPrimAtPath(spec.source_paths[0])
    if not source_root_prim.IsValid():
        raise ValueError(f"Source prim '{spec.source_paths[0]}' does not exist in {src_stage.GetRootLayer().realPath}.")
    _set_xform(root.GetPrim(), Gf.Matrix4d(1.0))

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_world = xform_cache.GetLocalToWorldTransform(source_root_prim)
    dst_mesh_prims = []
    for source_path, mesh_name in zip(spec.source_paths, spec.mesh_names, strict=True):
        src_mesh_prim = _first_mesh_under(src_stage, source_path)
        src_mesh_world = xform_cache.GetLocalToWorldTransform(src_mesh_prim)
        mesh_rel_to_root = src_mesh_world * root_world.GetInverse()
        dst_mesh = UsdGeom.Mesh.Define(dst_stage, f"/{spec.root_name}/{mesh_name}")
        _copy_mesh(src_mesh_prim, dst_mesh, mesh_rel_to_root)
        _bind_preview_surface(dst_stage, dst_mesh.GetPrim(), mesh_name, spec.color)
        dst_mesh_prims.append(dst_mesh.GetPrim())

    _author_physics(root.GetPrim(), dst_mesh_prims, spec)

    dst_stage.GetRootLayer().Save()
    return out_path


def main() -> None:
    """Run the RJ45 USD splitter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="~/isaaclab_assets/rj45/rj45_plug.usd", help="Combined RJ45 source USD.")
    parser.add_argument("--out", default="~/isaaclab_assets/rj45/", help="Output directory for split USDs.")
    parser.add_argument("--include-cable", action="store_true", help="Also export the cable mesh as a standalone USD.")
    parser.add_argument("--export-latch", action="store_true", help="Also export the latch as a separate rigid body.")
    args = parser.parse_args()

    src_path = Path(args.src).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    src_stage = Usd.Stage.Open(str(src_path))
    if src_stage is None:
        raise FileNotFoundError(f"Unable to open source USD: {src_path}")

    names = ["plug", "socket"]
    if args.export_latch:
        names.append("latch")
    if args.include_cable:
        names.append("cable")

    for name in names:
        out_path = _export_part(src_stage, out_dir, PART_SPECS[name])
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
