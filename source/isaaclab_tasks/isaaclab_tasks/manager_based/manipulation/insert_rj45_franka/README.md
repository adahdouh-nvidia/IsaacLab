# Franka RJ45 Insertion

Manager-based Franka Panda environment for a small RJ45 plug-insertion task inspired by Newton's RJ45 contact example.

## Asset Prep

Place the combined Newton RJ45 asset at:

```bash
~/isaaclab_assets/rj45/rj45_plug.usd
```

Then split it into Isaac-Lab-ready USDs:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/insert_rj45_franka/scripts/split_rj45_usd.py
```

This writes `rj45_plug_body.usd` and `rj45_socket.usd`. The plug USD contains both the jack body and the clip/latch as one rigid body, so the clip cannot fall away from the gripper. The cable is intentionally omitted because faithful cable behavior needs constraints that are not configured in this first port.

If the plug body and clip appear separated in the viewer, rerun the splitter above so the standalone plug USD is regenerated with the body and clip in one normalized rigid frame.

## Run

Primitive smoke test:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Insert-RJ45-Franka-Stub-v0 --visualizer newton env.sim=newton_mjwarp env.events=newton_mjwarp
```

USD-backed task after asset prep:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Insert-RJ45-Franka-v0 --visualizer newton env.sim=newton_mjwarp env.events=newton_mjwarp
```

Play variants are available as `Isaac-Insert-RJ45-Franka-Play-v0` and `Isaac-Insert-RJ45-Franka-Stub-Play-v0`.

## Design Notes

The plug starts pinched in the gripper at a fixed environment-frame pose near the Franka default TCP: `(0.485, 0.0, 0.39)`. Its reset orientation adds a 90-degree yaw so the exposed connector end points toward the socket. The socket starts close by at `(0.495, 0.0, 0.39)` so early training focuses on the final insertion motion instead of long-range reaching. The reset event does not read the TCP from `FrameTransformer` because body poses are stale until the next sim step.

The socket is kinematic, while the plug is one dynamic rigid object with gravity disabled for this insertion-only bootstrap. It includes the clip/latch mesh, so modeling the clip as a separate free body is avoided; adding a true hinged latch remains a focused follow-up.

The reward includes grasp-maintenance terms: one keeps a plug-frame grasp point behind the exposed connector near the TCP, and one rewards the finger joints staying at the tight grasp target. These terms are intentionally active for the whole episode so the policy does not trade away grasp stability while chasing insertion distance.

For Newton+MJWarp, Franka rigid-body gravity is disabled because the Panda USD has unauthored `MassAPI` on several links, causing Newton to use invalid mass/inertia fallbacks. You may still see cosmetic inertia warnings from Newton.
