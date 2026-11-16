"""Load each FR3 + end-effector model in a tabletop scene and check that it works.

For every model of :data:`MODELS` this builds the robot with the collection's own robot class next to a table (the
FR3 base 0.29 m below the tabletop, the rig height of the FR3 + Sharpa setups), then

* loads the home keyframe and steps, reporting any contact (the home pose must be collision-free) and any
  non-finite state or drift,
* solves the arm's ``palm`` IK for top-down poses 10 cm above the tabletop at robot-frame x 0.45-0.65 m and
  y -0.15..0.15 m, reports which of them the solver reaches, and writes each solution into the robot to compare
  the ``palm`` frame's forward kinematics with the target (that is what verifies the palm frame and the mount),
* opens and closes the hand through its own controller (:data:`CLOSED` for the dexterous hands, the single
  open/close command of a gripper) and reports the fingertips' travel and the largest joint velocity,
* renders the open and the closed hand side by side into ``<out-dir>/fr3_<name>.png``.

Notes on the checks:

* The IK is given ``max_samples=20`` (random restarts). The arm sub-controller's default single seeded solve
  (``max_samples=1``, chosen for frame-to-frame continuity) sticks in local minima for the large wrist
  reconfiguration these top-down poses need from the FR3's home pose, and its 0.4 rad/s rate limit would make
  the reconfiguration take tens of seconds of simulated time; neither is a property of the robot models.
* ``(0.45, 0.00)`` is out of the FR3's workspace with the palm down for the models whose palm frame is far from
  the flange (the long grippers: UMI 23 cm, ALOHA 15 cm, DEX-EE 19 cm): the fingers are level over the table at
  10 cm, so the arm's wrist has to be at the top of its reach.

Usage::

    python scripts/check_fr3_eef_models.py                      # all seven models
    python scripts/check_fr3_eef_models.py --models wuji umi
    python scripts/check_fr3_eef_models.py --out-dir /tmp/hands --cpu
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

import genesis as gs

from unienv_interface.world import WorldEnv
from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import CameraSensorNode, CameraSensorNodeOptions
from unienv_genesis_collection.robots.franka_fr3_aero import FrankaFR3WithAero
from unienv_genesis_collection.robots.franka_fr3_aloha import FrankaFR3WithAloha
from unienv_genesis_collection.robots.franka_fr3_dexee import FrankaFR3WithDexee
from unienv_genesis_collection.robots.franka_fr3_leap_v1 import FrankaFR3WithLeapV1
from unienv_genesis_collection.robots.franka_fr3_shadow import FrankaFR3WithShadow
from unienv_genesis_collection.robots.franka_fr3_umi import FrankaFR3WithUmi
from unienv_genesis_collection.robots.franka_fr3_wuji import FrankaFR3WithWuji
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.table_builder import TableBuilder

#: model name -> robot class.
MODELS = {
    "wuji": FrankaFR3WithWuji,
    "shadow": FrankaFR3WithShadow,
    "dexee": FrankaFR3WithDexee,
    "aero": FrankaFR3WithAero,
    "leap_v1": FrankaFR3WithLeapV1,
    "umi": FrankaFR3WithUmi,
    "aloha": FrankaFR3WithAloha,
}
#: The hand's closed pose, in the order of each class's ``HAND_JOINT_NAMES``; for the two grippers the single
#: open/close command's closed value.
CLOSED = {
    "wuji": [1.0, 0.4, 1.0, 0.8] + [1.3, 0.4, 1.2, 1.0] * 4,
    "shadow": [0.0, 1.2, 1.1] * 3 + [0.5, 0.0, 1.2, 1.1] + [0.5, 0.7, 0.0, -0.5, 1.0],
    "dexee": [0.0, 0.5, 1.0, 1.0] + [0.0, -1.0, 1.0, 1.0] * 2,
    "aero": [1.2, 1.0] * 4 + [0.9, 0.7, 0.9],
    "leap_v1": [1.2, 0.0, 1.0, 0.6] * 3 + [1.0, 0.3, 0.8, 0.8],
    "umi": 1.0,
    "aloha": 1.0,
}
TABLE_TOP = 0.6
#: The FR3 base sits this far below the tabletop (the FR3 + Sharpa rigs' height).
BASE_DROP = 0.29
#: Top-down palm poses to reach for, in the robot frame, 10 cm above the tabletop.
TARGET_X = (0.45, 0.55, 0.65)
TARGET_Y = (-0.15, 0.0, 0.15)
CLEARANCE = 0.10
#: The link the arm's ``eef_pose`` sub-controller drives (all seven models).
PALM = "palm"
#: A top-down palm: +x (out of the palm, or the gripper's approach) down, +z (along the fingers) away from the
#: robot, +y across them.
TOP_DOWN = np.stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--out-dir", default="/tmp/cookie_agent_quantumcookie/hands")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    if not gs._initialized:
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures: dict[str, list[str]] = {}
    for name in args.models:
        print(f"\n=== fr3_{name} " + "=" * 62)
        try:
            failures[name] = check(name, MODELS[name], out, args.steps)
        except Exception as exc:  # noqa: BLE001 - one broken model must not stop the others
            import traceback

            traceback.print_exc()
            failures[name] = [f"exception: {type(exc).__name__}: {exc}"]
        print(f"-> {name}: " + ("OK" if not failures[name] else "; ".join(failures[name])))
    print("\nSummary")
    for name in args.models:
        print(f"  {name:9s} {'OK' if not failures[name] else 'FAILED: ' + '; '.join(failures[name])}")
    return 1 if any(failures.values()) else 0


def check(name: str, cls, out: pathlib.Path, steps: int) -> list[str]:
    """Run the checks for one model; return the list of failures (empty when it worked)."""
    problems: list[str] = []
    world = GenesisWorld(scene_init_kwargs=dict(sim_options=gs.options.SimOptions(dt=1 / 30, substeps=5)))
    plane = PlaneBuilder(world, name="floor")
    table = TableBuilder(world, name="table", size=(0.9, 0.9, 0.02), pos=(0.55, 0.0, TABLE_TOP))
    robot = cls(world, name="franka", pos=(0.0, 0.0, TABLE_TOP - BASE_DROP))
    # framed on the hand at the FR3 home pose (the palm frame sits 0.12-0.34 m past the flange, at z 0.70-0.95)
    cam_pos, cam_lookat = (1.02, -0.62, 0.93), (0.55, 0.0, 0.87)
    camera = CameraSensorNode(
        world, name="camera",
        option=CameraSensorNodeOptions(backend="genesis", pos=cam_pos,
                                       quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
                                       res=(760, 620), fov=32, near=0.05, far=10.0),
        render_rgb=True,
    )
    env = WorldEnv(world, [plane, table, robot, camera], render_mode="rgb_array")
    env.reset(seed=0, reload=True)

    entity = robot.entity
    arm = robot.controller["arm"]
    sub_key, hand = _hand_controller(robot)
    mass = float(sum(float(np.asarray(link.get_mass().cpu()).ravel()[0]) for link in entity.links))
    print(f"  dofs {entity.n_dofs}, links {entity.n_links}, whole robot {mass:.3f} kg"
          f", hand joints {len(hand.joint_indexes)}")

    home = np.asarray(robot.keyframes["reset"].joint_position, dtype=float)
    open_cmd = _hand_action(hand, name, closed=False, device=world.device)
    home_pos, home_quat = arm.get_eef_pose_absolute()
    home_pos, home_quat = home_pos.detach().clone(), home_quat.detach().clone()

    # --- home pose: collision-free and steady ---------------------------------------------------------
    for _ in range(90):
        env.step({"arm": arm.get_action_world(home_pos, home_quat), sub_key: open_cmd})
    q, qd = _q(entity), _qd(entity)
    drift = float(np.abs(q[: home.size] - home).max())
    contacts, pairs = _contacts(entity)
    print(f"  home: |dq| {drift:.4f} rad, contacts {contacts} {pairs}, finite {bool(np.isfinite(q).all())}")
    if not (np.isfinite(q).all() and np.isfinite(qd).all()):
        problems.append("non-finite state at home")
    if drift > 0.05:
        problems.append(f"home pose not held (|dq| {drift:.3f} rad)")
    if contacts:
        problems.append(f"{contacts} contact(s) at the home pose {pairs}")

    # --- palm IK over the table -----------------------------------------------------------------------
    targets = [(x, y) for x in TARGET_X for y in TARGET_Y]
    quat = torch.tensor(_quat(TOP_DOWN), dtype=torch.float32, device=world.device).reshape(1, 4)
    ik_error, fk_error, solution = {}, {}, {}
    arm.ik_kwargs = {"max_samples": 20}
    full = _q(entity)
    for x, y in targets:
        pos = torch.tensor([[x, y, TABLE_TOP + CLEARANCE]], dtype=torch.float32, device=world.device)
        q_sol, error = arm.solve_ik_absolute(pos, quat)
        ik_error[(x, y)] = float(np.abs(np.asarray(error.cpu(), dtype=float)).max())
        solution[(x, y)] = np.asarray(q_sol.cpu(), dtype=float).ravel()
        full[list(arm.joint_indexes)] = solution[(x, y)]
        entity.set_dofs_position(full)  # also refreshes the link poses for the forward kinematics below
        reached = np.asarray(entity.get_link(PALM).get_pos().cpu(), dtype=float).ravel()
        fk_error[(x, y)] = float(np.linalg.norm(reached - np.array([x, y, TABLE_TOP + CLEARANCE])))
    arm.ik_kwargs = {}
    entity.set_dofs_position(_home_qpos(entity, home, arm))
    reachable = [t for t in targets if ik_error[t] < 0.005]
    missed = [f"({x:.2f},{y:+.2f})" for x, y in targets if ik_error[(x, y)] >= 0.005]
    worst_fk = max([fk_error[t] for t in reachable], default=float("nan"))
    print(f"  IK: {len(reachable)}/{len(targets)} top-down poses above the table reachable; on those the palm"
          f" frame's FK matches the target to {worst_fk * 1000:.2f} mm; unreachable: {', '.join(missed) or 'none'}")
    if len(reachable) < len(targets) - 1:
        problems.append(f"{len(missed)} of {len(targets)} top-down poses unreachable ({', '.join(missed)})")
    if reachable and worst_fk > 0.002:
        problems.append(f"palm FK {worst_fk * 1000:.0f} mm off the IK target")

    # --- open / close ---------------------------------------------------------------------------------
    frames, tips, joints = {}, {}, {}
    for state in ("open", "closed"):
        cmd = _hand_action(hand, name, closed=state == "closed", device=world.device)
        for _ in range(steps):
            env.step({"arm": arm.get_action_world(home_pos, home_quat), sub_key: cmd})
        q, qd = _q(entity), _qd(entity)
        tips[state] = _finger_tips(entity, cls)
        joints[state] = np.round(q[list(hand.joint_indexes)], 3)
        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            problems.append(f"non-finite state with the hand {state}")
        elif float(np.abs(qd).max()) > 60.0:
            problems.append(f"|qd| {np.abs(qd).max():.0f} rad/s with the hand {state}")
        rgb = env.render()
        frames[state] = np.asarray(rgb.cpu() if hasattr(rgb, "cpu") else rgb)
    travel = float(np.linalg.norm(tips["open"] - tips["closed"], axis=-1).max()) if len(tips["open"]) else 0.0
    print(f"  hand: fingertip travel {travel * 100:.1f} cm, max |qd| {float(np.abs(_qd(entity)).max()):.1f} rad/s")
    if travel < 0.005:
        problems.append("the fingers do not move between open and closed")
    print(f"    joint positions closed: {joints['closed']}")
    _save_pair(out / f"fr3_{name}.png", frames["open"], frames["closed"], name)
    print(f"  wrote {out / f'fr3_{name}.png'}")
    env.close()
    return problems


def _home_qpos(entity, home: np.ndarray, arm) -> np.ndarray:
    q = _q(entity)
    q[: home.size] = home
    return q


def _hand_controller(robot):
    """The end effector's sub-controller and its name: ``hand`` for a dexterous hand, ``gripper`` for a gripper."""
    subs = robot.controller._sub_controllers
    for key in ("hand", "gripper"):
        if key in subs:
            return key, subs[key]
    raise KeyError(f"no hand / gripper sub-controller in {list(subs)}")


def _hand_action(hand, name: str, closed: bool, device="cpu"):
    """The hand sub-controller's action: the joint target vector, or a gripper's single open/close command."""
    value = CLOSED[name]
    if not isinstance(value, list):
        return torch.tensor([value if closed else -value], dtype=torch.float32, device=device).reshape(1, 1)
    target = np.array(value if closed else [0.0] * len(value), dtype=np.float32)
    return torch.tensor(target, dtype=torch.float32, device=device).reshape(1, -1)


def _q(entity) -> np.ndarray:
    return np.asarray(entity.get_qpos().cpu(), dtype=float).ravel().copy()


def _qd(entity) -> np.ndarray:
    return np.asarray(entity.get_dofs_velocity().cpu(), dtype=float).ravel()


def _contacts(entity):
    """The number of contacts the robot has (with itself or anything else) and their link names."""
    try:
        contacts = entity.get_contacts()
    except Exception:  # noqa: BLE001
        return 0, ""
    if not contacts or contacts.get("link_a") is None:
        return 0, ""
    links = entity.scene.rigid_solver.links
    a = np.asarray(contacts["link_a"].cpu() if hasattr(contacts["link_a"], "cpu") else contacts["link_a"]).ravel()
    b = np.asarray(contacts["link_b"].cpu() if hasattr(contacts["link_b"], "cpu") else contacts["link_b"]).ravel()
    valid = contacts.get("valid_mask")
    if valid is not None:
        valid = np.asarray(valid.cpu() if hasattr(valid, "cpu") else valid).ravel().astype(bool)
        a, b = a[valid], b[valid]
    pairs = ", ".join(sorted({f"{links[int(i)].name}/{links[int(j)].name}" for i, j in zip(a, b)}))
    return int(a.size), (f"({pairs})" if pairs else "")


def _finger_tips(entity, cls) -> np.ndarray:
    """The positions of the fingertip / pad links the model's robot class reports for grasp detection."""
    module = sys.modules[cls.__module__]
    names = getattr(module, "FINGER_PAD_LINK_NAMES", None) or getattr(module, "GRIPPER_PAD_LINK_NAMES", ())
    return np.array([np.asarray(entity.get_link(n).get_pos().cpu(), dtype=float).ravel() for n in names])


def _quat(R: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` of a rotation matrix."""
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q, dtype=float)
    return q / np.linalg.norm(q)


def _save_pair(path: pathlib.Path, open_rgb: np.ndarray, closed_rgb: np.ndarray, name: str) -> None:
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.concatenate([np.asarray(open_rgb), np.asarray(closed_rgb)], axis=1))
    ImageDraw.Draw(im).text((10, 10), f"fr3_{name}: hand open (left) / closed (right)", fill=(255, 255, 0))
    im.save(path)


if __name__ == "__main__":
    sys.exit(main())
