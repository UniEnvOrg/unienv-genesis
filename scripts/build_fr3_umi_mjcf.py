"""Build the FR3 + UMI gripper MJCF from the mujoco_menagerie model in ``unienv_genesis_collection/assets``.

Writes ``franka_fr3_umi/fr3_umi.xml``: the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` with its Franka Hand
replaced by the UMI (Universal Manipulation Interface) gripper (``umi_gripper/umi_gripper.xml``, mujoco_menagerie
``umi_gripper``, MIT, pinned in ``umi_gripper/SOURCE.txt``).

Conversion:

* **Hand-held base made rigid.** Upstream the gripper's base plate floats on six joints
  (``gripper_joint_x`` / ``_y`` / ``_z`` slides and ``_rx`` / ``_ry`` / ``_rz`` hinges, 6-DOF compliance of the
  human hand holding it). On the FR3 the arm carries the gripper, so those six joints and their actuators are
  dropped and the base plate is bolted to the flange; the mesh, materials and textures of the base are unchanged.
* **Fingers.** The two finger carriages slide on the rail (``left_finger_joint`` / ``right_finger_joint``, 0 to
  50 mm each), coupled upstream through the ``split`` tendon; Genesis ignores tendons, so each gets its own
  position actuator (the class's gains) and the robot class drives both with one command. 0 is fully open (an
  83 mm gap between the finger plates), 42 mm per finger is closed. The carriages' return spring (the hand-held
  rig's ``stiffness=100``, as large as the actuator's own gain, which would leave the closed pose at half travel)
  is dropped: the actuator holds the fingers.
* **Mount.** Mounted as designed on its own base plate's rear face (the side away from the fingers), through a
  5 mm adapter, so the approach direction is out of the flange.
* **Palm frame.** A ``palm`` body between the two fingers' tips, 228 mm out along the approach direction from the
  base's rear face: +z along the approach direction (the fingers' way), +y along the closing direction (from the
  left finger to the right one, the model's +x) and +x across them.

The camera and the ArUco marker materials of the handheld rig are kept as they are (only the ``<camera>``
element is dropped).

Run from the repository root:  python scripts/build_fr3_umi_mjcf.py
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import ROBOTS, gripper_mount, hand_assets, hand_defaults, palm_frame, rename, set_pose, write_model

FINGER_JOINTS = ("left_finger_joint", "right_finger_joint")
#: Joint position gains (N/m) and the torque-free limits of the source's finger actuators.
KP, KD, FORCE = 100.0, 10.0, 50.0
#: Joint travel: 0 is open, this is closed (the fingers' plates meet at 41.7 mm).
CLOSED = 0.042
#: The grasp centre between the fingertips in the base frame: 228.4 mm along the fingers (their tips are at
#: z = 201.3 mm, the base's rear face at z = -27.1 mm) and centred in x.
PALM_ORIGIN = np.array([0.0, 0.003, 0.2013])
#: Palm frame axes in the base frame: +z along the approach direction, +y along the closing direction.
X_PALM, Y_PALM, Z_PALM = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
#: The base plate's rear face, 228.4 mm behind the grasp centre along -z.
BASE_BACK = 0.2284


def build() -> pathlib.Path:
    hand = ET.parse(ROBOTS / "umi_gripper" / "umi_gripper.xml").getroot()
    meshdir = hand.find("compiler").get("meshdir", "")
    tendon = hand.find("tendon")
    if tendon is not None:
        hand.remove(tendon)
    hand.remove(hand.find("actuator"))
    root = hand.find("worldbody").find("body")
    for j in list(root.findall("joint")):  # the handheld 6-DOF compliance
        root.remove(j)
    # The finger carriages' return spring (stiffness 100 N/m of the passive hand-held rig) is dropped: on the FR3
    # the position actuator holds the fingers, and a spring of the same magnitude as the actuator's gain would
    # leave the closed pose half way (the two balance at 21 mm of the 42 mm travel).
    for d in hand.find("default").iter("default"):
        if d.get("class") in ("finger_slide", "umi_finger_slide"):
            d.find("joint").attrib.pop("stiffness", None)
    for parent in root.iter():  # the hand-held GoPro camera is not used by the robot
        for cam in list(parent.findall("camera")):
            parent.remove(cam)
    rename(hand, {}, prefix="umi_")  # classes, materials, textures and meshes
    root.set("name", "umi_gripper")

    actuators = [ET.Element("position", name=f"{j}_actuator", joint=j, kp=f"{KP:g}", kv=f"{KD:g}",
                            ctrlrange=f"0 {CLOSED:g}", forcerange=f"-{FORCE:g} {FORCE:g}")
                 for j in FINGER_JOINTS]
    root.append(palm_frame(PALM_ORIGIN, X_PALM, Y_PALM, Z_PALM))
    set_pose(root, gripper_mount(X_PALM, Y_PALM, Z_PALM, PALM_ORIGIN - BASE_BACK * Z_PALM))
    return write_model(
        "umi",
        root,
        hand_defaults(hand),
        hand_assets(hand, "umi_gripper", meshdir),
        actuators,
        extra_excludes=(("fr3_link7", "umi_gripper"), ("fr3_link6", "umi_gripper")),
        n_hand_joints=2,
        home="0 0",
        comment="FR3 + UMI gripper (assets/robots/umi_gripper from mujoco_menagerie, MIT).",
    )


if __name__ == "__main__":
    print(build())
