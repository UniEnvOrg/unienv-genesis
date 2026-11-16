"""Build the FR3 + ALOHA (ViperX) gripper MJCF from the mujoco_menagerie model in ``unienv_genesis_collection/assets``.

Writes ``franka_fr3_aloha/fr3_aloha.xml``: the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` with its Franka Hand
replaced by the parallel gripper of the ALOHA bimanual platform (``aloha/aloha.xml``, mujoco_menagerie ``aloha``,
BSD 3-Clause (Trossen Robotics), pinned in ``aloha/SOURCE.txt``). Only the gripper of the right ViperX arm is
kept: ``aloha/assets`` ships just the meshes it uses.

Conversion:

* **The gripper alone.** ``right/gripper_base`` (the gripper bar, its wrist mounting plate and the wrist camera)
  and its two fingers ``right/left_finger_link`` / ``right/right_finger_link`` are lifted out of the arm; the
  ViperX's own wrist (``right/wrist_rotate``) is dropped, the FR3's wrist takes its place. The rest of
  ``aloha.xml`` (both arms, the cameras, the ``joint_position_actuators.xml`` / ``keyframe_ctrl.xml`` includes)
  is not used.
* **Names.** ``right/gripper_base`` -> ``gripper_base``, ``right/left_finger_link`` -> ``left_finger``,
  ``right/right_finger_link`` -> ``right_finger``, the joints -> ``left_finger_joint`` / ``right_finger_joint``;
  the default classes (the ``vx300s`` hierarchy the gripper's gains and geoms live in), materials and meshes get
  the ``aloha_`` prefix, and the ``<camera>`` element is dropped.
* **Fingers.** Two prismatic joints along the rail, 0 to 41 mm each with a 2 to 37 mm control range (0 = the
  finger bases touching at the centre of the rail, 37 mm = fully open, a 74 mm gap); upstream they are coupled by
  a joint equality and driven by one actuator on the left finger. Genesis supports the equality, so it is kept
  (``left_finger_joint = right_finger_joint``) and both fingers get a position actuator (the model's
  ``kp=2000`` / ``kv=124``) for the single open/close command.
* **Mount.** Mounted as designed on the gripper's own wrist mounting plate (``vx300s_7_gripper_wrist_mount``, its
  rear face 38.7 mm behind the gripper bar), through a 5 mm adapter, so the approach direction is out of the
  flange.
* **Palm frame.** A ``palm`` body at the grasp centre between the two fingers' pads, 110.4 mm along the approach
  direction from the plate's rear face: +z along the approach direction, +y along the closing direction (from the
  left finger to the right one, the model's +x) and +x across them.

Run from the repository root:  python scripts/build_fr3_aloha_mjcf.py
"""

from __future__ import annotations

import copy
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import ROBOTS, gripper_mount, hand_defaults, palm_frame, rename, set_pose, write_model

NAMES = {"right/gripper_base": "gripper_base", "right/left_finger_link": "left_finger",
         "right/right_finger_link": "right_finger", "right/left_finger": "left_finger_joint",
         "right/right_finger": "right_finger_joint"}
FINGER_JOINTS = ("left_finger_joint", "right_finger_joint")
#: The joint equality that couples the fingers: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {"right_finger_joint": ("left_finger_joint", 1.0, 0.0)}
#: The triads of pad spheres, one per finger.
PAD_LINKS = ("left_finger", "right_finger")
#: The grasp centre between the pad triads in the base frame (m).
PALM_ORIGIN = np.array([0.0, 0.0034, 0.1104])
#: Palm frame axes in the base frame: +z along the approach direction, +y along the closing direction.
X_PALM, Y_PALM, Z_PALM = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
#: The wrist mounting plate's rear face, 38.7 mm behind the finger bases along -z.
BASE_BACK = 0.1491
#: The meshes of the gripper (the rest of the ALOHA arm's meshes are not shipped).
MESHES = ("vx300s_7_gripper_prop.stl", "vx300s_7_gripper_bar.stl", "vx300s_7_gripper_wrist_mount.stl",
          "vx300s_8_custom_finger_left.stl", "vx300s_8_custom_finger_right.stl", "d405_solid.stl")


def build() -> pathlib.Path:
    aloha = ET.parse(ROBOTS / "aloha" / "aloha.xml").getroot()
    meshdir = aloha.find("compiler").get("meshdir", "")
    base = next(b for b in aloha.find("worldbody").iter("body") if b.get("name") == "right/gripper_base")
    hand = ET.Element("mujoco", model="aloha_gripper")
    hand.append(ET.SubElement(hand, "compiler", angle="radian", autolimits="true", meshdir=meshdir))
    default = ET.SubElement(hand, "default")
    for d in aloha.find("default"):
        if isinstance(d.tag, str):
            default.append(copy.deepcopy(d))
    asset = ET.SubElement(hand, "asset")
    for a in aloha.find("asset"):
        if not (isinstance(a.tag, str) and (a.tag == "material" or pathlib.Path(a.get("file", "")).name in MESHES)):
            continue
        c = copy.deepcopy(a)
        if c.tag == "mesh":  # the merged model sits in a sibling directory
            c.set("file", f"../aloha/{meshdir.strip('/')}/{pathlib.Path(c.get('file')).name}")
        asset.append(c)
    world = ET.SubElement(hand, "worldbody")
    world.append(base)
    for parent in base.iter():  # the ViperX wrist camera is not used by the robot
        for cam in list(parent.findall("camera")):
            parent.remove(cam)
    rename(hand, NAMES, prefix="aloha_")

    actuators = [ET.Element("position", name=f"{j}_actuator", joint=j, **{"class": "aloha_finger"})
                 for j in FINGER_JOINTS]
    equality = [ET.Element("joint", joint1=dep, joint2=drv, polycoef=f"0 {m:g} {o:g} 0 0", solref="0.005 1")
                for dep, (drv, m, o) in COUPLINGS.items()]
    base.append(palm_frame(PALM_ORIGIN, X_PALM, Y_PALM, Z_PALM))
    set_pose(base, gripper_mount(X_PALM, Y_PALM, Z_PALM, PALM_ORIGIN - BASE_BACK * Z_PALM))
    return write_model(
        "aloha",
        base,
        hand_defaults(hand),
        list(asset),
        actuators,
        equality=equality,
        extra_excludes=(("fr3_link7", "gripper_base"), ("fr3_link6", "gripper_base")),
        n_hand_joints=2,
        home="0.037 0.037",
        comment="FR3 + ALOHA (ViperX) gripper (assets/robots/aloha from mujoco_menagerie, BSD 3-Clause).",
    )


if __name__ == "__main__":
    print(build())
