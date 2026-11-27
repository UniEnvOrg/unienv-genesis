"""The Wuji five-finger hand (right hand) as an end effector.

Mounted by the back of its palm, its palm facing out of the flange (``scripts/build_fr3_wuji_mjcf.py``, which writes
both the merged ``franka_fr3_wuji/fr3_wuji.xml`` and the mountable ``eef/wuji.xml`` from ``wuji_hand/right.xml``).

20 joints, each driven: per finger (thumb, index, middle, ring, pinky) joint1 (MCP flexion), joint2 (MCP abduction),
joint3 (PIP), joint4 (DIP). No coupled joints. The source model drives every joint with a 0.5 N m / rad position
actuator; :data:`HAND_KP` raises that to the scale of the Sharpa hand's gains (which hold a ~0.3 kg object), leaving
the model's per-joint ``forcerange`` in place as the torque limit.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import mjcf_mount_pose

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
HAND_JOINT_NAMES = [f"{f}_joint{j}" for f in FINGERS for j in range(1, 5)]
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_link4" for f in FINGERS)
#: Joint position gains (N m / rad) per joint index: MCP flexion, MCP abduction, PIP, DIP.
_KP = (5.0, 2.5, 2.0, 2.0)
HAND_KP = [k for _ in FINGERS for k in _KP]
HAND_KD = [0.1 if kp >= 2.5 else 0.05 for _ in FINGERS for kp in _KP]
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {}

#: The hand's sub-controller: per-joint position control of the 20 joints.
HAND_SUB_CONTROLLER = {
    "controller_type": "joint_position",
    "joint_names": HAND_JOINT_NAMES,
    "Kp": HAND_KP,
    "Kd": HAND_KD,
    "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
}


def wuji_hand_part() -> RobotPart:
    """The hand alone (``eef/wuji.xml``): sub-controller ``"hand"``, reset to :data:`HAND_HOME_JOINT_POSITIONS`,
    frame ``"eef"`` (the palm frame: +x out of the palm, +z along the fingers)."""
    path = get_asset_path("robots", "eef", "wuji.xml")
    mount_pos, mount_quat = mjcf_mount_pose(path)
    return RobotPart(
        description_file=path,
        sub_controllers={"hand": HAND_SUB_CONTROLLER},
        keyframe=RobotKeyframe(joint_position=HAND_HOME_JOINT_POSITIONS, joint_names=HAND_JOINT_NAMES),
        frames={"eef": PartFrame(PALM_LINK_NAME)},
        mount_pos=mount_pos,
        mount_quat=mount_quat,
    )
