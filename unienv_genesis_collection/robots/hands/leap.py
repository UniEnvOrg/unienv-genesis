"""The LEAP Hand v2 Adv as an end effector part, and its description (used by ``robots.franka_fr3_leap``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_leap`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import check_side, eef_part

FINGERS = ("index", "middle", "ring", "pinky")
#: The driven hand joints (each finger's dip follows its pip).
HAND_JOINT_NAMES = [f"{f}_{j}" for f in FINGERS for j in ("mcpf", "mcps", "pip")] + [
    "thumb_mcpf", "thumb_mcps", "thumb_ip", "palm_4_finger", "palm_thumb"]
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_{p}" for f in (*FINGERS, "thumb") for p in ("proximal", "middle", "distal")
                              if not (f == "thumb" and p == "middle"))
#: Finger PD gains (N m / rad; the motors give ~0.5 N m, the palm folds 1-1.4 N m).
HAND_KP = [3.0] * 15 + [5.0, 5.0]
HAND_KD = [0.1] * 17
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))

#: The hand's sub-controller.
HAND_SUB_CONTROLLER = {
    "controller_type": "joint_position",
    "joint_names": HAND_JOINT_NAMES,
    "Kp": HAND_KP,
    "Kd": HAND_KD,
    "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
}

#: side -> the end effector model (``eef/<model>.xml``).
_MODELS = {"right": "leap_right", "left": "leap_left"}


def part(side: str = "right") -> RobotPart:
    """The LEAP Hand v2 Adv alone: ``eef/leap_right.xml`` (right) or ``eef/leap_left.xml`` (left).

    Sub-controller ``"hand"``, reset to :data:`HAND_HOME_JOINT_POSITIONS`; frame ``"eef"``
    (:data:`PALM_LINK_NAME`) is what the arm's end-effector controller targets.
    """
    return eef_part(
        _MODELS[check_side(side)],
        {"hand": HAND_SUB_CONTROLLER},
        home=HAND_HOME_JOINT_POSITIONS,
        joint_names=HAND_JOINT_NAMES,
        eef_frame=PartFrame(PALM_LINK_NAME),
    )
