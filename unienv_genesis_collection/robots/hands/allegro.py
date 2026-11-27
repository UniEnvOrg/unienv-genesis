"""The Wonik Allegro hand as an end effector part, and its description (used by ``robots.franka_fr3_allegro``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_allegro`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import check_side, eef_part

FINGERS = ("ff", "mf", "rf", "th")  # index, middle, ring, thumb
HAND_JOINT_NAMES = [f"{f}j{j}" for f in FINGERS for j in range(4)]
PALM_LINK_NAME = "palm"
#: Contact-capable finger links (the fingertip bodies have no joint, so their geoms belong to the distal links).
FINGER_PAD_LINK_NAMES = tuple(f"{f}_{part}" for f in FINGERS for part in ("proximal", "medial", "distal"))
#: Finger PD gains (the menagerie model's actuator kp = 1 is too weak to hold a grasp).
HAND_KP = [3.0] * 16
HAND_KD = [0.1] * 16
#: Fingers straight, the thumb at its lowest (most opposed) base angle.
HAND_HOME_JOINT_POSITIONS = torch.tensor([0.0] * 12 + [0.263, 0.0, 0.0, 0.0])

#: The hand's sub-controller.
HAND_SUB_CONTROLLER = {
    "controller_type": "joint_position",
    "joint_names": HAND_JOINT_NAMES,
    "Kp": HAND_KP,
    "Kd": HAND_KD,
    "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
}

#: side -> the end effector model (``eef/<model>.xml``).
_MODELS = {"right": "allegro_right", "left": "allegro_left"}


def part(side: str = "right") -> RobotPart:
    """The Wonik Allegro hand alone: ``eef/allegro_right.xml`` (right) or ``eef/allegro_left.xml`` (left).

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
