"""The first-generation LEAP Hand as an end effector part, and its description (used by ``robots.franka_fr3_leap_v1``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_leap_v1`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import check_side, eef_part

FINGERS = ("if", "mf", "rf")
HAND_JOINT_NAMES = ([f"{f}_{j}" for f in FINGERS for j in ("mcp", "rot", "pip", "dip")]
                    + ["th_cmc", "th_axl", "th_mcp", "th_ipl"])
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_ds" for f in FINGERS) + ("th_ds",)
HAND_KP = [3.0] * len(HAND_JOINT_NAMES)  # the model's own position gain
HAND_KD = [0.1] * len(HAND_JOINT_NAMES)
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {}

#: The hand's sub-controller.
HAND_SUB_CONTROLLER = {
    "controller_type": "joint_position",
    "joint_names": HAND_JOINT_NAMES,
    "Kp": HAND_KP,
    "Kd": HAND_KD,
    "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
}

#: side -> the end effector model (``eef/<model>.xml``).
_MODELS = {"right": "leap_v1", "left": "leap_v1_left"}


def part(side: str = "right") -> RobotPart:
    """The first-generation LEAP Hand alone: ``eef/leap_v1.xml`` (right) or ``eef/leap_v1_left.xml`` (left).

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
