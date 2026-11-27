"""The Sharpa Wave hand as an end effector part, and its description (used by ``robots.franka_fr3_sharpa``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_sharpa`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import check_side, eef_part

_FINGER_JOINTS = ("MCP_FE", "MCP_AA", "PIP", "DIP")
HAND_JOINT_NAMES = (
    [f"thumb_{j}" for j in ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP")]
    + [f"{f}_{j}" for f in ("index", "middle", "ring") for j in _FINGER_JOINTS]
    + ["pinky_CMC"] + [f"pinky_{j}" for j in _FINGER_JOINTS]
)
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_{p}" for f in ("index", "middle", "ring", "pinky") for p in ("PP", "MP", "DP")) + ("thumb_PP", "thumb_DP")
#: Position gains per joint type (N m / rad), the model's actuators'.
_KP = {"CMC_FE": 6.95, "CMC_AA": 13.2, "MCP_FE": 4.76, "MCP_AA": 6.62, "IP": 0.9, "PIP": 0.9, "DIP": 0.9, "CMC": 1.38}
HAND_KP = [_KP[name.split("_", 1)[1]] for name in HAND_JOINT_NAMES]
HAND_KD = [0.05 if kp < 2.0 else 0.2 for kp in HAND_KP]
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
_MODELS = {"right": "sharpa_right", "left": "sharpa_left"}


def part(side: str = "right") -> RobotPart:
    """The Sharpa Wave hand alone: ``eef/sharpa_right.xml`` (right) or ``eef/sharpa_left.xml`` (left).

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
