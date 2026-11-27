"""The Tetheria Aero Hand Open as an end effector part, and its description (used by ``robots.franka_fr3_aero``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_aero`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import check_side, eef_part

FINGERS = ("index", "middle", "ring", "pinky")
#: The driven joints (each finger's DIP and the thumb's IP follow their driver).
HAND_JOINT_NAMES = ([f"{f}_{j}" for f in FINGERS for j in ("mcp_flex", "pip")]
                    + ["thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp"])
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_distal" for f in FINGERS) + ("thumb_distal",)
#: Position gains (N m / rad) of a light (0.45 kg) printed hand, and the torque limit the build script sets.
_KP = {"mcp_flex": 2.0, "pip": 1.5, "thumb_cmc_abd": 1.0, "thumb_cmc_flex": 1.5, "thumb_mcp": 1.5}
HAND_KP = [_KP[n] if n in _KP else _KP[n.split("_", 1)[1]] for n in HAND_JOINT_NAMES]
HAND_KD = [0.05] * len(HAND_JOINT_NAMES)
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {**{f"{f}_dip": (f"{f}_pip", 1.0, 0.0) for f in FINGERS}, "thumb_ip": ("thumb_mcp", 1.0, 0.0)}

#: The hand's sub-controller.
HAND_SUB_CONTROLLER = {
    "controller_type": "joint_position",
    "joint_names": HAND_JOINT_NAMES,
    "Kp": HAND_KP,
    "Kd": HAND_KD,
    "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
}

#: side -> the end effector model (``eef/<model>.xml``).
_MODELS = {"right": "aero", "left": "aero_left"}


def part(side: str = "right") -> RobotPart:
    """The Tetheria Aero Hand Open alone: ``eef/aero.xml`` (right) or ``eef/aero_left.xml`` (left).

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
