"""The Shadow Dexterous Hand as an end effector part, and its description (used by ``robots.franka_fr3_shadow``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_shadow`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import check_side, eef_part

FINGERS = ("ff", "mf", "rf")
#: The driven joints (each finger's distal joint follows its middle one).
HAND_JOINT_NAMES = ([f"{f}_{j}" for f in FINGERS for j in ("j4", "j3", "j2")]
                    + ["lf_j5", "lf_j4", "lf_j3", "lf_j2"]
                    + ["th_j5", "th_j4", "th_j3", "th_j2", "th_j1"])
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = ("ff_distal", "mf_distal", "rf_distal", "lf_distal", "th_distal")
#: Position gains (N m / rad): the model's own (1 N m / rad at the fingers, 0.4-1.5 at the thumb base, an
#: upstream ``forcerange`` of +-1 N m) raised so that the fingers hold a ~0.3 kg object.
_KP = {"j4": 1.5, "j3": 2.0, "j2": 1.0, "j5": 1.5, "th_j5": 1.0, "th_j4": 1.5, "th_j3": 1.0, "th_j2": 2.0, "th_j1": 1.5}
HAND_KP = [_KP[name if name.startswith("th_") else name.split("_")[1]] for name in HAND_JOINT_NAMES]
HAND_KD = [0.1 if kp >= 1.5 else 0.05 for kp in HAND_KP]
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {f"{f}_j1": (f"{f}_j2", 1.0, 0.0) for f in (*FINGERS, "lf")}

#: The hand's sub-controller.
HAND_SUB_CONTROLLER = {
    "controller_type": "joint_position",
    "joint_names": HAND_JOINT_NAMES,
    "Kp": HAND_KP,
    "Kd": HAND_KD,
    "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
}

#: side -> the end effector model (``eef/<model>.xml``).
_MODELS = {"right": "shadow", "left": "shadow_left"}


def part(side: str = "right") -> RobotPart:
    """The Shadow Dexterous Hand alone: ``eef/shadow.xml`` (right) or ``eef/shadow_left.xml`` (left).

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
