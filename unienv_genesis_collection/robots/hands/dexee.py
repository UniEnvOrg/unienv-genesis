"""The Shadow DEX-EE hand as an end effector part, and its description (used by ``robots.franka_fr3_dexee``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_dexee`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import eef_part

FINGERS = ("f0", "f1", "f2")
HAND_JOINT_NAMES = [f"{f}_j{j}" for f in FINGERS for j in range(4)]
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_distal" for f in FINGERS)
#: The ``mujoco.pid`` plugin's gains per joint: kp (N m / rad), kd.
_KP = (2.8, 2.5, 1.1, 0.6)
_KD = (0.03, 0.02, 0.01, 0.008)
HAND_KP = [k for _ in FINGERS for k in _KP]
HAND_KD = [k for _ in FINGERS for k in _KD]
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


def part() -> RobotPart:
    """The Shadow DEX-EE hand alone: ``eef/dexee.xml``.

    Sub-controller ``"hand"``, reset to :data:`HAND_HOME_JOINT_POSITIONS`; frame ``"eef"``
    (:data:`PALM_LINK_NAME`) is what the arm's end-effector controller targets.
    """
    return eef_part(
        'dexee',
        {"hand": HAND_SUB_CONTROLLER},
        home=HAND_HOME_JOINT_POSITIONS,
        joint_names=HAND_JOINT_NAMES,
        eef_frame=PartFrame(PALM_LINK_NAME),
    )
