"""The UMI gripper as an end effector part, and its description (used by ``robots.franka_fr3_umi``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_umi`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import eef_part

PALM_LINK_NAME = "palm"
GRIPPER_DRIVER_NAMES = ["left_finger_joint", "right_finger_joint"]
GRIPPER_PAD_LINK_NAMES = ["left_finger_holder", "right_finger_holder"]
#: The fingers' prismatic joints are coupled (upstream one tendon, ``right = left``): {dependent: (driver, m, o)}.
COUPLINGS = {"right_finger_joint": ("left_finger_joint", 1.0, 0.0)}
#: Travel of one finger: 0 open, 42 mm closed; the single command maps -1 .. +1 onto that.
CLOSED = 0.042
GRIPPER_HOME_JOINT_POSITIONS = torch.zeros(len(GRIPPER_DRIVER_NAMES))

#: The gripper's sub-controller.
GRIPPER_SUB_CONTROLLER = {
    "controller_type": "mimic_joint_position",
    "joint_names": GRIPPER_DRIVER_NAMES,
    "Kp": 100.0,
    "Kd": 10.0,
    # -1 = fully open (fingers at 0), +1 = fully closed (42 mm per finger)
    "multiplier": CLOSED / 2,
    "shift": CLOSED / 2,
    "control_low": -1,
    "control_high": 1,
    "is_grasping_fn": make_is_grasping(GRIPPER_PAD_LINK_NAMES),
    "approach_direction": [0.0, 0.0, 1.0],
    "open_direction": [0.0, 1.0, 0.0],
    "gripper_width": 0.083,
    "gripper_depth": 0.05,
    "gripper_open_action": -1.0,
    "gripper_close_action": 1.0,
    # The palm frame is the grasp centre between the fingertips (the tip frame itself).
    "eef_tip_offset": [0.0, 0.0, 0.0],
}


def part() -> RobotPart:
    """The UMI gripper alone: ``eef/umi.xml``.

    Sub-controller ``"gripper"``, reset to :data:`GRIPPER_HOME_JOINT_POSITIONS`; frame ``"eef"``
    (:data:`PALM_LINK_NAME`) is what the arm's end-effector controller targets.
    """
    return eef_part(
        'umi',
        {"gripper": GRIPPER_SUB_CONTROLLER},
        home=GRIPPER_HOME_JOINT_POSITIONS,
        joint_names=GRIPPER_DRIVER_NAMES,
        eef_frame=PartFrame(PALM_LINK_NAME),
    )
