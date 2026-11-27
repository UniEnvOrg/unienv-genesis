"""The ALOHA ViperX gripper as an end effector part, and its description (used by ``robots.franka_fr3_aloha``).

``part()`` is the end effector alone, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``; the module
docstring of ``robots.franka_fr3_aloha`` describes the model, its joints and its mounting.
"""

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping
from unienv_genesis_collection.robots.hands import eef_part

PALM_LINK_NAME = "palm"
GRIPPER_DRIVER_NAMES = ["left_finger_joint", "right_finger_joint"]
GRIPPER_PAD_LINK_NAMES = ["left_finger", "right_finger"]
#: The fingers are coupled by a joint equality: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {"right_finger_joint": ("left_finger_joint", 1.0, 0.0)}
#: Control travel of one finger (the real robot's software limits): 2 mm closed, 37 mm fully open.
OPEN, CLOSED = 0.037, 0.002
GRIPPER_HOME_JOINT_POSITIONS = torch.tensor([OPEN, OPEN])

#: The gripper's sub-controller.
GRIPPER_SUB_CONTROLLER = {
    "controller_type": "mimic_joint_position",
    "joint_names": GRIPPER_DRIVER_NAMES,
    # the model's own gains for the finger slides (kp / kv of the ViperX gripper)
    "Kp": 2000.0,
    "Kd": 124.0,
    # -1 = fully open (37 mm per finger), +1 = fully closed (2 mm)
    "multiplier": (CLOSED - OPEN) / 2,
    "shift": (CLOSED + OPEN) / 2,
    "control_low": -1,
    "control_high": 1,
    "is_grasping_fn": make_is_grasping(GRIPPER_PAD_LINK_NAMES),
    "approach_direction": [0.0, 0.0, 1.0],
    "open_direction": [0.0, 1.0, 0.0],
    "gripper_width": 2 * OPEN,
    "gripper_depth": 0.04,
    "gripper_open_action": -1.0,
    "gripper_close_action": 1.0,
    # The palm frame is the grasp centre between the finger pads (the tip frame itself).
    "eef_tip_offset": [0.0, 0.0, 0.0],
}


def part() -> RobotPart:
    """The ALOHA ViperX gripper alone: ``eef/aloha.xml``.

    Sub-controller ``"gripper"``, reset to :data:`GRIPPER_HOME_JOINT_POSITIONS`; frame ``"eef"``
    (:data:`PALM_LINK_NAME`) is what the arm's end-effector controller targets.
    """
    return eef_part(
        'aloha',
        {"gripper": GRIPPER_SUB_CONTROLLER},
        home=GRIPPER_HOME_JOINT_POSITIONS,
        joint_names=GRIPPER_DRIVER_NAMES,
        eef_frame=PartFrame(PALM_LINK_NAME),
    )
