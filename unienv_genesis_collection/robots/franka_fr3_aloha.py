"""Franka Research FR3 robot descriptor with the ALOHA (ViperX) parallel gripper.

The gripper of the ALOHA bimanual platform is mounted on its own wrist mounting plate on the FR3 flange, its
fingers pointing along the flange axis (``scripts/build_fr3_aloha_mjcf.py`` builds ``franka_fr3_aloha/fr3_aloha.xml``
from ``franka_fr3_hand/fr3_hand.xml`` and the right gripper of ``aloha/aloha.xml``, mujoco_menagerie ``aloha``).

DOF summary (qpos order):
  Arm:     7 revolute joints (fr3_joint1 .. fr3_joint7)
  Gripper: 2 prismatic joints (left_finger_joint, right_finger_joint) on the finger rail, 0 to 41 mm each with a
           2 to 37 mm control range: 0 is the finger bases touching at the centre of the rail and 37 mm per finger
           is fully open (a 74 mm gap). They are coupled by a joint equality (upstream one actuator on the left
           finger), so both are driven with one command.

Default ``"combined"`` controller:
  - ``"arm"``:     ``eef_pose`` (IK) on the gripper's ``palm`` frame, the grasp centre between the finger pads
    (+z along the approach direction, +y along the closing direction).
  - ``"gripper"``: ``mimic_joint_position`` (one scalar); ``-1`` opens, ``+1`` closes.

MJCF source: mujoco_menagerie ``aloha`` (BSD 3-Clause, Trossen Robotics), pinned in ``aloha/SOURCE.txt``.
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.hands.aloha import (  # the gripper's description, kept importable from here
    PALM_LINK_NAME,
    GRIPPER_DRIVER_NAMES,
    GRIPPER_PAD_LINK_NAMES,
    COUPLINGS,
    OPEN,
    CLOSED,
    GRIPPER_HOME_JOINT_POSITIONS,
    GRIPPER_SUB_CONTROLLER,
)
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, GRIPPER_HOME_JOINT_POSITIONS])

_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_names": ARM_JOINT_NAMES,
        "target_link": LinkAttachment(link_name=PALM_LINK_NAME),
        "max_joint_pos_vel": DROID_FRANKA_ARM_IK_VEL_LIMIT,
    },
    "gripper": GRIPPER_SUB_CONTROLLER,
}


class FrankaFR3WithAloha(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with the ALOHA (ViperX) parallel gripper.

    Gripper action convention::

        -1  ->  fully open  (37 mm per finger, a 74 mm gap)
        +1  ->  fully closed (2 mm per finger)

    Args:
        world: GenesisWorld instance
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller; ``sub_controllers=...`` overrides the defaults.
    """

    keyframes = {"reset": RobotKeyframe(joint_position=HOME_JOINT_POSITIONS, joint_names=ARM_JOINT_NAMES + GRIPPER_DRIVER_NAMES)}

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "franka_aloha",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_aloha", "fr3_aloha.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
