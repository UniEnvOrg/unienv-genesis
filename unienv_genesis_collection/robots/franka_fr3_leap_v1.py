"""Franka Research FR3 robot descriptor with a LEAP Hand (first generation).

The hand is mounted by the back of its palm on the FR3 flange, its palm facing along the flange axis
(``scripts/build_fr3_leap_v1_mjcf.py`` builds ``franka_fr3_leap_v1/fr3_leap_v1.xml`` from
``franka_fr3_hand/fr3_hand.xml`` and ``leap_hand/right_hand.xml``, mujoco_menagerie ``leap_hand``). This is the
first-generation LEAP Hand: :mod:`unienv_genesis_collection.robots.franka_fr3_leap` is the v2 Adv hand.

DOF summary (qpos order):
  Arm:  7 revolute joints (fr3_joint1 .. fr3_joint7)
  Hand: 16 joints, each driven: per finger (index ``if``, middle ``mf``, ring ``rf``) the MCP flexion (``mcp``),
        the abduction (``rot``), the PIP and the DIP; the thumb's ``cmc``, ``axl``, ``mcp`` and ``ipl``. There are
        no coupled joints (the v2 Adv hand's PIP/DIP tendon coupling is not in this model).

Default ``"combined"`` controller:
  - ``"arm"``:  ``eef_pose`` (IK) on the hand's ``palm`` frame (+x out of the palm, +z along the fingers).
  - ``"hand"``: ``joint_position``, the 16 joints (:data:`HAND_JOINT_NAMES`), with the model's own 3 N m / rad
    gains (they hold a ~0.3 kg object).

MJCF source: mujoco_menagerie ``leap_hand`` (MIT), pinned in ``leap_hand/SOURCE.txt``.
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.hands.leap_v1 import (  # the hand's description, kept importable from here
    FINGERS,
    HAND_JOINT_NAMES,
    PALM_LINK_NAME,
    FINGER_PAD_LINK_NAMES,
    HAND_KP,
    HAND_KD,
    HAND_HOME_JOINT_POSITIONS,
    COUPLINGS,
    HAND_SUB_CONTROLLER,
)
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, HAND_HOME_JOINT_POSITIONS])

_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_names": ARM_JOINT_NAMES,
        "target_link": LinkAttachment(link_name=PALM_LINK_NAME),
        "max_joint_pos_vel": DROID_FRANKA_ARM_IK_VEL_LIMIT,
    },
    "hand": HAND_SUB_CONTROLLER,
}


class FrankaFR3WithLeapV1(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with a 16-DOF LEAP Hand (v1).

    Args:
        world: GenesisWorld instance
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller; ``sub_controllers=...`` overrides the defaults.
    """

    keyframes = {"reset": RobotKeyframe(joint_position=HOME_JOINT_POSITIONS, joint_names=ARM_JOINT_NAMES + HAND_JOINT_NAMES)}

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "franka_leap_v1",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_leap_v1", "fr3_leap_v1.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
