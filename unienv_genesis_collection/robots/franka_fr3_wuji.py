"""Franka Research FR3 robot descriptor with a Wuji five-finger hand.

The hand (the Wuji hand already used on the Tianji robot) is mounted by the back of its palm on the FR3 flange,
its palm facing along the flange axis (``scripts/build_fr3_wuji_mjcf.py`` builds ``franka_fr3_wuji/fr3_wuji.xml``
from ``franka_fr3_hand/fr3_hand.xml`` and ``wuji_hand/right.xml``).

DOF summary (qpos order):
  Arm:  7 revolute joints (fr3_joint1 .. fr3_joint7)
  Hand: 20 joints, each driven: per finger (thumb, index, middle, ring, pinky) joint1 (MCP flexion),
        joint2 (MCP abduction), joint3 (PIP), joint4 (DIP).  No coupled joints.

Default ``"combined"`` controller:
  - ``"arm"``:  ``eef_pose`` (IK) on the hand's ``palm`` frame (+x out of the palm, +z along the fingers).
  - ``"hand"``: ``joint_position``, the 20 joints (:data:`HAND_JOINT_NAMES`).

The source model drives every joint with a 0.5 N m / rad position actuator; :data:`HAND_KP` raises that to the
scale of the FR3 + Sharpa hand's gains (which hold a ~0.3 kg object), leaving the model's per-joint
``forcerange`` in place as the torque limit.
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.hands.wuji import (  # the hand's description, kept importable from here
    FINGERS,
    HAND_JOINT_NAMES,
    PALM_LINK_NAME,
    FINGER_PAD_LINK_NAMES,
    _KP,
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


class FrankaFR3WithWuji(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with a 20-DOF Wuji hand.

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
        name: str = "franka_wuji",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_wuji", "fr3_wuji.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
