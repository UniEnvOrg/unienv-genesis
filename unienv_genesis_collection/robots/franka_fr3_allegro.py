"""Franka Research FR3 robot descriptor with a Wonik Allegro hand.

The Allegro (left- or right-hand model) sits on the FR3 flange through a 5 mm adapter, turned about the flange axis
like the Franka Hand (``scripts/build_fr3_allegro_mjcf.py`` builds ``franka_fr3_allegro/fr3_allegro_<side>.xml`` from
``franka_fr3_hand/fr3_hand.xml`` and ``wonik_allegro/<side>_hand.xml``). A bimanual setup is two of these nodes.

DOF summary (qpos / actuator order):
  Arm:  7 revolute joints (fr3_joint1 .. fr3_joint7)
  Hand: 16 joints         ({ff,mf,rf,th}j{0..3}: index, middle, ring, thumb)
  Total: 23 joints

Default ``"combined"`` controller:
  - ``"arm"``:  ``eef_pose`` (IK) on the Allegro ``palm`` body frame (fingers along its +z).
  - ``"hand"``: ``joint_position``, 16 finger joints (``HAND_KP`` / ``HAND_KD``).

MJCF sources: mujoco_menagerie ``franka_fr3`` (arm, Apache 2.0) and ``wonik_allegro`` (hand, BSD-3).
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.hands.allegro import (  # the hand's description, kept importable from here
    FINGERS,
    HAND_JOINT_NAMES,
    PALM_LINK_NAME,
    FINGER_PAD_LINK_NAMES,
    HAND_KP,
    HAND_KD,
    HAND_HOME_JOINT_POSITIONS,
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


class FrankaFR3WithAllegro(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with a 16-DOF Allegro hand.

    Args:
        world: GenesisWorld instance
        side: ``"left"`` or ``"right"``: which Allegro hand model is mounted.
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller; ``sub_controllers=...`` overrides the defaults.

    Example::

        left = FrankaFR3WithAllegro(world, side="left", name="left_franka", pos=(0.1, 0.3, -0.287))
        # {"arm": <eef pose>, "hand": (16,) finger joint targets}
    """

    keyframes = {"reset": RobotKeyframe(joint_position=HOME_JOINT_POSITIONS, joint_names=ARM_JOINT_NAMES + HAND_JOINT_NAMES)}

    def __init__(
        self,
        world: GenesisWorld,
        side: str = "right",
        name: str = "franka_allegro",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        self.side = side
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_allegro", f"fr3_allegro_{side}.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
