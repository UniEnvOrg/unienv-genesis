"""Franka Research FR3 robot descriptor with the UMI gripper.

The gripper is mounted on its own base plate's rear face on the FR3 flange, its fingers pointing along the flange
axis (``scripts/build_fr3_umi_mjcf.py`` builds ``franka_fr3_umi/fr3_umi.xml`` from ``franka_fr3_hand/fr3_hand.xml``
and ``umi_gripper/umi_gripper.xml``, mujoco_menagerie ``umi_gripper``).

The gripper is the hand-held UMI rig made rigid: its upstream six base joints (the human hand's compliance) and
its two fingers' tendon coupling are removed; the two fingers slide on the rail and are driven with one command,
like the FR3 + Robotiq 2F-85.

DOF summary (qpos order):
  Arm:     7 revolute joints (fr3_joint1 .. fr3_joint7)
  Gripper: 2 prismatic joints (left_finger_joint, right_finger_joint), 0 to 50 mm each; 0 is fully open (an
           83 mm gap between the finger plates), 42 mm per finger is closed.

Default ``"combined"`` controller:
  - ``"arm"``:     ``eef_pose`` (IK) on the gripper's ``palm`` frame, the grasp centre between the fingertips
    (+z along the approach direction, +y along the closing direction).
  - ``"gripper"``: ``mimic_joint_position`` (one scalar); ``-1`` opens, ``+1`` closes.

MJCF source: mujoco_menagerie ``umi_gripper`` (MIT), pinned in ``umi_gripper/SOURCE.txt``.
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
PALM_LINK_NAME = "palm"
GRIPPER_DRIVER_NAMES = ["left_finger_joint", "right_finger_joint"]
GRIPPER_PAD_LINK_NAMES = ["left_finger_holder", "right_finger_holder"]
#: The fingers' prismatic joints are coupled (upstream one tendon, ``right = left``): {dependent: (driver, m, o)}.
COUPLINGS = {"right_finger_joint": ("left_finger_joint", 1.0, 0.0)}
#: Travel of one finger: 0 open, 42 mm closed; the single command maps -1 .. +1 onto that.
CLOSED = 0.042
GRIPPER_HOME_JOINT_POSITIONS = torch.zeros(len(GRIPPER_DRIVER_NAMES))
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, GRIPPER_HOME_JOINT_POSITIONS])

_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_names": ARM_JOINT_NAMES,
        "target_link": LinkAttachment(link_name=PALM_LINK_NAME),
        "max_joint_pos_vel": DROID_FRANKA_ARM_IK_VEL_LIMIT,
    },
    "gripper": {
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
    },
}


class FrankaFR3WithUmi(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with the UMI gripper (2 driven finger joints).

    Gripper action convention::

        -1  ->  fully open  (finger joints at 0, an 83 mm gap)
        +1  ->  fully closed (finger joints at 42 mm)

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
        name: str = "franka_umi",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_umi", "fr3_umi.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
