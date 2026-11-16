"""Franka Research FR3 robot descriptor with a Shadow DEX-EE three-finger hand.

The hand is mounted by the back of its palm on the FR3 flange, its palm facing along the flange axis
(``scripts/build_fr3_dexee_mjcf.py`` builds ``franka_fr3_dexee/fr3_dexee.xml`` from ``franka_fr3_hand/fr3_hand.xml``
and ``shadow_dexee/shadow_dexee.xml``, mujoco_menagerie ``shadow_dexee``).

DOF summary (qpos order):
  Arm:  7 revolute joints (fr3_joint1 .. fr3_joint7)
  Hand: 12 joints, each driven: per finger (f0, f1, f2) J0 (abduction), J1, J2, J3. F0 opposes F1 and F2, so F0
        curls with positive J1 and the other two with negative J1. No coupled joints.

Default ``"combined"`` controller:
  - ``"arm"``:  ``eef_pose`` (IK) on the hand's ``palm`` frame (+x out of the palm, +z along the fingers).
  - ``"hand"``: ``joint_position``, the 12 joints (:data:`HAND_JOINT_NAMES`).

Gains: the model drives its joints through the ``mujoco.pid`` plugin (which Genesis does not implement); the build
script turns those into position actuators with the plugin's own ``kp`` / ``kd`` (2.8 / 0.03, 2.5 / 0.02, 1.1 /
0.01, 0.6 / 0.008 per joint), which is what :data:`HAND_KP` / :data:`HAND_KD` repeat here.

MJCF source: mujoco_menagerie ``shadow_dexee`` (Apache 2.0), pinned in ``shadow_dexee/SOURCE.txt``.
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

FINGERS = ("f0", "f1", "f2")
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
HAND_JOINT_NAMES = [f"{f}_j{j}" for f in FINGERS for j in range(4)]
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_distal" for f in FINGERS)
#: The ``mujoco.pid`` plugin's gains per joint: kp (N m / rad), kd.
_KP = (2.8, 2.5, 1.1, 0.6)
_KD = (0.03, 0.02, 0.01, 0.008)
HAND_KP = [k for _ in FINGERS for k in _KP]
HAND_KD = [k for _ in FINGERS for k in _KD]
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, HAND_HOME_JOINT_POSITIONS])
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {}

_DEFAULT_SUB_CONTROLLERS = {
    "arm": {
        "controller_type": "eef_pose",
        "joint_names": ARM_JOINT_NAMES,
        "target_link": LinkAttachment(link_name=PALM_LINK_NAME),
        "max_joint_pos_vel": DROID_FRANKA_ARM_IK_VEL_LIMIT,
    },
    "hand": {
        "controller_type": "joint_position",
        "joint_names": HAND_JOINT_NAMES,
        "Kp": HAND_KP,
        "Kd": HAND_KD,
        "is_grasping_fn": make_is_grasping(FINGER_PAD_LINK_NAMES),
    },
}


class FrankaFR3WithDexee(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with a 12-DOF Shadow DEX-EE hand.

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
        name: str = "franka_dexee",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_dexee", "fr3_dexee.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
