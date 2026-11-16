"""Franka Research FR3 robot descriptor with a Tetheria Aero Hand Open.

The hand is mounted by the back of its palm on the FR3 flange, its palm facing along the flange axis
(``scripts/build_fr3_aero_mjcf.py`` builds ``franka_fr3_aero/fr3_aero.xml`` from ``franka_fr3_hand/fr3_hand.xml``
and ``tetheria_aero_hand_open/right_hand.xml``, mujoco_menagerie ``tetheria_aero_hand_open``).

DOF summary (qpos order):
  Arm:  7 revolute joints (fr3_joint1 .. fr3_joint7)
  Hand: 16 joints: per finger (index, middle, ring, pinky) MCP flexion, PIP and DIP; the thumb's abduction, CMC
        flexion, MCP and IP. The four DIPs follow their PIP and the thumb's IP follows its MCP (upstream one
        tendon per finger; the equality constraints are in the MJCF), so 11 joints are driven
        (:data:`HAND_JOINT_NAMES`).

Upstream the hand is driven through six flexor tendons on their lengths plus one joint; Genesis ignores MJCF
tendons, so the build script gives each driven joint its own position actuator instead.

Default ``"combined"`` controller:
  - ``"arm"``:  ``eef_pose`` (IK) on the hand's ``palm`` frame (+x out of the palm, +z along the fingers).
  - ``"hand"``: ``joint_position``, the 11 driven joints.

MJCF source: mujoco_menagerie ``tetheria_aero_hand_open`` (Apache 2.0), pinned in
``tetheria_aero_hand_open/SOURCE.txt``.
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

FINGERS = ("index", "middle", "ring", "pinky")
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
#: The driven joints (each finger's DIP and the thumb's IP follow their driver).
HAND_JOINT_NAMES = ([f"{f}_{j}" for f in FINGERS for j in ("mcp_flex", "pip")]
                    + ["thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp"])
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = tuple(f"{f}_distal" for f in FINGERS) + ("thumb_distal",)
#: Position gains (N m / rad) of a light (0.45 kg) printed hand, and the torque limit the build script sets.
_KP = {"mcp_flex": 2.0, "pip": 1.5, "thumb_cmc_abd": 1.0, "thumb_cmc_flex": 1.5, "thumb_mcp": 1.5}
HAND_KP = [_KP[n] if n in _KP else _KP[n.split("_", 1)[1]] for n in HAND_JOINT_NAMES]
HAND_KD = [0.05] * len(HAND_JOINT_NAMES)
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, HAND_HOME_JOINT_POSITIONS])
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {**{f"{f}_dip": (f"{f}_pip", 1.0, 0.0) for f in FINGERS}, "thumb_ip": ("thumb_mcp", 1.0, 0.0)}

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


class FrankaFR3WithAero(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with a 16-joint (11 driven) Tetheria Aero Hand Open.

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
        name: str = "franka_aero",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_aero", "fr3_aero.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
