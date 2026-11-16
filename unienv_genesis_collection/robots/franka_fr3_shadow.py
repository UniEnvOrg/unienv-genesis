"""Franka Research FR3 robot descriptor with a Shadow Dexterous Hand.

The hand (the right-hand model) is mounted by the back of its palm on the FR3 flange, its palm facing along the
flange axis (``scripts/build_fr3_shadow_mjcf.py`` builds ``franka_fr3_shadow/fr3_shadow.xml`` from
``franka_fr3_hand/fr3_hand.xml`` and ``shadow_hand/right_hand.xml``, mujoco_menagerie ``shadow_hand``).

DOF summary (qpos order):
  Arm:  7 revolute joints (fr3_joint1 .. fr3_joint7)
  Hand: 22 joints: per finger (index, middle, ring) the knuckle abduction (j4), the proximal (j3), the middle
        (j2) and the distal (j1); the little finger additionally its metacarpal (lf_j5); the thumb's opposition
        (th_j5), proximal (th_j4), hub (th_j3), middle (th_j2) and distal (th_j1). Each finger's distal joint is
        coupled to its middle joint (upstream one tendon per finger), so 18 joints are driven and 4 follow
        (:data:`COUPLINGS`; the equality constraints are in the MJCF).

The model's forearm and its two wrist joints (WRJ1 / WRJ2) are dropped: on the FR3 the arm's own wrist takes
their place (see the build script).

Default ``"combined"`` controller:
  - ``"arm"``:  ``eef_pose`` (IK) on the hand's ``palm`` frame (+x out of the palm, +z along the fingers).
  - ``"hand"``: ``joint_position``, the 18 driven joints (:data:`HAND_JOINT_NAMES`).

MJCF source: mujoco_menagerie ``shadow_hand`` (Apache 2.0), pinned in ``shadow_hand/SOURCE.txt``.
"""

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots.franka_fr3_hand import ARM_HOME_JOINT_POSITIONS, DROID_FRANKA_ARM_IK_VEL_LIMIT
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

FINGERS = ("ff", "mf", "rf")
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
#: The driven joints (each finger's distal joint follows its middle one).
HAND_JOINT_NAMES = ([f"{f}_{j}" for f in FINGERS for j in ("j4", "j3", "j2")]
                    + ["lf_j5", "lf_j4", "lf_j3", "lf_j2"]
                    + ["th_j5", "th_j4", "th_j3", "th_j2", "th_j1"])
PALM_LINK_NAME = "palm"
FINGER_PAD_LINK_NAMES = ("ff_distal", "mf_distal", "rf_distal", "lf_distal", "th_distal")
#: Position gains (N m / rad): the model's own (1 N m / rad at the fingers, 0.4-1.5 at the thumb base, an
#: upstream ``forcerange`` of +-1 N m) raised so that the fingers hold a ~0.3 kg object.
_KP = {"j4": 1.5, "j3": 2.0, "j2": 1.0, "j5": 1.5, "th_j5": 1.0, "th_j4": 1.5, "th_j3": 1.0, "th_j2": 2.0, "th_j1": 1.5}
HAND_KP = [_KP[name if name.startswith("th_") else name.split("_")[1]] for name in HAND_JOINT_NAMES]
HAND_KD = [0.1 if kp >= 1.5 else 0.05 for kp in HAND_KP]
HAND_HOME_JOINT_POSITIONS = torch.zeros(len(HAND_JOINT_NAMES))
HOME_JOINT_POSITIONS = torch.cat([ARM_HOME_JOINT_POSITIONS, HAND_HOME_JOINT_POSITIONS])
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {f"{f}_j1": (f"{f}_j2", 1.0, 0.0) for f in (*FINGERS, "lf")}

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


class FrankaFR3WithShadow(RobotDescriptionNode):
    """Franka Research FR3 7-DOF arm with a 22-joint (18 driven) Shadow Dexterous Hand.

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
        name: str = "franka_shadow",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = _DEFAULT_SUB_CONTROLLERS
        super().__init__(
            world=world,
            description_file=str(get_asset_path("robots", "franka_fr3_shadow", "fr3_shadow.xml")),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
