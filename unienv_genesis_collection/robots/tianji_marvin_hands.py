"""Tianji Marvin dual-arm robot with another end effector on each arm in place of its Wuji hands.

The models (``tianji_marvin_<hand>/tianji_<hand>.xml``) are built by ``scripts/build_tianji_hand_mjcf.py`` from the
Tianji model and the hand's FR3 models: every hand body, joint and actuator is the FR3 model's, prefixed with its
side (``left_`` / ``right_``), and each hand has a ``<side>_palm`` frame (dexterous hands: +x out of the palm, +z along
the fingers; grippers: +z the approach, +y the closing direction).

Default ``"combined"`` controller (as :class:`~unienv_genesis_collection.robots.tianji_marvin_wuji.TianjiMarvinWuji`):
  - ``"left_arm"`` / ``"right_arm"``: ``eef_pose`` (IK) on the side's ``palm`` frame, no offset; the Tianji's arm
    gains and joint-velocity cap.
  - ``"left_hand"`` / ``"right_hand"``: the hand's own controller from its FR3 module (per-joint position control,
    or a gripper's one-command ``mimic_joint_position``), on the side's prefixed joints.

Hands: :data:`HANDS`.
"""

from __future__ import annotations

import copy
import importlib
import xml.etree.ElementTree as ET

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.entity_controllers import CONTROL_MODE_T
from unienv_genesis.nodes.entities.robot_node import RobotDescriptionNode, RobotKeyframe
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis_collection.asset_path import get_asset_path
from unienv_genesis_collection.robots import tianji_marvin_wuji as tj
from unienv_genesis_collection.robots.grasp_utils import make_is_grasping

#: hand -> (FR3 module whose hand / gripper controller it takes, the attribute naming its finger pad links)
HANDS = {
    "sharpa": ("franka_fr3_sharpa", "FINGER_PAD_LINK_NAMES"),
    "leap": ("franka_fr3_leap", "FINGER_PAD_LINK_NAMES"),
    "shadow": ("franka_fr3_shadow", "FINGER_PAD_LINK_NAMES"),
    "leap_v1": ("franka_fr3_leap_v1", "FINGER_PAD_LINK_NAMES"),
    "aero": ("franka_fr3_aero", "FINGER_PAD_LINK_NAMES"),
    "umi": ("franka_fr3_umi", "GRIPPER_PAD_LINK_NAMES"),
    "robotiq": ("franka_fr3_robotiq_2f85", "GRIPPER_PAD_LINK_NAMES"),
    "hand": ("franka_fr3_hand", "FINGER_PAD_LINK_NAMES"),
}
SIDES = {"left": "L", "right": "R"}
#: The Franka Hand's gripper controller names its joints by index (in its FR3 model); here by name.
_FRANKA_HAND_JOINTS = ("finger_joint1", "finger_joint2")


def model_path(hand: str) -> str:
    return str(get_asset_path("robots", f"tianji_marvin_{hand}", f"tianji_{hand}.xml"))


def hand_controller(hand: str, side: str) -> dict:
    """The side's hand sub-controller: the FR3 module's, on the prefixed joints and pads."""
    module, pads_attr = HANDS[hand]
    mod = importlib.import_module(f"unienv_genesis_collection.robots.{module}")
    sub = mod._DEFAULT_SUB_CONTROLLERS
    cfg = copy.copy(sub["hand"] if "hand" in sub else sub["gripper"])
    if "joint_indexes" in cfg:
        cfg.pop("joint_indexes")
        cfg["joint_names"] = list(_FRANKA_HAND_JOINTS)
    cfg["joint_names"] = [f"{side}_{j}" for j in cfg["joint_names"]]
    cfg["is_grasping_fn"] = make_is_grasping([f"{side}_{p}" for p in getattr(mod, pads_attr)])
    if "eef_tip_offset" in cfg:  # the arm targets the palm frame, which is already the grasp centre
        cfg["eef_tip_offset"] = [0.0, 0.0, 0.0]
    return cfg


def default_sub_controllers(hand: str) -> dict:
    out = {}
    for side, L in SIDES.items():
        arm = copy.copy(tj._DEFAULT_SUB_CONTROLLERS[f"{side}_arm"])
        arm["target_link"] = LinkAttachment(link_name=f"{side}_palm")
        out[f"{side}_arm"] = arm
        out[f"{side}_hand"] = hand_controller(hand, side)
    return out


def _keyframe(hand: str) -> RobotKeyframe:
    """The reset keyframe: the Tianji class's arm home, each hand's home from the model's ``home`` key."""
    root = ET.parse(model_path(hand)).getroot()
    names = [j.get("name") for j in root.find("worldbody").iter("joint")]
    q = [float(v) for v in root.find("keyframe").find("key").get("qpos").split()]
    home = dict(zip(names, q))
    for side, arm in (("left", tj._ARM_HOME_LEFT), ("right", tj._ARM_HOME_RIGHT)):
        for i, v in enumerate(arm, start=1):
            home[f"Joint{i}_{SIDES[side]}"] = float(v)
    return RobotKeyframe(joint_position=torch.tensor([home[n] for n in names]), joint_names=names)


class TianjiMarvinWithHand(RobotDescriptionNode):
    """Tianji Marvin 14-DOF dual-arm robot with ``hand`` on each arm (see the module docstring).

    Args:
        world: GenesisWorld instance
        hand: one of :data:`HANDS`
        name: Name for the robot entity
        pos: Base position (x, y, z)
        quat: Base orientation quaternion (w, x, y, z)
        controller_type: Controller mode. Defaults to ``"combined"``.
        **controller_kwargs: Forwarded to the controller; ``sub_controllers=...`` overrides the defaults.
    """

    def __init__(
        self,
        world: GenesisWorld,
        hand: str,
        name: str = "tianji",
        pos: tuple = None,
        quat: tuple = None,
        controller_type: CONTROL_MODE_T = "combined",
        **controller_kwargs,
    ):
        if hand not in HANDS:
            raise ValueError(f"unknown hand {hand!r}: one of {tuple(HANDS)}")
        self.hand = hand
        if controller_type == "combined" and "sub_controllers" not in controller_kwargs:
            controller_kwargs["sub_controllers"] = default_sub_controllers(hand)
        self.keyframes = {"reset": _keyframe(hand)}
        super().__init__(
            world=world,
            description_file=model_path(hand),
            name=name,
            keyframes=self.keyframes,
            fix_root=True,
            pos=pos,
            quat=quat,
            controller_type=controller_type,
            **controller_kwargs,
        )
