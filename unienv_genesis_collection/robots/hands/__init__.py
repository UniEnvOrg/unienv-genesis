"""Hands and grippers as robot parts (``RobotPart``), mounted on an arm's flange at run time (``robots.arms``).

Each module holds one end effector's description (joint names, gains, pad links, its sub-controller config), which
the merged ``franka_fr3_<name>`` / ``tianji_marvin_<name>`` robots import, and ``part(...)``: the end effector alone
(``assets/robots/eef/<model>.xml``, cut out of the merged FR3 model by ``scripts/build_eef_mjcf.py``).

Example::

    from unienv_genesis_collection.robots.arms.franka_fr3 import FrankaFR3Assembly
    from unienv_genesis_collection.robots import hands

    robot = FrankaFR3Assembly(world, hands.get_part("shadow", side="left"), name="franka")
"""

import importlib
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe
from unienv_genesis_collection.asset_path import get_asset_path

#: end effector -> its module in this package.
HANDS = ("aero", "allegro", "aloha", "dexee", "franka_hand", "leap", "leap_v1", "robotiq_2f85", "shadow", "sharpa",
         "umi", "wuji")


def get_part(hand: str, side: Optional[str] = None, **kwargs) -> RobotPart:
    """``robots.hands.<hand>.part(side=side, **kwargs)`` (one of :data:`HANDS`; ``side`` only for sided hands)."""
    if hand not in HANDS:
        raise ValueError(f"unknown end effector {hand!r}: one of {HANDS}")
    module = importlib.import_module(f"{__name__}.{hand}")
    if side is not None:
        kwargs["side"] = side
    return module.part(**kwargs)


def mjcf_mount_pose(path: Union[str, os.PathLike]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """``(mount_pos, mount_quat)`` of a mountable end effector model (``eef/<name>.xml``): its root body's pose on
    the flange, stored as the custom numerics ``mount_pos`` / ``mount_quat`` by ``scripts/build_eef_mjcf.py``."""
    custom = ET.parse(path).getroot().find("custom")
    values = {n.get("name"): tuple(float(v) for v in n.get("data").split()) for n in custom.iter("numeric")}
    return values["mount_pos"], values["mount_quat"]


def check_side(side: str, sides: Sequence[str] = ("right", "left")) -> str:
    if side not in sides:
        raise ValueError(f"side must be one of {tuple(sides)}, got {side!r}")
    return side


def eef_part(
    model: str,
    sub_controllers: Dict[str, Dict[str, Any]],
    home: Optional[Any] = None,
    joint_names: Optional[Sequence[str]] = None,
    eef_frame: Optional[PartFrame] = None,
    parent_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> RobotPart:
    """The end effector ``eef/<model>.xml`` as a :class:`RobotPart`.

    Args:
        model: the model name (``eef/<model>.xml``).
        sub_controllers: the end effector's sub-controllers (joints by name).
        home: reset joint positions of ``joint_names`` (default: zeros).
        joint_names: the joints ``home`` sets (default: every joint of the model, in model order).
        eef_frame: the frame the arm's end-effector controller targets (part frame ``"eef"``).
        parent_overrides: see :class:`RobotPart`.
    """
    path = get_asset_path("robots", "eef", f"{model}.xml")
    mount_pos, mount_quat = mjcf_mount_pose(path)
    if joint_names is None:
        joint_names = [j.get("name") for j in ET.parse(path).getroot().find("worldbody").iter("joint")]
    home = torch.zeros(len(joint_names)) if home is None else torch.as_tensor(home, dtype=torch.float32)
    return RobotPart(
        description_file=path,
        sub_controllers=sub_controllers,
        keyframe=RobotKeyframe(joint_position=home, joint_names=list(joint_names)),
        frames={"eef": eef_frame} if eef_frame is not None else {},
        mount_pos=mount_pos,
        mount_quat=mount_quat,
        parent_overrides=dict(parent_overrides or {}),
    )
