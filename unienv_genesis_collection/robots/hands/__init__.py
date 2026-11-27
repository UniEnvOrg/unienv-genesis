"""Hands and grippers as robot parts (``RobotPart``), mounted on an arm's flange at run time (``robots.arms``).

Each module also holds the end effector's description (joint names, gains, pad links, its sub-controller config)
that the merged ``franka_fr3_<name>`` / ``tianji_marvin_<name>`` robots use.
"""

import os
import xml.etree.ElementTree as ET
from typing import Tuple, Union


def mjcf_mount_pose(path: Union[str, os.PathLike]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """``(mount_pos, mount_quat)`` of a mountable end effector model (``eef/<name>.xml``): its root body's pose on
    the flange, stored as the custom numerics ``mount_pos`` / ``mount_quat`` by ``scripts/_fr3_eef_mjcf.write_eef_model``."""
    custom = ET.parse(path).getroot().find("custom")
    values = {n.get("name"): tuple(float(v) for v in n.get("data").split()) for n in custom.iter("numeric")}
    return values["mount_pos"], values["mount_quat"]
