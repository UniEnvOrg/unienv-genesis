"""The Franka Research FR3 arm as a robot part, to mount end effectors on at run time.

``franka_fr3_arm/fr3_arm.xml`` (``scripts/build_fr3_arm_mjcf.py``) is the arm of every merged ``franka_fr3_<name>``
model without an end effector. End effectors are mounted on its :data:`FLANGE` frame (:class:`FrankaFR3Assembly`);
the end effectors' mountable models (``eef/<name>.xml``) are posed in that frame.

MJCF source (Apache 2.0): https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3
"""

import math
from typing import Optional

import torch

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.entities.robot_assembly_node import (
    PartFrame,
    PartLink,
    PartMount,
    RobotAssemblyNode,
    RobotPart,
)
from unienv_genesis.nodes.entities.robot_node import RobotKeyframe
from unienv_genesis_collection.asset_path import get_asset_path

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
ARM_HOME_JOINT_POSITIONS = torch.tensor([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Max joint velocity used by the arm's IK controller (rad/s).  This
# rate-limits the per-step IK joint-position targets.  Originally the DROID
# real-robot teleop limit (0.2 rad/s —
# Ref: https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py#L10
# ); raised to 0.4 rad/s for simulation, where the conservative teleop limit
# made the arm too slow to track transport waypoints.
DROID_FRANKA_ARM_IK_VEL_LIMIT = 0.4

# Franka hardware per-joint velocity limits (rad/s) from franka_hardware.yaml.
# Intentionally retained as hardware/reference documentation for future
# controller/planner work, even though the active controller uses the DROID
# IK velocity limit above; NOT wired into any active controller.
FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS = torch.tensor([2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51])

# Previous DROID_-prefixed name, intentionally retained alongside the
# reference constant above as searchable hardware documentation. Prefer
# ``FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS``; both names refer to the same
# reference tensor and neither feeds the active controller.
DROID_FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS = FRANKA_HARDWARE_ARM_JOINT_VEL_LIMITS


#: The flange: on the flange plane of ``fr3_link7`` (+z 0.107 m), +z out of the flange, +x along the Franka Hand's
#: x axis (``fr3_link7`` x turned -45 deg about z). ``scripts/_fr3_eef_mjcf.FLANGE``.
FLANGE = PartFrame("fr3_link7", pos=(0.0, 0.0, 0.107), quat=(math.cos(-math.pi / 8), 0.0, 0.0, math.sin(-math.pi / 8)))


def fr3_arm_part(target_link=None, **arm_controller_kwargs) -> RobotPart:
    """The FR3 arm: sub-controller ``"arm"`` (``eef_pose`` IK on ``target_link``, at the DROID velocity limit),
    reset to :data:`ARM_HOME_JOINT_POSITIONS`, frame ``"flange"``."""
    arm = {
        "controller_type": "eef_pose",
        "joint_names": ARM_JOINT_NAMES,
        "target_link": target_link,
        "max_joint_pos_vel": DROID_FRANKA_ARM_IK_VEL_LIMIT,
        **arm_controller_kwargs,
    }
    return RobotPart(
        description_file=get_asset_path("robots", "franka_fr3_arm", "fr3_arm.xml"),
        sub_controllers={"arm": arm},
        keyframe=RobotKeyframe(joint_position=ARM_HOME_JOINT_POSITIONS, joint_names=ARM_JOINT_NAMES),
        frames={"flange": FLANGE},
    )


class FrankaFR3Assembly(RobotAssemblyNode):
    """The FR3 arm with an end effector part (``robots.hands``) mounted on its flange.

    The arm's ``"arm"`` sub-controller targets the end effector's ``"eef"`` frame (its palm frame), and the end
    effector's own sub-controllers follow it, e.g. ``{"arm": ..., "hand": ...}`` as in the merged
    ``FrankaFR3With<Hand>`` robots.

    Args:
        world: GenesisWorld instance
        eef: The end effector part, e.g. ``robots.hands.wuji.part()`` or ``robots.hands.get_part("shadow", "left")``.
        name: The robot's name (the arm's entity; the end effector is ``f"{name}_{eef_name}"``).
        pos, quat: Base pose.
        eef_name: The end effector's part name.
        **arm_controller_kwargs: Overrides of the arm's sub-controller config, over the end effector's
            ``parent_overrides["arm"]`` (by default ``target_link`` is the end effector's ``"eef"`` frame).
    """

    def __init__(
        self,
        world: GenesisWorld,
        eef: RobotPart,
        name: str = "franka",
        pos: Optional[tuple] = None,
        quat: Optional[tuple] = None,
        eef_name: str = "hand",
        **arm_controller_kwargs,
    ):
        arm_controller_kwargs = {**eef.parent_overrides.get("arm", {}), **arm_controller_kwargs}
        if "target_link" not in arm_controller_kwargs and "eef" in eef.frames:
            arm_controller_kwargs["target_link"] = PartLink.of_frame(eef_name, eef.frames["eef"])
        super().__init__(
            world,
            parts={"arm": fr3_arm_part(**arm_controller_kwargs), eef_name: eef},
            mounts=[PartMount(child=eef_name, parent="arm", parent_frame="flange")],
            name=name,
            pos=pos,
            quat=quat,
        )
