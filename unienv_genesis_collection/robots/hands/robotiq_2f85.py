"""The Robotiq 2F-85 gripper as an end effector part.

Its description stays in ``robots.franka_fr3_robotiq_2f85`` (the merged FR3 + Robotiq robot); ``part()`` is the
gripper alone, cut out of that model by ``scripts/build_eef_mjcf.py``.
"""

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots import franka_fr3_robotiq_2f85 as _fr3
from unienv_genesis_collection.robots.hands import eef_part

GRIPPER_SUB_CONTROLLER = _fr3._DEFAULT_SUB_CONTROLLERS["gripper"]
#: The arm gains the merged robot uses with this gripper, passed to the arm (``RobotPart.parent_overrides``).
ARM_OVERRIDES = {k: _fr3._DEFAULT_SUB_CONTROLLERS["arm"][k] for k in ("Kp", "Kd")}


def part() -> RobotPart:
    """The Robotiq 2F-85 alone: ``eef/robotiq_2f85.xml``.

    Sub-controller ``"gripper"`` (-1 open, +1 closed; its ``eef_tip_offset`` is the fingertip), reset with every
    joint at 0; frame ``"eef"`` (the gripper base ``g_base``, the flange) is what the arm's end-effector controller
    targets, with the arm gains :data:`ARM_OVERRIDES`.
    """
    return eef_part(
        "robotiq_2f85",
        {"gripper": GRIPPER_SUB_CONTROLLER},
        eef_frame=PartFrame(_fr3.EE_LINK_NAME),
        parent_overrides={"arm": ARM_OVERRIDES},
    )
