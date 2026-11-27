"""The Franka Hand (parallel gripper) as an end effector part.

Its description stays in ``robots.franka_fr3_hand`` (the merged FR3 + Franka Hand robot); ``part()`` is the gripper
alone, cut out of that model by ``scripts/build_eef_mjcf.py``.
"""

from unienv_genesis.nodes.entities.robot_assembly_node import PartFrame, RobotPart
from unienv_genesis_collection.robots import franka_fr3_hand as _fr3
from unienv_genesis_collection.robots.hands import eef_part

GRIPPER_JOINT_NAMES = ["finger_joint1", "finger_joint2"]
#: The merged robot's gripper sub-controller, on the joints by name (it names them by index in the merged model).
GRIPPER_SUB_CONTROLLER = {
    k: v for k, v in {**_fr3._DEFAULT_SUB_CONTROLLERS["gripper"], "joint_names": GRIPPER_JOINT_NAMES}.items()
    if k != "joint_indexes"
}
#: The fingertip point the merged robot's arm targets: ``hand`` + 0.1114 m along the approach.
EEF_FRAME = PartFrame(
    _fr3.EE_LINK_NAME, pos=tuple(_fr3._DEFAULT_SUB_CONTROLLERS["arm"]["target_link"].offset_pos),
)


def part() -> RobotPart:
    """The Franka Hand alone: ``eef/hand.xml``.

    Sub-controller ``"gripper"`` (-1 open, +1 closed), reset fully open; frame ``"eef"`` (:data:`EEF_FRAME`) is
    what the arm's end-effector controller targets.
    """
    return eef_part(
        "hand",
        {"gripper": GRIPPER_SUB_CONTROLLER},
        home=_fr3.GRIPPER_HOME_JOINT_POSITIONS,
        joint_names=GRIPPER_JOINT_NAMES,
        eef_frame=EEF_FRAME,
    )
