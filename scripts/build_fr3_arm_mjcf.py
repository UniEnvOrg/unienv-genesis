"""Build the FR3 arm alone, ``franka_fr3_arm/fr3_arm.xml``: the arm of ``franka_fr3_hand/fr3_hand.xml`` without the
Franka Hand, as in every merged ``franka_fr3_<name>`` model. End effectors are mounted on it at run time
(``RobotAssemblyNode``, the flange frame :data:`_fr3_eef_mjcf.FLANGE`).

Run from the repository root:  python scripts/build_fr3_arm_mjcf.py
"""

from _fr3_eef_mjcf import write_arm_model

if __name__ == "__main__":
    print(write_arm_model())
