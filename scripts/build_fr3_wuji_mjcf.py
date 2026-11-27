"""Build the FR3 + Wuji hand MJCF from the model in ``unienv_genesis_collection/assets``.

Writes ``franka_fr3_wuji/fr3_wuji.xml``: the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` with its Franka Hand
replaced by the Wuji five-finger hand (``wuji_hand/right.xml``, the hand already used on the Tianji robot in
``tianji_marvin_wuji.py``).

The upstream model (a MuJoCo export of the Wuji hand) keeps the palm's two geoms and the five finger chains as
separate world bodies: the palm geoms are welded to the world, the fingers hang off the world by their own
bodies. Here they are collected under one root body (``wuji_hand``, the palm) so the hand can be mounted, and the
``right_`` prefix is dropped from the names:

* ``right_finger1_linkN`` / ``_jointN`` -> ``thumb_linkN`` / ``thumb_jointN``, ``right_finger2..5_*`` ->
  ``index`` / ``middle`` / ``ring`` / ``pinky`` (the model numbers its fingers from the thumb; the four fingers'
  bases run from the thumb side at +y to the pinky at -y, so finger 2 is the index).
* Default classes, materials and mesh names get the ``wuji_`` prefix. Every joint is driven by its own position
  actuator in the source; the source's ``kp`` is 0.5 N m / rad, so the robot class raises the gains
  (``franka_fr3_wuji.HAND_KP``) and keeps the source's ``forcerange``.

* **Mount.** By the back of the palm: the palm block is 32 mm thick along the palm normal (+x, x from -17.8 mm to
  +14.1 mm) and 81 x 105 mm across; its back face (x = -17.8 mm) is set on the FR3 flange through a 5 mm adapter,
  so the palm's +x (the side the fingers curl to) points out of the flange and the fingers (+z) lie level when the
  flange points down.
* **Palm frame.** A ``palm`` body at the centre of the four finger bases, ``(-6.6, -2.7, 85.0)`` mm in the root
  frame, with the model's own axes: +x out of the palm, +z along the fingers, +y toward the thumb.

Run from the repository root:  python scripts/build_fr3_wuji_mjcf.py
"""

from __future__ import annotations

import copy
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import ROBOTS, hand_assets, mount_transform, palm_frame, rename, set_pose, write_model

#: The model's finger number -> the finger it is (numbered from the thumb).
FINGERS = (("finger1", "thumb"), ("finger2", "index"), ("finger3", "middle"), ("finger4", "ring"), ("finger5", "pinky"))
#: Centre of the four finger bases (index / middle / ring / pinky) in the root frame (m).
PALM_ORIGIN = np.array([-0.0066341, -0.0027121, 0.085042])
#: The palm block's back face (its +x is out of the palm), in the root frame (m).
PALM_BACK_X = -0.0178


def build() -> pathlib.Path:
    hand = ET.parse(ROBOTS / "wuji_hand" / "right.xml").getroot()
    rename(hand, {f"right_{old}": new for old, new in FINGERS}, prefix="wuji_")
    meshdir = hand.find("compiler").get("meshdir", "")

    # The root body: the palm (its two geoms were welded to the world), the five fingers and the palm frame.
    root = ET.Element("body", name="wuji_hand")
    world = hand.find("worldbody")
    for e in list(world):
        if isinstance(e.tag, str):
            world.remove(e)
            root.append(e)
    x, y, z = np.eye(3)
    root.append(palm_frame(PALM_ORIGIN, x, y, z))
    set_pose(root, mount_transform(x, y, z, np.array([PALM_BACK_X, *PALM_ORIGIN[1:]])))

    # The hand's default block only describes its joints (the source has no class): it becomes a class of its
    # own, which the root body passes down to every hand joint.
    default = ET.SubElement(ET.Element("default"), "default", {"class": "wuji"})
    for e in hand.find("default"):
        if isinstance(e.tag, str):
            default.append(copy.deepcopy(e))
    return write_model(
        "wuji",
        root,
        [default],
        hand_assets(hand, "wuji_hand", meshdir),
        [a for a in hand.find("actuator") if isinstance(a.tag, str)],
        extra_excludes=(("fr3_link7", "wuji_hand"), ("fr3_link6", "wuji_hand")),
        n_hand_joints=20,
        hand_childclass="wuji",
        comment="FR3 + Wuji hand (assets/robots/wuji_hand, the hand of the Tianji robot).",
    )


if __name__ == "__main__":
    print(build())
