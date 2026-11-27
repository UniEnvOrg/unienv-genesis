"""Build the FR3 + LEAP Hand (v1) MJCF from the mujoco_menagerie model in ``unienv_genesis_collection/assets``.

Writes ``franka_fr3_leap_v1/fr3_leap_v1.xml``: the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` with its Franka
Hand replaced by the LEAP Hand (``leap_hand/right_hand.xml``, mujoco_menagerie ``leap_hand``, MIT, pinned in
``leap_hand/SOURCE.txt``). This is the first-generation LEAP Hand; the v2 Adv hand this collection already has
(``franka_fr3_leap``) is a different model.

Conversion:

* **Names.** The bodies keep the model's short names (``if_bs`` / ``if_px`` / ``if_md`` / ``if_ds``: the index
  finger's base, proximal, medial and distal links, ``th_mp`` / ``th_bs`` / ``th_px`` / ``th_ds``: the thumb's),
  except the root palm, which becomes ``leap_palm`` (``palm`` is the frame this repository's robot classes
  drive). Default classes, materials and meshes get the ``leap1_`` prefix. The joints keep their names
  (``if_mcp``, ``if_rot``, ``if_pip``, ``if_dip``, ...
  ).
* **Actuators.** Nineteen joints, sixteen of them actuated (three fingers x 4 + the thumb's 4), each by its own
  position actuator with the model's gains. There are no coupled joints (unlike the v2 Adv hand).
* **Mount.** By the back of the palm: the palm block is 60 mm thick along the palm normal (the model's -z) and
  149 x 155 mm across; its back face (the model's +z, the top of the housing) sits on the FR3 flange through a
  5 mm adapter, so the palm's +x (the side the fingers curl to, the model's -z) points out of the flange and the
  fingers (the model's +x) lie level when the flange points down. The wrist connector at the back of the housing
  then points along the flange plane, away from the arm.
* **Palm frame.** A ``palm`` body at the centre of the three finger bases, ``(-7.1, -22.4, -18.7)`` mm in the
  root frame, with +x out of the palm (the model's -z), +z along the fingers (the model's +x) and +y across the
  fingers (the model's +y).

* **Left hand.** ``--side left`` builds ``franka_fr3_leap_v1/fr3_leap_v1_left.xml`` from ``left_hand.xml``.
  Upstream's left model is its own export, not a mirror of the right one's frames; its palm frame is placed by
  the same rule (centre of the finger bases, the same axes) and its back face measured on the compiled model
  (its geometry within 35 mm of the flange axis ends 43.2 mm behind the palm frame, as on the right model). The
  right model's :data:`PALM_BACK` (78.3 mm) leaves a 35 mm gap between the adapter and the palm; it is kept so the
  existing right model (and the data collected with it) is unchanged.

Run from the repository root:  python scripts/build_fr3_leap_v1_mjcf.py
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import ROBOTS, hand_assets, hand_defaults, mount_transform, palm_frame, rename, set_pose, write_model

FINGERS = ("if", "mf", "rf")
#: The driven joints, in controller order (each finger's 4, then the thumb's 4).
HAND_JOINTS = ([f"{f}_{j}" for f in FINGERS for j in ("mcp", "rot", "pip", "dip")]
               + ["th_cmc", "th_axl", "th_mcp", "th_ipl"])
#: Centre of the three finger bases in the root frame (m).
PALM_ORIGIN = np.array([-0.00707, -0.02243, -0.01870])
#: Palm frame axes in the root frame: +x out of the palm (the model's -z), +y across the fingers, +z along them.
X_PALM, Y_PALM, Z_PALM = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
#: The palm block's back face, 78.3 mm behind the palm frame along -x (the model's +z: 59.6 mm).
PALM_BACK = 0.0783


#: The left model (upstream's own export, not a mirror of the right one's frames): the centre of its three finger
#: bases and its palm's back face, measured the same way; the palm axes in its root frame are the right model's.
LEFT_PALM_ORIGIN = np.array([-0.0825, -0.0403, 0.0078])
LEFT_PALM_BACK = 0.0432


def build(side: str = "right") -> pathlib.Path:
    origin, back = (PALM_ORIGIN, PALM_BACK) if side == "right" else (LEFT_PALM_ORIGIN, LEFT_PALM_BACK)
    hand = ET.parse(ROBOTS / "leap_hand" / f"{side}_hand.xml").getroot()
    meshdir = hand.find("compiler").get("meshdir", "")
    sensor = hand.find("sensor")
    if sensor is not None:
        hand.remove(sensor)
    actuators = [a for a in hand.find("actuator") if isinstance(a.tag, str)]
    rename(hand, {"palm": "leap_palm"}, prefix="leap1_")
    root = hand.find("worldbody").find("body")
    root.append(palm_frame(origin, X_PALM, Y_PALM, Z_PALM))
    set_pose(root, mount_transform(X_PALM, Y_PALM, Z_PALM, origin - back * X_PALM))

    bodies = {b.get("name") for b in root.iter("body")}
    contacts = [(e.get("body1"), e.get("body2")) for e in hand.find("contact")
                if isinstance(e.tag, str) and {e.get("body1"), e.get("body2")} <= bodies]
    return write_model(
        "leap_v1" if side == "right" else "leap_v1_left",
        root,
        hand_defaults(hand),
        hand_assets(hand, "leap_hand", meshdir),
        actuators,
        contacts=contacts,
        extra_excludes=(("fr3_link7", "leap_palm"), ("fr3_link6", "leap_palm")),
        n_hand_joints=16,
        comment=f"FR3 + LEAP Hand v1{"" if side == "right" else ", left"} (assets/robots/leap_hand from mujoco_menagerie, MIT).",
        directory="franka_fr3_leap_v1",
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("right", "left", "both"), default="right")
    side_ = ap.parse_args().side
    for side in (("right", "left") if side_ == "both" else (side_,)):
        print(build(side))
