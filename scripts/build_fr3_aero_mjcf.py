"""Build the FR3 + Tetheria Aero Hand MJCF from the mujoco_menagerie model in ``unienv_genesis_collection/assets``.

Writes ``franka_fr3_aero/fr3_aero.xml``: the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` with its Franka Hand
replaced by the Tetheria Aero Hand Open (``tetheria_aero_hand_open/right_hand.xml``, mujoco_menagerie
``tetheria_aero_hand_open``, Apache 2.0, pinned in ``tetheria_aero_hand_open/SOURCE.txt``).

Conversion:

* **Names.** The ``right_`` prefix goes: ``right_index_f_link`` -> ``index_base``, ``right_index_proximal_link``
  -> ``index_proximal``, ``right_index_middle_link`` -> ``index_middle``, ``right_index_distal_link`` ->
  ``index_distal`` (and ``middle`` / ``ring`` / ``pinky`` likewise), ``right_t_link`` -> ``thumb_base``,
  ``right_thumb_mcp_link`` -> ``thumb_mcp``, ``right_thumb_proximal_link`` -> ``thumb_proximal``,
  ``right_thumb_distal_link`` -> ``thumb_distal``; joints ``right_index_mcp_flex`` -> ``index_mcp_flex``,
  ``right_index_pip`` -> ``index_pip``, ``right_index_dip`` -> ``index_dip``, ``right_thumb_cmc_abd`` ->
  ``thumb_cmc_abd``, and so on. The model's own ``palm`` body becomes ``palm_base`` (the name ``palm`` is the
  frame this repository's robot classes drive), and default classes, materials and meshes get the ``aero_``
  prefix.
* **Tendons dropped, joints actuated directly.** Upstream the six flexor tendons are driven by position
  actuators on their lengths and five spatial tendons act as return springs; Genesis ignores MJCF tendons, so
  the tendon and spring blocks and their six actuators are dropped and each finger joint gets its own position
  actuator (11 in all). The model's own five joint equalities (``pip = dip`` on the four fingers, ``thumb_mcp =
  thumb_ip``) are kept, so those five joints follow their driver.
* **Mount.** By the back of the palm: the palm block is 38 mm thick along the palm normal (+x of the palm
  frame, the model's +z) and its back face is 19 mm behind the palm frame; it sits on the FR3 flange through a
  5 mm adapter, the palm facing out along the flange axis and the fingers level when the flange points down. The
  model's wrist plate (``tetheria_mount``, which carries the palm and extends 13 cm behind it) then points along
  the flange plane, away from the arm.
* **Palm frame.** A ``palm`` body at the centre of the four finger bases, ``(132.6, 12.9, -43.9)`` mm in the
  mount's frame, with +x out of the palm (the model's +z, where the fingers curl to), +z along the fingers (the
  model's +x) and +y toward the thumb (the model's -y).

``--side left`` builds ``franka_fr3_aero/fr3_aero_left.xml`` from ``left_hand.xml`` (upstream's mirror of the right
model): the same names (``left_`` dropped), the palm frame mirrored with it.

Run from the repository root:  python scripts/build_fr3_aero_mjcf.py
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import ROBOTS, hand_assets, hand_defaults, mount_transform, palm_frame, rename, set_pose, write_model

FINGERS = ("index", "middle", "ring", "pinky")
NAMES = {f"right_{f}_f_link": f"{f}_base" for f in FINGERS}
NAMES.update({f"right_{f}_proximal_link": f"{f}_proximal" for f in FINGERS})
NAMES.update({f"right_{f}_middle_link": f"{f}_middle" for f in FINGERS})
NAMES.update({f"right_{f}_distal_link": f"{f}_distal" for f in FINGERS})
NAMES.update({"right_t_link": "thumb_base", "right_thumb_mcp_link": "thumb_mcp",
              "right_thumb_proximal_link": "thumb_proximal", "right_thumb_distal_link": "thumb_distal",
              "palm": "palm_base"})
NAMES.update({f"right_{f}_mcp_flex": f"{f}_mcp_flex" for f in FINGERS})
NAMES.update({f"right_{f}_pip": f"{f}_pip" for f in FINGERS})
NAMES.update({f"right_{f}_dip": f"{f}_dip" for f in FINGERS})
NAMES.update({f"right_thumb_{j}": f"thumb_{j}" for j in ("cmc_abd", "cmc_flex", "mcp", "ip")})
#: The driven joints, in controller order: each finger's flexion and PIP (the DIP follows the PIP), the thumb's
#: abduction, CMC flexion and MCP (the IP follows the MCP).
HAND_JOINTS = ([f"{f}_{j}" for f in FINGERS for j in ("mcp_flex", "pip")]
               + ["thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp"])
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {**{f"{f}_dip": (f"{f}_pip", 1.0, 0.0) for f in FINGERS}, "thumb_ip": ("thumb_mcp", 1.0, 0.0)}
#: Centre of the four finger bases in the root frame (m).
PALM_ORIGIN = np.array([0.1326, 0.0129, -0.0439])
#: Palm frame axes in the root frame: +x out of the palm (the model's +z), +y toward the thumb, +z along the
#: fingers (the model's +x).
X_PALM, Y_PALM, Z_PALM = np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])
#: The palm block's back face, 19 mm behind the palm frame along -x (the model's -z).
PALM_BACK = 0.019
#: Joint position gains (N m / rad) and torque limits: the light (0.45 kg) open-source hand's own scale.
KP = {"mcp_flex": 2.0, "pip": 1.5, "thumb_cmc_abd": 1.0, "thumb_cmc_flex": 1.5, "thumb_mcp": 1.5}
FORCE = 2.0


def _side_names(side: str) -> dict:
    """:data:`NAMES` for ``side`` (the left model's names start with ``left_``)."""
    return {(f"{side}_" + k[len("right_"):] if k.startswith("right_") else k): v for k, v in NAMES.items()}


def build(side: str = "right") -> pathlib.Path:
    names = _side_names(side)
    # The left model is upstream's mirror of the right one across the root's xz plane: the palm frame's origin
    # mirrors (y -> -y); its +x (out of the palm) and +z (along the fingers) stay, so +y points away from the thumb.
    origin = PALM_ORIGIN * (np.array([1.0, -1.0, 1.0]) if side == "left" else 1.0)
    hand = ET.parse(ROBOTS / "tetheria_aero_hand_open" / f"{side}_hand.xml").getroot()
    meshdir = hand.find("compiler").get("meshdir", "")
    for tag in ("tendon", "sensor", "custom"):
        el = hand.find(tag)
        if el is not None:
            hand.remove(el)
    hand.remove(hand.find("actuator"))
    actuators = [ET.Element("position", name=f"{j}_actuator", joint=j, kp=f"{_kp(j):g}", kv="0.05",
                            ctrlrange=f"0 {_range(hand, j, names):g}", forcerange=f"-{FORCE:g} {FORCE:g}")
                 for j in HAND_JOINTS]
    rename(hand, names, prefix="aero_")
    root = hand.find("worldbody").find("body")
    root.set("name", "aero_hand")
    root.append(palm_frame(origin, X_PALM, Y_PALM, Z_PALM))
    set_pose(root, mount_transform(X_PALM, Y_PALM, Z_PALM, origin - PALM_BACK * X_PALM))

    bodies = {b.get("name") for b in root.iter("body")}
    contacts = [(e.get("body1"), e.get("body2")) for e in hand.find("contact")
                if isinstance(e.tag, str) and {e.get("body1"), e.get("body2")} <= bodies]
    equality = [ET.Element("joint", joint1=dep, joint2=drv, polycoef=f"0 {m:g} {o:g} 0 0", solref="0.005 1")
                for dep, (drv, m, o) in COUPLINGS.items()]
    return write_model(
        "aero" if side == "right" else "aero_left",
        root,
        hand_defaults(hand),
        hand_assets(hand, "tetheria_aero_hand_open", meshdir),
        actuators,
        equality=equality,
        contacts=contacts,
        extra_excludes=(("fr3_link7", "aero_hand"), ("fr3_link6", "aero_hand")),
        n_hand_joints=16,
        hand_childclass="aero_tetheria_rh",
        comment=f"FR3 + Tetheria Aero Hand Open{"" if side == "right" else ", left"} (assets/robots/tetheria_aero_hand_open from mujoco_menagerie, Apache 2.0).",
        directory="franka_fr3_aero",
    )


def _kp(joint: str) -> float:
    """The joint's position gain: ``KP`` is keyed by joint type, e.g. ``index_mcp_flex`` -> ``mcp_flex``."""
    return KP[joint] if joint in KP else KP[joint.split("_", 1)[1]]


def _range(hand: ET.Element, joint: str, names: dict = NAMES) -> float:
    """The joint's upper limit, read from the (not yet renamed) model."""
    upstream = {new: old for old, new in names.items()}[joint]
    for j in hand.iter("joint"):
        if j.get("name") == upstream:
            return float(j.get("range").split()[1])
    raise KeyError(joint)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("right", "left", "both"), default="right")
    side_ = ap.parse_args().side
    for side in (("right", "left") if side_ == "both" else (side_,)):
        print(build(side))
