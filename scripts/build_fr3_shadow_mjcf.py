"""Build the FR3 + Shadow Hand MJCF from the mujoco_menagerie model in ``unienv_genesis_collection/assets``.

Writes ``franka_fr3_shadow/fr3_shadow.xml``: the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` with its Franka Hand
replaced by the Shadow Dexterous Hand (``shadow_hand/right_hand.xml``, mujoco_menagerie ``shadow_hand``, Apache
2.0, pinned in ``shadow_hand/SOURCE.txt``).

Conversion:

* **Names.** The ``rh_`` prefix goes, and the bodies are named after what they are: ``hand_palm`` (the palm, the
  model's ``rh_palm``), ``ff_knuckle`` / ``ff_proximal`` / ``ff_middle`` / ``ff_distal`` (index),
  ``mf_*`` / ``rf_*`` (middle / ring), ``lf_metacarpal`` / ``lf_knuckle`` / ``lf_proximal`` / ``lf_middle`` /
  ``lf_distal`` (little), ``th_base`` / ``th_proximal`` / ``th_hub`` / ``th_middle`` / ``th_distal`` (thumb), and
  the joints after the upstream numbering: ``ff_j4`` (knuckle abduction), ``ff_j3`` (proximal), ``ff_j2``
  (middle), ``ff_j1`` (distal), ``lf_j5`` (metacarpal), ``th_j5`` (opposition) ... ``th_j1``. Default classes,
  materials and meshes get the ``shadow_`` prefix.
* **Forearm and wrist dropped.** The model's root is a 3 kg, 21 cm forearm housing (``rh_forearm``) with the
  hand's wrist gimbal (``rh_wrist``, ``rh_WRJ1`` / ``rh_WRJ2``) in front of it; bolted to the FR3 flange that
  housing would sit inside ``fr3_link6`` / ``fr3_link7``, and the FR3's own wrist takes its place. The hand is
  re-rooted at the palm, so 22 joints (the fingers) remain instead of the upstream 24.
* **Coupled joints.** Each finger's distal joint follows its middle joint (upstream one tendon per finger,
  ``rh_FFJ0`` = ``FFJ2`` + ``FFJ1``): the tendon actuators become position actuators on the middle joints and
  ``ff_j1 = ff_j2`` etc. become joint equality constraints (Genesis ignores tendons but supports equalities).
* **Mount.** By the back of the palm: the palm block is 25 mm thick along the palm normal (the model's -y) and
  its back face is 19 mm behind the finger bases; it sits on the FR3 flange through a 5 mm adapter, the palm
  facing out along the flange axis and the fingers level when the flange points down.
* **Palm frame.** A ``palm`` body at the centre of the four finger bases, ``(0, 0, 93.9)`` mm in the palm
  frame, with +x out of the palm (the model's -y, where the fingers curl to), +z along the fingers (the model's
  +z) and +y toward the thumb (the model's +x).

* **Left hand.** ``--side left`` builds ``franka_fr3_shadow/fr3_shadow_left.xml`` from ``shadow_hand/left_hand.xml``
  (upstream's mirror of the right model across the palm's x): the same names (the ``lh_`` prefix goes), the same
  palm frame constants (+x out of the palm, +z along the fingers; +y then points away from the thumb) and mount.

Run from the repository root:  python scripts/build_fr3_shadow_mjcf.py [--side right|left|both]
"""

from __future__ import annotations

import copy
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import ROBOTS, hand_assets, hand_defaults, mount_transform, palm_frame, rename, set_pose, write_model

#: The fingers: the model's joint prefix -> the new name stem (the index finger is ``FF`` upstream).
FINGERS = (("FF", "ff"), ("MF", "mf"), ("RF", "rf"), ("LF", "lf"))
#: Body renames (``rh_`` dropped, named after what each link is).
BODIES = {
    "rh_palm": "hand_palm",
    "rh_ffknuckle": "ff_knuckle", "rh_ffproximal": "ff_proximal", "rh_ffmiddle": "ff_middle", "rh_ffdistal": "ff_distal",
    "rh_mfknuckle": "mf_knuckle", "rh_mfproximal": "mf_proximal", "rh_mfmiddle": "mf_middle", "rh_mfdistal": "mf_distal",
    "rh_rfknuckle": "rf_knuckle", "rh_rfproximal": "rf_proximal", "rh_rfmiddle": "rf_middle", "rh_rfdistal": "rf_distal",
    "rh_lfmetacarpal": "lf_metacarpal", "rh_lfknuckle": "lf_knuckle", "rh_lfproximal": "lf_proximal",
    "rh_lfmiddle": "lf_middle", "rh_lfdistal": "lf_distal",
    "rh_thbase": "th_base", "rh_thproximal": "th_proximal", "rh_thhub": "th_hub", "rh_thmiddle": "th_middle",
    "rh_thdistal": "th_distal",
}
#: The joints of the four fingers, upstream numbering: J4 knuckle abduction, J3 proximal, J2 middle, J1 distal.
_FINGER_JOINTS = (("J4", "j4"), ("J3", "j3"), ("J2", "j2"), ("J1", "j1"))
_JOINTS = {f"rh_{up}{n}": f"{new}_{n.lower()}" for up, new in FINGERS for n, _ in _FINGER_JOINTS}
_JOINTS.update({"rh_LFJ5": "lf_j5"})
_JOINTS.update({f"rh_THJ{i}": f"th_j{i}" for i in range(1, 6)})
#: Joints that follow another through an equality constraint: {dependent: (driver, multiplier, offset)}.
COUPLINGS = {f"{new}_j1": (f"{new}_j2", 1.0, 0.0) for _, new in FINGERS}
#: Centre of the four finger bases in the (new) root frame, m.
PALM_ORIGIN = np.array([0.0, 0.0, 0.0939])
#: The palm frame's axes in the root frame: +x out of the palm, +y toward the thumb, +z along the fingers.
X_PALM, Y_PALM, Z_PALM = np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
#: The palm block's back face, 19 mm behind the palm frame along -x (the model's +y: 66.6 mm at the widest).
PALM_BACK = 0.019


def build(side: str = "right") -> pathlib.Path:
    p = {"right": "rh_", "left": "lh_"}[side]
    hand = ET.parse(ROBOTS / "shadow_hand" / f"{side}_hand.xml").getroot()
    meshdir = hand.find("compiler").get("meshdir", "")
    for tag in ("tendon", "sensor"):
        el = hand.find(tag)
        if el is not None:
            hand.remove(el)

    # Re-root at the palm: the forearm (3 kg) and the wrist gimbal are replaced by the FR3 wrist.
    palm = next(b for b in hand.find("worldbody").iter("body") if b.get("name") == f"{p}palm")
    palm.set("pos", "0 0 0")
    palm.attrib.pop("quat", None)
    for j in palm.findall("joint"):  # the palm's own wrist hinge (rh_WRJ1) is replaced by the flange
        palm.remove(j)
    names = {p + k[len("rh_"):]: v for k, v in {**BODIES, **_JOINTS}.items()}
    rename(hand, names, prefix="shadow_")

    actuators = []
    for a in hand.find("actuator"):
        if not isinstance(a.tag, str):
            continue
        joint = a.get("joint")
        if joint in (f"{p}WRJ1", f"{p}WRJ2"):
            continue
        c = copy.deepcopy(a)
        if a.get("tendon") is not None:  # one tendon per finger (J2 + J1): drive J2, J1 follows
            finger = a.get("tendon")[len(p):-len("J0")].lower()
            c.attrib.pop("tendon")
            c.set("joint", f"{finger}_j2")
            c.set("name", f"{finger}_j2_actuator")
            c.set("ctrlrange", f"0 {float(a.get('ctrlrange', '0 3.1415').split()[1]) / 2:.6g}")
        actuators.append(c)

    palm_frame_body = palm_frame(PALM_ORIGIN, X_PALM, Y_PALM, Z_PALM)
    palm.append(palm_frame_body)
    mount_point = PALM_ORIGIN - PALM_BACK * X_PALM
    set_pose(palm, mount_transform(X_PALM, Y_PALM, Z_PALM, mount_point))

    equality = [ET.Element("joint", joint1=dep, joint2=drv, polycoef=f"0 {m:g} {o:g} 0 0", solref="0.005 1")
                for dep, (drv, m, o) in COUPLINGS.items()]
    # Contact exclusions, minus the ones whose bodies were dropped (the forearm / wrist pair).
    bodies = {b.get("name") for b in palm.iter("body")}
    contacts = [(e.get("body1"), e.get("body2")) for e in hand.find("contact")
                if isinstance(e.tag, str) and {e.get("body1"), e.get("body2")} <= bodies]
    return write_model(
        "shadow" if side == "right" else "shadow_left",
        palm,
        hand_defaults(hand),
        hand_assets(hand, "shadow_hand", meshdir),
        actuators,
        equality=equality,
        contacts=contacts,
        extra_excludes=(("fr3_link7", "hand_palm"), ("fr3_link6", "hand_palm")),
        n_hand_joints=22,
        hand_childclass=f"shadow_{side}_hand",
        comment=f"FR3 + Shadow Hand{"" if side == "right" else ", left"} (assets/robots/shadow_hand from mujoco_menagerie, Apache 2.0).",
        directory="franka_fr3_shadow",
    )





if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("right", "left", "both"), default="right")
    for side in (("right", "left") if ap.parse_args().side == "both" else (ap.parse_args().side,)):
        print(build(side))
