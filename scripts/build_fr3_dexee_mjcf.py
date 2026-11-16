"""Build the FR3 + Shadow DEX-EE MJCF from the mujoco_menagerie model in ``unienv_genesis_collection/assets``.

Writes ``franka_fr3_dexee/fr3_dexee.xml``: the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` with its Franka Hand
replaced by the Shadow DEX-EE three-finger hand (``shadow_dexee/shadow_dexee.xml``, mujoco_menagerie
``shadow_dexee``, Apache 2.0, pinned in ``shadow_dexee/SOURCE.txt``).

Conversion:

* **Names.** ``F0/`` (the finger's mounting frame) -> ``f0_mount``, ``F0/finger_base`` -> ``f0_finger_base``, ``F0/finger_knuckle`` -> ``f0_knuckle``,
  ``F0/finger_proximal`` -> ``f0_proximal``, ``F0/finger_middle`` -> ``f0_middle``,
  ``F0/finger_distal`` -> ``f0_distal`` (and ``F1`` / ``F2`` likewise), joints ``F0/J0`` -> ``f0_j0`` ...
  ``F0/J3`` -> ``f0_j3``. Default classes, materials and meshes get the ``dexee_`` prefix.
* **Tree.** Upstream, the base (``hand_base``) and the three fingers are all welded to the world; the fingers
  become children of the base, which becomes the hand's root.
* **Debug geometry.** The model's group-5 contact-free ``*_CollisionGeom_*`` capsules and cylinders (its own
  visualisation of the collision shapes) are dropped.
* **Actuators.** The model drives its 12 joints through the ``mujoco.pid`` plugin, which Genesis does not
  implement (its actuators are silently inert there): each becomes a plain position actuator with the plugin's
  own ``kp`` / ``kd`` (``J0`` 2.8 / 0.03, ``J1`` 2.5 / 0.02, ``J2`` 1.1 / 0.01, ``J3`` 0.6 / 0.008), the same
  ``ctrlrange`` and ``forcerange``. The ``mujoco.pid`` extension block, its ``jointactuatorfrc`` sensors and the
  ``<custom>`` contact numerics are dropped.
* **Mount.** By the back of the palm: the base is a 148 mm wide block whose back face (the model's -y, the side
  opposite the fingers' curl and the logo) is 74 mm behind the finger bases; it sits on the FR3 flange through a
  5 mm adapter, the palm facing out along the flange axis and the fingers level when the flange points down. The
  base's ``gravcomp="1"`` and the fingers' are kept.
* **Palm frame.** A ``palm`` body at the centre of the three finger bases, ``(0, -2.7, 17)`` mm in the base
  frame, with +x out of the palm (the model's +y: the direction the two fingers on the palm side, F1 and F2,
  curl to), +z along the fingers and +y across the fingers.

Run from the repository root:  python scripts/build_fr3_dexee_mjcf.py
"""

from __future__ import annotations

import copy
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import ROBOTS, hand_assets, hand_defaults, mount_transform, palm_frame, rename, set_pose, write_model

FINGERS = ("f0", "f1", "f2")
#: The body renames: the model's ``F0/finger_base`` ... -> one flat namespace.
_PARTS = (("finger_base", "finger_base"), ("finger_knuckle", "knuckle"), ("finger_proximal", "proximal"),
          ("finger_middle", "middle"), ("finger_distal", "distal"))
NAMES = {f"F{i}/": f"f{i}_mount" for i in range(3)}  # the fingers' (massless) mounting frames
NAMES.update({f"F{i}/{part}": f"f{i}_{new}" for i in range(3) for part, new in _PARTS})
NAMES.update({f"F{i}/J{j}": f"f{i}_j{j}" for i in range(3) for j in range(4)})
#: The ``mujoco.pid`` gains of each joint, by joint index (from the model's ``<extension>``).
PID = {0: (2.8, 0.03), 1: (2.5, 0.02), 2: (1.1, 0.01), 3: (0.6, 0.008)}
#: Centre of the three finger bases in the base frame (m).
PALM_ORIGIN = np.array([0.0, -0.0027, 0.017])
#: Palm frame axes in the base frame: +x out of the palm (the model's +y), +y across the fingers, +z along them.
X_PALM, Y_PALM, Z_PALM = np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
#: The base's back face, 74 mm behind the palm frame along -x (the model's -y).
PALM_BACK = 0.074


def build() -> pathlib.Path:
    hand = ET.parse(ROBOTS / "shadow_dexee" / "shadow_dexee.xml").getroot()
    meshdir = hand.find("compiler").get("meshdir", "")
    for tag in ("extension", "sensor", "custom"):
        el = hand.find(tag)
        if el is not None:
            hand.remove(el)
    world = hand.find("worldbody")
    base = next(b for b in world if b.tag == "body" and b.get("name") == "hand_base")
    for e in list(world):
        if e is base:
            continue
        world.remove(e)
        if isinstance(e.tag, str) and e.tag == "body":  # the three fingers, welded to the world upstream
            base.append(e)
    base.set("name", "dexee_hand")

    # The model carries a set of group-5, contact-free "CollisionGeom" capsules/cylinders (its own visualisation
    # of the collision shapes); they hide the hand in a render and serve no purpose here.
    for parent in list(base.iter()):
        for g in list(parent.findall("geom")):
            if "CollisionGeom" in (g.get("name") or ""):
                parent.remove(g)

    actuators = []
    for a in hand.find("actuator"):
        if not isinstance(a.tag, str):
            continue
        joint = a.get("joint")
        finger, index = joint.split("/")
        kp, kd = PID[int(index[1:])]
        actuators.append(ET.Element("position", name=f"{NAMES[joint]}_actuator", joint=NAMES[joint],
                                    kp=f"{kp:g}", kv=f"{kd:g}", ctrlrange=a.get("ctrlrange"),
                                    forcerange=a.get("forcerange")))
    rename(hand, NAMES, prefix="dexee_")
    base.append(palm_frame(PALM_ORIGIN, X_PALM, Y_PALM, Z_PALM))
    set_pose(base, mount_transform(X_PALM, Y_PALM, Z_PALM, PALM_ORIGIN - PALM_BACK * X_PALM))

    bodies = {b.get("name") for b in base.iter("body")}
    contacts = [(e.get("body1"), e.get("body2")) for e in hand.find("contact")
                if isinstance(e.tag, str) and {e.get("body1"), e.get("body2")} <= bodies]
    return write_model(
        "dexee",
        base,
        hand_defaults(hand),
        hand_assets(hand, "shadow_dexee", meshdir),
        actuators,
        contacts=contacts,
        extra_excludes=(("fr3_link7", "dexee_hand"), ("fr3_link6", "dexee_hand")),
        n_hand_joints=12,
        comment="FR3 + Shadow DEX-EE (assets/robots/shadow_dexee from mujoco_menagerie, Apache 2.0).",
    )


if __name__ == "__main__":
    print(build())
