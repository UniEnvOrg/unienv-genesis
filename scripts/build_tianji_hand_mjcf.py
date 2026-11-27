"""Build the Tianji Marvin + <hand> MJCFs: the Tianji's two arms with another end effector on each in place of the Wuji.

Writes ``tianji_marvin_<hand>/tianji_<hand>.xml`` for each hand of :data:`SOURCES` from the Tianji model
(``tianji_marvin_ccs_696/tianji_marvin_CCS_real_limit_wuji.xml``) and the hand's FR3 models (``franka_fr3_*``,
built by ``scripts/build_fr3_*_mjcf.py``; one per side, or one for both sides for a symmetric gripper):

* **Wuji removed.** Its bodies, default class, actuators, contact exclusions and the assets only it used go; the
  arms, their defaults, actuators and exclusions stay as they are.
* **Hand copied.** Each side's hand subtree (the bodies under the FR3's ``fr3_link7``) with everything it needs: the
  FR3 model's default block (wrapped as one class, ``<side>_src``, so the hand keeps exactly the defaults it was
  built with), its assets, joint actuators, equality constraints (joint couplings and the Robotiq's linkage
  ``connect`` constraints) and internal contact exclusions. Tendons and the actuators on them are not copied
  (Genesis ignores tendons; the robot classes drive the joints). Every name and every
  reference gets the side's prefix (``left_`` / ``right_``): ``left_palm``, ``left_index_PP`` ...
* **Mount.** Inline, as the Tianji carries its Wuji: the hand's ``palm`` frame on the flange axis (``Link7``'s -y),
  with the hand's end nearest the arm seated flush on ``Link7``'s face (95 mm out): the geometry of its rigid part
  (the palm and what is fixed to it, not the fingers, which move: the LEAP v2 Adv's thumb sits at the wrist end)
  within :data:`FOOTPRINT` of the palm's z axis touches the face. A dexterous hand's palm axes are the
  Wuji's on that arm (fingers along the flange axis, the palm facing the way the Wuji's does); a gripper's approach
  (+z) is along the flange axis and its closing direction (+y) along the Wuji's palm normal, as a thumb-to-fingers
  pinch closes. :data:`DORSAL` hands (the LEAP v2 Adv, whose thumb sits at the wrist end of its palm) are mounted by
  the back of the hand instead, as on the FR3: the palm normal (+x) along the flange axis, the thumb side (+y) the
  Wuji's, the back seated on the face. (The Wuji itself sits 14 mm into the face; the FR3 models' adapter gaps are
  not carried over.)
* **Adapters dropped.** :data:`DROP_GEOMS`: the Aero's Tetheria mounting bracket (its root body's mesh and two
  collision boxes) is an adapter for its own arm interface, angled and reaching 8 cm in front of the palm.
* **Palm frame.** Every hand has a ``<side>_palm`` body, which the robot class's arm controllers target with no
  offset. The Franka Hand and the Robotiq 2F-85 have none in their FR3 models: one is added by the grippers'
  convention (the UMI's): between the two fingers' tips, +z along the approach, +y along the closing direction.
* **Keyframe.** ``home``: the Tianji's arm joints and each hand's own home (its FR3 model's ``home`` after the arm).

Geometry (the palm frame, the seat, the fingertips) is measured on the compiled models with MuJoCo.

Run from the repository root:  python scripts/build_tianji_hand_mjcf.py [hand ...]   (default: all)
"""

from __future__ import annotations

import copy
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import ROBOTS, T_of, fmt, quat_from_matrix, set_pose

TIANJI = ROBOTS / "tianji_marvin_ccs_696" / "tianji_marvin_CCS_real_limit_wuji.xml"
SIDES = {"left": "L", "right": "R"}
#: The hand's FR3 models: ``{hand: {side: model}}`` (a symmetric gripper uses one model for both sides).
SOURCES = {
    "sharpa": {s: f"franka_fr3_sharpa/fr3_sharpa_{s}.xml" for s in SIDES},
    "leap": {s: f"franka_fr3_leap/fr3_leap_{s}.xml" for s in SIDES},
    "shadow": {"left": "franka_fr3_shadow/fr3_shadow_left.xml", "right": "franka_fr3_shadow/fr3_shadow.xml"},
    "leap_v1": {"left": "franka_fr3_leap_v1/fr3_leap_v1_left.xml", "right": "franka_fr3_leap_v1/fr3_leap_v1.xml"},
    "aero": {"left": "franka_fr3_aero/fr3_aero_left.xml", "right": "franka_fr3_aero/fr3_aero.xml"},
    "umi": {s: "franka_fr3_umi/fr3_umi.xml" for s in SIDES},
    "robotiq": {s: "franka_fr3_robotiq_2f85/fr3_robotiq_2f85.xml" for s in SIDES},
    "hand": {s: "franka_fr3_hand/fr3_hand.xml" for s in SIDES},
}
#: Grippers without a ``palm`` body: ``(frame body, approach axis in it, (left finger body, right finger body))``.
ADD_PALM = {
    "hand": ("hand", (0.0, 0.0, 1.0), ("left_finger", "right_finger")),
    "robotiq": ("g_base", (0.0, 0.0, 1.0), ("g_left_pad", "g_right_pad")),
}
#: Bodies whose own geoms are dropped on the Tianji: arm adapters that are not part of the hand (the Aero's root
#: carries the Tetheria mounting bracket, angled for its own arm interface).
DROP_GEOMS = {"aero": ("aero_hand",)}
#: Hands mounted by the back of the hand (palm normal along the flange axis) rather than inline: the LEAP v2 Adv's
#: thumb sits at the wrist end of its palm, and its own arm interface is the bracket behind the palm.
DORSAL = ("leap",)
#: The grippers (approach along the flange axis); the rest are dexterous hands (fingers along it).
GRIPPERS = ("umi", "robotiq", "hand")
#: The flange axis in ``Link7``'s frame (the Wuji hangs off ``Link7`` along -y).
AXIS = np.array([0.0, -1.0, 0.0])
#: Radius around the palm's z axis within which the wrist end / rear face is seated on ``Link7``'s face (m).
FOOTPRINT = 0.035

NAMED = ("name", "joint", "joint1", "joint2", "body1", "body2", "mesh", "material", "texture", "site", "tendon",
         "class", "childclass", "target", "objname")
ASSETS = ("mesh", "material", "texture", "hfield", "skin")


def _xyaxes_R(el: ET.Element) -> np.ndarray:
    if el.get("xyaxes") is not None:
        v = np.array([float(x) for x in el.get("xyaxes").split()])
        x, y = v[:3] / np.linalg.norm(v[:3]), v[3:] / np.linalg.norm(v[3:])
        return np.stack([x, y, np.cross(x, y)], axis=1)
    return T_of(el)[:3, :3]


def _compile(xml: str | pathlib.Path):
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_kinematics(m, d)
    return m, d


def _geom_points(m, d, bodies: set[int]) -> np.ndarray:
    """World points of the geoms of ``bodies``: mesh vertices, box corners, sphere / capsule / cylinder extremes."""
    import mujoco

    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in bodies:
            continue
        R, t, s = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
        kind = m.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            local = m.mesh_vert[a:a + n]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            local = np.concatenate([np.eye(3), -np.eye(3)]) * s[0]
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            ring = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]) * s[0]
            ends = [-s[1], s[1]] if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else [-s[1] - s[0], -s[1], s[1], s[1] + s[0]]
            local = np.concatenate([ring + [0, 0, z] for z in ends])
        else:
            continue
        pts.append(local @ R.T + t)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def _subtree_ids(m, root_ids: list[int]) -> set[int]:
    out = set(root_ids)
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p in root_ids:
                out.add(b)
                break
            p = m.body_parentid[p]
    return out


def _prefix_tree(root: ET.Element, p: str) -> None:
    """Prefix every name and reference in ``root``'s subtree (body / joint / geom / site names, assets, classes)."""
    for e in root.iter():
        if not isinstance(e.tag, str):
            continue
        if e.tag == "mesh" and e.get("name") is None and e.get("file") is not None:
            e.set("name", pathlib.Path(e.get("file")).stem)  # MuJoCo's default name for an unnamed mesh
        if e.tag == "texture" and e.get("name") is None and e.get("file") is not None:
            e.set("name", pathlib.Path(e.get("file")).stem)
        for attr in NAMED:
            v = e.get(attr)
            if v is not None:
                e.set(attr, p + v)


def _hand_home(src: ET.Element) -> list[str]:
    key = src.find("keyframe").find("key")
    return key.get("qpos").split()[7:]


def _add_palm(src: ET.Element, model_path: pathlib.Path, hand: str) -> None:
    """Add the grippers' ``palm`` body to the FR3 model's tree ``src`` (see the module docstring)."""
    frame_name, approach, fingers = ADD_PALM[hand]
    m, d = _compile(model_path)
    import mujoco

    fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    Rf, tf = d.xmat[fb].reshape(3, 3), d.xpos[fb]
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in fingers]
    tips, centres = [], []
    for i in ids:
        p = (_geom_points(m, d, _subtree_ids(m, [i])) - tf) @ Rf  # in the frame body
        tips.append(p @ np.asarray(approach))
        centres.append(p.mean(0))
    a = np.asarray(approach, dtype=float)
    tip = max(t.max() for t in tips)
    y = centres[1] - centres[0]
    y = y - a * (y @ a)
    y /= np.linalg.norm(y)
    mid = 0.5 * (centres[0] + centres[1])
    origin = mid - a * (mid @ a) + a * tip
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(y, a), y, a], axis=1)
    T[:3, 3] = origin
    frame = next(b for b in src.iter("body") if b.get("name") == frame_name)
    palm = ET.SubElement(frame, "body", name="palm")
    set_pose(palm, T)
    ET.SubElement(palm, "site", name="palm_site", size="0.005", rgba="1 0.5 0 0.5", group="4")


def _tianji_base() -> tuple[ET.Element, dict]:
    """The Tianji model without its Wuji hands (mesh paths made relative to a sibling folder), and per side
    ``{side: (Link7 element, R(Link7 -> Wuji palm axes))}``."""
    t = ET.parse(TIANJI).getroot()
    meshdir = t.find("compiler").attrib.pop("meshdir", "")
    for a in t.find("asset"):
        if isinstance(a.tag, str) and a.get("file") is not None:
            a.set("file", f"../{TIANJI.parent.name}/{meshdir}{a.get('file')}")
    links, removed_bodies, removed_joints = {}, set(), set()
    for side, L in SIDES.items():
        l7 = next(b for b in t.iter("body") if b.get("name") == f"Link7_{L}")
        wuji = next(b for b in l7.findall("body") if b.get("name") == f"{side}_wuji_palm")
        links[side] = (l7, _xyaxes_R(wuji))
        removed_bodies |= {b.get("name") for b in wuji.iter("body")}
        removed_joints |= {j.get("name") for j in wuji.iter("joint")}
        l7.remove(wuji)
    default = t.find("default")
    default.remove(next(c for c in default if c.get("class") == "wuji_hand"))
    act = t.find("actuator")
    for a in list(act):
        if isinstance(a.tag, str) and a.get("joint") in removed_joints:
            act.remove(a)
    con = t.find("contact")
    for e in list(con):
        if isinstance(e.tag, str) and ({e.get("body1"), e.get("body2")} & removed_bodies):
            con.remove(e)
    # Drop the assets nothing references any more (the Wuji's meshes and materials).
    used = {v for e in t.iter() if isinstance(e.tag, str) and e.tag not in ASSETS
            for attr in ("mesh", "material", "texture") if (v := e.get(attr)) is not None}
    used |= {e.get("texture") for e in t.find("asset") if isinstance(e.tag, str) and e.tag == "material" and e.get("texture") and e.get("name") in used}
    asset = t.find("asset")
    for a in list(asset):
        if isinstance(a.tag, str) and a.tag in ASSETS and (a.get("name") or pathlib.Path(a.get("file", "")).stem) not in used:
            asset.remove(a)
    t.remove(t.find("keyframe"))
    return t, links


def _link7_face() -> float:
    """``Link7``'s face along the flange axis (-y), within :data:`FOOTPRINT` of it (m)."""
    import mujoco

    m, d = _compile(TIANJI)
    out = []
    for L in SIDES.values():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"Link7_{L}")
        p = (_geom_points(m, d, {b}) - d.xpos[b]) @ d.xmat[b].reshape(3, 3)
        near = p[np.linalg.norm(p[:, [0, 2]], axis=1) < FOOTPRINT]
        out.append(-near[:, 1].min())
    return float(np.mean(out))


def build(hand: str) -> pathlib.Path:
    import mujoco

    tianji, links = _tianji_base()
    face = _link7_face()
    home_arm = ET.parse(TIANJI).getroot().find("keyframe").find("key").get("qpos").split()
    arm_home = {"left": home_arm[0:7], "right": home_arm[27:34]}  # the Tianji's key: L arm, L Wuji, R arm, R Wuji
    hand_home, report = {}, []
    for side, src_rel in SOURCES[hand].items():
        p = f"{side}_"
        src_path = ROBOTS / src_rel
        src = ET.parse(src_path).getroot()
        for b in src.iter("body"):
            if b.get("name") in DROP_GEOMS.get(hand, ()):
                for g in b.findall("geom"):
                    b.remove(g)
        if hand in ADD_PALM:
            _add_palm(src, src_path, hand)
        if hand in ADD_PALM or hand in DROP_GEOMS:
            tmp = src_path.with_name(f".{src_path.stem}_tianji.xml")
            ET.ElementTree(src).write(tmp, encoding="unicode")
            m, d = _compile(tmp)
            tmp.unlink()
        else:
            m, d = _compile(src_path)
        l7_el = next(b for b in src.iter("body") if b.get("name") == "fr3_link7")
        roots = [b for b in l7_el.findall("body")]
        # Mount (see the module docstring), from the compiled FR3 model: the hand's geometry in its palm frame.
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731

        def T_world(i):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = d.xmat[i].reshape(3, 3), d.xpos[i]
            return T

        T_w_palm = T_world(bid("palm"))
        ids = _subtree_ids(m, [bid(r.get("name")) for r in roots])
        rigid = {b for b in ids if m.body_weldid[b] == m.body_weldid[bid("fr3_link7")]}  # the palm, not the fingers
        P = (_geom_points(m, d, rigid) - T_w_palm[:3, 3]) @ T_w_palm[:3, :3]
        k = 0 if hand in DORSAL else 2  # the palm axis that lies along the flange axis
        lat = [i for i in range(3) if i != k]
        seat = float(P[np.linalg.norm(P[:, lat], axis=1) < FOOTPRINT][:, k].min())  # wrist end / back / rear face
        link7, R_wuji = links[side]
        if hand in GRIPPERS:  # approach along the flange axis, closing along the Wuji's palm normal
            y = R_wuji[:, 0] - AXIS * (R_wuji[:, 0] @ AXIS)
            y /= np.linalg.norm(y)
            R_palm = np.stack([np.cross(y, AXIS), y, AXIS], axis=1)
        elif hand in DORSAL:  # the palm normal along the flange axis, the thumb side (+y) as the Wuji's
            y = R_wuji[:, 1] - AXIS * (R_wuji[:, 1] @ AXIS)
            y /= np.linalg.norm(y)
            R_palm = np.stack([AXIS, y, np.cross(AXIS, y)], axis=1)
        else:  # the Wuji's palm axes: fingers along the flange axis
            R_palm = R_wuji
        T_l7_palm = np.eye(4)
        T_l7_palm[:3, :3] = R_palm
        T_l7_palm[:3, 3] = AXIS * (face - seat)
        # The hand's pieces, prefixed.
        defaults = ET.Element("default", {"class": "src"})
        for c in src.find("default"):
            defaults.append(copy.deepcopy(c))
        names_assets = {}
        for r in roots:
            T_palm_r = np.linalg.inv(T_w_palm) @ T_world(bid(r.get("name")))
            r = copy.deepcopy(r)
            set_pose(r, T_l7_palm @ T_palm_r)
            if r.get("childclass") is None:
                r.set("childclass", "src")
            _prefix_tree(r, p)
            link7.append(r)
            report.append(f"{side}: palm {1000 * (face - seat):.0f} mm out")
        hand_bodies = {b.get("name") for r in roots for b in r.iter("body")}
        hand_joints = {j.get("name") for r in roots for j in r.iter("joint")}
        _prefix_tree(defaults, p)
        tianji.find("default").append(defaults)
        # Assets the hand references (materials pull in their textures).
        refs = {v for r in roots for e in r.iter() for a in ("mesh", "material") if (v := e.get(a)) is not None}
        refs |= {v for e in defaults.iter() for a in ("mesh", "material") if (v := e.get(a)) is not None and v.startswith(p)}
        refs = {v[len(p):] if v.startswith(p) else v for v in refs}
        src_assets = [a for a in src.find("asset") if isinstance(a.tag, str)]
        for a in src_assets:
            if a.tag == "mesh" and a.get("name") is None and a.get("file"):
                a.set("name", pathlib.Path(a.get("file")).stem)
        refs |= {a.get("texture") for a in src_assets if a.tag == "material" and a.get("name") in refs and a.get("texture")}
        for a in src_assets:
            if a.get("name") in refs:
                a = copy.deepcopy(a)
                _prefix_tree(a, p)
                tianji.find("asset").append(a)
        for a in src.find("actuator"):
            if isinstance(a.tag, str) and a.get("joint") in hand_joints:
                a = copy.deepcopy(a)
                _prefix_tree(a, p)
                tianji.find("actuator").append(a)
        if src.find("equality") is not None:
            eq = tianji.find("equality") if tianji.find("equality") is not None else ET.SubElement(tianji, "equality")
            for e in src.find("equality"):
                if isinstance(e.tag, str) and (e.get("joint1") in hand_joints or e.get("body1") in hand_bodies):
                    e = copy.deepcopy(e)
                    _prefix_tree(e, p)
                    eq.append(e)
        con = tianji.find("contact")
        for e in src.find("contact"):
            if isinstance(e.tag, str) and {e.get("body1"), e.get("body2")} <= hand_bodies:
                e = copy.deepcopy(e)
                _prefix_tree(e, p)
                con.append(e)
        L = SIDES[side]
        for r in roots:
            for arm_link in (f"Link7_{L}", f"Link6_{L}"):
                ET.SubElement(con, "exclude", body1=arm_link, body2=p + r.get("name"))
        hand_home[side] = _hand_home(src)
    # Keep the element order MuJoCo expects (equality after actuators is fine; keyframe last).
    qpos = arm_home["left"] + hand_home["left"] + arm_home["right"] + hand_home["right"]
    ET.SubElement(ET.SubElement(tianji, "keyframe"), "key", name="home", qpos=" ".join(qpos))
    tianji.set("model", f"tianji_{hand}")
    tianji.insert(0, ET.Comment(f" Generated by scripts/build_tianji_hand_mjcf.py from {TIANJI.name} and "
                                f"{', '.join(sorted(set(SOURCES[hand].values())))}; do not edit by hand. "))
    out = ROBOTS / f"tianji_marvin_{hand}" / f"tianji_{hand}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tianji, space="  ")
    ET.ElementTree(tianji).write(out, encoding="unicode")
    m, _ = _compile(out)  # must compile
    print(f"{out}: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, Link7 face {1000 * face:.1f} mm; " + "; ".join(report))
    return out


if __name__ == "__main__":
    for h in sys.argv[1:] or SOURCES:
        build(h)
