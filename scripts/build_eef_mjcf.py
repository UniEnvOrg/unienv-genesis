"""Cut the end effector out of each merged FR3 model, to mount it on an arm at run time.

For every merged ``franka_fr3_<x>/fr3_<y>.xml`` in :data:`MODELS` this writes ``eef/<y>.xml``: the body tree under
``fr3_link7`` (the hand or gripper) with exactly the elements of the merged model that belong to it, so that a
``RobotAssemblyNode`` mounting it on the FR3 arm (``franka_fr3_arm/fr3_arm.xml``) simulates the same robot as the
merged model:

* the bodies, with the default class they inherit in the merged model (the FR3 bodies carry ``childclass="fr3"``,
  so a hand without a class of its own inherits the arm's defaults; the root body gets that class explicitly);
* the default classes, assets (``file`` paths made relative to ``eef/``), tendons, actuators, equalities, contact
  exclusions / pairs and sensors that refer to the end effector only;
* the ``home`` keyframe's end effector part (after the arm's 7 joints) as its own ``home``.

Genesis loads a root body that has a pose under an extra ``world`` link, which an attach leaves behind, so the root
body is written at the origin and its pose on the flange (:data:`_fr3_eef_mjcf.FLANGE`) as the custom numerics
``mount_pos`` / ``mount_quat`` (``(w, x, y, z)``).

Run from the repository root:  python scripts/build_eef_mjcf.py [name ...]
"""

from __future__ import annotations

import argparse
import copy
import os
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from _fr3_eef_mjcf import FLANGE, ROBOTS, T_of, _write, fmt, quat_from_matrix

#: merged model (under ``assets/robots``) -> end effector model name (``eef/<name>.xml``).
MODELS = {
    "franka_fr3_hand/fr3_hand.xml": "hand",
    "franka_fr3_robotiq_2f85/fr3_robotiq_2f85.xml": "robotiq_2f85",
    "franka_fr3_umi/fr3_umi.xml": "umi",
    "franka_fr3_aloha/fr3_aloha.xml": "aloha",
    "franka_fr3_wuji/fr3_wuji.xml": "wuji",
    "franka_fr3_shadow/fr3_shadow.xml": "shadow",
    "franka_fr3_shadow/fr3_shadow_left.xml": "shadow_left",
    "franka_fr3_aero/fr3_aero.xml": "aero",
    "franka_fr3_aero/fr3_aero_left.xml": "aero_left",
    "franka_fr3_leap_v1/fr3_leap_v1.xml": "leap_v1",
    "franka_fr3_leap_v1/fr3_leap_v1_left.xml": "leap_v1_left",
    "franka_fr3_dexee/fr3_dexee.xml": "dexee",
    "franka_fr3_allegro/fr3_allegro_right.xml": "allegro_right",
    "franka_fr3_allegro/fr3_allegro_left.xml": "allegro_left",
    "franka_fr3_leap/fr3_leap_right.xml": "leap_right",
    "franka_fr3_leap/fr3_leap_left.xml": "leap_left",
    "franka_fr3_sharpa/fr3_sharpa_right.xml": "sharpa_right",
    "franka_fr3_sharpa/fr3_sharpa_left.xml": "sharpa_left",
}
#: The arm's joints in the merged models (the first 7 entries of their ``home`` keyframe).
N_ARM_QPOS = 7
#: Elements that may name a body / joint / site / tendon / geom, by attribute.
_REF_ATTRS = ("body", "body1", "body2", "joint", "joint1", "joint2", "site", "tendon", "tendon1", "tendon2",
              "geom", "geom1", "geom2", "objname", "jointinparent", "cranksite", "slidersite", "refsite")


def _classes_used(tree: ET.Element) -> set[str]:
    return {e.get(a) for e in tree.iter() for a in ("class", "childclass") if e.get(a)}


def _default_tree(default: ET.Element) -> dict[str, list[ET.Element]]:
    """``{class: [the default element of the class and its ancestors, outermost first]}``."""
    out: dict[str, list[ET.Element]] = {}

    def walk(el: ET.Element, chain: list[ET.Element]) -> None:
        for c in el:
            if c.tag == "default":
                here = chain + [c]
                if c.get("class"):
                    out[c.get("class")] = here
                walk(c, here)

    walk(default, [])
    return out


def _resolve(path_dir: pathlib.Path, sub: str, file: str) -> str:
    """The file's path (lexically normalized: the assets directory may be a symlink)."""
    return file if os.path.isabs(file) else os.path.normpath(os.path.join(path_dir, sub, file))


def extract(merged_rel: str, name: str) -> pathlib.Path:
    src_path = ROBOTS / merged_rel
    src = ET.parse(src_path).getroot()
    world = src.find("worldbody")
    parent_of = {c: p for p in world.iter() for c in p}
    link7 = next(b for b in world.iter("body") if b.get("name") == "fr3_link7")
    roots = [b for b in link7 if b.tag == "body"]
    if len(roots) != 1:
        raise ValueError(f"{merged_rel}: expected one end effector body under fr3_link7, found {len(roots)}")
    root = copy.deepcopy(roots[0])

    # The class the root body inherits in the merged model: the nearest childclass above it.
    inherited, e = None, link7
    while e is not None and inherited is None:
        inherited = e.get("childclass")
        e = parent_of.get(e)
    if root.get("childclass") is None and inherited is not None:
        root.set("childclass", inherited)

    names = {e.get("name") for e in root.iter() if e.get("name")}
    joints = [e.get("name") for e in root.iter("joint")] + [e.get("name") for e in root.iter("freejoint")]
    if any(e.tag == "freejoint" for e in root.iter()):
        raise ValueError(f"{merged_rel}: the end effector has a free joint")

    def ours(e: ET.Element) -> bool:
        refs = [e.get(a) for a in _REF_ATTRS if e.get(a)]
        refs += [j.get("joint") for j in e.iter("joint") if j is not e and j.get("joint")]  # tendon / equality children
        return bool(refs) and all(r in names or r in tendons for r in refs)

    tendons: set[str] = set()
    out = ET.Element("mujoco", model=f"eef_{name}")
    out.append(ET.Comment(f" Generated by scripts/build_eef_mjcf.py from {merged_rel}; do not edit by hand. "))
    out.append(ET.Comment(" The end effector of the merged model, to mount on an arm's flange at run time. "))

    compiler = src.find("compiler")
    sub_mesh = sub_tex = ""
    if compiler is not None:
        c = copy.deepcopy(compiler)
        sub_mesh = c.attrib.pop("meshdir", "")
        sub_tex = c.attrib.pop("texturedir", sub_mesh)
        c.attrib.pop("assetdir", None)
        out.append(c)
    if src.find("option") is not None:
        out.append(copy.deepcopy(src.find("option")))

    default_at = len(out)
    eef_dir = ROBOTS / "eef"

    mount = np.linalg.inv(FLANGE) @ T_of(root)
    for attr in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
        root.attrib.pop(attr, None)
    ET.SubElement(out, "worldbody").append(root)

    # Tendons first (actuators / equalities may refer to them), then everything else that is the end effector's.
    kept_all: list[ET.Element] = []
    for section in ("tendon", "actuator", "equality", "contact", "sensor"):
        el = src.find(section)
        if el is None:
            continue
        kept = []
        for e in el:
            if not isinstance(e.tag, str):
                continue
            if section == "tendon":
                if all(j.get("joint") in names for j in e.iter("joint")) and all(
                        s.get("site") in names for s in e.iter("site")):
                    kept.append(e)
                    tendons.add(e.get("name"))
            elif ours(e):
                kept.append(e)
        if kept:
            ET.SubElement(out, section).extend(copy.deepcopy(e) for e in kept)
            kept_all.extend(kept)

    # Defaults: the top-level settings and every top-level class tree holding a class the end effector uses.
    default = src.find("default")
    default_users: list[ET.Element] = []
    if default is not None:
        tree = _default_tree(default)
        used = _classes_used(root) | set().union(*(_classes_used(e) for e in kept_all))
        missing = sorted(used - set(tree))
        if missing:
            raise ValueError(f"{merged_rel}: classes {missing} are not defined")
        tops = {id(tree[c][0]) for c in used}
        # The settings of the used classes and their ancestors (a class nobody uses may name an undefined asset).
        default_users += [e for c in used for el in tree[c] for e in el if e.tag != "default"]
        default_users += [e for e in default if e.tag != "default"]
        d = ET.Element("default")
        out.insert(default_at, d)
        for child in default:
            if child.tag != "default" or id(child) in tops:
                d.append(copy.deepcopy(child))

    # Assets referenced by the end effector, directly or through its default classes (materials pull their textures).
    asset = src.find("asset")
    if asset is not None:
        users = [e for t in [root, *kept_all] for e in t.iter()] + default_users
        meshes = {e.get("mesh") for e in users if e.get("mesh")}
        materials = {e.get("material") for e in users if e.get("material")}
        hfields = {e.get("hfield") for e in users if e.get("hfield")}
        textures = {a.get("texture") for a in asset if a.tag == "material" and a.get("name") in materials}
        wanted = {"mesh": meshes, "material": materials, "texture": textures, "hfield": hfields}
        a_out = ET.Element("asset")
        out.insert(default_at + len(out.findall("default")), a_out)
        for a in asset:
            if not isinstance(a.tag, str):
                continue
            key = a.get("name") or (pathlib.Path(a.get("file", "")).stem if a.tag == "mesh" else None)
            if a.tag in wanted and key in wanted[a.tag]:
                c = copy.deepcopy(a)
                for attr in ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"):
                    if c.get(attr):
                        sub = sub_mesh if a.tag in ("mesh", "hfield") else sub_tex
                        c.set(attr, os.path.relpath(_resolve(src_path.parent, sub, c.get(attr)), eef_dir))
                if a.tag == "mesh" and not c.get("name"):
                    c.set("name", key)
                a_out.append(c)
        found = {a.get("name") or pathlib.Path(a.get("file", "")).stem for a in a_out}
        direct = [e for t in [root, *kept_all] for e in t.iter()]
        required = {e.get(a) for e in direct for a in ("mesh", "material", "hfield") if e.get(a)}
        required |= {a.get("texture") for a in a_out if a.tag == "material" and a.get("texture")}
        # Assets only a default names may be missing: the merged model loads as long as no element uses them.
        lost = sorted(required - found)
        if lost:
            raise ValueError(f"{merged_rel}: assets {lost} not found")


    key = src.find("keyframe")
    if key is not None:
        home = next((k for k in key if k.get("name") == "home"), None)
        if home is not None and home.get("qpos"):
            q = home.get("qpos").split()[N_ARM_QPOS:]
            ET.SubElement(ET.SubElement(out, "keyframe"), "key", name="home", qpos=" ".join(q))

    custom = ET.SubElement(out, "custom")
    custom.append(ET.Comment(" The root body's pose on the flange (+z out of the flange). "))
    ET.SubElement(custom, "numeric", name="mount_pos", data=fmt(mount[:3, 3]))
    ET.SubElement(custom, "numeric", name="mount_quat", data=fmt(quat_from_matrix(mount[:3, :3])))
    return _write(out, eef_dir / f"{name}.xml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help=f"end effectors to write (default: all of {sorted(MODELS.values())})")
    wanted = set(parser.parse_args().names) or set(MODELS.values())
    for merged, eef in MODELS.items():
        if eef in wanted:
            print(extract(merged, eef))
