"""Shared FR3 + end-effector MJCF assembly for ``scripts/build_fr3_*_mjcf.py``.

Every ``build_fr3_<name>_mjcf.py`` script converts one hand or gripper model to MJCF and hands the result to
:func:`write_model`, which merges it under the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` (the Franka Hand is
dropped) and writes ``assets/robots/franka_fr3_<name>/fr3_<name>.xml``.

Conventions shared by all the built models:

* **Flange.** The arm's flange is the Franka Hand mount of ``fr3_hand.xml``: ``fr3_link7`` +z at 0.107 m, with an
  adapter of :data:`ADAPTER` in front of it. Nothing is placed between the flange and the end effector except that
  adapter plate.
* **Palm frame (dexterous hands).** A ``palm`` body on the hand's root, at the centre of the finger bases, with
  +x out of the palm (the side the fingers curl to) and +z along the fingers (the Wuji / Allegro / LEAP / Sharpa
  convention). The hand is mounted by the *back of the palm*: the palm's rigid geometry is entirely on the far side
  of the flange plane, so the flange plane is tangent to the back of the palm, and +x of the palm points out of the
  flange. With the flange pointing down the palm therefore faces down and the fingers are level; +z of the palm
  points along the Franka Hand's x axis (``fr3_link7`` x turned -45 deg about z), the direction the Franka Hand's
  fingers separate in.
* **Palm frame (grippers).** The ``palm`` body is at the grasp centre between the fingertips, +z out of the flange
  (the approach direction, the fingertips' way) and +y along the closing direction. The gripper is mounted on its
  own rear face, so its geometry is a rigid extension of the flange.
* **Names.** Hand default classes, materials, textures and meshes get the hand's ``prefix`` (the FR3's are
  ``fr3``, ``visual``, ``collision`` ...); body and joint names stay as they are (the robot classes use them).
* **Home keyframe.** ``home`` holds the FR3 home pose and zeros for every hand joint; hands whose zero pose is
  not a valid open pose pass their own ``home``.
"""

from __future__ import annotations

import copy
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

ROBOTS = pathlib.Path(__file__).resolve().parents[1] / "unienv_genesis_collection" / "assets" / "robots"

#: ``fr3_link7`` -> flange, from ``fr3_hand.xml`` (the Franka Hand hangs at ``pos="0 0 0.107"``).
FLANGE_Z = 0.107
#: Thickness of the adapter plate between the flange and the end effector.
ADAPTER = 0.005
#: The FR3 ``home`` keyframe of ``fr3_hand.xml``.
FR3_HOME = "0 0 0 -1.57079 0 1.57079 -0.7853"
_C = np.sqrt(0.5)

#: Attributes that reference a name and must follow a rename.
NAMED = ("name", "joint", "joint1", "joint2", "body1", "body2", "mesh", "material", "texture", "site", "tendon")
#: Attributes that name a default class.
CLASSED = ("class", "childclass")


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` of a rotation matrix (the MJCF quaternion order).

    The branch on the largest diagonal term is needed for 180 deg rotations, where the usual off-diagonal
    differences all vanish (and the sign of the axis is otherwise ambiguous).
    """
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    if q[0] < 0:  # the same rotation, w >= 0
        q = -q
    return q / np.linalg.norm(q)


def fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def set_pose(el: ET.Element, T: np.ndarray) -> None:
    """Write a 4x4 transform as ``pos`` + ``quat`` (dropping any other orientation specifier)."""
    for attr in ("euler", "axisangle", "xyaxes", "zaxis"):
        el.attrib.pop(attr, None)
    el.set("pos", fmt(T[:3, 3]))
    el.set("quat", fmt(quat_from_matrix(T[:3, :3])))


def T_of(el: ET.Element | None) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        q = el.get("quat")
        if q is not None:
            q = np.array([float(v) for v in q.split()])
            w, x, y, z = q / np.linalg.norm(q)
            T[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        elif el.get("axisangle") is not None:
            axis, angle = el.get("axisangle").split()[:3], float(el.get("axisangle").split()[3])
            axis = np.array([float(v) for v in axis]) / np.linalg.norm([float(v) for v in axis])
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            T[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        if el.get("pos") is not None:
            T[:3, 3] = [float(v) for v in el.get("pos").split()]
    return T


def mount_transform(x_palm: np.ndarray, y_palm: np.ndarray, z_palm: np.ndarray, mount_point: np.ndarray):
    """Hand root pose on the flange.

    The palm frame's +x maps to ``fr3_link7`` +z (out of the flange) and its +z to the Franka Hand's x axis; a
    gripper's +z maps to +z and its +y to the Franka Hand's y axis instead (see :func:`gripper_mount`).

    Args:
        x_palm, y_palm, z_palm: the palm frame's axes in the hand root's frame.
        mount_point: the point of the hand (in the hand root's frame) that touches the adapter plate.

    Returns:
        The hand root's 4x4 pose in ``fr3_link7``.
    """
    D = np.stack([np.array([0.0, 0.0, 1.0]), np.array([-_C, -_C, 0.0]), np.array([_C, -_C, 0.0])], axis=1)
    A = np.stack([x_palm, y_palm, z_palm], axis=1)
    R = D @ A.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([0.0, 0.0, FLANGE_Z + ADAPTER]) - R @ np.asarray(mount_point, dtype=float)
    return T


def gripper_mount(x_palm: np.ndarray, y_palm: np.ndarray, z_palm: np.ndarray, mount_point: np.ndarray):
    """Gripper root pose on the flange: +z of the palm frame (the approach) out of the flange, +y (closing) along
    the Franka Hand's y axis, and the gripper's own rear face (:paramref:`mount_point`) on the adapter plate."""
    D = np.stack([np.array([_C, -_C, 0.0]), np.array([_C, _C, 0.0]), np.array([0.0, 0.0, 1.0])], axis=1)
    A = np.stack([x_palm, y_palm, z_palm], axis=1)
    R = D @ A.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([0.0, 0.0, FLANGE_Z + ADAPTER]) - R @ np.asarray(mount_point, dtype=float)
    return T


def rename(root: ET.Element, names: dict[str, str], prefix: str = "") -> None:
    """Apply a ``{old: new}`` body/joint name map and prefix default classes, materials, textures and meshes.

    ``names`` also maps a name *prefix*: an entry ``{"right_finger1": "thumb"}`` renames ``right_finger1_link2``
    to ``thumb_link2`` (an exact match is renamed to the new name as it is). Asset declarations (``mesh``,
    ``material``, ``texture``) get ``prefix``, and so do the elements that reference them.
    """
    assets = ("mesh", "material", "texture")

    def mapped(value: str) -> str:
        if value in names:
            return names[value]
        for old, new in names.items():
            if value.startswith(old + "_") or value.startswith(old + "/"):
                return new + "_" + value[len(old) + 1:]
        return value

    for e in root.iter():
        if not isinstance(e.tag, str):
            continue
        if e.tag in assets and e.get("name") is not None:  # the declaration
            e.set("name", prefix + e.get("name"))
        elif e.tag == "mesh" and e.get("file") is not None:  # an unnamed mesh: name it after its file
            e.set("name", prefix + pathlib.Path(e.get("file")).stem)
        for attr in NAMED:
            value = e.get(attr)
            if value is None or (attr == "name" and e.tag in assets):
                continue
            if attr in assets:  # a reference to a mesh / material / texture
                if not value.startswith(prefix):
                    e.set(attr, prefix + value)
            elif attr == "name" and e.tag in ("body", "joint", "site"):
                e.set(attr, mapped(value))
            else:
                e.set(attr, mapped(value))
        if e.tag == "default":
            if e.get("class") is not None:  # the declaration of a class
                e.set("class", prefix + e.get("class").replace("/", "_"))
            continue
        for attr in CLASSED:
            if e.get(attr) is not None:
                e.set(attr, prefix + e.get(attr).replace("/", "_"))


def hand_assets(hand: ET.Element, asset_dir: str, meshdir: str | None = None) -> list[ET.Element]:
    """The hand's ``<asset>`` children with ``file`` rewritten to point into ``assets/robots/<asset_dir>``."""
    out = []
    for a in hand.find("asset"):
        if not isinstance(a.tag, str):
            continue
        c = copy.deepcopy(a)
        if c.tag in ("mesh", "texture"):
            f = c.get("file")
            if not f.startswith("../") and not f.startswith("/"):
                sub = (meshdir or "").strip("/")
                c.set("file", f"../{asset_dir}/{sub + '/' if sub else ''}{f}")
        out.append(c)
    return out


def hand_defaults(hand: ET.Element) -> list[ET.Element]:
    """The hand's top-level default elements (copies), to be appended to the model's ``<default>``."""
    return [copy.deepcopy(d) for d in hand.find("default") if isinstance(d.tag, str)]


def load_arm() -> dict:
    """The FR3 arm of ``franka_fr3_hand/fr3_hand.xml``: defaults without the Franka Hand, meshes, link0 tree."""
    src = ET.parse(ROBOTS / "franka_fr3_hand" / "fr3_hand.xml").getroot()
    default = copy.deepcopy(src.find("default"))
    for d in list(default):
        if d.get("class") == "hand":
            default.remove(d)
    assets = [copy.deepcopy(a) for a in src.find("asset")
              if isinstance(a.tag, str)
              and not (a.tag == "mesh" and "franka_emika_panda" in a.get("file", ""))
              and not a.get("name", "").startswith("hand_")]
    arm = copy.deepcopy(src.find("worldbody").find("body"))
    link7 = next(b for b in arm.iter("body") if b.get("name") == "fr3_link7")
    link7.remove(next(b for b in link7 if b.tag == "body" and b.get("name") == "hand"))
    actuators = [copy.deepcopy(a) for a in src.find("actuator")
                 if isinstance(a.tag, str) and a.get("joint", "").startswith("fr3_")]
    return dict(default=default, assets=assets, arm=arm, link7=link7, actuators=actuators)


def write_model(
    name: str,
    hand_root: ET.Element,
    hand_defaults: list[ET.Element],
    assets: list[ET.Element],
    actuators: list[ET.Element],
    *,
    equality: list[ET.Element] | None = None,
    contacts: list[ET.Element] | None = None,
    extra_excludes: tuple[tuple[str, str], ...] = (),
    n_hand_joints: int,
    home: str | None = None,
    hand_childclass: str | None = None,
    comment: str = "",
) -> pathlib.Path:
    """Merge ``hand_root`` (already posed on the flange) into the FR3 and write the combined MJCF.

    Args:
        name: model name, ``fr3_<name>``; the file goes to ``franka_fr3_<name>/fr3_<name>.xml``.
        hand_root: the end effector's root body, posed in the ``fr3_link7`` frame by
            :func:`mount_transform` / :func:`gripper_mount`.
        hand_defaults: the hand's default elements (a class per hand).
        assets: the hand's asset elements (mesh ``file``s relative to the written file).
        actuators: the hand's actuator elements (the FR3 arm's are appended in front of them).
        equality: equality constraints of the hand (coupled joints).
        contacts: contact exclusions of the hand.
        extra_excludes: additional ``(body1, body2)`` contact exclusions, e.g. the arm to the hand body.
        n_hand_joints: number of hand joints (for the ``home`` keyframe when ``home`` is not given).
        home: the ``home`` keyframe's qpos for the hand joints (defaults to all zeros).
        hand_childclass: the hand's default class, set as the root body's ``childclass``. The FR3 bodies carry
            ``childclass="fr3"``, so without this every hand body and joint would inherit the arm's class (its
            joints' ``frictionloss`` of 1.137 N m alone stalls the fingers).
        comment: a comment written at the top of the file (provenance).
    """
    arm = load_arm()
    root = ET.Element("mujoco", model=f"fr3_{name}")
    root.append(ET.Comment(f" Generated by scripts/build_fr3_{name}_mjcf.py; do not edit by hand. "))
    if comment:
        root.append(ET.Comment(f" {comment} "))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", integrator="implicitfast")
    default = arm["default"]
    for d in hand_defaults:
        default.append(copy.deepcopy(d))
    root.append(default)
    asset = ET.SubElement(root, "asset")
    for a in arm["assets"]:
        asset.append(copy.deepcopy(a))
    for a in assets:
        asset.append(copy.deepcopy(a))
    if hand_childclass:
        hand_root.set("childclass", hand_childclass)
    world = ET.SubElement(root, "worldbody")
    world.append(arm["arm"])
    arm["link7"].append(hand_root)

    act = ET.SubElement(root, "actuator")
    act.extend(copy.deepcopy(a) for a in arm["actuators"])
    act.extend(copy.deepcopy(a) for a in (actuators or []))
    if equality:
        eq = ET.SubElement(root, "equality")
        eq.extend(copy.deepcopy(e) for e in equality)
    con = ET.SubElement(root, "contact")
    ET.SubElement(con, "exclude", body1="fr3_link0", body2="fr3_link1")
    for b1, b2 in extra_excludes:
        ET.SubElement(con, "exclude", body1=b1, body2=b2)
    for e in (contacts or []):
        ET.SubElement(con, "exclude", body1=e[0], body2=e[1])
    hand_home = home if home is not None else " ".join(["0"] * n_hand_joints)
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home", qpos=f"{FR3_HOME} {hand_home}".strip())
    path = ROBOTS / f"franka_fr3_{name}" / f"fr3_{name}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def palm_frame(origin, x_palm, y_palm, z_palm) -> ET.Element:
    """The ``palm`` frame body (with a small ``palm_site`` marker)."""
    body = ET.Element("body", name="palm")
    T = np.eye(4)
    T[:3, :3] = np.stack([x_palm, y_palm, z_palm], axis=1)
    T[:3, 3] = origin
    set_pose(body, T)
    ET.SubElement(body, "site", name="palm_site", size="0.005", rgba="1 0.5 0 0.5", group="4")
    return body
