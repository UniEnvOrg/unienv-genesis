"""Space metadata shared by the Genesis nodes and controllers.

Builds the ``metadata`` of the spaces the adaptor declares, using the unienv
vocabulary (``unienv_interface.space.space_utils.metadata_utils``). The
helpers here cover what several modules describe the same way: per-DOF
units and names of a rigid entity's joints, and end-effector / root poses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import genesis as gs

from unienv_interface.space.space_utils.metadata_utils import (
    action_metadata,
    metadata,
    proprio_metadata,
)

__all__ = [
    "DOF_QUANTITY_UNITS",
    "dof_descriptions",
    "dof_names",
    "dof_units",
    "joint_proprio_metadata",
    "joint_action_metadata",
    "position_metadata",
    "quaternion_metadata",
    "pose_frame",
    "frame_offset",
    "offset_extras",
    "eef_pose_extras",
    "base_frame_extras",
    "attach_extras",
    "XYZ_NAMES",
    "QUAT_WXYZ_NAMES",
]

XYZ_NAMES = ["x", "y", "z"]
QUAT_WXYZ_NAMES = ["qw", "qx", "qy", "qz"]

DOF_QUANTITY_UNITS: Dict[str, Tuple[str, str]] = {
    "position": ("m", "rad"),
    "velocity": ("m/s", "rad/s"),
    "force": ("N", "N*m"),
}
"""``quantity -> (unit of a translational DOF, unit of a rotational DOF)``."""

# Per joint type: (name suffix, translational?) for each of its DOFs.
_JOINT_DOFS = {
    gs.JOINT_TYPE.REVOLUTE: [("", False)],
    gs.JOINT_TYPE.PRISMATIC: [("", True)],
    gs.JOINT_TYPE.SPHERICAL: [("rx", False), ("ry", False), ("rz", False)],
    gs.JOINT_TYPE.FREE: [
        ("x", True), ("y", True), ("z", True),
        ("rx", False), ("ry", False), ("rz", False),
    ],
}


def dof_descriptions(
    entity: Any,
    dofs_idx_local: Optional[Sequence[int]] = None,
) -> List[Tuple[str, bool]]:
    """``(name, translational)`` of each DOF of a rigid entity.

    DOFs are listed in the order of ``dofs_idx_local`` (all DOFs of the
    entity when ``None``), the order the entity's ``get_dofs_*`` methods
    return them in. A single-DOF joint names its DOF after the joint; the DOFs
    of a free / spherical joint get ``_x`` .. ``_rz`` suffixes.
    """
    by_dof: Dict[int, Tuple[str, bool]] = {}
    for joint in entity.joints:
        idxs = [int(i) for i in joint.dofs_idx_local]
        if not idxs:
            continue
        layout = _JOINT_DOFS.get(joint.type)
        if layout is None or len(layout) != len(idxs):
            layout = [(str(i), False) for i in range(len(idxs))]
        for idx, (suffix, translational) in zip(idxs, layout):
            name = joint.name if len(idxs) == 1 else f"{joint.name}_{suffix}"
            by_dof[idx] = (name, translational)
    order = range(entity.n_dofs) if dofs_idx_local is None else dofs_idx_local
    return [by_dof[int(i)] for i in order]


def dof_names(entity: Any, dofs_idx_local: Optional[Sequence[int]] = None) -> List[str]:
    """Names of the selected DOFs (see :func:`dof_descriptions`)."""
    return [name for name, _ in dof_descriptions(entity, dofs_idx_local)]


def dof_units(
    entity: Any,
    dofs_idx_local: Optional[Sequence[int]] = None,
    quantity: str = "position",
) -> List[str]:
    """Per-DOF units of ``quantity`` (``"position"``, ``"velocity"`` or ``"force"``).

    Translational (prismatic / free-joint linear) DOFs are in ``m``, ``m/s``,
    ``N``; rotational (revolute / spherical / free-joint angular) DOFs in
    ``rad``, ``rad/s``, ``N*m``.
    """
    translational_unit, rotational_unit = DOF_QUANTITY_UNITS[quantity]
    return [
        translational_unit if translational else rotational_unit
        for _, translational in dof_descriptions(entity, dofs_idx_local)
    ]


def joint_proprio_metadata(
    entity: Any,
    dofs_idx_local: Optional[Sequence[int]],
    quantity: str,
    **extra: Any,
) -> Dict[str, Any]:
    """``kind="proprio"`` metadata of a per-DOF joint reading."""
    return proprio_metadata(
        units=dof_units(entity, dofs_idx_local, quantity),
        names=dof_names(entity, dofs_idx_local),
        **extra,
    )


def joint_action_metadata(
    entity: Any,
    dofs_idx_local: Optional[Sequence[int]],
    control: str,
    **extra: Any,
) -> Dict[str, Any]:
    """``kind="action"`` metadata of absolute per-DOF joint targets.

    ``control`` is ``"joint_position"`` or ``"joint_velocity"``.
    """
    quantity = {"joint_position": "position", "joint_velocity": "velocity"}[control]
    return action_metadata(
        control=control,
        mode="absolute",
        units=dof_units(entity, dofs_idx_local, quantity),
        names=dof_names(entity, dofs_idx_local),
        **extra,
    )


def pose_frame(has_base_frame: bool) -> str:
    """The ``frame`` of a pose expressed in a configured base frame, else the world.

    A ``"base"`` pose also records which base (:func:`base_frame_extras`).
    """
    return "base" if has_base_frame else "world"


def frame_offset(
    offset_pos: Optional[Sequence[float]],
    offset_quat: Optional[Sequence[float]] = None,
) -> Optional[Tuple[List[float], List[float]]]:
    """The virtual-frame offset a pose reader applies, as JSON lists, or ``None``.

    Mirrors the controllers and the end-effector state sensor: an offset is
    applied only when ``offset_pos`` is set, with ``offset_quat`` defaulting
    to the identity (an ``offset_quat`` without ``offset_pos`` is ignored by
    them, so it is here too). Returns ``(pos, quat_wxyz)``, or ``None`` when
    no offset is applied or it is the identity.
    """
    if offset_pos is None:
        return None
    pos = np.asarray(_as_numpy(offset_pos), dtype=np.float64).reshape(-1)
    quat = (
        np.array([1.0, 0.0, 0.0, 0.0])
        if offset_quat is None
        else np.asarray(_as_numpy(offset_quat), dtype=np.float64).reshape(-1)
    )
    identity = np.allclose(pos, 0.0, rtol=0.0, atol=1e-12) and (
        np.allclose(quat, [1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
        or np.allclose(quat, [-1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
    )
    if identity:
        return None
    return [float(v) for v in pos], [float(v) for v in quat]


def _as_numpy(value: Any) -> Any:
    """``value`` as something ``np.asarray`` takes (torch tensors moved to the CPU)."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def offset_extras(
    prefix: str,
    offset_pos: Optional[Sequence[float]],
    offset_quat: Optional[Sequence[float]] = None,
) -> Dict[str, List[float]]:
    """``<prefix>_offset_pos`` (m) / ``<prefix>_offset_quat`` (``w, x, y, z``) of a non-identity offset.

    The offset places a frame relative to the link it is attached to, in that
    link's frame: ``p = link_pos + link_quat * offset_pos``, ``q = link_quat *
    offset_quat``. Empty when there is no offset (see :func:`frame_offset`).
    """
    offset = frame_offset(offset_pos, offset_quat)
    if offset is None:
        return {}
    pos, quat = offset
    return {f"{prefix}_offset_pos": pos, f"{prefix}_offset_quat": quat}


def eef_pose_extras(
    link: Optional[str],
    offset_pos: Optional[Sequence[float]] = None,
    offset_quat: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """The ``link`` an EEF pose is read from, plus its virtual EEF offset.

    An EEF configured with an offset (e.g. a fingertip 0.11 m from the hand
    link) reports the pose of that offset frame, not of ``link``:
    ``eef_offset_pos`` / ``eef_offset_quat`` give it in ``link``'s frame
    (omitted without an offset).
    """
    extra: Dict[str, Any] = {}
    if isinstance(link, str):
        extra["link"] = link
    extra.update(offset_extras("eef", offset_pos, offset_quat))
    return extra


def base_frame_extras(
    base_link: Optional[str],
    offset_pos: Optional[Sequence[float]] = None,
    offset_quat: Optional[Sequence[float]] = None,
    base_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """What ``frame="base"`` refers to: the ``base_link`` (and ``base_entity``) plus its offset.

    With a base-frame offset, poses are relative to the offset frame
    ``base_offset_pos`` / ``base_offset_quat`` in ``base_link``'s frame.
    """
    extra: Dict[str, Any] = {}
    if isinstance(base_entity, str):
        extra["base_entity"] = base_entity
    if isinstance(base_link, str):
        extra["base_link"] = base_link
    extra.update(offset_extras("base", offset_pos, offset_quat))
    return extra


def position_metadata(kind: str = "proprio", *, frame: str, **extra: Any) -> Dict[str, Any]:
    """Metadata of a 3D position in metres (``names`` x, y, z)."""
    return metadata(kind, units="m", frame=frame, names=XYZ_NAMES, **extra)


def quaternion_metadata(kind: str = "proprio", *, frame: str, **extra: Any) -> Dict[str, Any]:
    """Metadata of a ``(w, x, y, z)`` orientation quaternion."""
    return metadata(kind, rotation="quat_wxyz", frame=frame, names=QUAT_WXYZ_NAMES, **extra)


def attach_extras(name: str, attach: Any) -> Dict[str, str]:
    """``source`` (the node name) plus the ``entity`` / ``link`` an attachment provider names.

    Providers that resolve their target otherwise (no ``entity_name`` /
    ``link_name`` string attributes) contribute nothing.
    """
    extra = {"source": name}
    for key, attr in (("entity", "entity_name"), ("link", "link_name")):
        value = getattr(attach, attr, None)
        if isinstance(value, str):
            extra[key] = value
    return extra
