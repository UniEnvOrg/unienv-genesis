"""A robot assembled at run time from parts mounted on each other, e.g. an arm and the hand on its flange.

Every combination of an arm and an end effector used to need its own merged description file. A
:class:`RobotAssemblyNode` instead loads each part from its own file (an arm, a hand, a gripper, ...) and mounts the
parts on each other with Genesis's ``RigidEntity.attach`` before the scene is built, so each arm and each end
effector is described once.

Each part stays a Genesis entity of its own, with its own DOFs, its own sub-controllers and its own reset keyframe;
Genesis simulates the attached parts as one kinematic tree, the DOFs of the parts following each other in the order
of :paramref:`RobotAssemblyNode.parts`. The node's controller is the union of the parts' sub-controllers
(:class:`AssemblyController`), so its action and observation spaces have the same nested layout as a
:class:`~unienv_genesis.nodes.entities.entity_controllers.CombinedController` over one merged entity.

A sub-controller may target a link of another part (:class:`PartLink`), e.g. an arm's end-effector pose controller
targeting the hand's palm frame. Genesis cannot solve IK through one entity for a link of another, so the node
resolves such a target to the nearest link of the controller's own part with the fixed offset between the two,
which requires the two links to be connected by fixed joints only (a hand rigidly mounted on the flange).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import genesis as gs
import torch
from genesis.engine import entities as gs_entities
from genesis.engine.entities.rigid_entity import RigidLink
from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
)

from unienv_interface.backends.pytorch import PyTorchArrayType
from unienv_genesis.options.attach_option import LinkAttachment
from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
from unienv_genesis.world.genesis_world import GenesisWorld
from .entity_controllers import CONTROL_MODE_MAP
from .entity_controllers.base import DiscreteEntityController, EntityController, resolve_joint_indexes
from .entity_controllers.combined_controller import CombinedController
from .robot_node import RobotDescriptionNode, RobotKeyframe

__all__ = [
    "PartFrame",
    "PartLink",
    "RobotPart",
    "PartMount",
    "AssemblyController",
    "RobotAssemblyNode",
]


@dataclass(frozen=True)
class PartFrame:
    """A frame fixed to a link of a part: ``(pos, quat)`` (``(w, x, y, z)``) in the link's frame."""
    link_name: str
    pos: Sequence[float] = (0.0, 0.0, 0.0)
    quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PartLink:
    """A link of a given part of the assembly, with an optional virtual offset (as :class:`LinkAttachment`).

    Used as a sub-controller's ``target_link`` (or ``base_frame``) to name a link of another part than the
    controller's own.
    """
    part: str
    link_name: str
    offset_pos: Optional[Sequence[float]] = None
    offset_quat: Optional[Sequence[float]] = None

    @classmethod
    def of_frame(cls, part: str, frame: PartFrame) -> "PartLink":
        return cls(part, frame.link_name, offset_pos=tuple(frame.pos), offset_quat=tuple(frame.quat))


@dataclass
class RobotPart:
    """One part of a robot assembly: its description file and what the robot does with it.

    Args:
        description_file: URDF / MJCF / USD file of the part. The file's root body must be the part's only tree
            and, for a mounted part, sit at the file's origin: Genesis loads a root body with a pose under an extra
            ``world`` link, and an attach would move only that link.
        sub_controllers: ``{name: config}`` of the part's sub-controllers, as for
            :class:`~unienv_genesis.nodes.entities.entity_controllers.CombinedController` (joint indexes / names
            are the part's own). Names are unique across the assembly.
        keyframe: the part's reset keyframe (its own joints; ``pos`` / ``quat`` only on the root part).
        frames: named frames of the part, e.g. an arm's ``"flange"`` (what end effectors are mounted on) or a
            hand's ``"eef"`` (what an arm's end-effector controller targets).
        morph_kwargs: extra keyword arguments of the part's Genesis morph.
        mount_pos, mount_quat: the pose of the part's base link in the frame it is mounted on (the parent's
            ``parent_frame``), e.g. a hand's pose on the flange. Identity by default.
        parent_overrides: ``{sub-controller name: {option: value}}`` this part asks of the sub-controllers of the
            part it is mounted on, e.g. the arm gains a gripper was tuned with. Applied by robot classes that
            assemble a given arm with any end effector (``FrankaFR3Assembly``), under their explicit options.
    """
    description_file: Union[str, os.PathLike]
    sub_controllers: Dict[str, Dict[str, Any]]
    keyframe: Optional[RobotKeyframe] = None
    frames: Dict[str, PartFrame] = field(default_factory=dict)
    morph_kwargs: Dict[str, Any] = field(default_factory=dict)
    mount_pos: Optional[Sequence[float]] = None
    mount_quat: Optional[Sequence[float]] = None
    parent_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class PartMount:
    """Mounts part ``child`` (at its ``mount_pos`` / ``mount_quat``) on the frame ``parent_frame`` of part ``parent``."""
    child: str
    parent: str
    parent_frame: str = "flange"


def _compose(pos_a, quat_a, pos_b, quat_b) -> Tuple[torch.Tensor, torch.Tensor]:
    """``T_a * T_b``."""
    return pos_a + quaternion_apply(quat_a, pos_b), quaternion_multiply(quat_a, quat_b)


def _invert(pos, quat) -> Tuple[torch.Tensor, torch.Tensor]:
    inv = quaternion_invert(quat)
    return -quaternion_apply(inv, pos), inv


def _as_pose(pos, quat) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = torch.zeros(3, dtype=torch.float64) if pos is None else torch.as_tensor(pos, dtype=torch.float64).reshape(3)
    quat = (torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64) if quat is None
            else torch.as_tensor(quat, dtype=torch.float64).reshape(4))
    return pos, quat / torch.linalg.norm(quat)


class AssemblyController(CombinedController):
    """The sub-controllers of every part of a :class:`RobotAssemblyNode`, each on its own part's entity.

    Same nested action / observation layout and methods as :class:`CombinedController`; ``entity`` is the root
    part's entity, and setting it (after a reload) re-binds every sub-controller to the current entity of its part.
    """

    controller_type = "assembly"

    def __init__(
        self,
        world: GenesisWorld,
        sub_controllers: Dict[str, EntityController],
        parts: Mapping[str, str],
        part_order: Sequence[str],
        entity_of: Callable[[str], gs_entities.RigidEntity],
    ) -> None:
        self._sub_controllers = dict(sub_controllers)
        self._parts = dict(parts)  # sub-controller name -> part name
        self._part_order = tuple(part_order)
        self._entity_of = entity_of
        observation_space, action_space = self._nested_spaces(world, self._sub_controllers)
        EntityController.__init__(self, world, entity_of(self._part_order[0]), observation_space, action_space)

    @property
    def entity(self) -> gs_entities.RigidEntity:
        return self._entity

    @entity.setter
    def entity(self, value: gs_entities.RigidEntity) -> None:
        self._entity = value
        for name, ctrl in self._sub_controllers.items():
            ctrl.entity = self._entity_of(self._parts[name])

    def part_of(self, name: str) -> str:
        """The part a sub-controller drives."""
        return self._parts[name]

    @property
    def max_joint_pos_vel(self) -> Optional[torch.Tensor]:
        """The position rate limits of all the assembly's DOFs, ``([batch_size], n_dofs)``, the parts' DOFs in
        assembly order (as they follow each other in the solver); ``inf`` where no sub-controller limits a DOF.

        ``None`` when a sub-controller is not a :class:`DiscreteEntityController` or has no limit.
        """
        batch_size = self.world.batch_size
        batched = batch_size is not None and batch_size > 0
        per_part = []
        for part in self._part_order:
            entity = self._entity_of(part)
            limit = torch.full(get_batch_aware_shape(batch_size, entity.n_dofs), float("inf"), device=self.world.device)
            for name, ctrl in self._sub_controllers.items():
                if self._parts[name] != part:
                    continue
                if not isinstance(ctrl, DiscreteEntityController):
                    return None
                v = ctrl.max_joint_pos_vel
                if v is None:
                    return None
                idx = ctrl.joint_indexes if ctrl.joint_indexes is not None else list(range(entity.n_dofs))
                if batched:
                    limit[:, idx] = v.to(limit.device)
                else:
                    limit[idx] = v.to(limit.device)
            per_part.append(limit)
        return torch.cat(per_part, dim=-1)


class RobotAssemblyNode(RobotDescriptionNode):
    """A robot assembled from :class:`RobotPart` s mounted on each other (see the module docstring).

    The root part (the first of :paramref:`parts`, the only one no :class:`PartMount` mounts) is placed at
    ``pos`` / ``quat`` and keeps the node's ``name`` as its entity name; every other part is the entity
    ``f"{name}_{part}"``, created right after its parent (Genesis requires attached entities to be created back to
    back, parents first, which is why one node creates all of them). The kinematics and planning helpers of
    :class:`RobotDescriptionNode` (``forward_kinematics``, ``solve_ik``, ``plan_path``) act on the root part.

    Args:
        world: The Genesis world.
        parts: ``{part name: part}``, parents before their children.
        mounts: how each non-root part is mounted.
        name: The robot's name (and its root part's entity name).
        pos, quat: The root part's base pose.
        sub_controllers: ``{name: config}`` replacing the parts' own sub-controllers; each config names its part
            under ``"part"``.
        fix_root: Whether the root part's base is fixed.
    """

    def __init__(
        self,
        world: GenesisWorld,
        parts: Mapping[str, RobotPart],
        mounts: Sequence[PartMount] = (),
        name: str = "robot",
        pos: Optional[tuple] = None,
        quat: Optional[tuple] = None,
        sub_controllers: Optional[Dict[str, Dict[str, Any]]] = None,
        control_timestep: Optional[float] = None,
        update_timestep: Optional[float] = None,
        fix_root: bool = True,
    ):
        self.parts: Dict[str, RobotPart] = dict(parts)
        self.mounts: Dict[str, PartMount] = {}
        for mount in mounts:
            if mount.child in self.mounts:
                raise ValueError(f"part {mount.child!r} is mounted twice")
            self.mounts[mount.child] = mount
        order = list(self.parts)
        if not order:
            raise ValueError("a robot assembly needs at least one part")
        for child, mount in self.mounts.items():
            if child not in self.parts or mount.parent not in self.parts:
                raise ValueError(f"mount {mount} names a part that is not in {order}")
            if order.index(mount.parent) > order.index(child):
                raise ValueError(f"part {mount.parent!r} must come before the part {child!r} mounted on it")
            if mount.parent_frame not in self.parts[mount.parent].frames:
                raise ValueError(
                    f"part {mount.parent!r} has no frame {mount.parent_frame!r} (frames: "
                    f"{sorted(self.parts[mount.parent].frames)})"
                )
        roots = [p for p in order if p not in self.mounts]
        if roots != order[:1]:
            raise ValueError(f"exactly one part, the first, must be unmounted (unmounted parts: {roots})")
        self.root_part = order[0]

        if sub_controllers is None:
            sub_controllers = {}
            for part_name, part in self.parts.items():
                for key, cfg in part.sub_controllers.items():
                    if key in sub_controllers:
                        raise ValueError(f"sub-controller name {key!r} is used by two parts")
                    sub_controllers[key] = {**cfg, "part": part_name}
        for key, cfg in sub_controllers.items():
            if cfg.get("part") not in self.parts:
                raise ValueError(f"sub-controller {key!r} must name its part under 'part' (one of {order})")
        self.sub_controller_configs = sub_controllers
        self.part_entities: Dict[str, gs_entities.RigidEntity] = {}
        self._resolved_targets: Dict[Tuple[str, str], LinkAttachment] = {}

        super().__init__(
            world,
            description_file=self.parts[self.root_part].description_file,
            name=name,
            control_timestep=control_timestep,
            update_timestep=update_timestep,
            fix_root=fix_root,
            pos=pos,
            quat=quat,
            controller_type="combined",  # unused: init_controller builds an AssemblyController
        )

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def entity_name(self, part: str) -> str:
        return self._name if part == self.root_part else f"{self._name}_{part}"

    def _part_morph(self, part_name: str):
        part = self.parts[part_name]
        file = str(part.description_file)
        kwargs: Dict[str, Any] = dict(part.morph_kwargs)
        if part_name == self.root_part:
            fixed = self.fix_root
            if self.pos is not None:
                kwargs["pos"] = self.pos
            if self.quat is not None:
                kwargs["quat"] = self.quat
        else:
            # A part loaded fixed to the world and then attached to a moving link must batch its fixed vertices.
            fixed = True
            kwargs.setdefault("batch_fixed_verts", True)
        base = os.path.basename(file)
        if base.endswith(".urdf"):
            return gs.morphs.URDF(file=file, fixed=fixed, **kwargs)
        if base.endswith(".xml"):
            return gs.morphs.MJCF(file=file, **kwargs)
        if base.endswith((".usd", ".usda", ".usdc")):
            return gs.morphs.USD(file=file, fixed=fixed, requires_jac_and_IK=True, **kwargs)
        raise ValueError(f"Unsupported robot description file format: {file}")

    def reload(self, *, priority: int = 0, seed=None, mask=None, **kwargs) -> None:
        if priority != 50:
            return
        self.part_entities = {}
        for part_name in self.parts:
            entity = self.world.scene.add_entity(self._part_morph(part_name), name=self.entity_name(part_name))
            mount = self.mounts.get(part_name)
            if mount is not None:
                part = self.parts[part_name]
                frame = self.parts[mount.parent].frames[mount.parent_frame]
                parent = self.part_entities[mount.parent]
                base = entity.links[0]
                pos, quat = _compose(*_as_pose(frame.pos, frame.quat), *_as_pose(part.mount_pos, part.mount_quat))
                pos, quat = _compose(pos, quat, *_as_pose(base.desc.pos, base.desc.quat))
                entity.attach(parent, frame.link_name, pos=pos.numpy(), quat=quat.numpy())
                # Every link must now be in the parent's kinematic tree: links left in a tree of their own follow the
                # parent kinematically but add no inertia to it and skip its self-collision filters.
                root = parent.get_link(frame.link_name).root_idx
                stray = [link.name for link in entity.links if link.root_idx != root]
                if stray:
                    raise ValueError(
                        f"part {part_name!r} ({part.description_file}): links {stray} stay out of the parent's "
                        f"kinematic tree after the attach. The file must hold a single tree whose root body sits at "
                        f"the origin; give its pose on the parent frame as RobotPart.mount_pos / mount_quat."
                    )
            self.part_entities[part_name] = entity
        if len(self.part_entities) > 1:
            # Genesis applies the contype / conaffinity bitmasks of an MJCF / USD entity to its own geom pairs only, so
            # the masks that keep a merged model's hand off its arm (e.g. the Wuji fingers' contype 2 against the FR3's
            # 1) would not act between two parts. Global masks act between the parts as within one entity; against
            # other entities they change nothing for the default masks (0xFFFF) that primitives and planes carry, nor
            # against MJCF / USD entities, whose masks stay local.
            for entity in self.part_entities.values():
                if getattr(entity, "_is_local_collision_mask", False):
                    entity._is_local_collision_mask = False
        self.entity = self.part_entities[self.root_part]

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def _resolve_link(self, own_part: str, target: PartLink) -> LinkAttachment:
        """A :class:`PartLink` as a :class:`LinkAttachment` on ``own_part``'s entity (see the module docstring)."""
        if target.part not in self.parts:
            raise ValueError(f"{target} names an unknown part (parts: {list(self.parts)})")
        if target.part == own_part:
            return LinkAttachment(target.link_name, offset_pos=target.offset_pos, offset_quat=target.offset_quat)
        key = (own_part, repr(target))
        if key not in self._resolved_targets:
            own = self.part_entities[own_part]
            link: RigidLink = self.part_entities[target.part].get_link(target.link_name)
            anchor = link
            while anchor.entity is not own:
                if anchor.n_dofs > 0:
                    raise ValueError(
                        f"{target}: the link is not rigidly mounted on part {own_part!r} (link "
                        f"{anchor.name!r} of part {target.part!r} has a joint)"
                    )
                if anchor.parent_idx < 0:
                    raise ValueError(f"{target}: the link is not mounted on part {own_part!r}")
                anchor = anchor._solver.links[anchor.parent_idx]

            def pose(l: RigidLink):
                p, q = l.get_pos(), l.get_quat()
                if p.dim() == 2:
                    p, q = p[0], q[0]
                return p.detach().double().cpu(), q.detach().double().cpu()

            offset = _compose(*_invert(*pose(anchor)), *pose(link))
            offset = _compose(*offset, *_as_pose(target.offset_pos, target.offset_quat))
            self._resolved_targets[key] = LinkAttachment(
                anchor.name, offset_pos=offset[0].tolist(), offset_quat=offset[1].tolist(),
            )
        return self._resolved_targets[key]

    def init_controller(self) -> AssemblyController:
        controllers, parts = {}, {}
        for key, cfg in self.sub_controller_configs.items():
            cfg = dict(cfg)
            part = cfg.pop("part")
            ctrl_class = CONTROL_MODE_MAP[cfg.pop("controller_type")]
            for attr in ("target_link", "base_frame"):
                if isinstance(cfg.get(attr), PartLink):
                    cfg[attr] = self._resolve_link(part, cfg[attr])
            controllers[key] = ctrl_class(world=self.world, entity=self.part_entities[part], **cfg)
            parts[key] = part
        return AssemblyController(
            self.world, controllers, parts, list(self.parts), lambda part: self.part_entities[part],
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _apply_rest(self, mask: Optional[torch.Tensor] = None) -> None:
        """Apply each part's reset keyframe to its entity (``reset_joint_position_noise``: ``{part: half-widths}``)."""
        noise = self.reset_joint_position_noise or {}
        for part_name, part in self.parts.items():
            kf = part.keyframe
            entity = self.part_entities.get(part_name)
            if kf is None or entity is None:
                continue
            if part_name == self.root_part:
                if kf.pos is not None:
                    entity.set_pos(kf.pos, envs_idx=mask)
                if kf.quat is not None:
                    entity.set_quat(kf.quat, envs_idx=mask)
            dofs_idx = resolve_joint_indexes(entity, joint_names=kf.joint_names)
            if kf.joint_position is not None:
                entity.set_dofs_position(
                    self._reset_joint_position(kf.joint_position, mask, noise.get(part_name)),
                    dofs_idx_local=dofs_idx,
                    envs_idx=mask,
                )
            if kf.joint_velocity is not None:
                entity.set_dofs_velocity(kf.joint_velocity, dofs_idx_local=dofs_idx, envs_idx=mask)

    def _reset_joint_position(self, joint_position, mask, noise=None):
        if noise is None:
            return joint_position
        return super()._reset_joint_position(joint_position, mask, noise)

    def _get_controller_max_joint_pos_vel(self) -> Optional[torch.Tensor]:
        """The root part's rate limits (``plan_path`` plans for the root part)."""
        v = super()._get_controller_max_joint_pos_vel()
        return None if v is None else v[: self.entity.n_dofs]
