"""Shared grasp-detection utilities for gripper robot definitions."""

import genesis as gs
import torch


def make_is_grasping(pad_link_names):
    """Create a grasp-detection callable checking contacts on specific pad links.

    The returned callable resolves and validates the configured pad links on
    the robot entity each time it is called (no identity caching), then
    reports whether any contact involves one of the pad links.  It works with
    both unbatched ``(n_contacts,)`` and batched ``(B, n_contacts)`` contact
    layouts; in the batched case the per-env ``valid_mask`` is honoured.
    """
    pad_link_names = tuple(pad_link_names)

    def is_grasping(robot_entity, target_entity):
        # Resolve pad link global indices directly, validating that every
        # configured pad link exists on this entity.
        resolved = {
            link.name: link.idx
            for link in robot_entity.links
            if link.name in pad_link_names
        }
        missing = [name for name in pad_link_names if name not in resolved]
        if missing:
            available = [link.name for link in robot_entity.links]
            raise ValueError(
                f"Grasp detection misconfigured for entity "
                f"'{getattr(robot_entity, 'name', '?')}': pad link(s) "
                f"{missing} not found. Available links: {available}"
            )
        pad_indices = [resolved[name] for name in pad_link_names]

        contacts = robot_entity.get_contacts(with_entity=target_entity)
        if contacts is None:
            # genesis-world >= 1.5: the contacts are unknown after a build, a reset or a state write until the
            # next step finds them anew; report no grasp until then.
            n_envs = robot_entity._solver.n_envs
            return torch.zeros((n_envs,) if n_envs > 0 else (), dtype=torch.bool, device=gs.device)

        # Check if any contact involves a pad link on the robot side.  The
        # comparison tensor is created on the contact tensor's device so CUDA
        # contact tensors never meet a CPU index tensor.
        link_a = contacts["link_a"]  # (n_contacts,) or (B, n_contacts)
        link_b = contacts["link_b"]
        pad_idx_t = torch.tensor(pad_indices, dtype=torch.long, device=link_a.device)
        is_pad = (
            (link_a.unsqueeze(-1) == pad_idx_t).any(dim=-1)
            | (link_b.unsqueeze(-1) == pad_idx_t).any(dim=-1)
        )  # (n_contacts,) or (B, n_contacts)

        if "valid_mask" not in contacts:
            # Unbatched: (n_contacts,) entries, already filtered — is_pad is (n_contacts,). (``force_a`` cannot tell:
            # genesis-world >= 1.5 sets it to None after a detection without a step.)
            return is_pad.any()
        # Batched: valid_mask is (B, n_contacts)
        return (contacts["valid_mask"] & is_pad).any(dim=-1)

    return is_grasping
