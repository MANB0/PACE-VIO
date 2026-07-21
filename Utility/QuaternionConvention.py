from __future__ import annotations

from collections.abc import Sequence

import pypose as pp
import torch


def xyzw_quaternion_to_so3(quaternion_xyzw: Sequence[float] | torch.Tensor) -> pp.LieTensor:
    """Convert an xyzw quaternion to PyPose SO3.

    PyPose uses the same xyzw storage order for SO3 tensors. HoloOcean
    ref_pose.csv also stores qx,qy,qz,qw, so no w-first reordering is applied.
    """
    q = torch.as_tensor(quaternion_xyzw, dtype=torch.float32).reshape(-1, 4)
    norm = q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    q = q / norm
    return pp.SO3(q)


def nwu_xyzw_quaternion_to_internal_so3(
    quaternion_xyzw: Sequence[float] | torch.Tensor,
    *,
    internal_world_frame: str,
) -> pp.LieTensor:
    """Convert a body-NWU to world-NWU xyzw quaternion into the optimizer frame."""
    rotation_nwu = xyzw_quaternion_to_so3(quaternion_xyzw)
    frame = str(internal_world_frame).strip().upper()
    if "NED" not in frame:
        return rotation_nwu

    nwu_to_ned = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
    rot_mat = rotation_nwu.matrix().float().reshape(-1, 3, 3)
    converted = nwu_to_ned.reshape(1, 3, 3) @ rot_mat @ nwu_to_ned.reshape(1, 3, 3)
    return pp.from_matrix(converted, pp.SO3_type).float()
