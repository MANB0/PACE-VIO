from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch


def visual_input_sha256(fields: Mapping[str, torch.Tensor]) -> str:
    """Return a deterministic fingerprint for visual observation tensors."""
    digest = hashlib.sha256()
    for name in sorted(fields):
        tensor = torch.as_tensor(fields[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()
