from __future__ import annotations

import pypose as pp
import torch


DTYPE = torch.float64
EPSILON = 1.0e-6


def _translation_accessor(tangent: torch.Tensor) -> torch.Tensor:
    pose = pp.se3(tangent.reshape(1, 6)).Exp()
    return pose.translation().reshape(3)


def _translation_by_act(tangent: torch.Tensor) -> torch.Tensor:
    pose = pp.se3(tangent.reshape(1, 6)).Exp()
    origin = torch.zeros(3, dtype=tangent.dtype, device=tangent.device)
    return pose.Act(origin).reshape(3)


def _central_difference(function, tangent: torch.Tensor) -> torch.Tensor:
    columns = []
    for column in range(6):
        delta = torch.zeros_like(tangent)
        delta[column] = EPSILON
        plus = function(tangent + delta)
        minus = function(tangent - delta)
        columns.append((plus - minus) / (2.0 * EPSILON))
    return torch.stack(columns, dim=1)


def test_exp_translation_accessor_and_act_origin_jacobians(capsys) -> None:
    tangent = torch.tensor(
        [0.31, -0.27, 0.18, 0.42, -0.33, 0.29],
        dtype=DTYPE,
        requires_grad=True,
    )
    assert bool((tangent[:3].abs() > 0.0).all())
    assert bool((tangent[3:].abs() > 0.0).all())

    accessor_value = _translation_accessor(tangent)
    act_value = _translation_by_act(tangent)
    accessor_autodiff = torch.autograd.functional.jacobian(
        _translation_accessor, tangent, vectorize=False
    )
    act_autodiff = torch.autograd.functional.jacobian(
        _translation_by_act, tangent, vectorize=False
    )
    accessor_fd = _central_difference(_translation_accessor, tangent.detach())
    act_fd = _central_difference(_translation_by_act, tangent.detach())

    accessor_rotation_error = (
        accessor_autodiff[:, 3:6] - accessor_fd[:, 3:6]
    ).abs().max()
    act_rotation_error = (act_autodiff[:, 3:6] - act_fd[:, 3:6]).abs().max()
    forward_error = (accessor_value - act_value).abs().max()

    print(f"forward_max_abs_difference={float(forward_error):.12e}")
    print(
        "translation_accessor_rotation_columns_max_abs_error="
        f"{float(accessor_rotation_error):.12e}"
    )
    print(
        "act_origin_rotation_columns_max_abs_error="
        f"{float(act_rotation_error):.12e}"
    )
    print("translation_accessor_rotation_columns_autodiff=")
    print(accessor_autodiff[:, 3:6])
    print("translation_accessor_rotation_columns_central_fd=")
    print(accessor_fd[:, 3:6])
    print("act_origin_rotation_columns_autodiff=")
    print(act_autodiff[:, 3:6])
    print("act_origin_rotation_columns_central_fd=")
    print(act_fd[:, 3:6])

    assert torch.allclose(accessor_value, act_value, atol=1.0e-12, rtol=1.0e-12)
    assert torch.isfinite(accessor_autodiff).all()
    assert torch.isfinite(act_autodiff).all()
    assert torch.isfinite(accessor_fd).all()
    assert torch.isfinite(act_fd).all()
    assert float(act_rotation_error) <= 1.0e-8

    captured = capsys.readouterr()
    print(captured.out, end="")
