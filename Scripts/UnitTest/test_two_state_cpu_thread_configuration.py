from types import SimpleNamespace

import torch

from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO


def test_two_state_cpu_thread_configuration_is_explicit_and_effective() -> None:
    original = torch.get_num_threads()
    config = SimpleNamespace(
        imu_factor_mode="two_state_fixed_lag",
        graph_type="disp",
        autodiff=True,
        parallel=False,
        vectorize=True,
        device="cpu",
        two_state_cpu_threads=2,
    )
    try:
        context = TwoFrame_PGO.init_context(config)
        assert context["two_state_cpu_threads"] == 2
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(original)


def test_non_two_state_mode_does_not_change_cpu_threads() -> None:
    original = torch.get_num_threads()
    config = SimpleNamespace(
        imu_factor_mode="legacy_pose_prior",
        graph_type="disp",
        autodiff=True,
        parallel=False,
        vectorize=True,
        device="cpu",
        two_state_cpu_threads=max(1, original - 1),
    )
    context = TwoFrame_PGO.init_context(config)
    assert context["two_state_cpu_threads"] == max(1, original - 1)
    assert torch.get_num_threads() == original
