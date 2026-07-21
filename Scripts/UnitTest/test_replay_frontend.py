from types import SimpleNamespace

import pytest
import torch

from Module.Frontend.Frontend import IFrontend, FrontendReplayViolation, ReplayFrontend


def test_replay_frontend_allocates_no_model_and_rejects_inference():
    config = SimpleNamespace()
    frontend = ReplayFrontend(config)

    assert frontend.provide_cov == (True, True)
    assert not hasattr(frontend, "model")
    assert vars(frontend) == {"config": config}
    assert not any(isinstance(value, torch.nn.Module) for value in vars(frontend).values())

    with pytest.raises(FrontendReplayViolation):
        frontend.estimate_pair(None, None)


@pytest.mark.parametrize(
    "entrypoint, args",
    [
        ("estimate_depth", (None,)),
        ("estimate_pair", (None, None)),
        ("estimate_triplet", (None, None)),
    ],
)
def test_replay_frontend_rejects_every_inference_entrypoint(entrypoint, args):
    frontend = ReplayFrontend(SimpleNamespace())

    with pytest.raises(FrontendReplayViolation):
        getattr(frontend, entrypoint)(*args)


def test_replay_frontend_is_registered_and_validates_empty_config():
    assert IFrontend.get_class("ReplayFrontend") is ReplayFrontend
    assert isinstance(IFrontend.instantiate("ReplayFrontend", SimpleNamespace()), ReplayFrontend)

    ReplayFrontend.is_valid_config(SimpleNamespace())
    with pytest.raises(KeyError):
        ReplayFrontend.is_valid_config(SimpleNamespace(unexpected=True))
