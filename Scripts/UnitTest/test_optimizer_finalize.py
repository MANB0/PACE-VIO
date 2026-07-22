from dataclasses import dataclass
from types import SimpleNamespace

from Module.Optimization.Interface import IOptimizer


@dataclass
class _Input:
    value: int


@dataclass
class _Output:
    value: int


class _FinalizeOptimizer(IOptimizer[_Input, dict, _Output]):
    @staticmethod
    def init_context(config) -> dict:
        return {"count": 0}

    @staticmethod
    def _optimize(context: dict, graph_data: _Input) -> tuple[dict, _Output]:
        context["count"] += int(graph_data.value)
        return context, _Output(context["count"])

    @staticmethod
    def _finalize_context(context: dict) -> tuple[dict, _Output]:
        return context, _Output(100 + context["count"])

    def write_graph_data(self, result: _Output | None, global_map: list[int]) -> None:
        if result is not None:
            global_map.append(result.value)


def test_finalize_protocol_sequential():
    optimizer = _FinalizeOptimizer(SimpleNamespace(parallel=False))
    outputs: list[int] = []
    optimizer.start_optimize(_Input(3))
    optimizer.write_map(outputs)
    final = optimizer.finalize_map(outputs)

    assert outputs == [3, 103]
    assert final is not None and final.value == 103


def test_finalize_protocol_parallel():
    optimizer = _FinalizeOptimizer(SimpleNamespace(parallel=True))
    outputs: list[int] = []
    try:
        optimizer.start_optimize(_Input(4))
        optimizer.write_map(outputs)
        final = optimizer.finalize_map(outputs)

        assert outputs == [4, 104]
        assert final is not None and final.value == 104
    finally:
        optimizer.terminate()
