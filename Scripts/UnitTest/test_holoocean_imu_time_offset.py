import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from DataLoader.Dataset.GeneralStereoIMU import GeneralStereoIMUSequence, IMUCSVLoader

_PREINT_SPEC = importlib.util.spec_from_file_location(
    "imu_preintegration_under_test_for_time_offset",
    Path("Module/IMUPreintegration.py"),
)
_PREINT_MODULE = importlib.util.module_from_spec(_PREINT_SPEC)
assert _PREINT_SPEC is not None and _PREINT_SPEC.loader is not None
_PREINT_SPEC.loader.exec_module(_PREINT_MODULE)
preintegrate_imu = _PREINT_MODULE.preintegrate_imu


def test_imu_loader_interpolates_range_boundaries_to_cover_camera_interval(tmp_path: Path):
    csv_path = tmp_path / "imu_data.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z",
                "0,0,0,0,0,0,-9.8",
                "10000000,0,0,0.1,0,0,-9.7",
                "20000000,0,0,0.2,0,0,-9.6",
                "30000000,0,0,0.3,0,0,-9.5",
                "40000000,0,0,0.4,0,0,-9.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loader = IMUCSVLoader(csv_path)

    start_ns = 3_333_333
    end_ns = 36_666_666
    time_ns, acc, gyro = loader.query_range(start_ns, end_ns)

    assert int(time_ns[0].item()) == start_ns
    assert int(time_ns[-1].item()) == end_ns
    assert time_ns.tolist() == [start_ns, 10_000_000, 20_000_000, 30_000_000, end_ns]
    assert gyro[0, 2].item() == pytest.approx(0.03333333)
    assert gyro[-1, 2].item() == pytest.approx(0.36666666)

    preint = preintegrate_imu(
        time_ns=time_ns,
        acc=acc,
        gyro=gyro,
        R0_world=None,
        gravity=9.8,
    )
    assert preint.dt_total == pytest.approx((end_ns - start_ns) * 1e-9)


def test_metadata_time_offset_is_used_when_auto_estimate_is_disabled():
    scene_root = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/clear_shallow")
    if not scene_root.exists():
        pytest.skip(f"HoloOcean test scene is unavailable: {scene_root}")

    seq = GeneralStereoIMUSequence(SimpleNamespace(
        root=str(scene_root),
        format="png",
        bl=0.225,
        gravity=9.8,
        imu_window_ns=100_000_000,
        imu_fallback_max_dt_ns=-1,
        auto_estimate_time_offset=False,
        imu_time_offset_ns=0,
        pose_output_frame="NWU",
    ))

    assert seq.imu_time_offset_ns == 0
    assert seq.imu_time_offset_source == "metadata.time_synchronization.camera_imu_time_offset_ns"


def test_metadata_time_offset_overrides_endpoint_auto_estimate():
    scene_root = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/clear_shallow")
    if not scene_root.exists():
        pytest.skip(f"HoloOcean test scene is unavailable: {scene_root}")

    seq = GeneralStereoIMUSequence(SimpleNamespace(
        root=str(scene_root),
        format="png",
        bl=0.225,
        gravity=9.8,
        imu_window_ns=100_000_000,
        imu_fallback_max_dt_ns=-1,
        auto_estimate_time_offset=True,
        imu_time_offset_ns=0,
        pose_output_frame="NWU",
    ))

    assert seq.imu_time_offset_ns == 0
    assert seq.imu_time_offset_source == "metadata.time_synchronization.camera_imu_time_offset_ns"
