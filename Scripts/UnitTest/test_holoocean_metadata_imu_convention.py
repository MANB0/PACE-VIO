import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch

from DataLoader.Dataset import GeneralStereoIMU as general_imu_module
from DataLoader.Dataset.GeneralStereoIMU import GeneralStereoIMUSequence
from Utility.IMUKinematics import (
    gravity_for_world_frame,
    imu_bias_sigma_to_continuous_random_walk_density,
    imu_sigma_to_continuous_density,
    vio_preintegrated_imu_residual,
)


HOLOOCEAN_CLEAR = Path("/mnt/e/文档/holoocean/code/recordings/batch_20260515_155653/clear_shallow")
HOLOOCEAN_ZED_RECT = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_zed100_closed_paths_smooth_20260705/normal_noise/clear_rectangle_path"
)
HOLOOCEAN_STATIC63_CIRCLE_NORMAL = Path(
    "/mnt/e/文档/holoocean/code/recordings/"
    "batch_clear_truth_paths_20260713_static63_variants/clear_circle_truth_normal_noise"
)
_PREINT_SPEC = importlib.util.spec_from_file_location(
    "imu_preintegration_under_test",
    Path("Module/IMUPreintegration.py"),
)
_PREINT_MODULE = importlib.util.module_from_spec(_PREINT_SPEC)
assert _PREINT_SPEC is not None and _PREINT_SPEC.loader is not None
_PREINT_SPEC.loader.exec_module(_PREINT_MODULE)
preintegrate_imu = _PREINT_MODULE.preintegrate_imu


def _holoocean_cfg(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        root=str(root),
        format="png",
        bl=0.225,
        gravity=9.8,
        imu_window_ns=100_000_000,
        imu_fallback_max_dt_ns=-1,
        auto_estimate_time_offset=True,
        imu_time_offset_ns=0,
        pose_output_frame="NWU",
        camera={},
    )


@pytest.mark.skipif(not HOLOOCEAN_CLEAR.exists(), reason="HoloOcean recording folder is unavailable")
def test_holoocean_metadata_converts_source_flu_to_macvo_internal_ned():
    seq = GeneralStereoIMUSequence(_holoocean_cfg(HOLOOCEAN_CLEAR))
    frame = seq[1]

    rotation = frame.imu.T_BS.rotation().matrix().reshape(3, 3).float()
    flu_to_ned = torch.diag(torch.tensor([1.0, -1.0, -1.0]))

    assert torch.allclose(rotation, flu_to_ned, atol=1e-6)
    assert getattr(frame, "imu_source_world_frame") == "NWU"
    assert getattr(frame, "imu_source_measurement_frame") == "FLU"
    assert getattr(frame, "imu_internal_world_frame") == "NED"
    assert getattr(frame, "imu_internal_measurement_frame") == "NED"
    assert getattr(frame, "imu_world_frame") == "NED"
    assert getattr(frame, "imu_measurement_frame") == "FLU"
    assert getattr(frame, "imu_time_offset_ns") == 0
    assert getattr(frame, "imu_time_offset_source") == "metadata.time_synchronization.camera_imu_time_offset_ns"


@pytest.mark.skipif(not HOLOOCEAN_CLEAR.exists(), reason="HoloOcean recording folder is unavailable")
def test_holoocean_metadata_imu_tbs_translation_uses_body_to_imu_in_internal_frame():
    seq = GeneralStereoIMUSequence(_holoocean_cfg(HOLOOCEAN_CLEAR))
    frame = seq[1]

    translation = frame.imu.T_BS.translation().reshape(3).float()

    assert torch.allclose(
        translation,
        torch.tensor([-0.097, 0.07, -0.06], dtype=torch.float32),
        atol=1e-6,
    )


@pytest.mark.skipif(not HOLOOCEAN_ZED_RECT.exists(), reason="ZED100 closed-path recording folder is unavailable")
def test_holoocean_loader_exposes_camera_to_imu_lever_arm_for_vio_residuals():
    seq = GeneralStereoIMUSequence(_holoocean_cfg(HOLOOCEAN_ZED_RECT))
    frame = seq[1]

    translation = frame.imu_vio_sensor_T_imu.translation().reshape(3).float()

    assert torch.allclose(
        translation,
        torch.tensor([-0.417, 0.18, 0.095], dtype=torch.float32),
        atol=1e-6,
    )


@pytest.mark.skipif(not HOLOOCEAN_CLEAR.exists(), reason="HoloOcean recording folder is unavailable")
def test_holoocean_metadata_summary_reports_actual_imu_tbs_translation(monkeypatch):
    meta = json.loads((HOLOOCEAN_CLEAR / "metadata.json").read_text(encoding="utf-8"))
    captured: list[str] = []

    monkeypatch.setattr(
        general_imu_module.Logger,
        "write",
        lambda _level, message: captured.append(str(message)),
    )

    general_imu_module._log_metadata_usage(
        HOLOOCEAN_CLEAR,
        meta,
        meta["camera"],
        meta["imu"],
        meta["extrinsics"],
    )

    summary = "\n".join(captured)
    assert "T_body_imu" in summary
    assert "internal NED" in summary
    assert "[-0.097, 0.07, -0.06]" in summary
    assert "GT velocity frame" in summary
    assert "world NWU" in summary


def test_gravity_sign_follows_world_z_direction():
    assert gravity_for_world_frame(9.8, "NWU") == pytest.approx(-9.8)
    assert gravity_for_world_frame(9.8, "NED") == pytest.approx(9.8)


def test_three_axis_sigma_conversion_preserves_each_axis():
    assert imu_sigma_to_continuous_density(
        [1.0, 2.0, 3.0], 100.0, "per-sample standard deviation"
    ) == pytest.approx((0.1, 0.2, 0.3))
    assert imu_bias_sigma_to_continuous_random_walk_density(
        [1.0, 2.0, 3.0], 400.0, "per-tick bias random-walk increment standard deviation"
    ) == pytest.approx((20.0, 40.0, 60.0))


@pytest.mark.skipif(
    not HOLOOCEAN_STATIC63_CIRCLE_NORMAL.exists(),
    reason="Static63 HoloOcean recording folder is unavailable",
)
def test_static63_bias_random_walk_uses_engine_update_rate_not_export_rate():
    seq = GeneralStereoIMUSequence(_holoocean_cfg(HOLOOCEAN_STATIC63_CIRCLE_NORMAL))
    frame = seq[1]

    assert seq.imu_calib_measurement_rate_hz == pytest.approx(100.0)
    assert seq.imu_calib_bias_random_walk_update_hz == pytest.approx(300.0)
    assert seq.imu_calib_acc_sigma == pytest.approx(0.0141258)
    assert seq.imu_calib_gyro_sigma == pytest.approx(0.00182898)
    assert seq.imu_calib_acc_w_sigma == pytest.approx(0.000386071)
    assert seq.imu_calib_gyro_w_sigma == pytest.approx(0.0000357864)
    assert getattr(frame, "imu_calib_bias_random_walk_update_hz") == pytest.approx(300.0)


@pytest.mark.skipif(not HOLOOCEAN_CLEAR.exists(), reason="HoloOcean recording folder is unavailable")
def test_holoocean_metadata_units_gravity_and_timestamp_are_exposed_on_frame():
    seq = GeneralStereoIMUSequence(_holoocean_cfg(HOLOOCEAN_CLEAR))
    frame = seq[1]

    assert seq.gravity == pytest.approx(9.8)
    assert seq.imu_calib_acc_sigma == pytest.approx(0.00277 / np.sqrt(300.0))
    assert seq.imu_calib_gyro_sigma == pytest.approx(0.00123 / np.sqrt(300.0))
    assert seq.imu_calib_acc_w_sigma == pytest.approx(0.00141 * np.sqrt(300.0))
    assert seq.imu_calib_gyro_w_sigma == pytest.approx(0.00388 * np.sqrt(300.0))
    assert seq.imu_calib_sigma_unit == "per-sample standard deviation"
    assert seq.imu_calib_bias_sigma_unit == "per-sample standard deviation"
    assert getattr(seq, "gravity_source") == "metadata.json"
    assert getattr(frame, "imu_acc_unit") == "m/s^2"
    assert getattr(frame, "imu_gyro_unit") == "rad/s"
    assert getattr(frame, "imu_timestamp_unit") == "ns"
    assert getattr(frame, "imu_metadata_gravity_m_s2") == pytest.approx(9.8)
    assert getattr(frame, "imu_gravity_source") == "metadata.json"


@pytest.mark.skipif(not HOLOOCEAN_CLEAR.exists(), reason="HoloOcean recording folder is unavailable")
def test_holoocean_metadata_rejects_incompatible_imu_units():
    meta = json.loads((HOLOOCEAN_CLEAR / "metadata.json").read_text(encoding="utf-8"))
    meta["imu"]["acc_unit"] = "ft/s^2"

    with pytest.raises(ValueError, match="acc_unit"):
        general_imu_module._validate_imu_metadata_conventions(meta)


def _pose_from_ref_row(row) -> pp.LieTensor:
    translation = torch.tensor([row["x"], row["y"], row["z"]], dtype=torch.float32).reshape(1, 3)
    quaternion = torch.tensor([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=torch.float32).reshape(1, 4)
    return pp.SE3(torch.cat([translation, quaternion], dim=-1))


def _pose_from_ref_row_as_internal_ned(row) -> pp.LieTensor:
    pose_nwu = _pose_from_ref_row(row)
    nwu_to_ned = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
    translation_ned = nwu_to_ned @ pose_nwu.translation().reshape(3)
    rotation_ned = pp.from_matrix(
        (nwu_to_ned @ pose_nwu.rotation().matrix().reshape(3, 3).float() @ nwu_to_ned).reshape(1, 3, 3),
        pp.SO3_type,
    )
    return pp.SE3(torch.cat([translation_ned.reshape(1, 3), rotation_ned.tensor().reshape(1, 4)], dim=-1))


def _velocity_from_ref_row(row) -> torch.Tensor:
    return torch.tensor([row["vx"], row["vy"], row["vz"]], dtype=torch.float32)


def _velocity_from_ref_row_as_internal_ned(row) -> torch.Tensor:
    nwu_to_ned = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))
    return nwu_to_ned @ _velocity_from_ref_row(row)


@pytest.mark.skipif(not HOLOOCEAN_CLEAR.exists(), reason="HoloOcean recording folder is unavailable")
def test_holoocean_source_nwu_and_internal_ned_preintegration_are_equivalent():
    imu = np.genfromtxt(HOLOOCEAN_CLEAR / "imu_data.csv", delimiter=",", names=True)
    ref = np.genfromtxt(HOLOOCEAN_CLEAR / "ref_pose.csv", delimiter=",", names=True)

    time_ns = torch.from_numpy(imu["timestamp"].astype(np.int64)).long()
    acc = torch.from_numpy(
        np.stack([imu["lin_acc_x"], imu["lin_acc_y"], imu["lin_acc_z"]], axis=1)
    ).float()
    gyro = torch.from_numpy(
        np.stack([imu["ang_vel_x"], imu["ang_vel_y"], imu["ang_vel_z"]], axis=1)
    ).float()
    flu_to_ned = torch.diag(torch.tensor([1.0, -1.0, -1.0]))

    source_nwu_errors = []
    internal_ned_errors = []
    for frame_idx in range(1, min(90, len(ref))):
        mask = (time_ns >= int(ref["timestamp"][frame_idx - 1])) & (time_ns <= int(ref["timestamp"][frame_idx]))
        if int(mask.sum()) < 2:
            continue

        pose_i_nwu = _pose_from_ref_row(ref[frame_idx - 1])
        pose_j_nwu = _pose_from_ref_row(ref[frame_idx])
        relative_nwu = pose_i_nwu.Inv() @ pose_j_nwu
        pose_i_ned = _pose_from_ref_row_as_internal_ned(ref[frame_idx - 1])
        pose_j_ned = _pose_from_ref_row_as_internal_ned(ref[frame_idx])
        relative_ned = pose_i_ned.Inv() @ pose_j_ned

        source_nwu_preint = preintegrate_imu(
            time_ns=time_ns[mask],
            acc=acc[mask],
            gyro=gyro[mask],
            R0_world=pose_i_nwu.rotation(),
            gravity=gravity_for_world_frame(9.8, "NWU"),
            sigma_acc=0.0,
            sigma_gyro=0.0,
        )
        internal_ned_preint = preintegrate_imu(
            time_ns=time_ns[mask],
            acc=(flu_to_ned @ acc[mask].T).T,
            gyro=(flu_to_ned @ gyro[mask].T).T,
            R0_world=pose_i_ned.rotation(),
            gravity=gravity_for_world_frame(9.8, "NED"),
            sigma_acc=0.0,
            sigma_gyro=0.0,
        )

        source_nwu_errors.append(float((source_nwu_preint.delta_R.Inv() @ relative_nwu.rotation()).Log().norm().item()))
        internal_ned_errors.append(float((internal_ned_preint.delta_R.Inv() @ relative_ned.rotation()).Log().norm().item()))

    assert source_nwu_errors
    assert float(np.median(source_nwu_errors)) < 3e-2
    assert float(np.median(internal_ned_errors)) == pytest.approx(
        float(np.median(source_nwu_errors)),
        abs=1e-6,
    )


@pytest.mark.skipif(not HOLOOCEAN_CLEAR.exists(), reason="HoloOcean recording folder is unavailable")
def test_holoocean_preintegration_is_consistent_with_vio_residual_blocks():
    imu = np.genfromtxt(HOLOOCEAN_CLEAR / "imu_data.csv", delimiter=",", names=True)
    ref = np.genfromtxt(HOLOOCEAN_CLEAR / "ref_pose.csv", delimiter=",", names=True)

    time_ns = torch.from_numpy(imu["timestamp"].astype(np.int64)).long()
    acc = torch.from_numpy(
        np.stack([imu["lin_acc_x"], imu["lin_acc_y"], imu["lin_acc_z"]], axis=1)
    ).float()
    gyro = torch.from_numpy(
        np.stack([imu["ang_vel_x"], imu["ang_vel_y"], imu["ang_vel_z"]], axis=1)
    ).float()

    residual_norms = []
    for frame_idx in range(1, min(90, len(ref))):
        mask = (time_ns >= int(ref["timestamp"][frame_idx - 1])) & (time_ns <= int(ref["timestamp"][frame_idx]))
        if int(mask.sum()) < 2:
            continue

        pose_i = _pose_from_ref_row_as_internal_ned(ref[frame_idx - 1])
        pose_j = _pose_from_ref_row_as_internal_ned(ref[frame_idx])
        flu_to_ned = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        preint = preintegrate_imu(
            time_ns=time_ns[mask],
            acc=(flu_to_ned @ acc[mask].T).T,
            gyro=(flu_to_ned @ gyro[mask].T).T,
            R0_world=pose_i.rotation(),
            gravity=gravity_for_world_frame(9.8, "NED"),
            sigma_acc=0.0,
            sigma_gyro=0.0,
        )

        residual = vio_preintegrated_imu_residual(
            from_pose=pose_i,
            to_pose=pose_j,
            prev_velocity_world=_velocity_from_ref_row_as_internal_ned(ref[frame_idx - 1]),
            curr_velocity_world=_velocity_from_ref_row_as_internal_ned(ref[frame_idx]),
            delta_R=preint.delta_R,
            delta_v=preint.delta_v,
            delta_p=preint.delta_p,
            dt_total=preint.dt_total,
        )
        residual_norms.append(residual.norm(dim=1).detach().cpu().numpy())

    assert residual_norms
    median_position, median_velocity, median_rotation = np.median(np.stack(residual_norms), axis=0)
    assert float(median_position) < 1e-3
    assert float(median_velocity) < 3e-2
    assert float(median_rotation) < 3e-2
