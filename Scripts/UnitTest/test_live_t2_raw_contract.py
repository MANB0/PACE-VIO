from types import SimpleNamespace

import pypose as pp
import torch

from Odometry.MACVO import MACVO
from Utility.LiveDashboard import LiveDashboard, LiveStateStore
from Utility.Point import point2pixel_NED
from Utility.TwoStateVIO import UVDFactor, solve_uvd_relative_pose_visual_only


def test_visual_only_uvd_solver_recovers_relative_pose_without_vio_state():
    torch.manual_seed(17)
    dtype = torch.float64
    points_Ci = torch.randn(160, 3, dtype=dtype)
    points_Ci[:, 0] = points_Ci[:, 0].abs() * 2.0 + 4.0
    points_Ci[:, 1:] *= 0.7
    intrinsic = torch.tensor(
        [[420.0, 0.0, 320.0], [0.0, 418.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    baseline = 0.12
    truth = pp.se3(torch.tensor(
        [[0.08, -0.025, 0.018, 0.012, -0.009, 0.017]], dtype=dtype
    )).Exp()
    points_Cj = truth.Act(points_Ci)
    target_uv = point2pixel_NED(points_Cj, intrinsic)
    target_disparity = intrinsic[0, 0] * baseline / points_Cj[:, 0:1]
    covariance = torch.eye(3, dtype=dtype).unsqueeze(0).repeat(points_Ci.shape[0], 1, 1)
    covariance[:, 0:2, 0:2] *= 0.04
    covariance[:, 2, 2] *= 0.01
    visual = UVDFactor(
        points_Ci=points_Ci,
        target_uv=target_uv,
        target_disparity=target_disparity,
        covariance_uvd=covariance,
        intrinsic=intrinsic,
        baseline=baseline,
        extrinsic_CI=pp.identity_SE3(1, dtype=dtype).tensor(),
        huber_delta=3.0,
    )

    estimate, diagnostics = solve_uvd_relative_pose_visual_only(
        visual, max_iterations=12
    )
    error = (pp.SE3(estimate).Inv() @ truth).Log().tensor().reshape(6)

    assert diagnostics["reason"] != "nonfinite"
    assert torch.isfinite(error).all()
    assert float(error.abs().max()) < 2.0e-5


def test_dashboard_committed_publish_uses_committed_index_and_matching_raw_pose():
    identity = pp.identity_SE3(3, dtype=torch.float64).tensor()
    committed = identity.clone()
    committed[1, 0] = 1.0
    committed[2, 0] = 99.0  # Current frontend frame; must not be published as committed.
    raw = identity[1:2].clone()
    raw[0, 1] = 2.0

    class Frames:
        def __init__(self):
            self.data = {
                "pose": committed,
                "time_ns": torch.tensor([0, 1_000_000_000, 2_000_000_000]),
                "imu_vio_sensor_T_imu": identity,
                "imu_vio_velocity_world": torch.zeros(3, 3),
                "imu_vio_acc_bias": torch.zeros(3, 3),
                "imu_vio_gyro_bias": torch.zeros(3, 3),
            }

        def __len__(self):
            return 3

    system = SimpleNamespace(
        graph=SimpleNamespace(frames=Frames()),
        Optimizer=SimpleNamespace(last_pair_diagnostics={
            "frame_idx": 1,
            "vio_backend": "isam2",
            "isam2_state_count": 2,
            "isam2_history_revision": True,
        }),
        _live_macvo_raw_poses={1: raw},
        _live_macvo_raw_last_diagnostics={"available": True, "frame_idx": 1},
        _pipeline_pending=None,
        _imu_static_init_diag={},
        _imu_static_initialized=True,
        _imu_static_zupt_active=False,
    )
    dashboard = LiveDashboard.__new__(LiveDashboard)
    dashboard._gt = {}
    dashboard._gt_anchor = None
    dashboard._meta = {}
    dashboard._stereo_cache = {}
    dashboard._last_stereo_timestamp_ns = None
    dashboard._last_stereo_encode_time = 0.0

    payload = dashboard._payload(system, None, "vio_committed")

    assert payload is not None
    assert payload["frame_idx"] == 1
    assert payload["timestamp_ns"] == 1_000_000_000
    assert payload["raw"] is not None
    assert payload["committed"] is not None
    assert payload["raw"] != payload["committed"]
    # The dashboard rebases the first published IMU-center pose to zero.  The
    # important contract is that it used frame 1 (not frame 2's x=99 pose)
    # and paired that committed state with frame 1's independent raw state.
    assert max(abs(value) for value in payload["committed"]) < 1.0e-12
    assert sum(value * value for value in payload["raw"]) > 1.0
    assert [
        item["frame_idx"] for item in payload["committed_history_revision"]
    ] == [0, 1]


def test_dashboard_imu_history_survives_committed_stage_without_frame_payload():
    identity = pp.identity_SE3(2, dtype=torch.float64).tensor()

    class Frames:
        def __init__(self):
            self.data = {
                "pose": identity.clone(),
                "time_ns": torch.tensor([0, 1_000_000_000]),
                "imu_vio_sensor_T_imu": identity.clone(),
                "imu_vio_velocity_world": torch.zeros(2, 3),
                "imu_vio_acc_bias": torch.zeros(2, 3),
                "imu_vio_gyro_bias": torch.zeros(2, 3),
            }

        def __len__(self):
            return 2

    raw = identity[1:2].clone()
    system = SimpleNamespace(
        graph=SimpleNamespace(frames=Frames()),
        Optimizer=SimpleNamespace(last_pair_diagnostics={"frame_idx": 1}),
        _live_macvo_raw_poses={1: raw},
        _live_macvo_raw_last_diagnostics={"available": True, "frame_idx": 1},
        _pipeline_pending=None,
        _imu_static_init_diag={},
        _imu_static_initialized=True,
        _imu_static_zupt_active=False,
    )
    frame = SimpleNamespace(imu=SimpleNamespace(
        time_ns=torch.tensor([900_000_000, 1_000_000_000]),
        acc=torch.tensor([[0.1, 0.2, 9.8], [0.2, 0.3, 9.7]]),
        gyro=torch.tensor([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]]),
    ))
    dashboard = LiveDashboard.__new__(LiveDashboard)
    dashboard._gt = {}
    dashboard._gt_anchor = None
    dashboard._meta = {}
    dashboard._stereo_cache = {}
    dashboard._last_stereo_timestamp_ns = None
    dashboard._last_stereo_encode_time = 0.0

    raw_payload = dashboard._payload(system, frame, "macvo_raw")
    committed_payload = dashboard._payload(system, None, "vio_committed")

    assert raw_payload is not None and committed_payload is not None
    assert len(raw_payload["imu_recent"]) == 2
    assert committed_payload["imu_recent"] == raw_payload["imu_recent"]
    assert committed_payload["raw_latest_frame_idx"] == 1
    assert committed_payload["committed_latest_frame_idx"] == 1


def test_dashboard_replay_stereo_is_separate_from_live_state_history():
    store = LiveStateStore(max_history=4)
    image0 = {
        "left": "data:image/jpeg;base64,left0",
        "right": "data:image/jpeg;base64,right0",
        "timestamp_ns": 100,
    }
    store.publish({
        "frame_idx": 0,
        "timestamp_ns": 100,
        "raw": [0.0, 0.0, 0.0],
        "stereo_images": image0,
    })
    # A throttled dashboard publication may still carry the previous encoded
    # image. It must not be mislabeled as frame 1 in replay storage.
    store.publish({
        "frame_idx": 1,
        "timestamp_ns": 200,
        "raw": [1.0, 0.0, 0.0],
        "stereo_images": image0,
    })

    replay = store.replay_snapshot(1)
    state = store.snapshot()

    assert replay is not None
    assert replay["frame_idx"] == 0
    assert replay["timestamp_ns"] == 100
    assert replay["stereo_images"] == image0
    assert [entry["frame_idx"] for entry in state["history"]] == [0, 1]
    assert all("stereo_images" not in entry for entry in state["history"])


def test_dashboard_history_revision_replaces_committed_points_by_frame_index():
    store = LiveStateStore(max_history=6)
    for frame_idx in range(3):
        store.publish({
            "frame_idx": frame_idx,
            "timestamp_ns": frame_idx * 100,
            "raw": [float(frame_idx), 0.0, 0.0],
            "committed": [float(frame_idx), 1.0, 0.0],
        })

    store.publish({
        "frame_idx": 2,
        "timestamp_ns": 200,
        "committed": [2.0, 2.0, 0.0],
        "committed_history_revision": [
            {"frame_idx": 0, "timestamp_ns": 0, "committed": [0.0, 3.0, 0.0]},
            {"frame_idx": 1, "timestamp_ns": 100, "committed": [1.0, 3.0, 0.0]},
            {"frame_idx": 2, "timestamp_ns": 200, "committed": [2.0, 3.0, 0.0]},
        ],
    })

    state = store.snapshot()
    assert [entry["frame_idx"] for entry in state["history"]] == [0, 1, 2]
    assert [entry["committed"][1] for entry in state["history"]] == [3.0, 3.0, 2.0]
    assert [entry["raw"][0] for entry in state["history"]] == [0.0, 1.0, 2.0]
    assert "committed_history_revision" not in state


def test_online_t2_compression_uses_visual_optimum_and_sets_warm_start():
    torch.manual_seed(23)
    dtype = torch.float64
    points_Ci = torch.randn(120, 3, dtype=dtype)
    points_Ci[:, 0] = points_Ci[:, 0].abs() * 2.0 + 4.0
    points_Ci[:, 1:] *= 0.6
    intrinsic = torch.tensor(
        [[420.0, 0.0, 320.0], [0.0, 418.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    baseline = 0.12
    truth_CjCi = pp.se3(torch.tensor(
        [[0.07, -0.03, 0.015, 0.01, -0.008, 0.014]], dtype=dtype
    )).Exp()
    points_Cj = truth_CjCi.Act(points_Ci)
    target_uv = point2pixel_NED(points_Cj, intrinsic)
    target_disparity = intrinsic[0, 0] * baseline / points_Cj[:, 0:1]
    uv_cov = torch.zeros(points_Ci.shape[0], 3, dtype=dtype)
    uv_cov[:, 0:2] = 0.04
    disparity_cov = torch.full((points_Ci.shape[0],), 0.01, dtype=dtype)
    identity = pp.identity_SE3(1, dtype=dtype).tensor()
    graph_identity = identity.float()
    graph_data = SimpleNamespace(
        points=SimpleNamespace(data={"pos_Tc": points_Ci}),
        observations=SimpleNamespace(data={
            "pixel2_uv": target_uv,
            "pixel2_disp": target_disparity,
            "pixel2_uv_cov": uv_cov,
            "pixel2_disp_cov": disparity_cov,
        }),
        images_intrinsic=intrinsic,
        baseline=torch.tensor([baseline], dtype=dtype),
        from_idx=torch.tensor([0]),
        frame_idx=torch.tensor([1]),
        from_pose=pp.SE3(graph_identity.clone()),
        init_motion=pp.SE3(graph_identity.clone()),
        visual_compressed_uvd_reference_CjCi=None,
        visual_compressed_uvd_hessian=None,
        visual_compressed_uvd_gradient=None,
        visual_compressed_uvd_robust_cost=None,
        visual_compressed_uvd_num_points=0,
        visual_compressed_uvd_num_inliers=0,
        visual_compressed_uvd_mean_mahalanobis_sq=None,
        visual_compressed_uvd_huber_delta=None,
    )
    system = MACVO.__new__(MACVO)
    system.Optimizer = SimpleNamespace(config=SimpleNamespace(
        imu_factor_mode="two_state_fixed_lag",
        two_state_visual_factor_mode="compressed_uvd",
        two_state_warm_start="macvo_pose",
        two_state_uvd_huber_delta=3.0,
        two_state_max_iterations=12,
        two_state_marginalization_eigenvalue_floor=1.0e-10,
    ))
    system._live_macvo_raw_poses = {0: graph_identity.clone()}
    system._live_macvo_raw_last_diagnostics = {}
    system.graph = SimpleNamespace(frames=SimpleNamespace(data={"pose": graph_identity.clone()}))

    system._update_live_macvo_raw_pose(graph_data)

    reference_error = (
        pp.SE3(graph_data.visual_compressed_uvd_reference_CjCi).Inv()
        @ truth_CjCi
    ).Log().tensor().reshape(6)
    expected_warm_start = pp.SE3(truth_CjCi.Inv().tensor().float())
    warm_start_error = (
        pp.SE3(graph_data.init_motion).Inv() @ expected_warm_start
    ).Log().tensor().reshape(6)
    assert system._live_macvo_raw_last_diagnostics["available"] is True
    assert system._live_macvo_raw_last_diagnostics["t2_compression_source"] == "online_visual_optimum"
    assert float(reference_error.abs().max()) < 2.0e-5
    assert float(warm_start_error.abs().max()) < 2.0e-5
    assert graph_data.visual_compressed_uvd_hessian.shape == (6, 6)
    assert graph_data.visual_compressed_uvd_gradient.shape == (6,)
    assert bool(torch.isfinite(graph_data.visual_compressed_uvd_hessian).all())
    assert graph_data.visual_compressed_uvd_num_points == points_Ci.shape[0]
