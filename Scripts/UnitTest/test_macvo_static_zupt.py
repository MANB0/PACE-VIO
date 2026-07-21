from types import SimpleNamespace

import pypose as pp
import torch

from Odometry.MACVO import MACVO


class _MatchTable:
    def __init__(self):
        self.data = {
            "pixel1_uv": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=torch.float32,
            )
        }


class _CountingOptimizer:
    def __init__(self):
        self.start_count = 0

    def start_optimize(self, graph_data):
        self.start_count += 1

    def get_graph_data(self, graph, frame_idx):
        return (graph, frame_idx)


class _CapturingFrames:
    def __init__(self):
        self.data = {}

    def __len__(self):
        if not self.data:
            return 0
        return next(iter(self.data.values())).shape[0]

    def push(self, frame_node):
        frame_idx = len(self)
        for name, value in frame_node.data.items():
            tensor = value.detach().clone()
            self.data[name] = tensor if name not in self.data else torch.cat([self.data[name], tensor], dim=0)
        return torch.tensor([frame_idx], dtype=torch.long)


def test_static_zupt_consumes_visual_observations_without_starting_optimizer():
    odom = object.__new__(MACVO)
    odom.min_num_point = 2
    odom.graph = SimpleNamespace(match=_MatchTable())
    odom.Optimizer = _CountingOptimizer()
    odom._pending_tracking_state = None
    odom._adaptive_enabled = False
    odom._adaptive_decision = None

    lost_track = odom._complete_tracking_step(
        frame1=SimpleNamespace(frame_idx=1),
        frame_idx=torch.tensor([1], dtype=torch.long),
        match_idx=torch.tensor([0, 1, 2], dtype=torch.long),
        match_obs=None,
        candidate_count=3,
        prev_pose=pp.identity_SE3(dtype=torch.float32),
        est_pose=pp.identity_SE3(dtype=torch.float32),
        imu_rel_pose=None,
        suppress_optimizer=True,
    )

    assert not lost_track
    assert odom.Optimizer.start_count == 0
    assert odom._pending_tracking_state["optimizer_skipped"]
    assert odom._pending_tracking_state["static_zupt_active"]
    assert odom._pending_tracking_state["match_idx_size_after_filter"] == 3


def test_static_zupt_initializes_state_then_releases_on_next_frame():
    odom = object.__new__(MACVO)
    odom._imu_static_initialized = False
    odom.imu_static_initialization_enable = True
    odom._imu_static_zupt_active = False
    odom._imu_static_time_chunks = []
    odom._imu_static_acc_chunks = []
    odom._imu_static_gyro_chunks = []
    odom._imu_static_last_time_ns = None
    odom._imu_static_initial_rotation = None
    odom._imu_static_anchor_pose = None
    odom._imu_static_init_diag = None
    odom.imu_static_initialization_duration_s = 3.0
    odom.imu_static_sigma_multiplier = 5.0
    odom.imu_static_gyro_mean_norm_max = 0.03
    odom.imu_static_acc_norm_error_max = 0.6
    odom.imu_sigma_acc = 0.0141258
    odom.imu_sigma_gyro = 0.00182898
    odom.prev_keyframe = (None, 0, None)
    odom.graph = SimpleNamespace(
        frames=SimpleNamespace(
            data={
                "pose": pp.identity_SE3(1, dtype=torch.float32).tensor(),
            }
        )
    )

    sample_count = 301
    time_ns = torch.arange(sample_count, dtype=torch.int64) * 10_000_000
    acc_bias = torch.tensor([0.0, 0.0, 0.03], dtype=torch.float32)
    gyro_bias = torch.tensor([0.01, -0.005, 0.003], dtype=torch.float32)
    acc = torch.tensor([0.0, 0.0, -9.8]).repeat(sample_count, 1) + acc_bias
    gyro = gyro_bias.repeat(sample_count, 1)
    frame = SimpleNamespace(
        imu_calib_measurement_rate_hz=100.0,
        imu_calib_acc_sigma=0.0141258,
        imu_calib_gyro_sigma=0.00182898,
    )

    hold_completion_frame = odom._try_static_imu_initialization(
        frame,
        time_ns,
        acc,
        gyro,
        gravity=9.8,
    )

    assert hold_completion_frame
    assert odom._imu_static_initialized
    assert odom._imu_static_zupt_active
    assert torch.allclose(odom._imu_vel_w, torch.zeros(3))
    assert torch.allclose(odom._imu_gyro_bias, gyro_bias, atol=1e-6)
    assert torch.allclose(odom._imu_acc_bias, acc_bias, atol=1e-5)
    assert torch.allclose(
        odom._imu_static_anchor_pose.tensor(),
        pp.identity_SE3(dtype=torch.float32).tensor(),
    )

    release_next_frame = odom._try_static_imu_initialization(
        frame,
        time_ns[-2:],
        acc[-2:],
        gyro[-2:],
        gravity=9.8,
    )

    assert not release_next_frame
    assert not odom._imu_static_zupt_active


def test_static_initialized_bias_survives_factorless_keyframe_and_feedback_sync():
    odom = object.__new__(MACVO)
    acc_bias = torch.tensor([0.02, -0.01, 0.03], dtype=torch.float32)
    gyro_bias = torch.tensor([0.001, -0.002, 0.003], dtype=torch.float32)
    odom._imu_acc_bias = acc_bias.clone()
    odom._imu_gyro_bias = gyro_bias.clone()
    odom._imu_vel_w = torch.zeros(3, dtype=torch.float32)
    odom.graph = SimpleNamespace(frames=_CapturingFrames())
    odom.Optimizer = SimpleNamespace(config=SimpleNamespace(imu_factor_mode="preintegrated_vio"))
    odom.imu_vio_velocity_feedback_enable = True
    odom.imu_vio_bias_feedback_enable = True

    frame = SimpleNamespace(
        stereo=SimpleNamespace(
            T_BS=pp.identity_SE3(1, dtype=torch.float32),
            frame_ns=3_000_000_000,
            K=torch.eye(3, dtype=torch.float32).unsqueeze(0),
            baseline=torch.tensor([0.12], dtype=torch.float32),
        )
    )
    frame_idx = odom.push_keyframe(frame, pp.identity_SE3(1, dtype=torch.float32))
    odom.prev_keyframe = (frame, int(frame_idx.item()), None)

    assert torch.allclose(odom.graph.frames.data["imu_vio_prev_acc_bias"][0], acc_bias)
    assert torch.allclose(odom.graph.frames.data["imu_vio_prev_gyro_bias"][0], gyro_bias)
    assert torch.allclose(odom.graph.frames.data["imu_vio_acc_bias"][0], acc_bias)
    assert torch.allclose(odom.graph.frames.data["imu_vio_gyro_bias"][0], gyro_bias)

    odom._imu_acc_bias.zero_()
    odom._imu_gyro_bias.zero_()
    odom._sync_optimized_vio_velocity_from_map()

    assert torch.allclose(odom._imu_acc_bias, acc_bias)
    assert torch.allclose(odom._imu_gyro_bias, gyro_bias)
