import numpy as np

from Utility.SlidingWindowVIO import (
    SlidingWindowConfig,
    SlidingWindowEdge,
    SlidingWindowSequence,
    optimize_sliding_window_sequence,
)


def test_sliding_window_vio_reduces_visual_position_drift_on_synthetic_sequence():
    n = 12
    time_ns = (np.arange(n, dtype=np.int64) * 100_000_000).astype(np.int64)
    t = time_ns.astype(np.float64) * 1e-9
    accel = np.array([0.4, 0.0, 0.0], dtype=np.float64)
    true_pos = np.stack(
        [
            0.5 * accel[0] * t**2,
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
        ],
        axis=1,
    )
    visual_pos = true_pos.copy()
    visual_pos[:, 0] *= 0.75
    visual_pos[:, 1] += np.linspace(0.0, 0.25, n)

    rot_identity = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n, axis=0)
    visual_vel = np.gradient(visual_pos, time_ns.astype(np.float64) * 1e-9, axis=0)
    visual_vel[0] = np.zeros(3, dtype=np.float64)
    edges = [
        SlidingWindowEdge(
            i=i,
            j=i + 1,
            dt=0.1,
            visual_delta_p_body=visual_pos[i + 1] - visual_pos[i],
            visual_delta_R=np.eye(3, dtype=np.float64),
            imu_delta_p_body=0.5 * accel * (0.1**2),
            imu_delta_v_body=accel * 0.1,
            imu_delta_R=np.eye(3, dtype=np.float64),
        )
        for i in range(n - 1)
    ]
    seq = SlidingWindowSequence(
        time_ns=time_ns,
        position_w=visual_pos,
        rotation_bw=rot_identity,
        velocity_w=visual_vel,
        edges=edges,
    )

    before = float(np.sqrt(np.mean(np.sum((visual_pos - true_pos) ** 2, axis=1))))
    result = optimize_sliding_window_sequence(
        seq,
        SlidingWindowConfig(
            window_size=6,
            stride=3,
            max_nfev=20,
            visual_position_weight=0.2,
                visual_rotation_weight=0.2,
                imu_position_weight=80.0,
                imu_velocity_weight=20.0,
                imu_rotation_weight=0.0,
            ),
        )
    after = float(np.sqrt(np.mean(np.sum((result.position_w - true_pos) ** 2, axis=1))))

    assert after < before * 0.5
    np.testing.assert_allclose(result.position_w[0], visual_pos[0], atol=1e-12)
