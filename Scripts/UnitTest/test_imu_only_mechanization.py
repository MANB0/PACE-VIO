import numpy as np

import Scripts.run_clear_circle_imu_only_mechanization as mech


def test_static_nwu_mechanization_keeps_pose_and_velocity():
    time_ns = np.array([0, 1_000_000_000], dtype=np.int64)
    acc = np.array([[0.0, 0.0, 9.8], [0.0, 0.0, 9.8]], dtype=np.float64)
    gyro = np.zeros((2, 3), dtype=np.float64)

    states = mech.mechanize_imu_nwu(
        time_ns=time_ns,
        acc_b=acc,
        gyro_b=gyro,
        p0=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        R0_bw=np.eye(3),
        v0_w=np.zeros(3, dtype=np.float64),
        gravity=9.8,
    )

    assert np.allclose(states.position_w[-1], [1.0, 2.0, 3.0], atol=1e-9)
    assert np.allclose(states.velocity_w[-1], [0.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(states.R_bw[-1], np.eye(3), atol=1e-9)


def test_constant_forward_acceleration_integrates_velocity_and_position():
    time_ns = np.array([0, 1_000_000_000], dtype=np.int64)
    acc = np.array([[1.0, 0.0, 9.8], [1.0, 0.0, 9.8]], dtype=np.float64)
    gyro = np.zeros((2, 3), dtype=np.float64)

    states = mech.mechanize_imu_nwu(
        time_ns=time_ns,
        acc_b=acc,
        gyro_b=gyro,
        p0=np.zeros(3, dtype=np.float64),
        R0_bw=np.eye(3),
        v0_w=np.zeros(3, dtype=np.float64),
        gravity=9.8,
    )

    assert np.allclose(states.velocity_w[-1], [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(states.position_w[-1], [0.5, 0.0, 0.0], atol=1e-9)


def test_imu_only_converts_imu_point_back_to_camera_point_with_lever_arm():
    states = mech.MechanizedStates(
        time_ns=np.array([0, 1], dtype=np.int64),
        position_w=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        velocity_w=np.zeros((2, 3), dtype=np.float64),
        R_bw=np.array(
            [
                np.eye(3),
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        ),
    )

    camera_states = mech.convert_imu_states_to_camera_point(
        states,
        camera_to_imu_body=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )

    assert np.allclose(camera_states.position_w, np.zeros((2, 3)), atol=1e-9)
