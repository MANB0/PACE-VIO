from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from Utility.OutputTrajectoryESKF3D import (
    STATE_DOF,
    CausalPoseOutputESKF3D,
    OutputESKF3DNoise,
    OutputESKF3DState,
    OutputESKFGate,
    boxminus,
    boxplus,
    measurement_residual_and_jacobian,
    predict_nominal,
    so3_exp,
    transition_jacobian,
)


def noise() -> OutputESKF3DNoise:
    return OutputESKF3DNoise(
        position_measurement_std=np.array([0.02, 0.025, 0.03]),
        rotation_measurement_std=np.array([0.008, 0.009, 0.01]),
        linear_acceleration_process_std=np.array([0.3, 0.35, 0.4]),
        angular_acceleration_process_std=np.array([0.1, 0.12, 0.14]),
    )


def random_state(generator: np.random.Generator) -> OutputESKF3DState:
    return OutputESKF3DState(
        position_W=generator.normal(size=3),
        velocity_W=generator.normal(size=3),
        rotation_WB=so3_exp(generator.normal(size=3) * 0.6),
        angular_velocity_B=generator.normal(size=3) * 0.5,
    )


def test_boxplus_boxminus_roundtrip_for_nonzero_3d_state():
    generator = np.random.default_rng(2)
    for _ in range(50):
        state = random_state(generator)
        delta = generator.normal(size=STATE_DOF) * 1.0e-4
        recovered = boxminus(boxplus(state, delta), state)
        assert np.max(np.abs(recovered - delta)) < 1.0e-10


def test_transition_jacobian_matches_central_difference_for_50_random_states():
    generator = np.random.default_rng(3)
    epsilon = 1.0e-6
    maximum_error = 0.0
    for _ in range(50):
        state = random_state(generator)
        dt = float(generator.uniform(0.005, 0.12))
        reference = predict_nominal(state, dt)
        numeric = np.zeros((STATE_DOF, STATE_DOF))
        for column in range(STATE_DOF):
            increment = np.zeros(STATE_DOF)
            increment[column] = epsilon
            plus = boxminus(
                predict_nominal(boxplus(state, increment), dt), reference
            )
            increment[column] = -epsilon
            minus = boxminus(
                predict_nominal(boxplus(state, increment), dt), reference
            )
            numeric[:, column] = (plus - minus) / (2.0 * epsilon)
        analytic = transition_jacobian(state, dt)
        maximum_error = max(maximum_error, float(np.max(np.abs(analytic - numeric))))
        assert np.max(np.abs(analytic - numeric)) < 1.0e-7
        assert np.max(np.abs(analytic[0:6, 6:12])) < 1.0e-14
    assert maximum_error < 1.0e-7


def test_measurement_jacobian_matches_central_difference_for_50_random_states():
    generator = np.random.default_rng(4)
    epsilon = 1.0e-6
    maximum_error = 0.0
    for _ in range(50):
        state = random_state(generator)
        measured_position = state.position_W + generator.normal(size=3) * 0.2
        measured_rotation = state.rotation_WB @ so3_exp(
            generator.normal(size=3) * 0.25
        )
        _, analytic = measurement_residual_and_jacobian(
            state, measured_position, measured_rotation
        )
        numeric = np.zeros((6, STATE_DOF))
        for column in range(STATE_DOF):
            increment = np.zeros(STATE_DOF)
            increment[column] = epsilon
            plus, _ = measurement_residual_and_jacobian(
                boxplus(state, increment), measured_position, measured_rotation
            )
            increment[column] = -epsilon
            minus, _ = measurement_residual_and_jacobian(
                boxplus(state, increment), measured_position, measured_rotation
            )
            numeric[:, column] = (plus - minus) / (2.0 * epsilon)
        maximum_error = max(maximum_error, float(np.max(np.abs(numeric + analytic))))
        assert np.max(np.abs(numeric + analytic)) < 1.0e-7
        assert np.max(np.abs(analytic[:, 3:6])) < 1.0e-14
        assert np.max(np.abs(analytic[:, 9:12])) < 1.0e-14
    assert maximum_error < 1.0e-7


def test_stationary_3d_filter_reduces_noise_and_keeps_covariance_psd():
    generator = np.random.default_rng(5)
    position_measurement = generator.normal(0.0, 0.02, size=(400, 3))
    rotation_measurement = Rotation.from_rotvec(
        generator.normal(0.0, 0.006, size=(400, 3))
    ).as_matrix()
    output_filter = CausalPoseOutputESKF3D(
        position_measurement[0], rotation_measurement[0], noise()
    )
    filtered_position = []
    filtered_rotation = []
    for index in range(400):
        output_filter.step(
            None if index == 0 else 0.02,
            position_measurement[index],
            rotation_measurement[index],
        )
        filtered_position.append(output_filter.state.position_W.copy())
        filtered_rotation.append(
            Rotation.from_matrix(output_filter.state.rotation_WB).as_rotvec()
        )
    filtered_position = np.asarray(filtered_position)
    filtered_rotation = np.asarray(filtered_rotation)
    assert np.var(filtered_position[100:], axis=0).max() < np.var(
        position_measurement[100:], axis=0
    ).max()
    assert np.var(filtered_rotation[100:], axis=0).max() < np.var(
        Rotation.from_matrix(rotation_measurement[100:]).as_rotvec(), axis=0
    ).max()
    assert np.linalg.eigvalsh(output_filter.covariance).min() >= -1.0e-12
    assert np.max(
        np.abs(output_filter.state.rotation_WB.T @ output_filter.state.rotation_WB - np.eye(3))
    ) < 1.0e-12


def test_filter_handles_nonplanar_translation_and_rotation():
    generator = np.random.default_rng(6)
    dt = 0.02
    time = np.arange(500) * dt
    truth_position = np.column_stack(
        [
            1.5 * np.sin(0.35 * time),
            0.8 * np.sin(0.61 * time + 0.2),
            0.25 * time + 0.15 * np.sin(0.42 * time),
        ]
    )
    truth_rotation = Rotation.from_euler(
        "xyz",
        np.column_stack(
            [
                0.15 * np.sin(0.4 * time),
                0.12 * np.cos(0.3 * time),
                0.45 * time,
            ]
        ),
    )
    measured_position = truth_position + generator.normal(0.0, 0.02, (500, 3))
    measured_rotation = truth_rotation * Rotation.from_rotvec(
        generator.normal(0.0, 0.006, (500, 3))
    )
    output_filter = CausalPoseOutputESKF3D(
        measured_position[0], measured_rotation[0].as_matrix(), noise()
    )
    filtered_position = []
    filtered_rotation = []
    for index in range(500):
        output_filter.step(
            None if index == 0 else dt,
            measured_position[index],
            measured_rotation[index].as_matrix(),
        )
        filtered_position.append(output_filter.state.position_W.copy())
        filtered_rotation.append(output_filter.state.rotation_WB.copy())
    filtered_position = np.asarray(filtered_position)
    filtered_rotation = np.asarray(filtered_rotation)
    assert np.ptp(filtered_position[:, 2]) > 1.0
    assert np.isfinite(filtered_position).all()
    assert np.isfinite(filtered_rotation).all()
    assert np.sqrt(np.mean(np.diff(filtered_position, n=2, axis=0) ** 2)) < np.sqrt(
        np.mean(np.diff(measured_position, n=2, axis=0) ** 2)
    )


def test_position_and_rotation_outliers_are_rejected_without_state_jump():
    output_filter = CausalPoseOutputESKF3D(
        np.zeros(3), np.eye(3), noise(), gate=OutputESKFGate()
    )
    for _ in range(100):
        output_filter.step(0.02, np.zeros(3), np.eye(3))
    before = output_filter.state.copy()
    diagnostics = output_filter.step(
        0.02,
        np.array([20.0, -10.0, 5.0]),
        so3_exp(np.array([0.9, -0.7, 1.1])),
    )
    assert diagnostics.position_action == "reject"
    assert diagnostics.rotation_action == "reject"
    assert np.linalg.norm(output_filter.state.position_W - before.position_W) < 1.0e-12
    assert np.linalg.norm(
        output_filter.state.angular_velocity_B - before.angular_velocity_B
    ) < 1.0e-12


def test_normal_turn_inflates_covariance_without_rejecting_measurement():
    output_filter = CausalPoseOutputESKF3D(
        np.zeros(3), np.eye(3), noise(), gate=OutputESKFGate()
    )
    for _ in range(100):
        output_filter.step(1.0 / 30.0, np.zeros(3), np.eye(3))
    output_filter.predict(1.0 / 30.0)
    diagnostics = output_filter.update(
        np.zeros(3), so3_exp(np.array([0.0, 0.0, 0.1]))
    )

    assert diagnostics.rotation_action == "inflate"
    assert np.isfinite(output_filter.state.rotation_WB).all()


def test_adaptive_process_scale_increases_for_persistent_finite_innovation():
    output_filter = CausalPoseOutputESKF3D(
        np.zeros(3), np.eye(3), noise(), adaptive_process=True
    )
    initial = output_filter.process_scale
    for _ in range(5):
        output_filter.step(0.02, np.array([0.2, 0.0, 0.0]), np.eye(3))
    assert output_filter.process_scale > initial
    assert 0.25 <= output_filter.process_scale <= 8.0
