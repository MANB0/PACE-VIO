import torch


def test_full_translation_prior_includes_velocity_term():
    from Utility.IMUKinematics import compose_imu_translation_prior

    delta_p_body = torch.tensor([0.01, -0.02, 0.03], dtype=torch.float32)
    velocity_world = torch.tensor([2.0, 0.5, -0.25], dtype=torch.float32)
    R_body_to_world = torch.eye(3, dtype=torch.float32)

    prior = compose_imu_translation_prior(
        delta_p_body=delta_p_body,
        velocity_world=velocity_world,
        R_body_to_world=R_body_to_world,
        dt_total=0.1,
    )

    expected = torch.tensor([0.21, 0.03, 0.005], dtype=torch.float32)
    assert torch.allclose(prior, expected, atol=1e-6)


def test_full_translation_prior_expresses_velocity_in_body_frame():
    from Utility.IMUKinematics import compose_imu_translation_prior

    delta_p_body = torch.zeros(3, dtype=torch.float32)
    velocity_world = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    R_body_to_world = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    prior = compose_imu_translation_prior(
        delta_p_body=delta_p_body,
        velocity_world=velocity_world,
        R_body_to_world=R_body_to_world,
        dt_total=2.0,
    )

    expected_body = torch.tensor([0.0, -2.0, 0.0], dtype=torch.float32)
    assert torch.allclose(prior, expected_body, atol=1e-6)


def test_velocity_propagation_does_not_add_gravity_twice():
    from Utility.IMUKinematics import propagate_imu_velocity_world

    velocity_world = torch.zeros(3, dtype=torch.float32)
    delta_v_body = torch.zeros(3, dtype=torch.float32)
    R_body_to_world = torch.eye(3, dtype=torch.float32)

    propagated = propagate_imu_velocity_world(
        velocity_world=velocity_world,
        delta_v_body=delta_v_body,
        R_body_to_world=R_body_to_world,
    )

    assert torch.allclose(propagated, torch.zeros(3), atol=1e-6)
