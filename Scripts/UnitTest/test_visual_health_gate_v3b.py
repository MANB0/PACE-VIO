from Utility.VisualHealthGateV3b import V3bConfig, VisualHealthGateV3b


def _signals(
    *,
    n_vis=200,
    r_p=0.0,
    imu_trans_loss=0.0,
    r_R=0.0,
    flow_cov=0.0,
    optimizer_skipped=False,
    has_tracking_state=True,
):
    return {
        "num_visual_residuals": n_vis,
        "median_flow_cov": flow_cov,
        "r_R_whitened_norm": r_R,
        "r_p_whitened_norm": r_p,
        "imu_trans_loss": imu_trans_loss,
        "visual_loss_per_residual": 0.0,
        "total_loss": 0.0,
        "optimizer_skipped": optimizer_skipped,
        "match_idx_size_after_filter": n_vis,
        "min_num_point": 10,
        "has_tracking_state": has_tracking_state,
    }


def test_two_level_severe_vc_enters_full_imu_probation():
    gate = VisualHealthGateV3b(
        V3bConfig(
            vc_mode="two_level",
            vc_severe_num_res_thr=30,
            vc_severe_sustain=1,
            fd_grace_enabled=False,
        )
    )

    decision = gate.update(_signals(n_vis=10), pair_id=1)

    assert decision.mode == "full_imu_probation"
    assert decision.use_imu_rotation is True
    assert decision.use_imu_translation is True
    assert decision.visual_collapse_triggered is True
    assert decision.visual_collapse_trigger_source == "severe"
    assert decision.reason == "visual_collapse_enter_full"


def test_two_level_zero_residual_initial_placeholder_does_not_enter_full_imu():
    gate = VisualHealthGateV3b(
        V3bConfig(
            vc_mode="two_level",
            vc_severe_num_res_thr=30,
            vc_severe_sustain=1,
            fd_grace_enabled=False,
        )
    )

    decision = gate.update(_signals(n_vis=0, has_tracking_state=False), pair_id=0)

    assert decision.mode == "rotation_only"
    assert decision.visual_collapse_raw is False
    assert decision.visual_collapse_triggered is False


def test_two_level_zero_residual_optimizer_skip_is_severe_visual_collapse():
    gate = VisualHealthGateV3b(
        V3bConfig(
            vc_mode="two_level",
            vc_severe_num_res_thr=30,
            vc_severe_sustain=1,
            fd_grace_enabled=False,
        )
    )

    decision = gate.update(_signals(n_vis=0, optimizer_skipped=True), pair_id=1)

    assert decision.mode == "full_imu_probation"
    assert decision.use_imu_translation is True
    assert decision.severe_vc_raw is True
    assert decision.visual_collapse_triggered is True
    assert decision.visual_collapse_trigger_source == "severe"


def test_full_divergence_exit_uses_existing_pair_update_semantics():
    gate = VisualHealthGateV3b(
        V3bConfig(
            vc_mode="two_level",
            vc_severe_num_res_thr=30,
            vc_severe_sustain=1,
            full_div_sustain=10,
            full_divergence_cooldown_frames=30,
            fd_grace_enabled=False,
        )
    )

    gate.update(_signals(n_vis=10), pair_id=1)

    last_full = None
    for pair_id in range(2, 12):
        last_full = gate.update(
            _signals(n_vis=200, r_p=11.0, imu_trans_loss=0.03),
            pair_id=pair_id,
        )

    assert last_full is not None
    assert last_full.mode == "full_imu_probation"
    assert last_full.full_divergence_counter == 10
    assert last_full.full_divergence_triggered is True

    exit_decision = gate.update(_signals(n_vis=200), pair_id=12)

    assert exit_decision.mode == "pure_macvo"
    assert exit_decision.reason == "full_divergence_exit_to_pure"
    # Cooldown is set during the transition and decremented once before logging.
    assert exit_decision.fd_cooldown_config == 30
    assert exit_decision.full_divergence_cooldown_remaining == 29
    assert exit_decision.cooldown_active is True
