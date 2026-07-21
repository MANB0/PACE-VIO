"""
VisualHealthGate-v2: Covariance-aware adaptive mode selector.

Uses frontend flow covariance (median_flow_cov) as primary trigger
for visual degradation detection. Retains hysteresis and safety
fallbacks from v1.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class V2State:
    """Internal state for v2 hysteresis."""
    cov_degraded_counter: int = 0
    cov_recovery_counter: int = 0
    translation_hold_counter: int = 0
    translation_active: bool = False
    prev_mode: str = "rotation_only"
    in_lost_track: bool = False


@dataclass
class VisualHealthGateV2Config:
    """Configuration for VisualHealthGate-v2."""

    # Default mode
    default_mode: str = "rotation_only"

    # Flow covariance thresholds
    flow_cov_enter_threshold: float = 0.1   # above → degraded visual
    flow_cov_exit_threshold: float = 0.05   # below → healthy visual

    # Hysteresis
    cov_degrade_enter_frames: int = 3       # consecutive degraded frames to enter
    cov_degrade_exit_frames: int = 20       # consecutive healthy frames to exit
    translation_min_hold_frames: int = 50   # minimum frames to hold translation

    # Safety
    r_p_conflict_threshold: float = 2.0         # disable translation if r_p above
    r_R_conflict_threshold: float = 1.0         # disable IMU rotation if r_R above
    min_keypoints: int = 20                     # lost track if below
    min_visual_residuals: int = 20


class VisualHealthGateV2:
    """v2: Flow-covariance gate with dual-threshold hysteresis."""

    def __init__(self, config: Optional[VisualHealthGateV2Config] = None):
        self.cfg = config or VisualHealthGateV2Config()
        self.state = V2State()

    def update(self, signals: dict) -> "AdaptiveDecision":
        """Evaluate per-frame-pair signals and return AdaptiveDecision."""
        # Import here to avoid circular dependency
        from Utility.VisualHealthGate import AdaptiveDecision

        d = AdaptiveDecision(mode=self.cfg.default_mode)

        # ── Extract signals ──────────────────────────────────────────────
        flow_cov = signals.get("median_flow_cov", float("nan"))
        n_vis = signals.get("num_visual_residuals", 0)
        n_kp = signals.get("num_selected_keypoints", signals.get("num_valid_points", 0))
        r_p = signals.get("r_p_whitened_norm", float("nan"))
        imu_ok = signals.get("imu_samples_available", True)
        meta_ok = signals.get("metadata_ok", True)
        v_loss = signals.get("visual_loss_raw_sum", float("inf"))
        per_loss = v_loss / max(n_vis, 1) if n_vis > 0 else float("inf")

        # ── Safety: missing data ─────────────────────────────────────────
        if self.cfg.default_mode != "pure_macvo":
            if not imu_ok or not meta_ok:
                d.mode = "pure_macvo"
                d.use_imu_rotation = False
                d.use_imu_translation = False
                d.reason = "imu_missing_fallback"
                d.hysteresis_state = "safety_fallback"
                return d

        # ── Safety: lost track ───────────────────────────────────────────
        if n_vis < self.cfg.min_visual_residuals or n_kp < self.cfg.min_keypoints:
            self.state.in_lost_track = True
            d.mode = "translation_only"
            d.use_imu_rotation = False
            d.use_imu_translation = True
            d.reason = "lost_track_translation"
            d.hysteresis_state = "lost_track"
            d.visual_health_score = 0.0
            d.degeneracy_score = 1.0
            return d

        self.state.in_lost_track = False

        # ── Safety: translation conflict ─────────────────────────────────
        trans_conflict = False
        if not (math.isnan(r_p) if isinstance(r_p, float) else False):
            try:
                if float(r_p) > self.cfg.r_p_conflict_threshold:
                    trans_conflict = True
            except (TypeError, ValueError):
                pass

        # ── Check if flow_cov is NaN ─────────────────────────────────────
        flow_cov_valid = not (math.isnan(flow_cov) if isinstance(flow_cov, float) else False)
        if not flow_cov_valid:
            # Fallback: use visual_loss_per_residual heuristic
            if per_loss > 0.05 and n_vis > self.cfg.min_visual_residuals:
                d.mode = "rotation_only"
                d.use_imu_rotation = True
                d.use_imu_translation = False
                d.reason = "cov_missing_fallback"
                d.hysteresis_state = "cov_missing"
            else:
                d.mode = self.cfg.default_mode
                d.use_imu_rotation = True
                d.use_imu_translation = False
                d.reason = "cov_missing_fallback"
                d.hysteresis_state = "cov_missing"
            return d

        # ── Primary decision: flow covariance ────────────────────────────
        try:
            fc = float(flow_cov)
        except (TypeError, ValueError):
            fc = 0.0

        is_degraded = fc > self.cfg.flow_cov_enter_threshold
        is_healthy = fc < self.cfg.flow_cov_exit_threshold

        # Update hysteresis counters
        if is_degraded:
            self.state.cov_degraded_counter += 1
            self.state.cov_recovery_counter = 0
        else:
            self.state.cov_degraded_counter = 0

        if is_healthy:
            self.state.cov_recovery_counter += 1
        else:
            self.state.cov_recovery_counter = 0

        # ── Decision logic ──────────────────────────────────────────────
        if trans_conflict:
            # Safety override: translation conflict → back to rotation
            d.mode = "rotation_only"
            d.use_imu_rotation = True
            d.use_imu_translation = False
            d.reason = "rp_conflict_rotation_fallback"
            d.hysteresis_state = "conflict_fallback"
            self.state.translation_active = False
            self.state.translation_hold_counter = 0
            return d

        # Check if we should enter translation mode
        if self.state.cov_degraded_counter >= self.cfg.cov_degrade_enter_frames:
            if not self.state.translation_active:
                self.state.translation_active = True
                self.state.translation_hold_counter = 0

        if self.state.translation_active:
            self.state.translation_hold_counter += 1

        # Check if we should exit translation mode
        if self.state.translation_active:
            if self.state.cov_recovery_counter >= self.cfg.cov_degrade_exit_frames and \
               self.state.translation_hold_counter >= self.cfg.translation_min_hold_frames:
                self.state.translation_active = False
                self.state.translation_hold_counter = 0

        # ── Output decision ──────────────────────────────────────────────
        if self.state.translation_active:
            d.mode = "translation_only"
            d.use_imu_rotation = False
            d.use_imu_translation = True
            if self.state.translation_hold_counter < self.cfg.translation_min_hold_frames:
                d.reason = "translation_hold"
                d.hysteresis_state = "hold_translation"
            elif is_degraded:
                d.reason = "cov_high_translation"
                d.hysteresis_state = "degraded"
            else:
                d.reason = "translation_hold"
                d.hysteresis_state = "hold_translation"
        else:
            if is_healthy:
                d.mode = self.cfg.default_mode  # rotation_only
                d.use_imu_rotation = True
                d.use_imu_translation = False
                d.reason = "cov_low_rotation"
                d.hysteresis_state = "healthy"
            elif self.state.cov_recovery_counter > 0:
                d.mode = self.cfg.default_mode
                d.use_imu_rotation = True
                d.use_imu_translation = False
                d.reason = "cov_recovery_pending"
                d.hysteresis_state = "recovering"
            else:
                d.mode = self.cfg.default_mode
                d.use_imu_rotation = True
                d.use_imu_translation = False
                d.reason = "cov_low_rotation"
                d.hysteresis_state = "neutral"

        # ── Health scores (for logging) ──────────────────────────────────
        d.visual_health_score = 1.0 if is_healthy else (0.0 if is_degraded else 0.5)
        d.degeneracy_score = 1.0 - d.visual_health_score

        return d
