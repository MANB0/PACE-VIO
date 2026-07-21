"""
VisualHealthGate-v1: Runtime adaptive mode selector for IMU constraints.

Decides per-frame-pair whether to use IMU rotation, IMU translation, or neither,
based only on runtime-available visual quality and motion signals.
Does NOT use GT, scene name, ATE, or oracle labels.

Design principle for v1:
  - Default: rotation_only (safest)
  - Healthy visual → rotation_only (or pure if configured)
  - Degraded visual → translation_only
  - Never selects full_imu by default (0/7 scenes optimal in oracle)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdaptiveDecision:
    """Output of VisualHealthGate per frame pair."""
    mode: str = "rotation_only"       # pure_macvo | rotation_only | translation_only | full_imu
    use_imu_rotation: bool = True
    use_imu_translation: bool = False
    reason: str = "default"
    visual_health_score: float = -1.0
    degeneracy_score: float = -1.0
    motion_abnormal_score: float = -1.0
    hysteresis_state: str = "init"


@dataclass
class VisualHealthGateConfig:
    """Configuration for VisualHealthGate-v1.

    All thresholds derived from 7×4 experiment statistics.
    """
    # Mode defaults
    default_mode: str = "rotation_only"
    healthy_mode: str = "rotation_only"   # rotation_only or pure_macvo
    degraded_mode: str = "translation_only"
    allow_full_imu: bool = False           # v1: never (oracle shows 0/7 optimal)

    # Visual health thresholds (in PER-RESIDUAL units: loss/n_vis)
    # clear_shallow: per-residual loss p50≈0.004, open_water: p50≈3.0
    min_visual_residuals: int = 50          # fewer than this → unhealthy
    healthy_per_residual_loss_max: float = 0.02   # below this → healthy
    degraded_per_residual_loss_min: float = 0.15  # above this → degraded
    healthy_depth_cov_max: float = 0.01     # m², below → healthy
    degraded_depth_cov_min: float = 0.05    # m², above → degraded

    # Motion abnormality thresholds
    est_delta_t_jump_threshold: float = 5.0  # m, above → abnormal motion
    est_delta_R_jump_threshold: float = 0.5  # rad, above → abnormal rotation

    # IMU conflict thresholds
    r_R_conflict_threshold: float = 1.0    # whitened norm above → conflict
    r_p_conflict_threshold: float = 2.0    # whitened norm above → conflict

    # Hysteresis
    degrade_enter_frames: int = 5      # consecutive degraded before enabling translation
    degrade_exit_frames: int = 10      # consecutive healthy before disabling translation
    translation_min_hold_frames: int = 10  # minimum frames to hold translation once active
    rotation_min_hold_frames: int = 5

    # Safety
    fallback_mode: str = "rotation_only"
    disable_translation_when_visual_healthy: bool = True
    disable_imu_on_missing_data: bool = True
    disable_translation_on_conflict: bool = True


class VisualHealthGate:
    """v1: Rule-based visual health gate with hysteresis."""

    def __init__(self, config: Optional[VisualHealthGateConfig] = None):
        self.cfg = config or VisualHealthGateConfig()

        # State
        self._prev_mode: str = self.cfg.default_mode
        self._translation_active: bool = False
        self._translation_frames_held: int = 0
        self._rotation_frames_held: int = 0

        # History for hysteresis
        self._degrade_history: deque[bool] = deque(maxlen=self.cfg.degrade_enter_frames)
        self._healthy_history: deque[bool] = deque(maxlen=self.cfg.degrade_exit_frames)

    def update(self, signals: dict) -> AdaptiveDecision:
        """Evaluate per-frame-pair signals and return AdaptiveDecision.

        Args:
            signals: dict with runtime-available keys:
                num_visual_residuals, num_valid_points, visual_loss,
                est_delta_t_norm, est_delta_R_angle,
                r_R_whitened_norm, r_p_whitened_norm,
                imu_samples_available (bool), metadata_ok (bool)
                (all optional; missing → use defaults)

        Returns:
            AdaptiveDecision with mode, use_imu_rotation, use_imu_translation
        """
        d = AdaptiveDecision()

        # ── Extract signals with defaults ──────────────────────────────
        n_vis = signals.get("num_visual_residuals", 0)
        n_pts = signals.get("num_valid_points", 0)
        v_loss = signals.get("visual_loss", float("inf"))
        delta_t = signals.get("est_delta_t_norm", 0.0)
        delta_R = signals.get("est_delta_R_angle", 0.0)
        r_R = signals.get("r_R_whitened_norm", float("nan"))
        r_p = signals.get("r_p_whitened_norm", float("nan"))
        imu_ok = signals.get("imu_samples_available", True)
        meta_ok = signals.get("metadata_ok", True)

        # ── Safety checks ─────────────────────────────────────────────
        if self.cfg.disable_imu_on_missing_data and (not imu_ok or not meta_ok):
            d.mode = "pure_macvo"
            d.use_imu_rotation = False
            d.use_imu_translation = False
            d.reason = "safety: imu data or metadata missing"
            d.hysteresis_state = "safety_fallback"
            self._prev_mode = d.mode
            return d

        # ── Visual health scoring ──────────────────────────────────────
        # Health score: higher = healthier visual (0..1)
        vis_score = 0.5  # neutral
        if n_vis > 0 and v_loss < float("inf"):
            # Healthy: many residuals + low loss
            loss_per_residual = v_loss / max(n_vis, 1)  # per-residual loss
            if loss_per_residual < self.cfg.healthy_per_residual_loss_max and n_vis >= self.cfg.min_visual_residuals:
                vis_score = 0.8
            elif loss_per_residual > self.cfg.degraded_per_residual_loss_min:
                vis_score = 0.2
            else:
                span = self.cfg.degraded_per_residual_loss_min - self.cfg.healthy_per_residual_loss_max
                if span > 0:
                    vis_score = 1.0 - (loss_per_residual - self.cfg.healthy_per_residual_loss_max) / span
                    vis_score = max(0.1, min(0.9, vis_score))

        d.visual_health_score = vis_score

        # ── Degeneracy scoring ─────────────────────────────────────────
        deg_score = 1.0 - vis_score  # simple inverse for v1
        d.degeneracy_score = deg_score

        # ── Motion abnormality ────────────────────────────────────────
        mot_score = 0.0
        if delta_t > self.cfg.est_delta_t_jump_threshold:
            mot_score += 0.5
        if delta_R > self.cfg.est_delta_R_jump_threshold:
            mot_score += 0.5
        d.motion_abnormal_score = min(mot_score, 1.0)

        # ── Decision logic ────────────────────────────────────────────
        is_healthy = vis_score > 0.6 and n_vis >= self.cfg.min_visual_residuals
        is_degraded = vis_score < 0.3 or mot_score > 0.5

        # Hysteresis: update history
        self._degrade_history.append(is_degraded)
        self._healthy_history.append(is_healthy)

        # Translation conflict check
        trans_conflict = False
        if self.cfg.disable_translation_on_conflict and not (math.isnan(r_p) if isinstance(r_p, float) else False):
            try:
                if r_p > self.cfg.r_p_conflict_threshold:
                    trans_conflict = True
            except (TypeError, ValueError):
                pass

        # Hysteresis: enable translation only after sustained degradation
        if self.cfg.degrade_enter_frames > 0:
            sustained_degrade = sum(self._degrade_history) >= self.cfg.degrade_enter_frames
        else:
            sustained_degrade = is_degraded

        if self.cfg.degrade_exit_frames > 0:
            sustained_healthy = sum(self._healthy_history) >= self.cfg.degrade_exit_frames
        else:
            sustained_healthy = is_healthy

        # Mode selection
        if sustained_degrade and not trans_conflict:
            # Degraded → use translation
            d.mode = self.cfg.degraded_mode
            d.use_imu_rotation = False
            d.use_imu_translation = True  # translation_only
            d.reason = "degraded visual: using translation_only"
            d.hysteresis_state = "degraded"
            self._translation_active = True
            self._translation_frames_held = 0
        elif sustained_healthy and self.cfg.disable_translation_when_visual_healthy:
            # Healthy → disable translation, keep rotation or go pure
            d.mode = self.cfg.healthy_mode  # "rotation_only" or "pure_macvo"
            if d.mode == "pure_macvo":
                d.use_imu_rotation = False
                d.use_imu_translation = False
            else:  # rotation_only
                d.use_imu_rotation = True
                d.use_imu_translation = False
            d.reason = "healthy visual: disabling translation"
            d.hysteresis_state = "healthy"
            self._translation_active = False
            self._translation_frames_held = 0
        else:
            # Default mode
            d.mode = self.cfg.default_mode  # rotation_only
            d.use_imu_rotation = True
            d.use_imu_translation = False
            d.reason = "default: rotation_only (neutral state)"
            d.hysteresis_state = "neutral"

            # Hold translation if active but not yet enough frames
            if self._translation_active:
                self._translation_frames_held += 1
                if self._translation_frames_held < self.cfg.translation_min_hold_frames:
                    d.mode = "translation_only"
                    d.use_imu_rotation = False
                    d.use_imu_translation = True
                    d.reason = "holding translation (min hold frames)"
                    d.hysteresis_state = "hold_translation"

        self._prev_mode = d.mode
        return d
