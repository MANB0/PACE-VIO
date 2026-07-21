"""
VisualHealthGate-v3a: Binary pure/full gate via running median flow covariance.

v3a Design:
  - Only TWO online output modes: pure_macvo, full_imu
  - Primary signal: running_median_flow_cov = median(last N median_flow_cov)
  - Enter full (path A): running_median_flow_cov > flow_cov_enter_full_threshold
  - Enter full (path B): optimizer lost track (sustained optimizer skip/low matches)
  - Cov bridge: bridge short NaN gaps using last valid running median
  - Hysteresis: full_hold_counter + stable_visual_exit to prevent oscillation
  - Rotation_only and translation_only are NEVER selected online.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VisualHealthGateV3Config:
    """Configuration for VisualHealthGate-v3a."""

    flow_cov_window_size: int = 50
    flow_cov_enter_full_threshold: float = 0.15
    flow_cov_exit_full_threshold: float = 0.08
    min_full_hold_frames: int = 50

    max_cov_bridge_frames: int = 10

    # Optimizer-level lost-track trigger
    optimizer_lost_track_enter_frames: int = 5
    min_effective_visual_residuals: int = 10
    min_match_idx_after_filter: int = 10

    # Exit from optimizer-lost-track full_imu
    stable_visual_exit_frames: int = 20

    # ── Controlled experiment: force/latch modes ────────────────────
    # Used ONLY for diagnostic experiments, NOT production v3a.
    force_mode: str = ""                # "pure", "full", or "" (normal)
    force_pure_for_first_n_pairs: int = 0  # force pure for first N pairs
    latch_full_on_first_trigger: bool = False  # once full, always full
    disable_exit_full: bool = False     # never exit full_imu once entered


@dataclass
class V3State:
    """Internal state for v3a hysteresis."""
    flow_cov_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    current_mode: str = "pure_macvo"
    entered_by_optimizer_lost_track: bool = False
    full_hold_counter: int = 0
    last_valid_flow_cov: float = float("nan")
    last_valid_running_median: float = float("nan")
    cov_missing_counter: int = 0
    optimizer_lost_track_counter: int = 0
    optimizer_lost_track_active: bool = False
    stable_visual_counter: int = 0


@dataclass
class V3Decision:
    """Output of VisualHealthGateV3 per frame pair."""
    mode: str = "pure_macvo"
    use_imu_rotation: bool = False
    use_imu_translation: bool = False
    reason: str = "default_pure"
    hysteresis_state: str = "init"

    median_flow_cov: float = float("nan")
    running_median_flow_cov: float = float("nan")
    flow_cov_window_size: int = 0
    flow_cov_enter_full_threshold: float = 0.0
    flow_cov_exit_full_threshold: float = 0.0

    last_valid_flow_cov: float = float("nan")
    last_valid_running_median_flow_cov: float = float("nan")
    cov_missing: bool = False
    cov_missing_counter: int = 0
    max_cov_bridge_frames: int = 0
    cov_missing_bridge_active: bool = False

    num_visual_residuals: int = 0
    num_selected_keypoints: int = 0
    visual_lost_track: bool = False
    lost_track_counter: int = 0
    lost_track_enter_frames: int = 0

    optimizer_skipped: bool = False
    match_idx_size_after_filter: int = 0
    num_keypoints_candidate: int = 0
    min_num_point: int = 10

    stable_visual_counter: int = 0
    stable_visual_exit_frames: int = 0

    full_hold_counter: int = 0
    current_mode_duration: int = 0
    fallback_triggered: bool = False


class VisualHealthGateV3:
    """v3a: Binary pure/full gate with cov bridge and optimizer-level lost-track trigger."""

    def __init__(self, config: Optional[VisualHealthGateV3Config] = None):
        self.cfg = config or VisualHealthGateV3Config()
        self.state = V3State()
        self.state.flow_cov_history = deque(maxlen=self.cfg.flow_cov_window_size)

    def update(self, signals: dict) -> V3Decision:
        d = V3Decision()
        d.flow_cov_window_size = self.cfg.flow_cov_window_size
        d.flow_cov_enter_full_threshold = self.cfg.flow_cov_enter_full_threshold
        d.flow_cov_exit_full_threshold = self.cfg.flow_cov_exit_full_threshold
        d.max_cov_bridge_frames = self.cfg.max_cov_bridge_frames
        d.stable_visual_exit_frames = self.cfg.stable_visual_exit_frames
        d.lost_track_enter_frames = self.cfg.optimizer_lost_track_enter_frames

        # ── Controlled experiment: force pure for first N pairs ─────
        if self.cfg.force_pure_for_first_n_pairs > 0:
            # Count calls via a simple counter stored on state
            if not hasattr(self.state, '_pair_call_count'):
                self.state._pair_call_count = 0
            self.state._pair_call_count += 1
            if self.state._pair_call_count <= self.cfg.force_pure_for_first_n_pairs:
                d.mode = "pure_macvo"
                d.use_imu_rotation = False
                d.use_imu_translation = False
                d.reason = "force_pure_first_n"
                d.hysteresis_state = "forced_pure"
                return d

        # ── Controlled experiment: force full_imu from start ─────────
        if self.cfg.force_mode == "full":
            d.mode = "full_imu"
            d.use_imu_rotation = True
            d.use_imu_translation = True
            d.reason = "force_full"
            d.hysteresis_state = "forced_full"
            return d

        # ── Controlled experiment: force pure_macvo ──────────────────
        if self.cfg.force_mode == "pure":
            d.mode = "pure_macvo"
            d.use_imu_rotation = False
            d.use_imu_translation = False
            d.reason = "force_pure"
            d.hysteresis_state = "forced_pure"
            return d

        # ── Extract signals ──────────────────────────────────────────
        flow_cov = signals.get("median_flow_cov", float("nan"))
        n_vis = signals.get("num_visual_residuals", 0)
        n_kp = signals.get("num_selected_keypoints", 0)
        imu_ok = signals.get("imu_samples_available", True)
        meta_ok = signals.get("metadata_ok", True)

        # Optimizer tracking state
        opt_skip = bool(signals.get("optimizer_skipped", False))
        match_cnt = int(signals.get("match_idx_size_after_filter", 0))
        kp_cand = int(signals.get("num_keypoints_candidate", 0))
        min_pts = int(signals.get("min_num_point", 10))
        has_tracking = bool(signals.get("has_tracking_state", False))

        d.num_visual_residuals = n_vis
        d.num_selected_keypoints = n_kp
        d.optimizer_skipped = opt_skip
        d.match_idx_size_after_filter = match_cnt
        d.num_keypoints_candidate = kp_cand
        d.min_num_point = min_pts

        # Validate flow_cov
        flow_cov_valid = False
        fc_val = float("nan")
        if flow_cov is not None:
            try:
                fc_val = float(flow_cov)
                if not math.isnan(fc_val) and not math.isinf(fc_val):
                    flow_cov_valid = True
            except (TypeError, ValueError):
                pass
        d.median_flow_cov = fc_val if flow_cov_valid else float("nan")

        # Safety: missing IMU/metadata
        if not imu_ok or not meta_ok:
            d.mode = "pure_macvo"
            d.use_imu_rotation = False
            d.use_imu_translation = False
            d.reason = "imu_missing_fallback"
            d.hysteresis_state = "safety_fallback"
            d.fallback_triggered = True
            self._reset_state()
            return d

        # ── Optimizer-level lost-track detection ─────────────────────
        tracking_failed = (
            opt_skip
            or match_cnt < self.cfg.min_match_idx_after_filter
            or n_vis < self.cfg.min_effective_visual_residuals
        )

        has_opt_data = (n_vis > 0 or n_kp > 0 or match_cnt > 0 or has_tracking)

        if has_opt_data:
            if tracking_failed:
                self.state.optimizer_lost_track_counter += 1
            else:
                self.state.optimizer_lost_track_counter = 0
            self.state.optimizer_lost_track_active = (
                self.state.optimizer_lost_track_counter >= self.cfg.optimizer_lost_track_enter_frames
            )
        else:
            self.state.optimizer_lost_track_counter = 0
            self.state.optimizer_lost_track_active = False

        d.visual_lost_track = self.state.optimizer_lost_track_active
        d.lost_track_counter = self.state.optimizer_lost_track_counter

        # Stable visual tracking
        if not tracking_failed and flow_cov_valid:
            self.state.stable_visual_counter += 1
        else:
            self.state.stable_visual_counter = 0
        d.stable_visual_counter = self.state.stable_visual_counter

        # ── Cov bridge ───────────────────────────────────────────────
        if flow_cov_valid:
            self.state.last_valid_flow_cov = fc_val
            self.state.cov_missing_counter = 0
            d.cov_missing = False
            d.cov_missing_counter = 0
            d.cov_missing_bridge_active = False
            self.state.flow_cov_history.append(fc_val)
            hist_list = list(self.state.flow_cov_history)
            running_median = float(np.median(hist_list)) if hist_list else fc_val
            self.state.last_valid_running_median = running_median
            d.running_median_flow_cov = running_median
        else:
            self.state.cov_missing_counter += 1
            d.cov_missing = True
            d.cov_missing_counter = self.state.cov_missing_counter
            if (self.state.cov_missing_counter <= self.cfg.max_cov_bridge_frames
                    and not math.isnan(self.state.last_valid_running_median)):
                d.cov_missing_bridge_active = True
                running_median = self.state.last_valid_running_median
                d.running_median_flow_cov = running_median
            else:
                d.cov_missing_bridge_active = False
                running_median = float("nan")
                d.running_median_flow_cov = float("nan")

        d.last_valid_flow_cov = self.state.last_valid_flow_cov
        d.last_valid_running_median_flow_cov = self.state.last_valid_running_median

        # ── Decision logic ───────────────────────────────────────────
        if self.state.current_mode == "pure_macvo":

            # Path A: optimizer lost track → full_imu
            if self.state.optimizer_lost_track_active:
                self.state.current_mode = "full_imu"
                self.state.entered_by_optimizer_lost_track = True
                self.state.full_hold_counter = 0
                self.state.stable_visual_counter = 0
                d.mode = "full_imu"
                d.use_imu_rotation = True
                d.use_imu_translation = True
                d.reason = "optimizer_lost_track_enter_full"
                d.hysteresis_state = "optimizer_lost_track_full"

            # Path B: flow_cov high → full_imu
            elif (not math.isnan(running_median)
                  and running_median > self.cfg.flow_cov_enter_full_threshold):
                self.state.current_mode = "full_imu"
                self.state.entered_by_optimizer_lost_track = False
                self.state.full_hold_counter = 0
                self.state.stable_visual_counter = 0
                d.mode = "full_imu"
                d.use_imu_rotation = True
                d.use_imu_translation = True
                d.reason = "flow_cov_enter_full"
                d.hysteresis_state = "enter_full"

            # Path C: no cov, no history, not lost-track → safe pure
            elif math.isnan(running_median) and not self.state.optimizer_lost_track_active:
                d.mode = "pure_macvo"
                d.use_imu_rotation = False
                d.use_imu_translation = False
                if (d.cov_missing
                        and self.state.cov_missing_counter > self.cfg.max_cov_bridge_frames):
                    d.reason = "cov_missing_no_history_pure"
                else:
                    d.reason = "flow_cov_low_pure"
                d.hysteresis_state = "pure_stable"

            # Path D: cov below threshold → pure
            else:
                d.mode = "pure_macvo"
                d.use_imu_rotation = False
                d.use_imu_translation = False
                if d.cov_missing_bridge_active:
                    d.reason = "cov_missing_bridge"
                else:
                    d.reason = "flow_cov_low_pure"
                d.hysteresis_state = "pure_stable"

        elif self.state.current_mode == "full_imu":
            self.state.full_hold_counter += 1

            # Minimum hold
            if self.state.full_hold_counter < self.cfg.min_full_hold_frames:
                d.mode = "full_imu"
                d.use_imu_rotation = True
                d.use_imu_translation = True
                d.reason = "full_hold"
                d.hysteresis_state = "hold_full"

            # Exit from optimizer-lost-track full
            elif self.state.entered_by_optimizer_lost_track:
                can_exit = (
                    not self.state.optimizer_lost_track_active
                    and self.state.stable_visual_counter >= self.cfg.stable_visual_exit_frames
                    and not opt_skip
                    and match_cnt >= min_pts
                    and n_vis >= self.cfg.min_effective_visual_residuals
                    and not math.isnan(running_median)
                    and running_median < self.cfg.flow_cov_exit_full_threshold
                )
                if can_exit:
                    self.state.current_mode = "pure_macvo"
                    self.state.entered_by_optimizer_lost_track = False
                    self.state.full_hold_counter = 0
                    self.state.optimizer_lost_track_counter = 0
                    d.mode = "pure_macvo"
                    d.use_imu_rotation = False
                    d.use_imu_translation = False
                    d.reason = "optimizer_lost_track_recovered_exit_pure"
                    d.hysteresis_state = "exit_full"
                else:
                    d.mode = "full_imu"
                    d.use_imu_rotation = True
                    d.use_imu_translation = True
                    d.reason = "optimizer_lost_track_hold_full"
                    d.hysteresis_state = "optimizer_lost_track_full"

            # Exit from flow_cov full
            else:
                if (not math.isnan(running_median)
                        and running_median < self.cfg.flow_cov_exit_full_threshold
                        and not opt_skip
                        and match_cnt >= min_pts):
                    self.state.current_mode = "pure_macvo"
                    self.state.entered_by_optimizer_lost_track = False
                    self.state.full_hold_counter = 0
                    d.mode = "pure_macvo"
                    d.use_imu_rotation = False
                    d.use_imu_translation = False
                    d.reason = "flow_cov_exit_to_pure"
                    d.hysteresis_state = "exit_full"
                else:
                    d.mode = "full_imu"
                    d.use_imu_rotation = True
                    d.use_imu_translation = True
                    if d.cov_missing_bridge_active:
                        d.reason = "cov_missing_short_keep"
                    else:
                        d.reason = "flow_cov_keep_full"
                    d.hysteresis_state = "full_stable"
        else:
            self._reset_state()
            d.mode = "pure_macvo"
            d.use_imu_rotation = False
            d.use_imu_translation = False
            d.reason = "unknown_state_fallback"
            d.hysteresis_state = "fallback"
            d.fallback_triggered = True

        d.full_hold_counter = self.state.full_hold_counter
        d.current_mode_duration = self.state.full_hold_counter

        # ── Controlled experiment: latch on first full ───────────────
        if self.cfg.latch_full_on_first_trigger:
            if d.mode == "full_imu":
                self.state._latched = True
            if getattr(self.state, '_latched', False):
                d.mode = "full_imu"
                d.use_imu_rotation = True
                d.use_imu_translation = True
                d.reason = d.reason + "_latched"

        # ── Controlled experiment: disable exit ──────────────────────
        if self.cfg.disable_exit_full and self.state.current_mode == "full_imu":
            d.mode = "full_imu"
            d.use_imu_rotation = True
            d.use_imu_translation = True
            if "exit" in d.reason or "recovered" in d.reason:
                d.reason = "exit_blocked_keep_full"

        return d

    def _reset_state(self) -> None:
        self.state.current_mode = "pure_macvo"
        self.state.entered_by_optimizer_lost_track = False
        self.state.full_hold_counter = 0
        self.state.optimizer_lost_track_counter = 0
        self.state.optimizer_lost_track_active = False
        self.state.stable_visual_counter = 0
