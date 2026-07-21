"""
VisualHealthGate-v3b: 3-detector state machine for MACVO adaptive mode selection.

v3b Design:
  - DEFAULT: rotation_only
  - 3 detectors: visual_collapse, full_divergence, rot_harm
  - 3 output modes: rotation_only, pure_macvo, full_imu
  - translation_only: NEVER selected online
  - Cooldowns prevent immediate re-entry after fallback

Detectors:
  1. visual_collapse: num_visual_residuals < 50 sustained 5  → enter full_imu
  2. full_divergence: r_p > 10 AND imu_trans_loss > 0.02 sustained 10 → exit to pure
  3. rot_harm:       r_R > 1.2 AND flow_cov > 1.0 sustained 20 → exit to pure

State machine:
  DEFAULT: rotation_only
    ├─[visual_collapse]→ full_imu_probation
    │    └─[full_divergence]→ pure_macvo + cooldown
    └─[rot_harm]→ pure_macvo + cooldown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class V3bConfig:
    """Configuration for VisualHealthGate-v3b."""

    # ── Default mode ──────────────────────────────────────────────────
    default_mode: str = "rotation_only"  # "rotation_only" | "pure_macvo"

    # ── Visual Collapse Detector ──────────────────────────────────────
    # Mode: "single_level" (original V3b) or "two_level" (Rule B)
    vc_mode: str = "single_level"  # "single_level" | "two_level"

    # single_level (original V3b): num_vis < thr × sustain → VC triggered
    visual_collapse_num_res_thr: int = 50
    visual_collapse_sustain: int = 5

    # two_level (Rule B): severe OR mild → VC triggered
    vc_severe_num_res_thr: int = 30
    vc_severe_sustain: int = 1
    vc_mild_num_res_thr: int = 50
    vc_mild_sustain: int = 5

    # ── Full Divergence Detector ──────────────────────────────────────
    full_div_rp_thr: float = 10.0
    full_div_imu_trans_loss_thr: float = 0.02
    full_div_sustain: int = 10

    # ── Rot Harm Detector ─────────────────────────────────────────────
    rot_harm_rR_thr: float = 1.2
    rot_harm_flow_cov_thr: float = 1.0
    rot_harm_sustain: int = 20

    # ── Cooldowns ─────────────────────────────────────────────────────
    rot_harm_cooldown_frames: int = 100
    full_divergence_cooldown_frames: int = 100

    # ── Full IMU probation ────────────────────────────────────────────
    full_imu_probation_min_frames: int = 30

    # ── Force/override (for controlled experiments only) ──────────────
    force_mode: str = ""  # "rotation_only", "pure_macvo", "full_imu", or "" (normal)

    # ── Ablation: velocity reset on full_imu entry ───────────────────
    reset_velocity_on_full_enter: bool = False
    velocity_reset_strategy: str = "zero"  # "zero" | "imu_aligned" (future)

    # ── Ablation D2: rerun current pair with full_imu on VC trigger ───
    d2_rerun_on_vc: bool = False  # D2 only
    d2_vc_sustain: int | None = None  # if set, override VC sustain for D2 (D2+D4)

    # ── FD-E: full_imu grace period (V3b++ Phase 1a) ─────────────────
    fd_grace_enabled: bool = False  # Default OFF — preserves V3b+ Rule B behavior
    fd_grace_period: int = 30       # Number of frames to suppress FD after entering full_imu


@dataclass
class V3bState:
    """Internal state for v3b state machine."""

    # Current mode
    current_mode: str = "rotation_only"
    mode_duration: int = 0
    previous_mode: str = ""
    state_name: str = "init"

    # Detector counters (sustained trigger)
    visual_collapse_counter: int = 0
    full_divergence_counter: int = 0
    rot_harm_counter: int = 0

    # Two-level VC (Rule B)
    severe_vc_counter: int = 0
    mild_vc_counter: int = 0

    # Detector raw (this frame)
    visual_collapse_raw: bool = False
    severe_vc_raw: bool = False
    mild_vc_raw: bool = False
    full_divergence_raw: bool = False
    rot_harm_raw: bool = False

    # Detector triggered (sustained)
    visual_collapse_triggered: bool = False
    severe_vc_triggered: bool = False
    mild_vc_triggered: bool = False
    full_divergence_triggered: bool = False
    rot_harm_triggered: bool = False
    # Source of visual_collapse trigger
    visual_collapse_trigger_source: str = ""  # "severe" | "mild" | ""

    # Cooldowns
    rot_harm_cooldown_remaining: int = 0
    full_divergence_cooldown_remaining: int = 0

    # Full IMU probation
    probation_counter: int = 0

    # Statistics
    total_mode_switches: int = 0
    first_full_enter_pair: int = -1
    first_full_exit_pair: int = -1
    first_rot_harm_pair: int = -1
    first_full_divergence_pair: int = -1

    # Per-detector first trigger tracking
    _vc_first_logged: bool = False
    _fd_first_logged: bool = False
    _rh_first_logged: bool = False

    # FD-E: full_imu episode tracking (V3b++ Phase 1a)
    full_imu_episode_frame_idx: int = 0
    fd_check_suppressed_by_grace: bool = False

    # Velocity reset tracking
    velocity_reset_triggered: bool = False
    velocity_reset_pair: int = -1
    velocity_reset_strategy_used: str = ""
    velocity_before_reset_norm: float = float("nan")
    velocity_after_reset_norm: float = float("nan")
    _velocity_reset_done_this_entry: bool = False  # prevent multiple resets per entry


@dataclass
class V3bDecision:
    """Per-frame-pair decision output of VisualHealthGateV3b."""

    mode: str = "rotation_only"
    use_imu_rotation: bool = True
    use_imu_translation: bool = False
    reason: str = "default"

    # Raw signals
    num_visual_residuals: int = 0
    median_flow_cov: float = float("nan")
    r_R_whitened_norm: float = float("nan")
    r_p_whitened_norm: float = float("nan")
    imu_trans_loss: float = float("nan")
    visual_loss_per_residual: float = float("nan")
    total_loss: float = float("nan")

    # Detector raw
    visual_collapse_raw: bool = False
    severe_vc_raw: bool = False
    mild_vc_raw: bool = False
    full_divergence_raw: bool = False
    rot_harm_raw: bool = False

    # Detector triggered
    visual_collapse_triggered: bool = False
    severe_vc_triggered: bool = False
    mild_vc_triggered: bool = False
    full_divergence_triggered: bool = False
    rot_harm_triggered: bool = False

    # Counters
    visual_collapse_counter: int = 0
    severe_vc_counter: int = 0
    mild_vc_counter: int = 0
    full_divergence_counter: int = 0
    rot_harm_counter: int = 0

    # Two-level VC config + source
    vc_mode: str = "single_level"
    visual_collapse_trigger_source: str = ""  # "severe" | "mild"
    visual_collapse_reason: str = ""  # e.g. "severe_visual_collapse_enter_full"
    severe_vc_threshold: int = 30
    severe_vc_sustain_config: int = 1
    mild_vc_threshold: int = 50
    mild_vc_sustain_config: int = 5

    # Cooldowns
    rot_harm_cooldown_remaining: int = 0
    full_divergence_cooldown_remaining: int = 0

    # State
    state_name: str = "init"
    previous_mode: str = ""
    probation_counter: int = 0
    cooldown_reason: str = ""

    # Velocity reset (ablation)
    velocity_reset_enabled: bool = False
    velocity_reset_triggered: bool = False
    velocity_reset_strategy: str = ""
    velocity_before_reset_norm: float = float("nan")
    velocity_after_reset_norm: float = float("nan")
    velocity_reset_pair: int = -1
    velocity_reset_reason: str = ""
    mode_transition_caused_velocity_reset: str = ""

    # VC sustain config (for ablation D4 logging)
    visual_collapse_sustain_config: int = 5

    # ── FD-E fields (V3b++ Phase 1a) ──────────────────────────────────
    fd_grace_enabled: bool = False
    fd_grace_period_config: int = 0
    full_imu_episode_frame_idx: int = 0
    fd_check_suppressed_by_grace: bool = False
    fd_grace_remaining: int = 0
    full_divergence_reason: str = "none"

    # ── Cooldown config (V3b++ Phase 1b) ──────────────────────────────
    fd_cooldown_config: int = 100
    rot_harm_cooldown_config: int = 100
    cooldown_active: bool = False

    # ── D2 rerun fields ──────────────────────────────────────────────
    d2_rerun_enabled: bool = False
    d2_rerun_triggered: bool = False
    d2_rerun_pair_id: int = -1
    d2_rerun_reason: str = ""
    d2_pre_rerun_mode: str = ""
    d2_post_rerun_mode: str = ""
    d2_committed_result_source: str = "original"
    d2_pre_rerun_est_delta_t_norm: float = float("nan")
    d2_post_rerun_est_delta_t_norm: float = float("nan")
    d2_pre_rerun_r_p_whitened_norm: float = float("nan")
    d2_post_rerun_r_p_whitened_norm: float = float("nan")
    d2_pre_rerun_num_vis_res: int = 0
    d2_post_rerun_num_vis_res: int = 0
    d2_rerun_failed: bool = False
    d2_rerun_failure_reason: str = ""


class VisualHealthGateV3b:
    """v3b: 3-detector state machine for adaptive mode selection."""

    def __init__(self, config: Optional[V3bConfig] = None):
        self.cfg = config or V3bConfig()
        self.state = V3bState()
        self.state.current_mode = self.cfg.default_mode

        # Map default mode
        if self.cfg.default_mode == "rotation_only":
            self.state.current_mode = "rotation_only"
        elif self.cfg.default_mode == "pure_macvo":
            self.state.current_mode = "pure_macvo"
        else:
            self.state.current_mode = "rotation_only"

    def update(self, signals: dict, pair_id: int = 0) -> V3bDecision:
        """Evaluate detectors and produce mode decision for the current frame pair.

        Args:
            signals: dict with keys: num_visual_residuals, median_flow_cov,
                     r_R_whitened_norm, r_p_whitened_norm, imu_trans_loss,
                     visual_loss_per_residual, total_loss
            pair_id: current frame pair index (for first-trigger tracking)
        """
        d = V3bDecision()
        self._update_mode_duration()

        # ── Force mode override ───────────────────────────────────────
        if self.cfg.force_mode:
            d = self._force_decision(d, self.cfg.force_mode)
            # Still track episode for consistency
            self._track_full_imu_episode()
            return d

        # ── Extract signals (handle None values safely) ─────────────────
        def _f(val, default=float("nan")):
            """Convert to float, returning default for None/NaN."""
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        n_vis = int(signals.get("num_visual_residuals") or 0)
        flow_cov = _f(signals.get("median_flow_cov"))
        r_R = _f(signals.get("r_R_whitened_norm"))
        r_p = _f(signals.get("r_p_whitened_norm"))
        imu_tl = _f(signals.get("imu_trans_loss"))
        vl_pr = _f(signals.get("visual_loss_per_residual"), default=float("inf"))
        tl = _f(signals.get("total_loss"))

        # Populate decision with raw signals
        d.num_visual_residuals = n_vis
        d.median_flow_cov = flow_cov
        d.r_R_whitened_norm = r_R
        d.r_p_whitened_norm = r_p
        d.imu_trans_loss = imu_tl
        d.visual_loss_per_residual = vl_pr if vl_pr < float("inf") else float("nan")
        d.total_loss = tl

        # ── Evaluate detectors (raw this-frame) ────────────────────────
        import math

        # Visual collapse: two-level (Rule B) or single-level (original V3b)
        d.vc_mode = self.cfg.vc_mode
        optimizer_skipped = bool(signals.get("has_tracking_state", False) and signals.get("optimizer_skipped", False))
        if self.cfg.vc_mode == "two_level":
            # ── Rule B: severe (<30×1) OR mild (<50×5) ─────────────────
            sev_raw = optimizer_skipped or (n_vis > 0 and n_vis < self.cfg.vc_severe_num_res_thr)
            mild_raw = (n_vis > 0 and n_vis < self.cfg.vc_mild_num_res_thr)
            d.severe_vc_raw = sev_raw
            d.mild_vc_raw = mild_raw
            d.visual_collapse_raw = sev_raw or mild_raw

            # Update sustained counters
            self._update_counter("severe_vc", sev_raw, self.cfg.vc_severe_sustain)
            self._update_counter("mild_vc", mild_raw, self.cfg.vc_mild_sustain)

            # Determine trigger source
            if self.state.severe_vc_triggered:
                self.state.visual_collapse_triggered = True
                self.state.visual_collapse_trigger_source = "severe"
            elif self.state.mild_vc_triggered:
                self.state.visual_collapse_triggered = True
                self.state.visual_collapse_trigger_source = "mild"
            else:
                self.state.visual_collapse_triggered = False
                self.state.visual_collapse_trigger_source = ""

            # Legacy counter for state machine compatibility
            self.state.visual_collapse_counter = max(
                self.state.severe_vc_counter, self.state.mild_vc_counter)

            # Populate two-level fields in decision
            d.severe_vc_counter = self.state.severe_vc_counter
            d.mild_vc_counter = self.state.mild_vc_counter
            d.severe_vc_triggered = self.state.severe_vc_triggered
            d.mild_vc_triggered = self.state.mild_vc_triggered
            d.severe_vc_threshold = self.cfg.vc_severe_num_res_thr
            d.severe_vc_sustain_config = self.cfg.vc_severe_sustain
            d.mild_vc_threshold = self.cfg.vc_mild_num_res_thr
            d.mild_vc_sustain_config = self.cfg.vc_mild_sustain
        else:
            # ── single_level (original V3b): <50×5 ─────────────────────
            vc_raw = optimizer_skipped or (n_vis > 0 and n_vis < self.cfg.visual_collapse_num_res_thr)
            d.visual_collapse_raw = vc_raw
            d.severe_vc_raw = False
            d.mild_vc_raw = False
            self._update_counter("visual_collapse", vc_raw, self.cfg.visual_collapse_sustain)
            d.severe_vc_counter = 0
            d.mild_vc_counter = 0
            d.severe_vc_triggered = False
            d.mild_vc_triggered = False

        # Full divergence: r_p > thr AND imu_trans_loss > thr
        fd_raw = False
        if not (math.isnan(r_p) or math.isnan(imu_tl)):
            fd_raw = (r_p > self.cfg.full_div_rp_thr and
                      imu_tl > self.cfg.full_div_imu_trans_loss_thr)
        d.full_divergence_raw = fd_raw

        # Rot harm: r_R > thr AND flow_cov > thr
        rh_raw = False
        if not (math.isnan(r_R) or math.isnan(flow_cov)):
            rh_raw = (r_R > self.cfg.rot_harm_rR_thr and
                      flow_cov > self.cfg.rot_harm_flow_cov_thr)
        d.rot_harm_raw = rh_raw

        # ── State machine ──────────────────────────────────────────────
        prev_mode = self.state.current_mode
        self._run_state_machine(pair_id)

        # ── FD-E: full_imu episode tracking (V3b++ Phase 1a) ───────────
        # MUST run BEFORE FD counter update to avoid one-frame lag
        prev_was_full_imu = prev_mode in ("full_imu_probation", "full_imu_active")
        in_full_imu_now = self.state.current_mode in ("full_imu_probation", "full_imu_active")
        if in_full_imu_now:
            if not prev_was_full_imu:
                self.state.full_imu_episode_frame_idx = 1
                if self.cfg.fd_grace_enabled:
                    # FD-E: explicitly reset FD counter on new episode entry
                    # Prevents counter leakage from pure_macvo phase (audit fix)
                    self.state.full_divergence_counter = 0
                    self.state.full_divergence_triggered = False
            else:
                self.state.full_imu_episode_frame_idx += 1
        else:
            self.state.full_imu_episode_frame_idx = 0

        # ── Update sustained counters (non-VC) ─────────────────────────
        # FD-E: during grace period, suppress FD counter and trigger
        in_grace_now = (self.cfg.fd_grace_enabled
                       and in_full_imu_now
                       and self.state.full_imu_episode_frame_idx > 0
                       and self.state.full_imu_episode_frame_idx <= self.cfg.fd_grace_period)
        if in_grace_now:
            self.state.full_divergence_raw = fd_raw
            self.state.full_divergence_counter = 0
            self.state.full_divergence_triggered = False
            self.state.fd_check_suppressed_by_grace = True
        else:
            self._update_counter("full_divergence", fd_raw, self.cfg.full_div_sustain)
            self.state.fd_check_suppressed_by_grace = False
        # Rot harm: never suppressed by FD grace
        self._update_counter("rot_harm", rh_raw, self.cfg.rot_harm_sustain)

        # ── Decrement cooldowns ────────────────────────────────────────
        if self.state.rot_harm_cooldown_remaining > 0:
            self.state.rot_harm_cooldown_remaining -= 1
        if self.state.full_divergence_cooldown_remaining > 0:
            self.state.full_divergence_cooldown_remaining -= 1

        # ── Update probation counter ───────────────────────────────────
        if self.state.current_mode in ("full_imu_probation", "full_imu_active"):
            self.state.probation_counter += 1

        # ── Populate decision output ───────────────────────────────────
        d.mode = self.state.current_mode
        d.state_name = self.state.state_name
        d.previous_mode = prev_mode
        d.probation_counter = self.state.probation_counter

        # Detectors
        d.visual_collapse_triggered = self.state.visual_collapse_triggered
        d.full_divergence_triggered = self.state.full_divergence_triggered
        d.rot_harm_triggered = self.state.rot_harm_triggered
        d.visual_collapse_trigger_source = self.state.visual_collapse_trigger_source
        # Reason annotation for VC source
        if self.state.visual_collapse_triggered:
            if self.state.visual_collapse_trigger_source == "severe":
                d.visual_collapse_reason = "severe_visual_collapse_enter_full"
            elif self.state.visual_collapse_trigger_source == "mild":
                d.visual_collapse_reason = "mild_visual_collapse_enter_full"
            else:
                d.visual_collapse_reason = "visual_collapse_enter_full"

        # Counters
        d.visual_collapse_counter = self.state.visual_collapse_counter
        d.full_divergence_counter = self.state.full_divergence_counter
        d.rot_harm_counter = self.state.rot_harm_counter

        # Cooldowns
        d.rot_harm_cooldown_remaining = self.state.rot_harm_cooldown_remaining
        d.full_divergence_cooldown_remaining = self.state.full_divergence_cooldown_remaining
        d.fd_cooldown_config = self.cfg.full_divergence_cooldown_frames
        d.rot_harm_cooldown_config = self.cfg.rot_harm_cooldown_frames
        d.cooldown_active = (self.state.rot_harm_cooldown_remaining > 0
                            or self.state.full_divergence_cooldown_remaining > 0)

        # Reason and cooldown reason
        d.reason = self.get_reason()
        d.cooldown_reason = self._get_cooldown_reason()

        # Velocity reset (ablation) — propagate state to decision
        d.velocity_reset_enabled = self.cfg.reset_velocity_on_full_enter
        d.velocity_reset_triggered = self.state.velocity_reset_triggered
        d.velocity_reset_strategy = self.state.velocity_reset_strategy_used
        d.velocity_before_reset_norm = self.state.velocity_before_reset_norm
        d.velocity_after_reset_norm = self.state.velocity_after_reset_norm
        d.velocity_reset_pair = self.state.velocity_reset_pair
        d.velocity_reset_reason = self.state.state_name
        d.mode_transition_caused_velocity_reset = (
            f"{self.state.previous_mode}->{self.state.current_mode}"
            if self.state.velocity_reset_triggered else ""
        )

        # One-shot: clear the trigger after propagating to decision
        self.state.velocity_reset_triggered = False

        # IMU switches
        if self.state.current_mode == "rotation_only":
            d.use_imu_rotation = True
            d.use_imu_translation = False
        elif self.state.current_mode == "pure_macvo":
            d.use_imu_rotation = False
            d.use_imu_translation = False
        elif self.state.current_mode in ("full_imu_probation", "full_imu_active"):
            d.use_imu_rotation = True
            d.use_imu_translation = True

        # VC sustain config (for ablation D4 logging)
        d.visual_collapse_sustain_config = self.cfg.visual_collapse_sustain

        # ── D2 rerun: populate decision fields ─────────────────────────
        d.d2_rerun_enabled = self.cfg.d2_rerun_on_vc

        # ── FD-E fields (V3b++ Phase 1a) ──────────────────────────────
        d.fd_grace_enabled = self.cfg.fd_grace_enabled
        d.fd_grace_period_config = self.cfg.fd_grace_period
        d.full_imu_episode_frame_idx = self.state.full_imu_episode_frame_idx
        d.fd_check_suppressed_by_grace = self.state.fd_check_suppressed_by_grace
        d.fd_grace_remaining = max(0, self.cfg.fd_grace_period - self.state.full_imu_episode_frame_idx) if (in_full_imu_now and self.cfg.fd_grace_enabled) else 0
        # full_divergence_reason
        if self.state.fd_check_suppressed_by_grace:
            d.full_divergence_reason = "suppressed_by_fd_grace"
        elif self.state.full_divergence_triggered:
            d.full_divergence_reason = "sustained_triggered"
        elif self.state.full_divergence_counter > 0:
            d.full_divergence_reason = "counting"
        elif d.full_divergence_raw:
            d.full_divergence_reason = "raw_true"
        else:
            d.full_divergence_reason = "raw_false"

        # D2 rerun logic (after FD-E fields populated)
        if self.cfg.d2_rerun_on_vc and self.state.visual_collapse_triggered:
            d.d2_rerun_triggered = True
            d.d2_rerun_pair_id = pair_id
            d.d2_rerun_reason = "visual_collapse_rerun_current_pair_full_imu"
            d.d2_pre_rerun_mode = prev_mode
            d.d2_post_rerun_mode = self.state.current_mode
            # Pre-rerun values from signals (post-rerun filled by MACVO)
            d.d2_pre_rerun_est_delta_t_norm = _f(signals.get("est_delta_t_norm"))
            d.d2_pre_rerun_r_p_whitened_norm = r_p
            d.d2_pre_rerun_num_vis_res = n_vis

        return d

    def _update_mode_duration(self):
        """Increment mode duration counter."""
        self.state.mode_duration += 1

    def _update_counter(self, counter_name: str, raw: bool, sustain: int):
        """Update a sustained-trigger counter. Reset on miss, trigger on sustain."""
        if raw:
            current = getattr(self.state, f"{counter_name}_counter")
            setattr(self.state, f"{counter_name}_counter", current + 1)
        else:
            setattr(self.state, f"{counter_name}_counter", 0)

        counter_val = getattr(self.state, f"{counter_name}_counter")
        triggered = counter_val >= sustain
        setattr(self.state, f"{counter_name}_triggered", triggered)
        setattr(self.state, f"{counter_name}_raw", raw)

    def _run_state_machine(self, pair_id: int):
        """Execute V3b state machine transitions."""
        current = self.state.current_mode

        # ── Priority 1: Full divergence (must exit full_imu) ──────────
        if current in ("full_imu_probation", "full_imu_active"):
            if self.state.full_divergence_triggered:
                self._transition_to("pure_macvo", "full_divergence_exit_to_pure",
                                    "full_divergence_active")
                self.state.full_divergence_cooldown_remaining = (
                    self.cfg.full_divergence_cooldown_frames)
                if self.state.first_full_exit_pair < 0:
                    self.state.first_full_exit_pair = pair_id
                if not self.state._fd_first_logged:
                    self.state._fd_first_logged = True
                    self.state.first_full_divergence_pair = pair_id
                return

        # ── Priority 2: Rot harm (must exit rotation_only) ────────────
        if current == "rotation_only" and self.state.rot_harm_cooldown_remaining <= 0:
            if self.state.rot_harm_triggered:
                self._transition_to("pure_macvo", "rot_harm_exit_to_pure",
                                    "rot_harm_active")
                self.state.rot_harm_cooldown_remaining = (
                    self.cfg.rot_harm_cooldown_frames)
                if self.state.first_rot_harm_pair < 0:
                    self.state.first_rot_harm_pair = pair_id
                return

        # ── Priority 3: Visual collapse (enter full_imu) ──────────────
        # From rotation_only (original path)
        if current == "rotation_only" and self.state.full_divergence_cooldown_remaining <= 0:
            if self.state.visual_collapse_triggered:
                self._transition_to("full_imu_probation", "visual_collapse_enter_full",
                                    "full_imu_probation")
                self.state.probation_counter = 0
                if self.state.first_full_enter_pair < 0:
                    self.state.first_full_enter_pair = pair_id
                return

        # From pure_macvo (NEW: rescue path when frontend fails while pure)
        # rot_harm_cooldown does NOT block this — only full_divergence_cooldown blocks
        if current == "pure_macvo" and self.state.full_divergence_cooldown_remaining <= 0:
            if self.state.visual_collapse_triggered:
                self._transition_to("full_imu_probation", "visual_collapse_from_pure_enter_full",
                                    "full_imu_probation")
                self.state.probation_counter = 0
                if self.state.first_full_enter_pair < 0:
                    self.state.first_full_enter_pair = pair_id
                return

        # ── Full_imu probation → active (optional) ────────────────────
        if current == "full_imu_probation":
            if self.state.probation_counter >= self.cfg.full_imu_probation_min_frames:
                if not self.state.full_divergence_triggered:
                    self.state.state_name = "full_imu_active"

        # ── Cooldown: stay in pure_macvo ──────────────────────────────
        if current == "pure_macvo":
            if self.state.rot_harm_cooldown_remaining > 0:
                self.state.state_name = "pure_cooldown_rot_harm"
            elif self.state.full_divergence_cooldown_remaining > 0:
                self.state.state_name = "pure_cooldown_full_div"
            else:
                # Cooldowns expired — could return to rotation_only
                # But we stay in pure_macvo until a future detector triggers
                self.state.state_name = "pure_idle"

        # ── Default: stay in current mode ─────────────────────────────
        if self.state.state_name == "init":
            self.state.state_name = f"{current}_stable"

    def _transition_to(self, new_mode: str, reason: str, state_name: str):
        """Execute a mode transition."""
        self.state.previous_mode = self.state.current_mode
        self.state.current_mode = new_mode
        self.state.mode_duration = 0
        self.state.state_name = state_name
        self.state.total_mode_switches += 1

        # Reset probation counter on mode change
        if new_mode == "full_imu_probation":
            self.state.probation_counter = 0

        # ── Velocity reset on full_imu entry (ablation) ──────────────
        if (self.cfg.reset_velocity_on_full_enter
                and new_mode == "full_imu_probation"
                and not self.state._velocity_reset_done_this_entry):
            self.state.velocity_reset_triggered = True
            self.state.velocity_reset_strategy_used = self.cfg.velocity_reset_strategy
            self.state._velocity_reset_done_this_entry = True
        elif new_mode != "full_imu_probation":
            # Reset the guard when exiting full_imu
            self.state._velocity_reset_done_this_entry = False
            self.state.velocity_reset_triggered = False
            self.state.velocity_before_reset_norm = float("nan")
            self.state.velocity_after_reset_norm = float("nan")

        # Store reason in state for logging
        self.state._last_reason = reason

    def get_reason(self) -> str:
        """Get the reason string for the current mode decision."""
        current = self.state.current_mode
        state_name = self.state.state_name

        if hasattr(self.state, "_last_reason"):
            return getattr(self.state, "_last_reason", state_name)

        # Fallback reasons
        if current == "rotation_only":
            return "default_rotation_only"
        elif current in ("full_imu_probation", "full_imu_active"):
            return "full_imu_active"
        elif current == "pure_macvo":
            if self.state.rot_harm_cooldown_remaining > 0:
                return "pure_rot_harm_cooldown"
            elif self.state.full_divergence_cooldown_remaining > 0:
                return "pure_full_div_cooldown"
            return "pure_idle"
        return "unknown"

    def _get_cooldown_reason(self) -> str:
        """Get cooldown reason for logging."""
        parts = []
        if self.state.rot_harm_cooldown_remaining > 0:
            parts.append(f"rot_harm({self.state.rot_harm_cooldown_remaining})")
        if self.state.full_divergence_cooldown_remaining > 0:
            parts.append(f"full_div({self.state.full_divergence_cooldown_remaining})")
        return ";".join(parts) if parts else "none"

    def _force_decision(self, d: V3bDecision, mode: str) -> V3bDecision:
        """Return a forced decision for controlled experiments."""
        d.mode = mode
        d.reason = f"force_{mode}"
        d.state_name = "forced"
        if mode == "rotation_only":
            d.use_imu_rotation = True
            d.use_imu_translation = False
        elif mode == "pure_macvo":
            d.use_imu_rotation = False
            d.use_imu_translation = False
        elif mode == "full_imu":
            d.use_imu_rotation = True
            d.use_imu_translation = True
        # FD-E fields for force mode
        d.fd_grace_enabled = self.cfg.fd_grace_enabled
        d.fd_grace_period_config = self.cfg.fd_grace_period
        d.full_imu_episode_frame_idx = self.state.full_imu_episode_frame_idx
        d.fd_check_suppressed_by_grace = self.state.fd_check_suppressed_by_grace
        # Cooldown config
        d.fd_cooldown_config = self.cfg.full_divergence_cooldown_frames
        d.rot_harm_cooldown_config = self.cfg.rot_harm_cooldown_frames
        d.cooldown_active = (self.state.rot_harm_cooldown_remaining > 0
                            or self.state.full_divergence_cooldown_remaining > 0)
        return d

    def _track_full_imu_episode(self):
        """Update full_imu episode counter for FD-E grace period (V3b++ Phase 1a).
        Simplified: just update the counter based on current mode."""
        prev_was_full_imu = getattr(self.state, '_prev_was_full_imu', False)
        in_full_imu_now = self.state.current_mode in ("full_imu_probation", "full_imu_active")
        if in_full_imu_now:
            if not prev_was_full_imu:
                self.state.full_imu_episode_frame_idx = 1
                if self.cfg.fd_grace_enabled:
                    self.state.full_divergence_counter = 0
                    self.state.full_divergence_triggered = False
            else:
                self.state.full_imu_episode_frame_idx += 1
        else:
            self.state.full_imu_episode_frame_idx = 0
        self.state._prev_was_full_imu = in_full_imu_now
