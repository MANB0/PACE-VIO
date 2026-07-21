# V3b Murky Ablation D2+D4 — Run Instructions

## Script Path

```
Scripts/run_v3b_murky_ablation_d2d4_sequential.py
```

## Manual Launch Command

```bash
conda activate macvo
python Scripts/run_v3b_murky_ablation_d2d4_sequential.py
```

### Dry-run (print plan, do not execute)

```bash
python Scripts/run_v3b_murky_ablation_d2d4_sequential.py --dry-run
```

## Output Directory

```
Results/v3b_murky_ablation_d2d4_YYYYMMDD_HHMMSS/
```

Example: `Results/v3b_murky_ablation_d2d4_20260521_120000/`

Structure:
```
Results/v3b_murky_ablation_d2d4_20260521_120000/
  trial_1/
    murky_coast/
      poses.csv
      frame_pair_diagnostics.csv
      adaptive_decisions.csv
      config.yaml
      run.log
      completed.ok
      ...
  trial_2/
  trial_3/
```

## What the Script Does

1. Runs **murky_coast × 3 trials** strictly sequentially.
2. Each trial uses:
   - `--adaptive-v3b` — enables V3b state machine
   - `--v3b-d2-rerun-on-vc` — enables D2 (re-run current pair with full_imu on VC trigger)
   - `--v3b-visual-collapse-sustain 1` — enables D4 (sustain=1)
3. All other V3b thresholds remain at default values.
4. MACVO's own tqdm progress bar is preserved — output streams directly to the terminal.
5. After each run, files are flattened and `completed.ok` is written.

## Strictly Sequential — No Parallelism

- ✅ 3 runs × 1 scene only, one after another.
- ✅ Second trial starts only after the first finishes.
- ✅ No multiprocessing, no concurrent.futures, no xargs -P, no GNU parallel.

## MACVO tqdm Preserved

The script uses `subprocess.run(..., stdout=None, stderr=None)` so MACVO's stdout connects directly to the parent terminal. This preserves the TTY and allows MACVO's tqdm progress bar to update in-place with `\r` carriage returns.

No outer tqdm wrapper is used — only MACVO's own internal progress bars.

## Failure Handling

If a run fails (non-zero exit code), the script immediately aborts and prints:

- Config (D2+D4)
- Scene
- Trial
- Run directory
- run.log path
- Return code

Fix the issue and re-run — completed runs will be skipped via `completed.ok`.

## Interrupt and Resume

- Press `Ctrl+C` to interrupt safely.
- Completed runs are saved; re-run the script to continue from where you left off.
- The script checks for `completed.ok` markers and skips already-finished runs.

## D2 Verification (adaptive_decisions.csv)

After each run, the following D2 fields should be checked:

| Field | Expected Value |
|-------|----------------|
| `d2_rerun_enabled` | `1` |
| `d2_rerun_triggered` | `1` on VC trigger pair |
| `d2_committed_result_source` | `full_imu_rerun` |
| `d2_rerun_failed` | `0` |
| `d2_post_rerun_est_delta_t_norm` | Lower than `d2_pre_rerun_est_delta_t_norm` |
| `d2_post_rerun_r_p_whitened_norm` | Lower than `d2_pre_rerun_r_p_whitened_norm` |

Completed.ok also records:
- `D2_rerun_enabled`
- `rerun_current_pair_triggered_count`
- `first_rerun_pair`
- `committed_result_source`
- `d2_rerun_failed_count`

## After Runs Complete

Tell the agent:

> murky ablation D2+D4 run 已完成，开始 analyze.

This triggers the analysis phase: comparing D2+D4 ATE against D4-only, V3b original, and fixed full baselines.
