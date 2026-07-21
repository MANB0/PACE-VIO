# V3b+ D4 7×3 Formal Evaluation — Run Instructions

## Script Path

```
Scripts/run_v3bplus_d4_7x3_sequential.py
```

## Manual Launch Command

```bash
conda activate macvo
python Scripts/run_v3bplus_d4_7x3_sequential.py
```

### Dry-run (print plan, do not execute)

```bash
python Scripts/run_v3bplus_d4_7x3_sequential.py --dry-run
```

## Output Directory

```
Results/v3bplus_d4_7x3_YYYYMMDD_HHMMSS/
```

Example: `Results/v3bplus_d4_7x3_20260521_120000/`

Structure:
```
Results/v3bplus_d4_7x3_20260521_120000/
  trial_1/
    turbid_harbor/
    clear_shallow/
    deep_dark/
    caustic_shallow/
    dam_inspection/
    murky_coast/
    open_water/
  trial_2/ ...
  trial_3/ ...
```

Each scene directory contains:
```
poses.csv
frame_pair_diagnostics.csv
adaptive_decisions.csv
config.yaml
run.log
completed.ok
...
```

## What the Script Does

1. Runs **7 scenes × 3 trials = 21 runs** strictly sequentially.
2. Each trial uses `--adaptive-v3b --v3b-visual-collapse-sustain 1` (D4).
3. **D2 is NOT enabled**, velocity reset is NOT enabled, translation_only is NOT enabled.
4. All other V3b thresholds remain at default values.
5. MACVO's own tqdm progress bar is preserved — output streams directly to the terminal.
6. No outer tqdm wrapper — only simple `print()` statements for scene/trial status.

## Strictly Sequential — No Parallelism

- ✅ 21 runs across 7 scenes × 3 trials, one after another.
- ✅ All trial_1 scenes complete before trial_2 starts.
- ✅ No multiprocessing, no concurrent.futures, no xargs -P, no GNU parallel.

## MACVO tqdm Preserved — No Outer tqdm

The script uses `subprocess.run(..., stdout=None, stderr=None)` so MACVO's stdout connects directly to the parent terminal. This preserves the TTY and allows MACVO's tqdm progress bar to update in-place with `\r` carriage returns.

**No outer tqdm wrapper is used.** Only MACVO's own internal progress bars appear. The script prints simple status lines without tqdm.

## Failure Handling

If a run fails (non-zero exit code), the script immediately aborts and prints:

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

## V3b+ D4 Verification (completed.ok)

After each run, `completed.ok` records:

| Field | Expected Value |
|-------|----------------|
| `ablation_name` | `V3bplus_D4` |
| `visual_collapse_sustain_config` | `1` |
| `D2_rerun_enabled` | `false` |
| `first_full_enter_pair` | varies by scene |
| `visual_collapse_first_trigger_pair` | varies by scene |

## After Runs Complete

Tell the agent:

> V3b+ D4 7×3 run 已完成，开始 analyze.

This triggers the analysis phase: comparing V3b+ D4 ATE against V3b original 7×3, D4 murky-only, and fixed full baselines across all 7 scenes.
