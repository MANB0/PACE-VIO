# V3b+ Rule B Two-Level VC Guard — Sanity Run Instructions

## Script Path

```
Scripts/run_ruleB_sanity_sequential.py
```

## Manual Launch Command

```bash
conda activate macvo
python Scripts/run_ruleB_sanity_sequential.py
```

### Dry-run (print plan, do not execute)

```bash
python Scripts/run_ruleB_sanity_sequential.py --dry-run
```

## Output Directory

```
Results/v3bplus_ruleB_sanity_YYYYMMDD_HHMMSS/
```

Example: `Results/v3bplus_ruleB_sanity_20260521_140000/`

Structure:
```
Results/v3bplus_ruleB_sanity_20260521_140000/
  murky_coast/
    poses.csv
    frame_pair_diagnostics.csv
    adaptive_decisions.csv
    config.yaml
    run.log
    completed.ok
    ...
  open_water/
    ...
  dam_inspection/
    ...
```

## What the Script Does

1. Runs **3 scenes × 1 trial** strictly sequentially:
   - `murky_coast` — verify severe VC triggers at pair 1
   - `open_water` — verify no false VC trigger at pair 832
   - `dam_inspection` — verify no regression vs D4
2. Each run uses:
   - `--adaptive-v3b` — enables V3b state machine
   - `--v3b-vc-mode two_level` — enables Rule B
   - `--v3b-vc-severe-thr 30 --v3b-vc-severe-sustain 1` — severe VC
   - `--v3b-vc-mild-thr 50 --v3b-vc-mild-sustain 5` — mild VC
3. **D2 is NOT enabled. Velocity reset is NOT enabled.**
4. All other V3b thresholds remain at default values.
5. MACVO's own tqdm progress bar is preserved — output streams directly to the terminal.
6. No outer tqdm wrapper — only simple `print()` statements for scene status.

## Strictly Sequential — No Parallelism

- ✅ 3 runs, one after another.
- ✅ No multiprocessing, no concurrent.futures, no xargs -P, no GNU parallel.

## MACVO tqdm Preserved — No Outer tqdm

The script uses `subprocess.run(..., stdout=None, stderr=None)` so MACVO's stdout connects directly to the parent terminal. This preserves the TTY and allows MACVO's tqdm progress bar to update in-place with `\r` carriage returns.

**No outer tqdm wrapper is used.** Only MACVO's own internal progress bars appear.

## Failure Handling

If a run fails (non-zero exit code), the script immediately aborts and prints:

- Scene
- Run directory
- run.log path
- Return code

Fix the issue and re-run — completed runs will be skipped via `completed.ok`.

## Interrupt and Resume

- Press `Ctrl+C` to interrupt safely.
- Completed runs are saved; re-run the script to continue from where you left off.
- The script checks for `completed.ok` markers and skips already-finished runs.

## Verification Checklist (after all 3 runs)

| Scene | Expected VC Trigger | Expected Mode |
|-------|:---:|:---:|
| murky_coast | severe_vc_triggered=1 at pair 1 | full_imu at pair 1 |
| open_water | no VC trigger at pair 832 (38 ≥ 30) | rotation_only → pure_macvo |
| dam_inspection | severe/mild may trigger; ATM accepted | near_oracle ATE |

Check `adaptive_decisions.csv`:
```python
import pandas as pd
# Murky
ad = pd.read_csv('Results/.../murky_coast/adaptive_decisions.csv')
print(ad[ad['severe_vc_triggered']==1]['pair_id'].min())  # should be 1

# Open water
ad2 = pd.read_csv('Results/.../open_water/adaptive_decisions.csv')
print(ad2['visual_collapse_triggered'].sum())  # should be 0

# Dam
ad3 = pd.read_csv('Results/.../dam_inspection/adaptive_decisions.csv')
print(ad3['vc_mode'].iloc[0])  # should be "two_level"
```

## After Runs Complete

Tell the agent:

> Rule B sanity run 已完成，开始 analyze.
