# V3b Murky Ablation D4 — Run Instructions

## Script Path

```
Scripts/run_v3b_murky_ablation_d4_sequential.py
```

## Manual Launch Command

```bash
conda activate macvo
python Scripts/run_v3b_murky_ablation_d4_sequential.py
```

### Dry-run (print plan, do not execute)

```bash
python Scripts/run_v3b_murky_ablation_d4_sequential.py --dry-run
```

## Output Directory

```
Results/v3b_murky_ablation_d4_YYYYMMDD_HHMMSS/
```

Example: `Results/v3b_murky_ablation_d4_20260520_171500/`

## What the Script Does

1. Runs **murky_coast × 3 trials** strictly sequentially.
2. Each trial uses `--adaptive-v3b --v3b-visual-collapse-sustain 1` (D4 ablation).
3. All other V3b thresholds remain at default values.
4. MACVO's own tqdm progress bar is preserved — output streams directly to the terminal.
5. After each run, files are flattened and `completed.ok` is written.

## Strictly Sequential — No Parallelism

- ✅ 3 runs × 1 scene only, one after another.
- ✅ Second trial starts only after the first finishes.
- ✅ No multiprocessing, no concurrent.futures, no xargs -P, no GNU parallel.

## MACVO tqdm Preserved

The script uses `subprocess.run(..., stdout=None, stderr=None)` so MACVO's stdout connects directly to the parent terminal. This preserves the TTY and allows MACVO's tqdm progress bar to update in-place with `\r` carriage returns.

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

## After Runs Complete

Tell the agent:

> murky ablation D4 run 已完成，开始 analyze.

This triggers the analysis phase: comparing D4 ATE against baseline V3b results, checking `visual_collapse_first_trigger_pair`, `first_full_enter_pair`, and evaluating whether the early trigger reduces rotation_only frame contamination.
