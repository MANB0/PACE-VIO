# Full-sequence paper evaluation outputs

When `ref_pose.csv` is available, `Scripts/run_pace_vio.py` automatically creates
`<result-root>/<dataset>/paper_evaluation/` after a successful run.  State and
edge counts are derived from the first through last valid post-initialization
edge; no experiment length is hard coded.

The primary files are:

- `poses_final.csv`: complete-sequence final trajectory after the backend's
  native history revision.  Paper APE and RPE use this trajectory.
- `trajectory_evaluated.csv` and `ground_truth_evaluated.csv`: timestamp-paired
  final trajectories after applying exactly the declared evaluation alignment;
  these are the direct inputs for paper trajectory plots.
- `poses_online.csv`: causal state available after each backend commit.
- `poses_macvo_raw.csv`: visual-only MACVO trajectory at the IMU origin, when
  the run exported it.
- `ground_truth.csv`: normalized dataset ground truth.
- `metrics_per_state.csv`: translation, XY, rotation and axis-wise APE samples.
- `metrics_per_edge.csv`: adjacent-state translation and rotation RPE samples.
- `metrics_summary.json` and `metrics_summary.csv`: table-ready RMSE, P95,
  median and smoothness summaries for each trajectory.
- `run_summary.csv`: table-ready timing Median/P95, convergence, effective
  throughput and detector confusion statistics for the run.
- `timing_per_edge.csv`: frontend, factor construction, backend update, complete
  backend, commit latency, commit writeback and non-overlapping compute total.
- `solver_status.csv`: convergence and incremental-history status per edge.
- `motion_detection.csv` and `confusion_matrix.json`: detector outputs and,
  when explicit reference labels are supplied, TP/FN/FP/TN, precision and recall.
- `evaluation_alignment.json`: active range, reference point, world frame and
  the exact time/spatial alignment policy, including evaluated frames, states,
  edges and duration.
- `dataset_manifest.json` and `run_manifest.json`: input hashes, configuration,
  stereo-pair/IMU counts, measured durations and rates, code revision and
  hardware/software provenance.

Simulation defaults to independent first-state anchoring of estimate and GT.
No best-fit SE(3), Sim(3), scale, yaw or time adjustment is applied. A fixed
field-data time/SE(3) contract must be supplied explicitly with
`--evaluation-alignment-json`; scale is always fixed to one.
Ground-truth gaps create separate segments: APE keeps every associated sample,
while RPE and second-difference smoothness never bridge a missing interval.

## Complete experiment matrix

Copy `Config/paper_experiments.example.json`, replace only the dataset and
optional field-alignment paths, then run:

```bash
python Scripts/run_paper_experiments.py --manifest Config/paper_experiments.json
```

For every listed scene the runner executes the complete active sequence for
`Pose-iSAM2`, `UVD-iSAM2`, `PACE-Two`, `PACE-iSAM2`, and `PACE-VIO`. It never
passes a frame limit. `paper_run_summary.csv` contains the table-ready metrics
and timing statistics, while `paper_comparison_sources.json` records the exact
bundle used by each paper comparison. Per-state and per-edge samples remain in
the individual bundles for trajectory and CDF figures.
