# Paper rerun contract

This repository produces every numeric input required by the current paper
except reference labels for the optional motion-detector confusion matrices.
Those labels are external annotations and may be supplied with
`motion_reference_csv` in the experiment manifest.

## Fixed experiment matrix

Each dataset is processed over its complete active sequence with five runs:

| Run key | Visual factor | Backend | Conditional velocity factor |
|---|---|---|---|
| `pose_isam2` | Pose | iSAM2 | off |
| `uvd_isam2` | UVD | iSAM2 | off |
| `pace_two_state` | PACE | Two-state | off |
| `pace_isam2` | PACE | iSAM2 | off |
| `pace_vio` | PACE | iSAM2 | causal v2 detector |

The MACVO baseline is taken from the raw visual trajectory exported by the
`pace_vio` run, so it shares the same live frontend pass and timestamp range.

## Paper mapping

| Paper result | Source runs | Exported values |
|---|---|---|
| Overall simulation/field comparison | MACVO, `pace_two_state`, `pace_isam2`, `pace_vio` | XY and 3D APE RMSE/P95, evaluated trajectories, per-state errors |
| Visual-factor selection | `pose_isam2`, `uvd_isam2`, `pace_isam2` | translation/rotation APE and RPE RMSE/P95, factor-build and iSAM2-update Median/P95, convergence rate |
| Backend selection | `pace_two_state`, `pace_isam2` | translation/rotation APE and RPE RMSE/P95, backend-update Median/P95 |
| Conditional velocity factor | `pace_isam2`, `pace_vio` | translation/rotation APE and RPE RMSE/P95, detector decisions |
| Trajectory and CDF figures | the corresponding run bundle | aligned trajectory pairs and per-state/per-edge samples |

`first_pose` independently anchors estimate and GT only for synchronized
simulation evaluation. `fixed_se3` and `none` never receive an additional
first-pose alignment. Scale changes are rejected for every mode.

## Outputs

`paper_run_summary.csv` is the flat table index. The exact per-run sources are
recorded in `paper_comparison_sources.json`. Every run also preserves:

- final, online and raw MACVO trajectories;
- evaluated estimate/GT trajectory pairs;
- per-state APE and per-edge RPE samples;
- per-edge frontend, factor-build, backend-update and total timings;
- solver status, runtime configuration, dataset hashes and hardware/software
  provenance;
- motion-detector decisions and optional confusion statistics.
