# Validated real-time T2 snapshot and rollback point

This private repository is a code-only release of the MACVO + IMU real-time
T2 pipeline frozen on 2026-07-21.

## Included validated behavior

- MACVO processes the real stereo frames before the VIO backend consumes the
  matching visual output and IMU interval.
- The online T2 backend uses the compressed UVD visual factor and the standard
  local-frame IMU preintegration path.
- The dashboard publishes independent `MACVO raw`, `VIO committed`, and GT
  trajectories at the agreed IMU-center coordinate contract.
- The dashboard includes stereo images, continuously retained IMU samples,
  pan/zoom/reset controls, pipeline status lights, and a draggable replay
  timeline that works while new frames continue to arrive.

The minimal branch entry points are:

- `Scripts/run_realtime_t2.py`
- `Scripts/check_runtime.py`
- `Scripts/download_models.py`
- `Utility/LiveDashboard.py`
- `Utility/TwoStateVIO.py`
- `Utility/CompressedUVDFactorCache.py`

The regression contract is in:

- `Scripts/UnitTest/test_live_t2_raw_contract.py`

## Freeze provenance

- Frozen source files: 576
- Original source archive SHA-256:
  `6cb095a62184aeedc8014cbfe24716ab8d18d8af42ad3e16aa4ab300051da7b3`
- The full source tree and its per-file manifest remain on the rollback tag.

The only production model is intentionally not stored in Git. Download and
verify it with `python Scripts/download_models.py`:

- `Model/MACVO_FrontendCov.pth`

Validated model SHA-256:
`bec6edd7e195bab863132f1e9659cdd26e6eaeae7cfd24a626828de294cf5b3a`.

## Verification

```bash
pytest -q
```

The frozen environment reports `5 passed`.

The full pre-trim source remains on `main` and the immutable rollback tag
`realtime-t2-full-20260721`. This branch does not include datasets, generated
results, model weights, or historical diagnostic scripts.
