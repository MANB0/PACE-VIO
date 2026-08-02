# Validated real-time PACE-VIO snapshot and rollback point

This private repository is a code-only release of the PACE-VIO pipeline
validated on 2026-07-23. `T2` is retained only in legacy compatibility paths
and immutable historical result names.

## Included validated behavior

- MACVO processes the real stereo frames before the VIO backend consumes the
  matching visual output and IMU interval.
- The online PACE-VIO backend uses the compressed UVD visual factor and the standard
  local-frame IMU preintegration path.
- The optional iSAM2 backend consumes the same compressed UVD, local-frame IMU,
  and bias random-walk factor packets without first running the two-state
  optimizer.
- The dashboard publishes independent `MACVO raw`, `VIO committed`, and GT
  trajectories at the agreed IMU-center coordinate contract.
- The dashboard includes stereo images, continuously retained IMU samples,
  pan/zoom/reset controls, pipeline status lights, and a draggable replay
  timeline that works while new frames continue to arrive.
- Dataset metadata uses continuous-time IMU noise densities and one 4x4
  `T_CI`, with `p_C = T_CI p_I`. Raw IMU FLU samples remain in their native
  frame during initialization and preintegration.

The minimal branch entry points are:

- `Scripts/run_pace_vio.py`
- `Scripts/check_runtime.py`
- `Scripts/download_models.py`
- `Utility/LiveDashboard.py`
- `Utility/TwoStateVIO.py`
- `Utility/PACEISAM2Backend.py`
- `Utility/PACEFactorPacket.py`
- `Utility/CompressedUVDFactorCache.py`

The regression contracts are in `Scripts/UnitTest`, including live frontend
ordering, static initialization modes, the single-`T_CI` frame contract,
factor-packet consistency, and the iSAM2 backend.

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
python -m pytest -q Scripts/UnitTest
```

The validated environment reports `32 passed`. A 100-frame smoke replay using
the real MACVO frontend, fixed three-second IMU initialization, and the iSAM2
backend completed successfully on 2026-07-23. Its runtime contract reported:

- `live MACVO stereo frontend (no visual cache)`
- `isam2 + PACE compressed-UVD factor packets`
- `standard_local_frame_preintegration`
- `IMU center for VIO output`

Model weights remain intentionally excluded from Git. The bootstrap/download
scripts fetch and verify the required model before runtime.

The full pre-trim source remains on `main` and the immutable rollback tag
`realtime-t2-full-20260721`. This branch does not include datasets, generated
results, model weights, or historical diagnostic scripts.
