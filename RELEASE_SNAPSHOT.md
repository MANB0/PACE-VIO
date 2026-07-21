# Validated real-time T2 snapshot

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

The principal entry points are:

- `Scripts/run_real_t2_pipeline.py`
- `Scripts/run_progress_dashboard.py`
- `Utility/LiveDashboard.py`
- `Utility/TwoStateVIO.py`
- `Utility/CompressedUVDFactorCache.py`

The regression contract is in:

- `Scripts/UnitTest/test_live_t2_raw_contract.py`

## Freeze provenance

- Frozen source files: 576
- Original source archive SHA-256:
  `6cb095a62184aeedc8014cbfe24716ab8d18d8af42ad3e16aa4ab300051da7b3`
- Per-file hashes: `FROZEN_SOURCE_MANIFEST.sha256`

The two pretrained model files are intentionally not stored in Git because
one exceeds GitHub's normal 100 MB object limit. Download them using the links
in the upstream `README.md` and place them at:

- `Model/MACVO_FrontendCov.pth`
- `Model/MACVO_posenet.pkl`

Their exact validated hashes remain in `FROZEN_SOURCE_MANIFEST.sha256`.

## Verification

```bash
pytest -q -o addopts= Scripts/UnitTest/test_live_t2_raw_contract.py
```

The frozen environment reports `5 passed`. The `addopts` override disables the
repository-wide jaxtyping import hook, whose installed Typeguard version cannot
parse the legacy shape annotation syntax during collection.

This snapshot does not include datasets, generated results, or model weights.
The field-dataset conversion helper and sequence description present in the
frozen tree are retained, but they do not modify or participate in the
validated real-time T2 production path.
