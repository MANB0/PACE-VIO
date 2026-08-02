# PACE-VIO iSAM2 backend

This directory contains the C++/Pybind incremental backend used by the
minimal realtime entry point when `--vio-backend isam2` is selected. The
original two-state solver remains available as `--vio-backend two_state` for
controlled comparisons and rollback.

Both backends consume the same visual-inertial packet contract. The visual
payload is selected with `--visual-factor`:

- `pose`: a robust relative-pose `BetweenFactor<Pose3>`;
- `uvd`: native point-level nonlinear UVD factors, reprojected whenever iSAM2
  relinearizes the connected poses;
- `pace`: the compressed local visual information `(T_ref, A, c)` reconstructed
  from `(H, g)`; this is the production default.

Every packet also carries:

- state pose `T_WI` at the IMU origin in MACVO internal NED;
- cached preintegration residual order `[p, v, R]` and full covariance;
- bias correction Jacobian and bias random-walk covariance;
- project pose tangent `[translation, rotation]`, explicitly permuted where a
  GTSAM factor expects `[rotation, translation]`.

It intentionally does not use GTSAM's independent preintegration
implementation. The IMU, bias and coordinate contracts remain PACE-VIO's.

Build the in-process extension in the active Python environment:

```bash
export GTSAM_DIR=/absolute/path/to/lib/cmake/GTSAM  # omit for a standard install
bash Scripts/build_pace_vio_isam2.sh
python Scripts/UnitTest/test_t2_isam2_backend.py
```

Run the realtime pipeline:

```bash
python Scripts/run_pace_vio.py \
  --dataset /absolute/path/to/sequence \
  --vio-backend isam2 \
  --visual-factor pace \
  --static-init-mode fixed \
  --static-init-duration-s 3.0
```

For the controlled visual-factor comparison, replace `pace` with `pose` or
`uvd` while keeping `--vio-backend isam2` and every other option unchanged.

The standalone bundle runner below is retained for factor-level audit and
cross-language regression; it is not the production realtime path.

```bash
python Scripts/export_t2_isam2_bundle.py \
  --tensor-map /path/to/tensor_map.npz \
  --output-dir /tmp/t2_isam2_bundle \
  --start-frame 90 --end-frame 299

cmake -S cpp/t2_isam2 -B build/pace_vio_isam2 \
  -DGTSAM_DIR=/home/admin1/.local/lib/cmake/GTSAM
cmake --build build/pace_vio_isam2 -j"$(nproc)"
ctest --test-dir build/pace_vio_isam2 --output-on-failure

build/pace_vio_isam2/pace_vio_isam2_runner \
  --bundle /tmp/t2_isam2_bundle \
  --output /tmp/t2_isam2_result
```
