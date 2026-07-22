# T2-iSAM2 backend

This directory contains the C++/Pybind incremental backend used by the
minimal realtime entry point when `--vio-backend isam2` is selected. The
original two-state solver remains available as `--vio-backend two_state` for
controlled comparisons and rollback.

Both backends consume the same `T2FactorPacket` contract:

- state pose `T_WI` at the IMU origin in MACVO internal NED;
- cached preintegration residual order `[p, v, R]` and full covariance;
- bias correction Jacobian and bias random-walk covariance;
- compressed UVD visual factor `(T_ref, A, c)` reconstructed from `(H, g)`;
- T2 right tangent `[translation, rotation]`, explicitly permuted to GTSAM's
  right tangent `[rotation, translation]`.

It intentionally does not use the older MACVO relative-pose sidecar or
GTSAM's independent preintegration implementation. Only the nonlinear solver
and historical-state management change; the visual, IMU, bias and coordinate
contracts remain T2's.

Build the in-process extension in the active Python environment:

```bash
export GTSAM_DIR=/absolute/path/to/lib/cmake/GTSAM  # omit for a standard install
bash Scripts/build_t2_isam2.sh
python Scripts/UnitTest/test_t2_isam2_backend.py
```

Run the realtime pipeline:

```bash
python Scripts/run_realtime_t2.py \
  --dataset /absolute/path/to/sequence \
  --vio-backend isam2 \
  --static-init-mode fixed \
  --static-init-duration-s 3.0
```

The standalone bundle runner below is retained for factor-level audit and
cross-language regression; it is not the production realtime path.

```bash
python Scripts/export_t2_isam2_bundle.py \
  --tensor-map /path/to/tensor_map.npz \
  --output-dir /tmp/t2_isam2_bundle \
  --start-frame 90 --end-frame 299

cmake -S cpp/t2_isam2 -B build/t2_isam2 \
  -DGTSAM_DIR=/home/admin1/.local/lib/cmake/GTSAM
cmake --build build/t2_isam2 -j"$(nproc)"
ctest --test-dir build/t2_isam2 --output-on-failure

build/t2_isam2/t2_isam2_runner \
  --bundle /tmp/t2_isam2_bundle \
  --output /tmp/t2_isam2_result
```
