# Stereo + IMU dataset contract

The runner accepts one dataset directory:

```text
dataset/
  left/<timestamp_ns>.png
  right/<timestamp_ns>.png
  imu_data.csv
  metadata.json
  ref_pose.csv              # optional, dashboard/evaluation only
```

Left and right filenames are integer nanosecond timestamps and must match.
`imu_data.csv` uses:

```text
timestamp,qx,qy,qz,qw,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z
```

The IMU gyroscope and accelerometer are body-frame measurements in `rad/s`
and `m/s^2`. The acceleration measurement includes gravity. `metadata.json`
must describe camera intrinsics/baseline, continuous-time IMU noise densities,
and one camera/IMU extrinsic. Image filenames and IMU CSV timestamps are always
integer nanoseconds and are assumed synchronized with zero offset. These are
fixed runtime contracts, not metadata fields, and no endpoint offset is
estimated. IMU sample intervals are read from the timestamps rather than a
nominal frequency field. See
[`examples/metadata.holocean.example.json`](../examples/metadata.holocean.example.json).

`ref_pose.csv` is not consumed by the estimator. When present it may provide
GT to the dashboard with columns:

```text
timestamp,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz
```

`metadata.extrinsics` contains exactly one 4x4 matrix, `T_CI`, with the strict
contract `p_C = T_CI p_I`. `I` is the raw IMU CSV measurement frame and `C` is
MACVO's internal camera frame (x-forward, y-right, z-down). Translation is the
IMU origin expressed in `C`. The estimator keeps acceleration, angular rate,
biases and preintegrated deltas in the raw IMU frame; `T_CI` is used only to
compose `T_WI = T_WC T_CI` and to express visual factors at the camera.

For the validated HoloOcean contract, world truth is NWU and IMU measurements
are FLU. The corresponding aligned-socket rotation in `T_CI` is
`diag(1,-1,-1)`. No separate or hidden FLU/NED rotation is applied. The full
matrix also carries the lever arm so the committed VIO position is reported at
the IMU center. A different rig must provide its calibrated full `T_CI`.
There is no `coordinate_convention` metadata block: the frame relation is fully
encoded by `T_CI`, while the raw-IMU and internal-world conventions above are a
fixed runtime contract.

If a sequence has a static prefix, the default `adaptive` mode accepts it once
stationarity, bias precision, gravity-direction precision and recent-window
stability pass. Use `--static-init-mode fixed --static-init-duration-s <sec>`
when an audited fixed interval is required. For a sequence without a static
prefix, pass `--static-init-mode off`; this starts immediately with weaker
zero-bias and zero-velocity initial values and is not equivalent to static
initialization.

For controlled ablation only, `--static-init-state-policy zero` keeps the
detected fixed/adaptive boundary while discarding its estimated attitude and
biases. The production default remains `estimated`.
