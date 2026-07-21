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
must describe camera intrinsics/baseline, continuous or per-sample IMU noise,
time synchronization, coordinate conventions, and camera/IMU extrinsics. See
[`examples/metadata.holocean.example.json`](../examples/metadata.holocean.example.json).

`ref_pose.csv` is not consumed by the estimator. When present it may provide
GT to the dashboard with columns:

```text
timestamp,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz
```

For the validated HoloOcean contract, world is NWU, body/camera is aligned
NWU, and IMU measurements are FLU. The loader converts these to the internal
frame and uses metadata extrinsics so the committed VIO position is reported
at the IMU center. Do not silently reuse this convention for a different
sensor rig: update the metadata and validate the transform first.

If a sequence has a verified static prefix, use the default three-second IMU
initialization. For a sequence without a static prefix, pass
`--static-init-duration-s 0`; this starts immediately with a weaker zero-bias,
zero-velocity initialization and is not equivalent to static initialization.
