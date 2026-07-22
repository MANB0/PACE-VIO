# Minimal release boundary

This default branch is the sole maintained implementation and source of truth
for the realtime MACVO T2 VIO project. Historical full-source branches and tags
are read-only audit and rollback material, not parallel implementations.

Kept runtime surfaces:

- live MACVO stereo frontend and uncertainty prediction;
- online compressed-UVD T2 visual factor;
- standard local-frame IMU preintegration with switchable two-state fixed-lag
  and incremental iSAM2 backends consuming the same T2 factor packets;
- IMU-center coordinate conversion from dataset metadata;
- realtime dashboard and replay controls;
- generic stereo/IMU dataset loader;
- focused realtime contract regression tests.

Removed surfaces:

- training programs and training datasets;
- unrelated baseline odometry systems;
- historical experiment, plotting, audit and report scripts;
- generated results, datasets and pretrained binaries;
- stale absolute-path launch files and unused submodule declarations.

The full pre-trim source remains as read-only history at Git tag
`realtime-t2-full-20260721`. Any recovered code must be reviewed and integrated
into this branch rather than developed on the historical snapshot.
