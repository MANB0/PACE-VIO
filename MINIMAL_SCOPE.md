# Minimal release boundary

Kept runtime surfaces:

- live MACVO stereo frontend and uncertainty prediction;
- online compressed-UVD T2 visual factor;
- standard local-frame IMU preintegration and two-state fixed-lag backend;
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

The full pre-trim source remains at Git tag `realtime-t2-full-20260721`.
