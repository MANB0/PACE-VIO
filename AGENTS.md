# AGENTS.md instructions

- 我说的不一定对，你说的不一定错；你需要辩证地看待问题，先基于证据和工程逻辑思考，再实事求是地回答我认为正确的答案。
- 不要因为我提出一个方向就默认同意。对于关键技术判断，需要明确区分：已有证据支持什么、仍不确定什么、哪些结论只是推断。
- HoloOcean 项目路径为 `E:/文档/holoocean`。该项目有一个独立 agent；如果需要了解 HoloOcean 数据生成、坐标系、传感器定义或项目实现细节，应先生成一段明确的问题提示词，交由用户询问该 HoloOcean agent。
- HoloOcean agent 已确认当前数据契约：`imu_data.csv` 中四元数为 IMUSocket FLU -> world FLU，角速度和线加速度均为 FLU body frame，线加速度包含重力，静止期望读数为 `[0, 0, +9.8]`；`ref_pose.csv` 位置为 world NWU，相机姿态为 body NWU -> world NWU，速度 `vx/vy/vz` 为 world NWU，角速度为 body NWU 四元数差分。
- HoloOcean agent 已确认 IMU 噪声参数来自 `metadata.json`/生成脚本中的手动传感器配置，`AccelSigma`、`AngVelSigma`、`AccelBiasSigma`、`AngVelBiasSigma` 的 `sigma_unit` 是 `per-sample standard deviation`，噪声在传感器端真实加入；bias random walk 生效但 bias 本身不写入 CSV。
- HoloOcean agent 已确认 camera/IMU/ref_pose 时间戳来自同一个 `tick_index/TICKS_PER_SEC*1e9`，当前导出数据应视为时间同步，offset 为 0。
