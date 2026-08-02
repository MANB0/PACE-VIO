# AGENTS.md instructions

- 本仓库当前默认分支是 PACE-VIO 的唯一生产实现和唯一源码真相源。所有新功能、缺陷修复、测试、文档和发布都必须基于本分支完成。
- `T2` 是 PACE 压缩视觉因子的历史工程代号。新代码、文档和界面统一使用 `PACE-VIO`；旧 `T2*` 标识只能作为兼容接口或历史结果字段保留，并须明确标注为 legacy。
- `main`、`realtime-t2-full-20260721` 及其他完整工程副本仅作为只读历史归档和逐文件回退依据，不得作为并行实现继续开发，也不得未经审计把历史代码整体复制回生产路径。
- 我说的不一定对，你说的不一定错；你需要辩证地看待问题，先基于证据和工程逻辑思考，再实事求是地回答我认为正确的答案。
- 不要因为我提出一个方向就默认同意。对于关键技术判断，需要明确区分：已有证据支持什么、仍不确定什么、哪些结论只是推断。
- HoloOcean 项目路径为 `E:/文档/holoocean`。该项目有一个独立 agent；如果需要了解 HoloOcean 数据生成、坐标系、传感器定义或项目实现细节，应先生成一段明确的问题提示词，交由用户询问该 HoloOcean agent。
- HoloOcean agent 已确认当前数据契约：`imu_data.csv` 中四元数为 IMUSocket FLU -> world FLU，角速度和线加速度均为 FLU body frame，线加速度包含重力，静止期望读数为 `[0, 0, +9.8]`；`ref_pose.csv` 位置为 world NWU，相机姿态为 body NWU -> world NWU，速度 `vx/vy/vz` 为 world NWU，角速度为 body NWU 四元数差分。
- HoloOcean agent 已确认 IMU 噪声参数来自 `metadata.json`/生成脚本中的手动传感器配置，`AccelSigma`、`AngVelSigma`、`AccelBiasSigma`、`AngVelBiasSigma` 的 `sigma_unit` 是 `per-sample standard deviation`，噪声在传感器端真实加入；bias random walk 生效但 bias 本身不写入 CSV。
- HoloOcean agent 已确认 camera/IMU/ref_pose 时间戳来自同一个 `tick_index/TICKS_PER_SEC*1e9`，当前导出数据应视为时间同步，offset 为 0。
