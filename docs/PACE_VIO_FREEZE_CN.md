# PACE-VIO minimal 冻结说明

## 冻结范围

本版本在 minimal 实时工程中加入可选的增量 iSAM2 求解器。iSAM2 与原
PACE-VIO 的 two-state 与 iSAM2 后端共用同一个 `PACEFactorPacket`，不会重新生成或重新加权测量：

- 状态位姿：IMU 中心的 `T_WB`；
- 状态切空间：`[p, phi, v, ba, bg]`；
- IMU 残差与 covariance：`[p, v, R]`；
- 视觉因子：压缩 UVD 的 `(T_ref, sqrt_information, residual_offset)`；
- bias：与 PACE-VIO-2S 相同的 random-walk 因子；
- 坐标输出：IMU 中心、世界 NWU。

`--vio-backend two_state` 保留原两状态求解器，`--vio-backend isam2`
启用本版本。iSAM2 是 PACE 因子的增量求解方式，不是独立的 GTSAM
VIO 测量模型。

## 已验证内容

2026-07-22 使用原始双目图像与原始 IMU 重新运行了圆形、矩形和直线
normal-noise 完整序列，没有使用视觉缓存。固定静止初始化为 3.0 s，活动帧
分别为：

| 场景 | edge 数 | 最终状态数 | 活动帧范围 |
|---|---:|---:|---:|
| circle | 1799 | 1800 | 90--1889 |
| rectangle | 1799 | 1800 | 90--1889 |
| straight | 539 | 540 | 90--629 |

最终导出由一次不增加新因子的 iSAM2 全历史快照生成，避免混合周期性历史
修订与最新两状态结果。完整快照的 `pose/velocity/ba/bg` 帧范围严格一致。

位置误差二阶差分 RMS 为 `0.000129 / 0.000177 / 0.000129 m`。相较修复前
实时导出分别降低约 `84.5% / 87.9% / 90.8%`。iSAM2 更新中位耗时约为
`10.83 / 10.84 / 4.69 ms`，MACVO 前端中位耗时约为 `249 ms`，因此当前
吞吐瓶颈仍为视觉前端。

iSAM2 明显改善局部 RPE 和轨迹平滑度，但没有在每个场景都降低全局 ATE。
该限制指向视觉相对平移的低频系统偏差，而不是 iSAM2、IMU sigma 或窗口
长度问题。

## 构建与验收

```bash
export GTSAM_DIR=/absolute/path/to/lib/cmake/GTSAM  # 标准安装可省略
bash Scripts/build_pace_vio_isam2.sh

python -m pytest -q \
  Scripts/UnitTest/test_optimizer_finalize.py \
  Scripts/UnitTest/test_t2_isam2_backend.py \
  Scripts/UnitTest/test_t2_factor_packet.py \
  Scripts/UnitTest/test_live_t2_raw_contract.py \
  Scripts/UnitTest/test_static_initialization_modes.py
```

冻结前结果为 22 项 Python 测试全部通过，C++ 因子测试通过。

## 已知边界

- 首次 clone 后必须针对当前 Python 和 GTSAM ABI 编译扩展，不能复用其他机器
  的 `build/`；
- 模型权重和数据集不进入仓库；
- 周期性发布完整历史轨迹存在稀疏耗时峰值，当前 2--4 Hz 前端下可运行；
- 后续精度研究应冻结本求解器，单独处理视觉平移均值偏差和 covariance 校准。
