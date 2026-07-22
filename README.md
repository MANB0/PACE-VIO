# MACVO Realtime T2 VIO

这是经过裁切的实时双目视觉惯性里程计工程。运行顺序固定为：

```text
双目图像 -> MACVO 前端 -> 当前帧对 UVD 压缩视觉因子
原始 IMU  -> 局部坐标系预积分 ----------------------> 两状态 T2 后端
                                                     -> IMU 中心轨迹 + 实时网页
```

## 唯一实现

当前默认分支是本项目唯一维护和发布的实现，也是后续修改的唯一源码基线。仓库中的历史完整分支和冻结标签只用于审计、差异比较与灾难恢复，不属于另一套可并行演进的产品代码。

因此，后续功能必须直接在本实现中完成并通过本仓库测试；不得依赖 `macvo-dev` 的实验脚本、历史视觉缓存或完整分支中的隐式运行路径。

该分支不读取视觉缓存。MACVO 必须先完成当前帧对的视觉计算，T2 后端随后使用同一帧对的视觉量和对应时间区间 IMU。网页显示独立的 `MACVO raw`、`VIO committed` 和可选 GT，并支持轨迹缩放/拖动、双目图像、IMU 曲线及运行中回放。

## 已验证环境

- Ubuntu/WSL2，Python 3.10
- NVIDIA GPU，CUDA 12.1
- PyTorch 2.4.0，PyPose 0.9.5
- 640x480 双目图像至少约 6 GB VRAM

CPU 负责 T2 求解，GPU 负责 MACVO 神经网络前端。没有 CUDA GPU 时当前前端不能运行。

## 1. Clone 与环境

```bash
git clone --depth 1 https://github.com/MANB0/macvo-realtime-t2-vio.git
cd macvo-realtime-t2-vio
bash Scripts/bootstrap_conda.sh macvo-t2
conda activate macvo-t2
```

脚本会安装经验证的 CUDA 12.1 PyTorch、其余依赖，并下载和校验唯一需要的模型 `Model/MACVO_FrontendCov.pth`。也可以手动安装后执行：

```bash
python Scripts/download_models.py
python Scripts/check_runtime.py
pytest -q
```

## 2. 数据

数据目录最少包含：

```text
my_sequence/
  left/<timestamp_ns>.png
  right/<timestamp_ns>.png
  imu_data.csv
  metadata.json
```

`ref_pose.csv` 仅用于 GT 展示与评估，不是估计器输入。完整字段和坐标契约见 [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md)。

## 3. 直接运行

启动时存在静止段的数据默认使用自适应初始化：

```bash
python Scripts/run_realtime_t2.py \
  --dataset /absolute/path/to/my_sequence
```

自适应模式根据静止性、gyro-bias均值标准误差、重力方向标准误差和
最近窗口稳定性决定何时完成，不假设静止段必须恰好为三秒。实际接受
时长和统计量写入结果目录的 `static_initialization.json`。

固定时长模式必须显式给出时长，例如固定使用前三秒：

```bash
python Scripts/run_realtime_t2.py \
  --dataset /absolute/path/to/my_sequence \
  --static-init-mode fixed \
  --static-init-duration-s 3.0
```

网页默认在 [http://127.0.0.1:8765/](http://127.0.0.1:8765/) 打开。结果写入 `Results/realtime_t2/<dataset-name>/`。

### iSAM2 增量后端

默认 `--vio-backend two_state` 保留已验证的两状态 T2 求解器。iSAM2 后端消费同一份
`T2FactorPacket`，其中包含压缩 UVD 视觉因子、局部坐标系 IMU 预积分因子和 bias
random-walk 因子；它不会先运行两状态求解器再平滑其位姿。

iSAM2 需要 GTSAM C++ 开发文件。若 GTSAM 安装在非标准位置，先设置
`GTSAM_DIR`，然后构建扩展：

```bash
export GTSAM_DIR=/absolute/path/to/lib/cmake/GTSAM
bash Scripts/build_t2_isam2.sh
```

冻结版本的实现边界、测试和三场景结果记录在
[`docs/T2_ISAM2_FREEZE_CN.md`](docs/T2_ISAM2_FREEZE_CN.md)。

启动固定三秒初始化的 iSAM2 实时流水线：

```bash
python Scripts/run_realtime_t2.py \
  --dataset /absolute/path/to/my_sequence \
  --vio-backend isam2 \
  --static-init-mode fixed \
  --static-init-duration-s 3.0
```

页面会逐帧追加最新 committed 状态，并在 iSAM2 发布历史修订时按 `frame_idx`
替换已有 committed 轨迹点。`MACVO raw`、GT、双目回放和 IMU 历史不会被该修订覆盖。

没有静止起始段的数据必须使用 `off` 模式：

```bash
python Scripts/run_realtime_t2.py \
  --dataset /absolute/path/to/my_sequence \
  --static-init-mode off
```

这会使用较弱的零速度/零 bias 初值，并不等价于可靠的静止初始化。建议先检查配置而不运行网络：

```bash
python Scripts/run_realtime_t2.py \
  --dataset /absolute/path/to/my_sequence \
  --dry-run
```

常用选项：

```text
--seq-to 300              只运行前 300 帧
--mode serial             串行前端/后端，用于调试
--no-live-display         关闭网页
--dashboard-port 8766     修改网页端口
--cpu-threads 4           T2 求解线程数
--vio-backend two_state   默认：原两状态 T2 后端
--vio-backend isam2       增量 iSAM2 后端，需要先构建 C++ 扩展
--static-init-mode adaptive  默认：自适应静止初始化
--static-init-mode fixed     固定窗口，必须同时指定时长
--static-init-mode off       不假设存在静止起始段
--static-init-state-policy zero  仅用于消融：保留静止边界，但丢弃估计姿态和 bias
```

`--static-init-state-policy` 默认是 `estimated`。`zero` 只允许与 `fixed` 或
`adaptive` 配合，用于比较静止估计值是否带来收益；它不是生产推荐配置。

## 运行契约

- VIO 输出位置以 IMU 中心为参考点。
- `MACVO raw` 保留纯视觉历史；网页对比时根据 metadata 外参转换到约定参考点。
- IMU 预积分只使用 body-frame 原始测量、dt、bias 线性化点和噪声参数；重力仅进入因子残差。
- 默认 T2 为 `two_state_fixed_lag + compressed_uvd`，不是 T0 相对位姿因子。
- 四个 IMU sigma 和相机/IMU 外参优先从 `metadata.json` 读取，不用配置文件中的兜底值覆盖有效 metadata。

## 历史回退

私有仓库 `main` 和标签 `realtime-t2-full-20260721` 保存裁切前的只读快照；它们不是当前实现，也不接受独立功能开发。需要恢复历史代码时，应先逐文件审计，再把必要改动移植到当前默认分支并运行本仓库回归测试。

本工程基于 MAC-VO，许可证见 [LICENSE](LICENSE)。模型来自 [MAC-VO model release](https://github.com/MAC-VO/MAC-VO/releases/tag/model)。
