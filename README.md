# MACVO Realtime T2 VIO

这是经过裁切的实时双目视觉惯性里程计工程。运行顺序固定为：

```text
双目图像 -> MACVO 前端 -> 当前帧对 UVD 压缩视觉因子
原始 IMU  -> 局部坐标系预积分 ----------------------> 两状态 T2 后端
                                                     -> IMU 中心轨迹 + 实时网页
```

该分支不读取视觉缓存。MACVO 必须先完成当前帧对的视觉计算，T2 后端随后使用同一帧对的视觉量和对应时间区间 IMU。网页显示独立的 `MACVO raw`、`VIO committed` 和可选 GT，并支持轨迹缩放/拖动、双目图像、IMU 曲线及运行中回放。

## 已验证环境

- Ubuntu/WSL2，Python 3.10
- NVIDIA GPU，CUDA 12.1
- PyTorch 2.4.0，PyPose 0.9.5
- 640x480 双目图像至少约 6 GB VRAM

CPU 负责 T2 求解，GPU 负责 MACVO 神经网络前端。没有 CUDA GPU 时当前前端不能运行。

## 1. Clone 与环境

```bash
git clone https://github.com/MANB0/macvo-realtime-t2-vio.git
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

有三秒静止起始段的数据：

```bash
python Scripts/run_realtime_t2.py \
  --dataset /absolute/path/to/my_sequence
```

网页默认在 [http://127.0.0.1:8765/](http://127.0.0.1:8765/) 打开。结果写入 `Results/realtime_t2/<dataset-name>/`。

没有静止起始段的数据必须显式关闭静止初始化：

```bash
python Scripts/run_realtime_t2.py \
  --dataset /absolute/path/to/my_sequence \
  --static-init-duration-s 0
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
```

## 运行契约

- VIO 输出位置以 IMU 中心为参考点。
- `MACVO raw` 保留纯视觉历史；网页对比时根据 metadata 外参转换到约定参考点。
- IMU 预积分只使用 body-frame 原始测量、dt、bias 线性化点和噪声参数；重力仅进入因子残差。
- 默认 T2 为 `two_state_fixed_lag + compressed_uvd`，不是 T0 相对位姿因子。
- 四个 IMU sigma 和相机/IMU 外参优先从 `metadata.json` 读取，不用配置文件中的兜底值覆盖有效 metadata。

## 回退与源码范围

私有仓库 `main` 和标签 `realtime-t2-full-20260721` 保存裁切前的完整可用快照；本分支只保留生产运行路径、运行配置、实时网页和关键回归测试。因此裁切错误可以逐文件比对或直接回到完整标签，不会丢失原实现。

本工程基于 MAC-VO，许可证见 [LICENSE](LICENSE)。模型来自 [MAC-VO model release](https://github.com/MAC-VO/MAC-VO/releases/tag/model)。
