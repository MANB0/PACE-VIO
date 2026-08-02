# PACE-VIO 生产仓库命名契约

正式名称：

- **PACE-VIO**
- **Pointwise Uncertainty-Aware Compressed Estimation for Visual-Inertial Odometry**
- **点级不确定性感知压缩视觉惯性里程计**

实现变体：

- `PACE-VIO-2S`：两状态后端；
- `PACE-VIO-iSAM2`：增量 iSAM2 后端；
- `PACEFactorPacket`：两种后端共享的求解器无关因子包。

新部署和文档必须优先使用：

```bash
python Scripts/run_pace_vio.py
bash Scripts/build_pace_vio_isam2.sh
```

旧的 `T2FactorPacket`、`IncrementalT2ISAM2Backend`、
`run_realtime_t2.py` 和 `build_t2_isam2.sh` 只作为兼容别名或转发入口保留。
历史结果目录、内部源文件名和现有 GitHub 仓库 slug 不做破坏性重命名。

公开名称检索记录和完整迁移表位于开发工作区
`docs/PACE_VIO_NAMING_AUDIT_CN.md`。截至 2026-07-24，没有发现同名的
VIO、SLAM 或机器人定位算法；存在一个无关的员工管理商业产品
`PaceVIO`，因此本结论不等同于法律或商标审查。
