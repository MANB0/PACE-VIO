# V3b 7×3 Sequential Evaluation — Run Instructions

## 脚本路径

```
Scripts/run_v3b_7x3_sequential.py
```

## 手动启动命令

```bash
cd /home/admin1/macvo-dev
conda activate macvo
python Scripts/run_v3b_7x3_sequential.py
```

### 干运行（预览 run 计划，不执行）

```bash
python Scripts/run_v3b_7x3_sequential.py --dry-run
```

## 输出目录

```
Results/v3b_7x3_YYYYMMDD_HHMMSS/
├── trial_1/
│   ├── turbid_harbor/
│   ├── clear_shallow/
│   ├── deep_dark/
│   ├── caustic_shallow/
│   ├── dam_inspection/
│   ├── murky_coast/
│   └── open_water/
├── trial_2/
│   └── ... (same 7 scenes)
└── trial_3/
    └── ... (same 7 scenes)
```

每个 run 目录包含：

- `poses.csv`
- `frame_pair_diagnostics.csv`
- `adaptive_decisions.csv`
- `config.yaml`
- `run.log`
- `completed.ok`（标记文件，用于重跑检测）

## 运行规则

1. **21 个 run 严格顺序运行** — 一个完成后才开始下一个。
2. **不会并行** — 没有 multiprocessing、没有 xargs -P、没有 GNU parallel、没有后台 &。
3. **tqdm 进度条保留** — 显示 `[N/21] scene trial_N` 进度和 ATE。
4. **MACVO 内部 tqdm 不关闭** — 每个 MACVO 进程的进度条输出到终端。
5. **失败即停止** — 如果某个 run 失败，脚本立即退出并打印：
   - scene
   - trial
   - run 目录
   - run.log 路径
   - 返回码
6. **可中断** — Ctrl+C 中断后，已完成的结果保留。重新运行会自动跳过已完成 run。
7. **跳过已完成 run** — `completed.ok` 存在即跳过。可以手动删除某个 run 的 `completed.ok` 来强制重跑。
8. **stdout/stderr** — 每个 run 的完整输出写入 `run.log`。终端也显示关键状态行。

## 中断后继续

直接重新运行同一个命令：

```bash
python Scripts/run_v3b_7x3_sequential.py
```

脚本会自动检测各 run 目录下的 `completed.ok` 标记文件，跳过已完成 run，从中断点继续。

注：结果目录名包含时间戳。如果希望继续原目录，需要用 `--result-root` 指定原目录：

```bash
python Scripts/run_v3b_7x3_sequential.py --result-root Results/v3b_7x3_YYYYMMDD_HHMMSS/
```

## Run 完成后

全部 21 个 run 完成后，或中断后希望分析已有结果时，通知 Agent：

> "run 已完成，开始 analyze。"

Agent 会读取 `Results/v3b_7x3_*/` 目录并生成分析报告。
