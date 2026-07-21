#!/usr/bin/env python3
"""Serve a small web dashboard for MACVO batch-run progress."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


WORKDIR = Path("/home/admin1/macvo-dev")
DEFAULT_PORT = 8765


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def read_tail(path: Path | None, *, max_lines: int = 120) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as exc:
        return [f"[dashboard] failed to read log {path}: {exc}"]
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def csv_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            line_count = sum(1 for _ in f)
    except Exception:
        return 0
    return max(0, line_count - 1)


def ref_pose_total_frames(scene_root: Path) -> int | None:
    ref_pose = scene_root / "ref_pose.csv"
    count = csv_data_row_count(ref_pose)
    return count if count > 0 else None


def latest_file(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.rglob(filename) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_pose_bundle(root: Path) -> dict[str, Any]:
    poses = latest_file(root, "poses.csv")
    if poses is None:
        return {
            "poses_path": None,
            "pose_frame_path": None,
            "has_poses": False,
            "has_pose_frame": False,
        }
    pose_frame = poses.parent / "pose_coordinate_frame.txt"
    return {
        "poses_path": str(poses),
        "pose_frame_path": str(pose_frame),
        "has_poses": True,
        "has_pose_frame": pose_frame.exists(),
    }


def diagnostics_progress(
    result_dir: Path,
    scene_root: Path,
    *,
    total_frames_limit: int | None = None,
) -> dict[str, Any]:
    diagnostics = latest_file(result_dir, "frame_pair_diagnostics.csv")
    total = ref_pose_total_frames(scene_root)
    if total is not None and total_frames_limit is not None and total_frames_limit > 0:
        total = min(total, total_frames_limit)
    if diagnostics is None:
        return {
            "diagnostics_path": None,
            "diagnostics_rows": 0,
            "current_frame": None,
            "total_frames": total,
            "percent": None,
            "updated_at": None,
        }

    rows = read_csv_rows(diagnostics)
    current_frame: int | None = None
    for row in reversed(rows):
        raw = row.get("frame_idx", "") or row.get("frame_j", "") or row.get("to_idx", "")
        try:
            current_frame = int(float(raw))
            break
        except Exception:
            continue
    if current_frame is None and rows:
        current_frame = len(rows)

    percent: float | None = None
    if total and current_frame is not None:
        percent = max(0.0, min(100.0, 100.0 * float(current_frame + 1) / float(total)))

    return {
        "diagnostics_path": str(diagnostics),
        "diagnostics_rows": len(rows),
        "current_frame": current_frame,
        "total_frames": total,
        "percent": percent,
        "updated_at": diagnostics.stat().st_mtime,
    }


def latest_progress_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("trial", ""), row.get("scene", ""), row.get("variant", ""))
        latest[key] = row
    return latest


def command_result_root(cmd: str) -> str | None:
    try:
        parts = shlex.split(cmd)
    except Exception:
        parts = cmd.split()
    for idx, part in enumerate(parts):
        if part == "--resultRoot" and idx + 1 < len(parts):
            return parts[idx + 1]
        if part.startswith("--resultRoot="):
            return part.split("=", 1)[1]
    return None


def running_macvo_processes() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,ppid,etime,pcpu,pmem,cmd"],
            cwd=str(WORKDIR),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    processes: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines()[1:]:
        if "MACVO.py" not in line:
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, elapsed, pcpu, pmem, cmd = parts
        processes.append(
            {
                "pid": pid,
                "ppid": ppid,
                "elapsed": elapsed,
                "cpu_percent": pcpu,
                "mem_percent": pmem,
                "result_dir": command_result_root(cmd),
                "cmd": cmd,
            }
        )
    return processes


def gpu_status() -> list[dict[str, Any]]:
    query = "index,name,utilization.gpu,memory.used,memory.total"
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "utilization_gpu_percent": parts[2],
                "memory_used_mib": parts[3],
                "memory_total_mib": parts[4],
            }
        )
    return rows


def status_from_progress(
    row: dict[str, str] | None,
    pose_info: dict[str, Any],
    diag_info: dict[str, Any],
    is_running: bool,
) -> str:
    if row:
        status = row.get("status", "").strip()
        if status:
            return status
    if is_running:
        return "running"
    if pose_info.get("has_poses") and pose_info.get("has_pose_frame"):
        return "complete_unlogged"
    if diag_info.get("diagnostics_path") and int(diag_info.get("diagnostics_rows") or 0) > 0:
        return "partial"
    return "pending"


def collect_dashboard_state(
    result_root: Path,
    *,
    log_paths: list[Path] | None = None,
    include_system: bool = True,
    now: float | None = None,
) -> dict[str, Any]:
    result_root = result_root.expanduser().resolve()
    manifest_rows = read_csv_rows(result_root / "run_manifest.csv")
    progress_rows = read_csv_rows(result_root / "progress.csv")
    progress_by_key = latest_progress_by_key(progress_rows)
    processes = running_macvo_processes() if include_system else []

    runs: list[dict[str, Any]] = []
    for row in manifest_rows:
        trial = row.get("trial", "")
        scene = row.get("scene", "")
        variant = row.get("variant", "")
        result_dir = Path(row.get("result_dir", ""))
        if not result_dir.is_absolute():
            result_dir = (WORKDIR / result_dir).resolve()
        scene_root = Path(row.get("scene_root", ""))
        if not scene_root.is_absolute():
            scene_root = (WORKDIR / scene_root).resolve()

        progress = progress_by_key.get((trial, scene, variant))
        try:
            total_frames_limit = int(row.get("seq_to", ""))
        except (TypeError, ValueError):
            total_frames_limit = None
        active_result_dir: Path | None = None
        if progress and progress.get("status", "").strip() == "running":
            raw_active_result_dir = progress.get("active_result_dir", "").strip()
            if raw_active_result_dir:
                active_result_dir = Path(raw_active_result_dir)
                if not active_result_dir.is_absolute():
                    active_result_dir = (WORKDIR / active_result_dir).resolve()
                else:
                    active_result_dir = active_result_dir.resolve()
        monitored_result_dir = active_result_dir or result_dir
        pose_info = latest_pose_bundle(monitored_result_dir)
        diag_info = diagnostics_progress(
            monitored_result_dir,
            scene_root,
            total_frames_limit=total_frames_limit,
        )
        running = any(
            proc.get("result_dir")
            and Path(str(proc["result_dir"])).resolve() == monitored_result_dir
            for proc in processes
        )
        status = status_from_progress(progress, pose_info, diag_info, running)

        updated_at = diag_info.get("updated_at")
        stale_seconds = None
        if updated_at is not None:
            stale_seconds = max(0.0, float(now if now is not None else time.time()) - float(updated_at))

        runs.append(
            {
                "trial": trial,
                "scene": scene,
                "variant": variant,
                "status": status,
                "return_code": "" if not progress else progress.get("return_code", ""),
                "runtime_s": "" if not progress else progress.get("runtime_s", ""),
                "result_dir": str(result_dir),
                "active_result_dir": None if active_result_dir is None else str(active_result_dir),
                "scene_root": str(scene_root),
                **pose_info,
                **diag_info,
                "stale_seconds": stale_seconds,
            }
        )

    counters = {
        "total": len(runs),
        "ok": sum(1 for run in runs if run["status"] == "ok"),
        "failed": sum(1 for run in runs if run["status"] in {"failed", "timeout", "incomplete_output"}),
        "running": sum(1 for run in runs if run["status"] == "running"),
        "pending": sum(1 for run in runs if run["status"] == "pending"),
        "partial": sum(1 for run in runs if run["status"] == "partial"),
        "complete_unlogged": sum(1 for run in runs if run["status"] == "complete_unlogged"),
    }

    logs = []
    for log_path in log_paths or []:
        logs.append({"path": str(log_path), "tail": read_tail(log_path)})

    return {
        "generated_at": time.time() if now is None else now,
        "result_root": str(result_root),
        "manifest_path": str(result_root / "run_manifest.csv"),
        "progress_path": str(result_root / "progress.csv"),
        "counters": counters,
        "runs": runs,
        "processes": processes,
        "gpus": gpu_status() if include_system else [],
        "logs": logs,
    }


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MACVO Run Progress Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f6fa;
      --panel: #ffffff;
      --line: #d8e0ea;
      --text: #1f2937;
      --muted: #607085;
      --accent: #1667d9;
      --ok: #0f8a5f;
      --bad: #b42318;
      --run: #9a6700;
      --pending: #52606d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 16px 20px;
      background: #172033;
      color: #fff;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    header h1 { font-size: 20px; margin: 0; }
    header .meta { color: #cbd5e1; font-size: 13px; }
    main { padding: 16px 20px 32px; max-width: 1500px; margin: 0 auto; }
    .cards {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .card .label { font-size: 12px; color: var(--muted); }
    .card .value { font-size: 26px; font-weight: 700; margin-top: 4px; }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 14px;
      overflow: hidden;
    }
    section h2 {
      font-size: 16px;
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
    }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px 9px; vertical-align: top; }
    th { background: #f8fafc; color: #334155; text-align: left; position: sticky; top: 0; }
    code, pre { font-family: Consolas, "Liberation Mono", monospace; }
    pre {
      margin: 0;
      padding: 12px 14px;
      max-height: 360px;
      overflow: auto;
      background: #101828;
      color: #e5e7eb;
      font-size: 12px;
      line-height: 1.45;
    }
    .path {
      color: var(--muted);
      max-width: 360px;
      overflow-wrap: anywhere;
      font-size: 12px;
    }
    .badge {
      display: inline-block;
      min-width: 74px;
      text-align: center;
      padding: 3px 7px;
      border-radius: 999px;
      color: white;
      font-size: 12px;
      font-weight: 700;
    }
    .ok { background: var(--ok); }
    .failed, .timeout, .incomplete_output { background: var(--bad); }
    .running { background: var(--run); }
    .pending { background: var(--pending); }
    .partial { background: #7c3aed; }
    .complete_unlogged { background: #2563eb; }
    .bar {
      min-width: 150px;
      height: 12px;
      background: #e5e7eb;
      border-radius: 99px;
      overflow: hidden;
      margin-top: 4px;
    }
    .bar > div { height: 100%; background: var(--accent); width: 0%; }
    .small { color: var(--muted); font-size: 12px; }
    .nowrap { white-space: nowrap; }
    .launch-panel {
      padding: 12px 14px;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .launch-button {
      border: 0;
      border-radius: 6px;
      padding: 9px 14px;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    .launch-button:disabled { opacity: 0.55; cursor: default; }
    @media (max-width: 900px) {
      .cards { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      table { font-size: 12px; }
      th, td { padding: 6px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>MACVO Run Progress Dashboard</h1>
      <div class="meta" id="root"></div>
    </div>
    <div class="meta" id="updated">loading...</div>
  </header>
  <main>
    <div class="cards" id="cards"></div>
    <section id="launch-section" hidden>
      <h2>Manual launch</h2>
      <div class="launch-panel">
        <button class="launch-button" id="launch-button" type="button">Start prepared full run</button>
        <span class="small" id="launch-status">The run starts only after this button is pressed.</span>
      </div>
    </section>
    <section>
      <h2>Runs</h2>
      <div style="overflow:auto">
        <table>
          <thead>
            <tr>
              <th>scene</th><th>variant</th><th>status</th><th>frame progress</th>
              <th>runtime</th><th>rc</th><th>outputs</th><th>result dir</th>
            </tr>
          </thead>
          <tbody id="runs"></tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>Processes</h2>
      <div style="overflow:auto">
        <table>
          <thead><tr><th>pid</th><th>elapsed</th><th>cpu</th><th>mem</th><th>result dir</th></tr></thead>
          <tbody id="processes"></tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>GPU</h2>
      <div style="overflow:auto">
        <table>
          <thead><tr><th>idx</th><th>name</th><th>util</th><th>memory</th></tr></thead>
          <tbody id="gpus"></tbody>
        </table>
      </div>
    </section>
    <section>
      <h2>Logs</h2>
      <div id="logs"></div>
    </section>
  </main>
  <script>
    const refreshMs = __REFRESH_MS__;
    function pct(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return "";
      return `${Number(v).toFixed(1)}%`;
    }
    function esc(s) {
      return String(s ?? "").replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }
    function badge(status) {
      return `<span class="badge ${esc(status)}">${esc(status)}</span>`;
    }
    function renderCards(c) {
      const items = [
        ["total", c.total], ["ok", c.ok], ["running", c.running],
        ["pending", c.pending], ["partial", c.partial], ["failed", c.failed]
      ];
      document.getElementById("cards").innerHTML = items.map(([k, v]) =>
        `<div class="card"><div class="label">${esc(k)}</div><div class="value">${esc(v)}</div></div>`
      ).join("");
    }
    function renderRuns(runs) {
      document.getElementById("runs").innerHTML = runs.map(run => {
        const percent = run.percent ?? 0;
        const frame = run.current_frame === null || run.current_frame === undefined
          ? "no diagnostics yet"
          : `${run.current_frame + 1} / ${run.total_frames ?? "?"} (${pct(run.percent)})`;
        const outputs = [
          run.has_poses ? "poses" : "",
          run.has_pose_frame ? "frame" : "",
          run.diagnostics_path ? "diag" : ""
        ].filter(Boolean).join(" / ") || "-";
        return `<tr>
          <td class="nowrap">${esc(run.scene)}</td>
          <td class="nowrap">${esc(run.variant)}</td>
          <td>${badge(run.status)}</td>
          <td>${esc(frame)}<div class="bar"><div style="width:${Math.max(0, Math.min(100, percent))}%"></div></div>
              <div class="small">${run.stale_seconds === null || run.stale_seconds === undefined ? "" : `updated ${Math.round(run.stale_seconds)}s ago`}</div></td>
          <td class="nowrap">${esc(run.runtime_s)}</td>
          <td class="nowrap">${esc(run.return_code)}</td>
          <td class="nowrap">${esc(outputs)}</td>
          <td class="path">${esc(run.result_dir)}</td>
        </tr>`;
      }).join("");
    }
    function renderProcesses(processes) {
      document.getElementById("processes").innerHTML = processes.length ? processes.map(p =>
        `<tr><td>${esc(p.pid)}</td><td>${esc(p.elapsed)}</td><td>${esc(p.cpu_percent)}%</td><td>${esc(p.mem_percent)}%</td><td class="path">${esc(p.result_dir)}</td></tr>`
      ).join("") : `<tr><td colspan="5" class="small">No MACVO.py process detected.</td></tr>`;
    }
    function renderGpu(gpus) {
      document.getElementById("gpus").innerHTML = gpus.length ? gpus.map(g =>
        `<tr><td>${esc(g.index)}</td><td>${esc(g.name)}</td><td>${esc(g.utilization_gpu_percent)}%</td><td>${esc(g.memory_used_mib)} / ${esc(g.memory_total_mib)} MiB</td></tr>`
      ).join("") : `<tr><td colspan="4" class="small">nvidia-smi unavailable or no GPU detected.</td></tr>`;
    }
    function renderLogs(logs) {
      document.getElementById("logs").innerHTML = logs.length ? logs.map(log =>
        `<h3 style="font-size:13px;margin:10px 14px">${esc(log.path)}</h3><pre>${esc((log.tail || []).join("\n"))}</pre>`
      ).join("") : `<pre>No log file configured.</pre>`;
    }
    function renderLaunch(launch) {
      const section = document.getElementById("launch-section");
      const button = document.getElementById("launch-button");
      const status = document.getElementById("launch-status");
      section.hidden = !launch?.enabled;
      if (!launch?.enabled) return;
      button.disabled = Boolean(launch.running);
      status.textContent = launch.running
        ? `Run process started (pid ${launch.pid ?? "?"}). The progress dashboard will take over shortly.`
        : "The run starts only after this button is pressed.";
    }
    async function launchPreparedRun() {
      if (!window.confirm("Start the prepared full sequence now?")) return;
      const button = document.getElementById("launch-button");
      const status = document.getElementById("launch-status");
      button.disabled = true;
      status.textContent = "Starting...";
      try {
        const response = await fetch("/api/launch", {method: "POST"});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
        status.textContent = `Run process started (pid ${data.pid}).`;
      } catch (err) {
        button.disabled = false;
        status.textContent = "Launch failed: " + err;
      }
    }
    async function refresh() {
      try {
        const response = await fetch("/api/status", {cache: "no-store"});
        const data = await response.json();
        document.getElementById("root").textContent = data.result_root;
        document.getElementById("updated").textContent =
          "updated " + new Date(data.generated_at * 1000).toLocaleString();
        renderCards(data.counters);
        renderRuns(data.runs);
        renderProcesses(data.processes);
        renderGpu(data.gpus);
        renderLogs(data.logs);
        renderLaunch(data.launch);
      } catch (err) {
        document.getElementById("updated").textContent = "refresh failed: " + err;
      }
    }
    refresh();
    document.getElementById("launch-button").addEventListener("click", launchPreparedRun);
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>
"""


@dataclass
class DashboardConfig:
    result_root: Path
    log_paths: list[Path]
    refresh_s: float
    launch_script: Path | None = None
    launch_log: Path | None = None
    launch_process: subprocess.Popen[Any] | None = None
    launch_lock: threading.Lock = field(default_factory=threading.Lock)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MACVOProgressDashboard/1.0"

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, text: str) -> None:
        raw = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[dashboard] " + fmt % args + "\n")

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        config: DashboardConfig = self.server.dashboard_config  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        if path == "/api/status":
            payload = collect_dashboard_state(
                config.result_root,
                log_paths=config.log_paths,
                include_system=True,
            )
            process = config.launch_process
            payload["launch"] = {
                "enabled": config.launch_script is not None,
                "running": process is not None and process.poll() is None,
                "pid": None if process is None else process.pid,
            }
            self._send_json(payload)
            return
        if path in {"/", "/index.html"}:
            page = HTML_PAGE.replace("__REFRESH_MS__", str(int(max(config.refresh_s, 0.5) * 1000)))
            self._send_html(page)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook
        config: DashboardConfig = self.server.dashboard_config  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        if path != "/api/launch":
            self.send_error(404, "not found")
            return
        if config.launch_script is None:
            self._send_json({"error": "manual launch is disabled"}, status=403)
            return
        with config.launch_lock:
            if config.launch_process is not None and config.launch_process.poll() is None:
                self._send_json(
                    {"error": "prepared run is already running", "pid": config.launch_process.pid},
                    status=409,
                )
                return
            launch_log = config.launch_log or (config.result_root / "manual_launch.log")
            launch_log.parent.mkdir(parents=True, exist_ok=True)
            with launch_log.open("a", encoding="utf-8") as stream:
                stream.write(f"\n[dashboard] manual launch at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                stream.flush()
                config.launch_process = subprocess.Popen(
                    ["/bin/bash", str(config.launch_script)],
                    cwd=WORKDIR,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self._send_json({"status": "started", "pid": config.launch_process.pid})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--log", dest="logs", action="append", type=Path, default=[])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--refresh-s", type=float, default=2.0)
    parser.add_argument(
        "--launch-script",
        type=Path,
        default=None,
        help="Optional exact shell script exposed through a manual start button.",
    )
    parser.add_argument("--launch-log", type=Path, default=None)
    parser.add_argument("--once", action="store_true", help="Print one JSON status snapshot and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root
    if not result_root.is_absolute():
        result_root = (WORKDIR / result_root).resolve()
    logs = []
    for log_path in args.logs:
        logs.append(log_path if log_path.is_absolute() else (WORKDIR / log_path).resolve())
    launch_script = args.launch_script
    if launch_script is not None:
        launch_script = (
            launch_script.resolve()
            if launch_script.is_absolute()
            else (WORKDIR / launch_script).resolve()
        )
        if not launch_script.is_file():
            raise FileNotFoundError(f"launch script does not exist: {launch_script}")
    launch_log = args.launch_log
    if launch_log is not None:
        launch_log = launch_log.resolve() if launch_log.is_absolute() else (WORKDIR / launch_log).resolve()

    if args.once:
        print(
            json.dumps(
                collect_dashboard_state(result_root, log_paths=logs, include_system=True),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    server = ThreadingHTTPServer((args.host, int(args.port)), DashboardHandler)
    server.dashboard_config = DashboardConfig(  # type: ignore[attr-defined]
        result_root=result_root,
        log_paths=logs,
        refresh_s=max(float(args.refresh_s), 0.5),
        launch_script=launch_script,
        launch_log=launch_log,
    )
    print(f"Serving MACVO progress dashboard")
    print(f"  Result root: {result_root}")
    if logs:
        print("  Logs:")
        for log in logs:
            print(f"    - {log}")
    if launch_script is not None:
        print(f"  Manual launch: {launch_script}")
    print(f"  URL: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
