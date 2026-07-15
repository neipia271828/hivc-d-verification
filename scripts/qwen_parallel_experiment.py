"""GPU並列実験のorchestrator。

1 GPU・1 worker を基本とし、条件ごとに shard を複数GPUで同時実行する。
blocked schedule（control → consulting → hivc_d）で条件を順次実行する。

主な責務:
- GPU検出・事前検査（VRAM、温度、他プロセス）
- shard 生成と worker 起動
- `gpu_metrics.csv` への30秒間隔監視
- 温度閾値・thermal slowdown 検出と pause 要求
- shard 結果の整合性検査と `master_manifest.json` / `merge_report.json` 生成

worker プロセスは `scripts/qwen_parallel_worker.py` を使う。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "hivc_sim"))
sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import merge_config_and_cli  # noqa: E402
from llm_turn_game_common import CONDITIONS  # noqa: E402
from qwen_two_agent_experiment import ARG_TYPES, CLI_DEFAULTS  # noqa: E402
from turn_game_metrics import compute_summary_metrics  # noqa: E402


SAMPLE_INTERVAL = 30
PAUSE_REQUEST_FILE = "pause_request"
ALLOWED_COMPUTE_MODES = {"Default", "Exclusive_Process", "E. Process", "E. Thread"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _framework_info() -> dict[str, object]:
    info: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("torch", "transformers", "numpy"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = m.__version__
        except Exception:
            info[f"{mod}_version"] = None
    return info


def _persona_file_hash(cfg: dict[str, object]) -> str | None:
    path = cfg.get("role_file") or cfg.get("personas_file")
    if path is None:
        return None
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return _sha256_file(p)


@dataclass
class Shard:
    shard_id: str
    condition: str
    gpu_id: int
    seed_start: int
    seed_count: int
    shard_dir: Path
    pause_file: Path
    process: Any = None
    pid: int | None = None
    status: str = "pending"
    exit_code: int | None = None
    output_file: str | None = None
    row_count: int | None = None
    gpu_uuid: str | None = None
    gpu_name: str | None = None
    vram_used_mb: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    skip: bool = False


def _nvidia_smi_query(gpu_ids: list[int] | None, fields: list[str]) -> list[dict[str, str]]:
    """nvidia-smi --query-gpu を実行してCSV行を dict リストで返す。"""
    cmd = ["nvidia-smi"]
    if gpu_ids:
        cmd.append(f"--id={','.join(str(g) for g in gpu_ids)}")
    cmd.extend(["--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
    except Exception as exc:
        raise RuntimeError(f"nvidia-smi 実行失敗: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi エラー: {result.stderr.strip()}")
    rows: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(fields):
            continue
        rows.append(dict(zip(fields, parts)))
    return rows


def detect_gpus() -> list[int]:
    """nvidia-smi から使用可能なGPUインデックスを検出する。"""
    fields = ["index", "compute_mode"]
    rows = _nvidia_smi_query(None, fields)
    indices: list[int] = []
    for row in rows:
        try:
            idx = int(row["index"])
        except ValueError:
            continue
        mode = row["compute_mode"]
        if mode in ALLOWED_COMPUTE_MODES:
            indices.append(idx)
    return indices


def get_gpu_snapshot(gpu_ids: list[int]) -> list[dict[str, object]]:
    """GPUの瞬時スナップショットを取得する。"""
    fields = [
        "index",
        "uuid",
        "name",
        "compute_mode",
        "memory.free",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "utilization.gpu",
        "utilization.memory",
        "pstate",
        "clocks_throttle_reasons.thermal",
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.hw_slowdown",
    ]
    rows = _nvidia_smi_query(gpu_ids, fields)
    out: list[dict[str, object]] = []
    for row in rows:
        try:
            temp = int(row["temperature.gpu"])
        except ValueError:
            temp = None
        try:
            mem_free = int(row["memory.free"])
        except ValueError:
            mem_free = None
        try:
            mem_used = int(row["memory.used"])
        except ValueError:
            mem_used = None
        try:
            mem_total = int(row["memory.total"])
        except ValueError:
            mem_total = None
        thermal_active = "Active" in (row.get("clocks_throttle_reasons.thermal") or "")
        hw_thermal_active = "Active" in (row.get("clocks_throttle_reasons.hw_thermal_slowdown") or "")
        hw_slowdown_active = "Active" in (row.get("clocks_throttle_reasons.hw_slowdown") or "")
        out.append(
            {
                "index": int(row["index"]),
                "uuid": row["uuid"],
                "name": row["name"],
                "compute_mode": row["compute_mode"],
                "memory_free_mb": mem_free,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "temperature": temp,
                "power_draw": row.get("power.draw"),
                "power_limit": row.get("power.limit"),
                "utilization_gpu": row.get("utilization.gpu"),
                "utilization_memory": row.get("utilization.memory"),
                "pstate": row.get("pstate"),
                "thermal_throttle": thermal_active,
                "hw_thermal_slowdown": hw_thermal_active,
                "hw_slowdown": hw_slowdown_active,
            }
        )
    return out


def get_compute_apps() -> list[dict[str, str]]:
    """nvidia-smi からCUDA computeプロセス一覧を取得する。"""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
    except Exception:
        return []
    if result.returncode != 0:
        return []
    apps: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        apps.append(
            {
                "pid": parts[0],
                "process_name": parts[1],
                "used_memory": parts[2],
                "gpu_uuid": parts[3],
            }
        )
    return apps


def compute_shards(
    conditions: list[str],
    seed: int,
    games: int,
    gpu_ids: list[int],
    workers_per_gpu: int,
) -> list[Shard]:
    """条件ごとに同一のseed分割をGPU workerに割り当てる。"""
    worker_gpus = [gpu for gpu in gpu_ids for _ in range(workers_per_gpu)]
    total_workers = len(worker_gpus)
    base = games // total_workers
    remainder = games % total_workers
    counts = [base + (1 if i < remainder else 0) for i in range(total_workers)]

    shards: list[Shard] = []
    shards_dir = None  # 後で設定
    for condition in conditions:
        start = seed
        for i, count in enumerate(counts):
            if count == 0:
                continue
            end = start + count - 1
            gpu_id = worker_gpus[i]
            shard_id = f"{condition}-gpu{gpu_id}-seed{start}-{end}"
            shards.append(
                Shard(
                    shard_id=shard_id,
                    condition=condition,
                    gpu_id=gpu_id,
                    seed_start=start,
                    seed_count=count,
                    shard_dir=Path(),
                    pause_file=Path(),
                )
            )
            start += count
    return shards


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_shard_manifest(shard_dir: Path) -> dict[str, Any] | None:
    return _read_json(shard_dir / "shard_manifest.json")


class MasterLogger:
    def __init__(self, master_dir: Path) -> None:
        self.master_dir = master_dir
        self.log_path = master_dir / "master.log"

    def log(self, message: str) -> None:
        line = f"[{_now_iso()}] {message}"
        print(line)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU並列実験 orchestrator")
    parser.add_argument("--config", default=None, help="YAML設定ファイル")
    parser.add_argument("--parallel", action="store_true", default=False, help="並列モード")
    parser.add_argument("--gpus", nargs="+", type=int, default=None, help="使用GPU ID")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="GPUあたりworker数")
    parser.add_argument("--temperature-warning", type=int, default=80)
    parser.add_argument("--temperature-stop-scheduling", type=int, default=83)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model-peak-mb", type=int, default=10000, help="モデル読込みピークMB")
    parser.add_argument("--gpu-vram-safety-factor", type=float, default=1.25)
    parser.add_argument("--condition-order", nargs="*", default=None, choices=list(CONDITIONS) + ["all"])

    # 実験引数（config 上書き用）
    for key, typ in ARG_TYPES.items():
        arg_name = "--" + key.replace("_", "-")
        kwargs: dict[str, Any] = {"default": None}
        if key == "conditions":
            kwargs["nargs"] = "*"
            kwargs["choices"] = list(CONDITIONS) + ["all"]
        elif key == "output_dir":
            kwargs["type"] = str
        elif typ is bool:
            kwargs["type"] = str
        elif typ is list:
            kwargs["nargs"] = "*"
        elif typ is int:
            kwargs["type"] = int
        else:
            kwargs["type"] = str
        parser.add_argument(arg_name, **kwargs)
    return parser


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    cli_overrides: dict[str, object] = {}
    for key in ARG_TYPES:
        value = getattr(args, key, None)
        if value is not None:
            cli_overrides[key] = value
    cfg = merge_config_and_cli(args.config, cli_overrides, CLI_DEFAULTS, ARG_TYPES)
    cfg["model_path"] = str(Path(str(cfg["model_path"])).expanduser())
    # 並列実行では live stream は無効（shard worker も上書きする）
    cfg["live_jsonl"] = None
    conditions = cfg["conditions"]
    if "all" in conditions:
        conditions = list(CONDITIONS)
    cfg["conditions"] = conditions
    return cfg


def _prepare_master_dir(cfg: dict[str, Any]) -> Path:
    output_dir = Path(str(cfg["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_gpu_ids(args: argparse.Namespace) -> list[int]:
    if args.gpus:
        return args.gpus
    gpus = detect_gpus()
    if not gpus:
        raise RuntimeError("使用可能なGPUが検出できませんでした。--gpus を明示指定してください。")
    return gpus


def _pre_check(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    gpu_ids: list[int],
    logger: MasterLogger,
) -> tuple[list[dict[str, object]], list[Shard]]:
    """起動前検査を実施する。失敗した場合は RuntimeError を投げる。"""
    logger.log("GPU検出中...")
    snapshot = get_gpu_snapshot(gpu_ids)
    if len(snapshot) != len(gpu_ids):
        raise RuntimeError("指定GPUの一部が nvidia-smi で検出できません")

    for info in snapshot:
        mode = info["compute_mode"]
        if mode not in ALLOWED_COMPUTE_MODES:
            raise RuntimeError(f"GPU {info['index']} compute_mode={mode} は使用不可です")

    # 温度検査
    for info in snapshot:
        temp = info.get("temperature")
        if temp is None:
            raise RuntimeError(f"GPU {info['index']} の温度が取得できません")
        if temp >= args.temperature_stop_scheduling:
            raise RuntimeError(f"GPU {info['index']} 温度 {temp}C >= 停止閾値 {args.temperature_stop_scheduling}C")

    # VRAM検査
    required_per_gpu = args.model_peak_mb * args.workers_per_gpu * args.gpu_vram_safety_factor
    for info in snapshot:
        free = info.get("memory_free_mb")
        if free is None:
            raise RuntimeError(f"GPU {info['index']} VRAM空きが取得できません")
        if free < required_per_gpu:
            raise RuntimeError(
                f"GPU {info['index']} VRAM空き {free}MiB が不足 "
                f"(必要 {required_per_gpu}MiB = {args.model_peak_mb} * {args.workers_per_gpu} * {args.gpu_vram_safety_factor})"
            )

    # 他のCUDAプロセス検出
    apps = get_compute_apps()
    if apps:
        pids = ", ".join(f"{a['pid']}({a['process_name']})" for a in apps)
        raise RuntimeError(f"他のCUDAプロセスが検出されました: {pids}")

    # shard 生成
    conditions = cfg["conditions"]
    if args.condition_order:
        order = args.condition_order
        if "all" in order:
            order = list(CONDITIONS)
        if set(order) != set(conditions):
            raise RuntimeError("--condition-order は --conditions と同じ集合である必要があります")
        conditions = [c for c in order if c in conditions]
        cfg["conditions"] = conditions

    shards = compute_shards(
        conditions,
        cfg["seed"],
        cfg["games"],
        gpu_ids,
        args.workers_per_gpu,
    )

    # shard 一意性・カバレッジ検査
    expected_seeds = set(range(cfg["seed"], cfg["seed"] + cfg["games"]))
    for condition in conditions:
        cond_shards = [s for s in shards if s.condition == condition]
        actual_seeds: set[int] = set()
        for s in cond_shards:
            actual_seeds.update(range(s.seed_start, s.seed_start + s.seed_count))
        if actual_seeds != expected_seeds:
            raise RuntimeError(
                f"condition {condition} のseedカバレッジが不正: {sorted(actual_seeds)} != {sorted(expected_seeds)}"
            )

    return snapshot, shards


def _write_master_manifest(master_dir: Path, data: dict[str, Any]) -> None:
    _write_json(master_dir / "master_manifest.json", data)


def _load_master_manifest(master_dir: Path) -> dict[str, Any] | None:
    return _read_json(master_dir / "master_manifest.json")


def _create_master_manifest(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    gpu_ids: list[int],
    shards: list[Shard],
    master_dir: Path,
) -> dict[str, Any]:
    git_sha = _git_sha()
    config_hash = _sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str))
    persona_hash = _persona_file_hash(cfg)
    framework_info = _framework_info()

    manifest = {
        "run_id": master_dir.name,
        "started_at": _now_iso(),
        "status": "pre_check_passed",
        "config_path": args.config,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "framework_info": framework_info,
        "persona_hash": persona_hash,
        "model_path": cfg["model_path"],
        "conditions": cfg["conditions"],
        "games": cfg["games"],
        "seed": cfg["seed"],
        "gpus": gpu_ids,
        "workers_per_gpu": args.workers_per_gpu,
        "temperature_warning": args.temperature_warning,
        "temperature_stop_scheduling": args.temperature_stop_scheduling,
        "resume": args.resume,
        "shards": [_shard_entry(s) for s in shards],
    }
    _write_master_manifest(master_dir, manifest)
    return manifest


def _shard_entry(shard: Shard, include_status: bool = True) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "shard_id": shard.shard_id,
        "condition": shard.condition,
        "gpu_id": shard.gpu_id,
        "seed_start": shard.seed_start,
        "seed_count": shard.seed_count,
        "shard_dir": str(shard.shard_dir),
        "shard_manifest_path": str(shard.shard_dir / "shard_manifest.json"),
    }
    if include_status:
        entry.update(
            {
                "status": shard.status,
                "exit_code": shard.exit_code,
                "pid": shard.pid,
                "output_file": shard.output_file,
                "row_count": shard.row_count,
                "gpu_uuid": shard.gpu_uuid,
                "gpu_name": shard.gpu_name,
                "vram_used_mb": shard.vram_used_mb,
                "started_at": shard.started_at,
                "finished_at": shard.finished_at,
            }
        )
    return entry


def _apply_resume(master_dir: Path, manifest: dict[str, Any], shards: list[Shard]) -> None:
    """--resume 時に完了済みshardをスキップする。"""
    if not manifest.get("resume"):
        return
    existing = _load_master_manifest(master_dir)
    if not existing:
        return
    if existing.get("config_hash") != manifest["config_hash"]:
        raise RuntimeError("resume 時の config_hash が一致しません。新規runを開始してください。")
    completed_ids = {
        s["shard_id"]
        for s in existing.get("shards", [])
        if s.get("status") == "completed" and s.get("exit_code") == 0
    }
    for shard in shards:
        if shard.shard_id in completed_ids:
            shard.skip = True
            shard.status = "completed"
            shard.exit_code = 0


def _launch_worker(shard: Shard, cfg: dict[str, Any], args: argparse.Namespace) -> None:
    shard.shard_dir.mkdir(parents=True, exist_ok=True)
    shard.started_at = _now_iso()
    shard.status = "running"

    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_DIR / "qwen_parallel_worker.py"),
        "--condition",
        shard.condition,
        "--seed",
        str(shard.seed_start),
        "--games",
        str(shard.seed_count),
        "--output-dir",
        str(shard.shard_dir),
        "--gpu-id",
        str(shard.gpu_id),
        "--pause-file",
        str(shard.pause_file),
    ]
    if args.config:
        cmd.extend(["--config", args.config])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(shard.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    log_path = shard.shard_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as logfile:
        proc = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=env,
            start_new_session=True,
        )
    shard.process = proc
    shard.pid = proc.pid
    (shard.shard_dir / "pid").write_text(str(proc.pid), encoding="utf-8")


def _finalize_shard(shard: Shard) -> None:
    proc = shard.process
    if proc is None:
        return
    try:
        exit_code = proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return
    shard.exit_code = exit_code
    shard.finished_at = _now_iso()
    (shard.shard_dir / "exit_code").write_text(str(exit_code), encoding="utf-8")

    manifest = _load_shard_manifest(shard.shard_dir)
    if manifest:
        shard.status = manifest.get("status", "failed")
        shard.row_count = manifest.get("row_count")
        shard.output_file = manifest.get("output_file")
        shard.gpu_uuid = manifest.get("gpu_uuid")
        shard.gpu_name = manifest.get("gpu_name")
        shard.vram_used_mb = manifest.get("vram_used_mb")
    elif exit_code == 0:
        shard.status = "completed"
    else:
        shard.status = "failed"


def _update_manifest_shards(master_dir: Path, manifest: dict[str, Any], shards: list[Shard]) -> None:
    manifest["shards"] = [_shard_entry(s) for s in shards]
    _write_master_manifest(master_dir, manifest)


def _wait_for_shards(shards: list[Shard], state: dict[str, Any], state_lock: threading.Lock) -> None:
    launched = [s for s in shards if s.process is not None]
    while True:
        still_running = False
        for shard in launched:
            if shard.process is None:
                continue
            if shard.process.poll() is None:
                still_running = True
                continue
            _finalize_shard(shard)
            with state_lock:
                state["workers"] = [w for w in state["workers"] if w.shard_id != shard.shard_id]
        if not still_running:
            break
        time.sleep(0.5)


def _monitor_gpus(
    gpu_ids: list[int],
    state: dict[str, Any],
    state_lock: threading.Lock,
    master_dir: Path,
    stop_event: threading.Event,
    args: argparse.Namespace,
    logger: MasterLogger,
) -> None:
    metrics_path = master_dir / "gpu_metrics.csv"
    fieldnames = [
        "timestamp",
        "gpu_index",
        "gpu_uuid",
        "gpu_name",
        "utilization_gpu",
        "utilization_memory",
        "memory_used_mb",
        "memory_total_mb",
        "temperature",
        "power_draw_w",
        "power_limit_w",
        "pstate",
        "thermal_throttle",
        "hw_thermal_slowdown",
        "hw_slowdown",
        "worker_pids",
        "worker_shard_ids",
    ]
    first = True
    high_temp_since: dict[int, dt.datetime] = {}
    pause_files_created = False

    while not stop_event.is_set():
        try:
            snapshot = get_gpu_snapshot(gpu_ids)
        except Exception as exc:
            logger.log(f"GPU監視取得失敗: {exc}")
            stop_event.wait(SAMPLE_INTERVAL)
            continue

        now = dt.datetime.now(dt.timezone.utc).astimezone()
        with state_lock:
            workers = list(state["workers"])
            stop_scheduling = state.get("stop_scheduling", False)

        for info in snapshot:
            gpu_idx = info["index"]
            workers_for_gpu = [w for w in workers if w.gpu_id == gpu_idx and w.status == "running"]
            pids = [str(w.pid) for w in workers_for_gpu if w.pid]
            shard_ids = [w.shard_id for w in workers_for_gpu]

            row = {
                "timestamp": now.isoformat(),
                "gpu_index": gpu_idx,
                "gpu_uuid": info.get("uuid"),
                "gpu_name": info.get("name"),
                "utilization_gpu": info.get("utilization_gpu"),
                "utilization_memory": info.get("utilization_memory"),
                "memory_used_mb": info.get("memory_used_mb"),
                "memory_total_mb": info.get("memory_total_mb"),
                "temperature": info.get("temperature"),
                "power_draw_w": info.get("power_draw"),
                "power_limit_w": info.get("power_limit"),
                "pstate": info.get("pstate"),
                "thermal_throttle": info.get("thermal_throttle"),
                "hw_thermal_slowdown": info.get("hw_thermal_slowdown"),
                "hw_slowdown": info.get("hw_slowdown"),
                "worker_pids": ",".join(pids),
                "worker_shard_ids": ",".join(shard_ids),
            }
            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if first:
                    writer.writeheader()
                    first = False
                writer.writerow(row)

            temp = info.get("temperature")
            if temp is not None:
                if temp >= args.temperature_warning:
                    logger.log(f"WARNING: GPU {gpu_idx} 温度 {temp}C >= {args.temperature_warning}C")
                if temp >= args.temperature_stop_scheduling:
                    if not stop_scheduling:
                        logger.log(f"STOP: GPU {gpu_idx} 温度 {temp}C >= {args.temperature_stop_scheduling}C。新規shard起動を停止します")
                        with state_lock:
                            state["stop_scheduling"] = True
                    high_temp_since.setdefault(gpu_idx, now)
                else:
                    high_temp_since.pop(gpu_idx, None)

            thermal_any = info.get("thermal_throttle") or info.get("hw_thermal_slowdown") or info.get("hw_slowdown")
            high_duration = False
            if gpu_idx in high_temp_since:
                high_duration = (now - high_temp_since[gpu_idx]).total_seconds() >= 60

            if thermal_any or high_duration:
                if not pause_files_created:
                    logger.log("THERMAL PAUSE: 実行中workerにpause要求を出します")
                    with state_lock:
                        state["thermal_pause"] = True
                        for w in state["workers"]:
                            if w.status == "running":
                                w.pause_file.write_text("thermal", encoding="utf-8")
                    pause_files_created = True

        stop_event.wait(SAMPLE_INTERVAL)


def _merge_results(
    master_dir: Path,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    shards: list[Shard],
    logger: MasterLogger,
) -> int:
    """shard結果を結合する。成功なら0、失敗なら1を返す。"""
    all_completed = all(s.status == "completed" and s.exit_code == 0 for s in shards)
    checks: list[dict[str, Any]] = []
    condition_rows: dict[str, list[dict[str, str]]] = {c: [] for c in cfg["conditions"]}

    # 未完了・失敗shardの有無
    checks.append({"name": "all_shards_completed", "passed": all_completed})
    if not all_completed:
        failed = [s.shard_id for s in shards if s.status != "completed" or s.exit_code != 0]
        logger.log(f"shard未完了・失敗: {failed}")

    # 読み込み
    if all_completed:
        for shard in shards:
            csv_path = shard.shard_dir / f"{shard.condition}_games.csv"
            rows = _read_csv(csv_path)
            condition_rows[shard.condition].extend(rows)

    # 行スキーマ一致
    all_rows: list[dict[str, str]] = []
    for rows in condition_rows.values():
        all_rows.extend(rows)
    schema_ok = True
    if all_rows:
        expected_keys = set(all_rows[0].keys())
        for i, row in enumerate(all_rows):
            if set(row.keys()) != expected_keys:
                schema_ok = False
                logger.log(f"行スキーマ不一致: row {i}")
                break
    checks.append({"name": "row_schema_consistent", "passed": schema_ok})

    # seed 重複・欠損
    expected_seeds = set(range(cfg["seed"], cfg["seed"] + cfg["games"]))
    seed_ok = True
    for condition, rows in condition_rows.items():
        seeds = set()
        for r in rows:
            try:
                seeds.add(int(r["seed"]))
            except (ValueError, TypeError):
                seed_ok = False
                logger.log(f"condition {condition} seed値が不正: {r.get('seed')}")
                break
        if set(seeds) != expected_seeds:
            seed_ok = False
            logger.log(f"condition {condition} seed集合が不一致: {sorted(seeds)} != {sorted(expected_seeds)}")
    checks.append({"name": "condition_seed_set_match", "passed": seed_ok})

    # turn 重複
    seen: set[tuple[str, str, str]] = set()
    dup_ok = True
    for row in all_rows:
        key = (row.get("condition", ""), row.get("seed", ""), row.get("turn", ""))
        if key in seen:
            dup_ok = False
            logger.log(f"重複ターン: {key}")
            break
        seen.add(key)
    checks.append({"name": "no_duplicate_turn", "passed": dup_ok})

    # shard manifest のハッシュ一致
    hash_ok = True
    for shard in shards:
        sm = _load_shard_manifest(shard.shard_dir)
        if not sm:
            hash_ok = False
            logger.log(f"shard_manifest.json 読み込み失敗: {shard.shard_dir}")
            continue
        for key in ("config_hash", "git_sha", "persona_hash"):
            if sm.get(key) != manifest.get(key):
                hash_ok = False
                logger.log(f"{shard.shard_id} {key} mismatch: {sm.get(key)} != {manifest.get(key)}")
        if sm.get("framework_info") != manifest.get("framework_info"):
            hash_ok = False
            logger.log(f"{shard.shard_id} framework_info mismatch")
    checks.append({"name": "shard_hashes_match_master", "passed": hash_ok})

    all_checks_pass = all(c["passed"] for c in checks)
    merge_status = "merged" if (all_completed and all_checks_pass) else "failed"

    output_files: dict[str, Any] = {}
    if merge_status == "merged":
        # 条件別CSV
        for condition, rows in condition_rows.items():
            rows_sorted = sorted(rows, key=lambda r: (r.get("seed", ""), r.get("turn", "")))
            path = master_dir / f"{condition}_games.csv"
            _write_csv(path, rows_sorted)
            output_files[f"{condition}_games.csv"] = _sha256_file(path)

        all_sorted = sorted(all_rows, key=lambda r: (r.get("condition", ""), r.get("seed", ""), r.get("turn", "")))
        all_path = master_dir / "all_games.csv"
        _write_csv(all_path, all_sorted)
        output_files["all_games.csv"] = _sha256_file(all_path)

        # summary.csv
        summary_rows: list[dict[str, object]] = []
        for condition in cfg["conditions"]:
            rows = condition_rows[condition]
            unique_games = len({r["seed"] for r in rows})
            metrics = compute_summary_metrics(rows)
            summary_rows.append({"condition": condition, "games": unique_games, **metrics})
        summary_path = master_dir / "summary.csv"
        _write_csv(summary_path, summary_rows)
        output_files["summary.csv"] = _sha256_file(summary_path)

    merge_report = {
        "run_id": master_dir.name,
        "merged_at": _now_iso(),
        "status": merge_status,
        "all_completed": all_completed,
        "checks": checks,
        "row_counts": {c: len(rows) for c, rows in condition_rows.items()},
        "expected_games_per_condition": cfg["games"],
        "output_files": output_files,
    }
    merge_report_path = master_dir / "merge_report.json"
    _write_json(merge_report_path, merge_report)

    manifest["merge_status"] = merge_status
    manifest["merge_report_path"] = str(merge_report_path)
    manifest["finished_at"] = _now_iso()
    manifest["status"] = "completed" if merge_status == "merged" else "failed"
    _write_master_manifest(master_dir, manifest)

    if merge_status == "merged":
        logger.log(f"結合完了: {len(all_rows)} 行を {master_dir} へ出力")
        return 0
    else:
        logger.log("結合失敗: merge_report.json を確認してください")
        return 1


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = _load_config(args)
    master_dir = _prepare_master_dir(cfg)
    logger = MasterLogger(master_dir)
    shards_dir = master_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    try:
        gpu_ids = _resolve_gpu_ids(args)
    except Exception as exc:
        logger.log(f"GPU検出失敗: {exc}")
        sys.exit(1)

    # shard 生成
    try:
        _snapshot, shards = _pre_check(args, cfg, gpu_ids, logger)
    except Exception as exc:
        logger.log(f"事前検査失敗: {exc}")
        # master_manifest に最低限記録して終了
        manifest = _create_master_manifest(args, cfg, gpu_ids, [], master_dir)
        manifest["status"] = "pre_check_failed"
        manifest["error"] = str(exc)
        _write_master_manifest(master_dir, manifest)
        sys.exit(1)

    # shard_dir / pause_file を設定
    for shard in shards:
        shard.shard_dir = shards_dir / shard.shard_id
        shard.pause_file = shard.shard_dir / PAUSE_REQUEST_FILE

    # master manifest 作成
    manifest = _create_master_manifest(args, cfg, gpu_ids, shards, master_dir)
    if args.resume:
        _apply_resume(master_dir, manifest, shards)

    # スキップ情報を master manifest に反映
    _update_manifest_shards(master_dir, manifest, shards)

    state: dict[str, Any] = {
        "workers": [],
        "stop_scheduling": False,
        "thermal_pause": False,
        "gpus": gpu_ids,
    }
    state_lock = threading.Lock()
    stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=_monitor_gpus,
        args=(gpu_ids, state, state_lock, master_dir, stop_event, args, logger),
        daemon=True,
    )
    monitor_thread.start()

    try:
        for condition in cfg["conditions"]:
            if state.get("stop_scheduling"):
                logger.log(f"stop_scheduling: {condition} をスキップ")
                break

            condition_shards = [s for s in shards if s.condition == condition]
            logger.log(f"条件 {condition} の {len(condition_shards)} shard を起動")

            for shard in condition_shards:
                if state.get("stop_scheduling"):
                    logger.log("新規shard起動停止")
                    break
                if shard.skip:
                    logger.log(f"{shard.shard_id} は resume によりスキップ")
                    continue
                _launch_worker(shard, cfg, args)
                with state_lock:
                    state["workers"].append(shard)
                _update_manifest_shards(master_dir, manifest, shards)

            _wait_for_shards(condition_shards, state, state_lock)
            _update_manifest_shards(master_dir, manifest, shards)

            if state.get("stop_scheduling"):
                logger.log("stop_scheduling: 後続条件をスキップ")
                break

    except Exception as exc:
        logger.log(f"実行中エラー: {exc}\n{traceback.format_exc()}")
    finally:
        stop_event.set()
        monitor_thread.join(timeout=5)

    # 残ったworkerを待機
    _wait_for_shards([s for s in shards if s.process is not None], state, state_lock)
    _update_manifest_shards(master_dir, manifest, shards)

    # 結合
    try:
        exit_code = _merge_results(master_dir, cfg, manifest, shards, logger)
    except Exception as exc:
        logger.log(f"結合処理エラー: {exc}\n{traceback.format_exc()}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
