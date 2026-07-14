"""GPUサーバーでリモート実験を起動するスクリプト。

このMacのターミナルから1コマンドで:
  1. GPUサーバーにSSH接続
  2. 実験をバックグラウンド起動（nohup）

実験結果は `scripts/download_gpu_logs.py` でローカルに取得し、
`scripts/local_preview.py` でオフライン再生する（新運用）。

使い方:
  # デフォルトconfigでリモート実験を起動
  python3 scripts/gpu_run.py

  # 実験configを指定
  python3 scripts/gpu_run.py --experiment-config configs/experiment.yaml

  # 実験ログを tail で確認
  python3 scripts/gpu_run.py --logs

  # GPUサーバーの状態確認
  python3 scripts/gpu_run.py --status

  # 実験を停止
  python3 scripts/gpu_run.py --stop
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_yaml  # noqa: E402


def load_gpu_config(config_path: str = "configs/gpu_server.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        print(f"ERROR: GPU server config not found: {path}", file=sys.stderr)
        print("  configs/gpu_server.yaml を作成してください。", file=sys.stderr)
        sys.exit(1)
    return load_yaml(path)


def build_ssh_cmd(cfg: dict, remote_cmd: str, background: bool = False) -> list[str]:
    """SSH接続してリモートコマンドを実行するコマンドラインを構築。"""
    ssh_key = os.path.expanduser(cfg["ssh_key"])
    cmd = [
        "ssh",
        "-p", str(cfg["ssh_port"]),
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        f"{cfg['ssh_user']}@{cfg['ssh_host']}",
    ]
    if background:
        cmd.append("-f")
    cmd.append(remote_cmd)
    return cmd


def build_remote_prefix(cfg: dict) -> str:
    """リモートコマンドの前置き（PATH設定 + cd + venv activate）。"""
    venv = cfg.get("remote_venv", ".venv")
    project_dir = cfg["remote_project_dir"]
    return (
        f"export PATH=$HOME/bin:$PATH && "
        f"cd {project_dir} && "
        f"source {venv}/bin/activate"
    )


def start_experiment(cfg: dict, experiment_config: str) -> subprocess.CompletedProcess:
    """GPUサーバー上で実験をバックグラウンド起動。"""
    prefix = build_remote_prefix(cfg)
    remote_cmd = (
        f'{prefix} && '
        f'nohup python3 scripts/qwen_two_agent_experiment.py '
        f'--config {experiment_config} '
        f'</dev/null > /tmp/hivc_experiment.log 2>&1 &'
        f' echo "experiment PID: $!"'
    )
    print("Starting experiment on GPU server...")
    result = subprocess.run(
        build_ssh_cmd(cfg, remote_cmd),
        capture_output=True, text=True, timeout=15
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}", file=sys.stderr)
    return result


def stop_remote(cfg: dict) -> None:
    """GPUサーバー上の実験を停止。"""
    prefix = build_remote_prefix(cfg)
    remote_cmd = (
        f'{prefix} && '
        f'pkill -f "qwen_two_agent_experiment" 2>/dev/null; '
        f'echo "stopped"'
    )
    print("Stopping experiment on GPU server...")
    result = subprocess.run(
        build_ssh_cmd(cfg, remote_cmd),
        capture_output=True, text=True, timeout=15
    )
    print(result.stdout.strip() or "stopped")


def show_logs(cfg: dict) -> None:
    """GPUサーバー上の実験ログをtail -fで表示。"""
    ssh_key = os.path.expanduser(cfg["ssh_key"])
    cmd = [
        "ssh", "-p", str(cfg["ssh_port"]),
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        f"{cfg['ssh_user']}@{cfg['ssh_host']}",
        "tail -f /tmp/hivc_experiment.log"
    ]
    print("Tailing experiment logs (Ctrl+C to stop)...")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopped tailing.")


def show_status(cfg: dict) -> None:
    """GPUサーバー上のプロセス状態とGPU使用状況を表示。"""
    prefix = build_remote_prefix(cfg)
    remote_cmd = (
        f'{prefix} && '
        f'echo "=== Running processes ===" && '
        f'ps aux | grep -E "qwen_two_agent" | grep -v grep || echo "none" && '
        f'echo "=== GPU ===" && '
        f'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"'
    )
    print("Checking GPU server status...")
    result = subprocess.run(
        build_ssh_cmd(cfg, remote_cmd),
        capture_output=True, text=True, timeout=15
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print(f"stderr: {result.stderr.strip()}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPUサーバーでリモート実験を起動する。"
    )
    parser.add_argument("--gpu-config", default="configs/gpu_server.yaml",
                        help="GPUサーバーSSH接続設定ファイル")
    parser.add_argument("--experiment-config", default=None,
                        help="実験用config（GPUサーバー側で使用）。未指定時はgpu_configの値を使用。")
    parser.add_argument("--logs", action="store_true",
                        help="実験ログをtail -fで表示")
    parser.add_argument("--stop", action="store_true",
                        help="GPUサーバー上の実験を停止")
    parser.add_argument("--status", action="store_true",
                        help="GPUサーバーの状態を表示")
    args = parser.parse_args()

    cfg = load_gpu_config(args.gpu_config)

    if args.logs:
        show_logs(cfg)
        return

    if args.stop:
        stop_remote(cfg)
        return

    if args.status:
        show_status(cfg)
        return

    experiment_config = args.experiment_config or cfg.get("experiment_config", "configs/experiment.yaml")
    start_experiment(cfg, experiment_config)

    print("\n--- コマンド ---")
    print("  ログ確認:  python3 scripts/gpu_run.py --logs")
    print("  状態確認:  python3 scripts/gpu_run.py --status")
    print("  停止:      python3 scripts/gpu_run.py --stop")
    print("  ログ取得:  python3 scripts/download_gpu_logs.py")
    print("  ローカルプレビュー: python3 scripts/local_preview.py")


if __name__ == "__main__":
    main()
