"""GPUサーバーでリモート実験を起動し、ライブ可視化を自動開始するスクリプト。

このMacのターミナルから1コマンドで:
  1. GPUサーバーにSSH接続
  2. 実験をバックグラウンド起動（nohup）
  3. live_server.py --ngrok をバックグラウンド起動
  4. ngrokの公開URLを取得
  5. このMacのブラウザを自動オープン

使い方:
  # デフォルトconfigでリモート実験 + ライブ可視化
  python3 scripts/gpu_run.py

  # 実験configを指定
  python3 scripts/gpu_run.py --experiment-config configs/experiment.yaml

  # 実験のみ（可視化なし）
  python3 scripts/gpu_run.py --no-visualize

  # サーバーのみ（実験は手動で別ターミナルから起動済み）
  python3 scripts/gpu_run.py --server-only

  # 実験のログを tail で確認
  python3 scripts/gpu_run.py --logs

  # 実験とサーバーを停止
  python3 scripts/gpu_run.py --stop

  # GPUサーバーの状態確認
  python3 scripts/gpu_run.py --status
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
import json
import webbrowser
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
        print("  テンプレート: configs/gpu_server.yaml.example", file=sys.stderr)
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
        cmd.append("-f")  # バックグラウンド実行
    cmd.append(remote_cmd)
    return cmd


def build_remote_prefix(cfg: dict) -> str:
    """リモートコマンドの前置き（PATH設定 + cd）。"""
    ngrok_path = cfg.get("remote_ngrok_path", "~/bin/ngrok")
    venv = cfg.get("remote_venv", ".venv")
    project_dir = cfg["remote_project_dir"]
    return (
        f"export PATH=$HOME/bin:$PATH && "
        f"cd {project_dir} && "
        f"source {venv}/bin/activate && "
        f"export PATH=$(dirname {ngrok_path}):$PATH"
    )


def start_experiment(cfg: dict, experiment_config: str) -> subprocess.CompletedProcess:
    """GPUサーバー上で実験をバックグラウンド起動。"""
    prefix = build_remote_prefix(cfg)
    live_jsonl = cfg.get("live_jsonl", "hivc_sim/results/turn_game/experiment/stream.jsonl")
    remote_cmd = (
        f'{prefix} && '
        f'nohup python3 scripts/qwen_two_agent_experiment.py '
        f'--config {experiment_config} '
        f'--live-jsonl {live_jsonl} '
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


def start_server(cfg: dict) -> subprocess.CompletedProcess:
    """GPUサーバー上でlive_server.py --ngrokをバックグラウンド起動。"""
    prefix = build_remote_prefix(cfg)
    live_jsonl = cfg.get("live_jsonl", "hivc_sim/results/turn_game/experiment/stream.jsonl")
    port = cfg.get("live_port", 8765)
    remote_cmd = (
        f'{prefix} && '
        f'nohup python3 scripts/live_server.py '
        f'--file {live_jsonl} '
        f'--port {port} '
        f'--ngrok '
        f'</dev/null > /tmp/hivc_server.log 2>&1 &'
        f' echo "server PID: $!"'
    )
    print("Starting live server with ngrok on GPU server...")
    result = subprocess.run(
        build_ssh_cmd(cfg, remote_cmd),
        capture_output=True, text=True, timeout=15
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}", file=sys.stderr)
    return result


def get_ngrok_url(cfg: dict, timeout: int = 20) -> str | None:
    """GPUサーバー上のngrok APIから公開URLを取得。"""
    port = cfg.get("live_port", 8765)
    # SSH経由でngrokのローカルAPIにアクセス
    ssh_key = os.path.expanduser(cfg["ssh_key"])
    for _ in range(timeout):
        time.sleep(1)
        try:
            result = subprocess.run(
                [
                    "ssh", "-p", str(cfg["ssh_port"]),
                    "-i", ssh_key,
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=5",
                    f"{cfg['ssh_user']}@{cfg['ssh_host']}",
                    "curl -s http://localhost:4040/api/tunnels 2>/dev/null"
                ],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                tunnels = json.loads(result.stdout)
                for tunnel in tunnels.get("tunnels", []):
                    url = tunnel.get("public_url", "")
                    if url.startswith("https://"):
                        return url
        except Exception:
            continue
    return None


def open_browser(url: str) -> None:
    """ブラウザでURLを開く。"""
    print(f"Opening browser: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        print(f"  ブラウザを開けませんでした。手動でアクセスしてください: {url}")


def stop_remote(cfg: dict) -> None:
    """GPUサーバー上の実験とサーバーを停止。"""
    prefix = build_remote_prefix(cfg)
    remote_cmd = (
        f'{prefix} && '
        f'pkill -f "qwen_two_agent_experiment" 2>/dev/null; '
        f'pkill -f "live_server.py" 2>/dev/null; '
        f'pkill -f "ngrok http" 2>/dev/null; '
        f'echo "stopped"'
    )
    print("Stopping experiment and server on GPU server...")
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
        f'ps aux | grep -E "qwen_two_agent|live_server|ngrok" | grep -v grep || echo "none" && '
        f'echo "=== GPU ===" && '
        f'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available" && '
        f'echo "=== ngrok tunnels ===" && '
        f'curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); [print(t[\'public_url\']) for t in d.get(\'tunnels\',[])]" 2>/dev/null || echo "ngrok not running"'
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
        description="GPUサーバーでリモート実験を起動し、ライブ可視化を自動開始する。"
    )
    parser.add_argument("--gpu-config", default="configs/gpu_server.yaml",
                        help="GPUサーバーSSH接続設定ファイル")
    parser.add_argument("--experiment-config", default=None,
                        help="実験用config（GPUサーバー側で使用）。未指定時はgpu_configの値を使用。")
    parser.add_argument("--no-visualize", action="store_true",
                        help="ライブ可視化を起動しない（実験のみ）")
    parser.add_argument("--server-only", action="store_true",
                        help="サーバーのみ起動（実験は別途手動起動済みを想定）")
    parser.add_argument("--no-browser", action="store_true",
                        help="ブラウザを自動で開かない")
    parser.add_argument("--logs", action="store_true",
                        help="実験ログをtail -fで表示")
    parser.add_argument("--stop", action="store_true",
                        help="GPUサーバー上の実験とサーバーを停止")
    parser.add_argument("--status", action="store_true",
                        help="GPUサーバーの状態を表示")
    args = parser.parse_args()

    cfg = load_gpu_config(args.gpu_config)

    # --logs
    if args.logs:
        show_logs(cfg)
        return

    # --stop
    if args.stop:
        stop_remote(cfg)
        return

    # --status
    if args.status:
        show_status(cfg)
        return

    # 通常実行: 実験 + サーバー + ブラウザ
    experiment_config = args.experiment_config or cfg.get("experiment_config", "configs/experiment.yaml")

    if not args.server_only:
        start_experiment(cfg, experiment_config)
        # 実験の初期化（モデルロード等）を少し待つ
        print("Waiting for experiment to initialize...")
        time.sleep(3)

    if not args.no_visualize:
        start_server(cfg)
        print("Waiting for ngrok tunnel...")
        public_url = get_ngrok_url(cfg, timeout=20)
        if public_url:
            visualize_url = f"{public_url}/visualize"
            stream_url = f"{public_url}/stream.jsonl"
            print(f"\n{'='*60}")
            print(f"  Live visualizer ready!")
            print(f"  Visualizer:  {visualize_url}")
            print(f"  JSONL stream: {stream_url}")
            print(f"{'='*60}\n")
            print("ブラウザで「ライブモード」ボタン →")
            print(f"  {stream_url} を指定してください。")
            if not args.no_browser:
                open_browser(visualize_url)
        else:
            print("WARNING: ngrok URLを取得できませんでした。", file=sys.stderr)
            print("  GPUサーバーにSSH接続して /tmp/hivc_server.log を確認してください。", file=sys.stderr)

    print("\n--- コマンド ---")
    print("  ログ確認:  python3 scripts/gpu_run.py --logs")
    print("  状態確認:  python3 scripts/gpu_run.py --status")
    print("  停止:      python3 scripts/gpu_run.py --stop")


if __name__ == "__main__":
    main()
