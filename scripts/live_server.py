"""REQUIREMENTS 可視化用の最小 HTTP サーバー。

実験スクリプトが --live-jsonl で追記する JSONL ファイルを CORS ヘッダ付きで配信し、
visualize_game.html のライブモードからポーリングできるようにする。

使い方:
  # ターミナル1: 実験を実行（JSONLを追記）
  python3 scripts/qwen_two_agent_experiment.py \
    --config configs/experiment.yaml \
    --live-jsonl hivc_sim/results/turn_game/experiment/stream.jsonl

  # ターミナル2: サーバーを起動（ローカルアクセス）
  python3 scripts/live_server.py \
    --file hivc_sim/results/turn_game/experiment/stream.jsonl \
    --port 8765

  # リモートGPUサーバーの場合: ngrokトンネルを同時起動
  python3 scripts/live_server.py \
    --file hivc_sim/results/turn_game/experiment/stream.jsonl \
    --port 8765 --ngrok

  # ブラウザで表示されたURL/visualize を開き「ライブモード」ボタン →
  # 表示されたURL/stream.jsonl を指定

SSHポートフォワーディングを使う場合:
  ssh -L 8765:localhost:8765 user@gpu-server
  → ローカルPCのブラウザで http://localhost:8765/visualize にアクセス
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class CORSJSONLHandler(http.server.BaseHTTPRequestHandler):
    jsonl_path: str = ""

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/stream.jsonl", "/"):
            self._serve_jsonl()
        elif path == "/visualize":
            self._serve_html()
        else:
            self.send_response(404)
            self._send_cors()
            self.end_headers()

    def _serve_jsonl(self) -> None:
        p = Path(self.jsonl_path)
        if not p.exists():
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/jsonl; charset=utf-8")
            self.end_headers()
            return
        data = p.read_bytes()
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "application/jsonl; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_html(self) -> None:
        html_path = Path(__file__).resolve().parent / "visualize_game.html"
        if not html_path.exists():
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            return
        data = html_path.read_bytes()
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args) -> None:
        pass


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def _find_ngrok() -> str | None:
    """ngrokバイナリのパスを返す。見つからなければNone。"""
    return shutil.which("ngrok")


def _start_ngrok(port: int) -> subprocess.Popen | None:
    """ngrokをサブプロセスとして起動し、Popenオブジェクトを返す。

    ngrokのローカルAPI（http://localhost:4040）から公開URLを取得して表示する。
    """
    ngrok_path = _find_ngrok()
    if ngrok_path is None:
        print("ERROR: ngrokがインストールされていません。", file=sys.stderr)
        print("  macOS:   brew install ngrok/ngrok/ngrok", file=sys.stderr)
        print("  Linux:   https://ngrok.com/download からダウンロード", file=sys.stderr)
        print("  インストール後: ngrok config add-authtoken <YOUR_TOKEN>", file=sys.stderr)
        return None

    print(f"Starting ngrok tunnel for port {port}...")
    proc = subprocess.Popen(
        [ngrok_path, "http", str(port), "--log=stdout"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # ngrokのAPIエンドポイントから公開URLを取得（起動待ち）
    public_url = None
    for _ in range(15):  # 最大15秒待機
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2) as resp:
                tunnels = json.loads(resp.read().decode())
                for tunnel in tunnels.get("tunnels", []):
                    url = tunnel.get("public_url", "")
                    if url.startswith("https://"):
                        public_url = url
                        break
            if public_url:
                break
        except Exception:
            continue

    if public_url:
        print(f"\n{'='*60}")
        print(f"  ngrok tunnel active!")
        print(f"  Visualizer:  {public_url}/visualize")
        print(f"  JSONL stream: {public_url}/stream.jsonl")
        print(f"{'='*60}\n")
        print("ブラウザで上記URL/visualize を開き「ライブモード」ボタン →")
        print(f"  {public_url}/stream.jsonl を指定してください。")
        print("\nPress Ctrl+C to stop.\n")
    else:
        print("WARNING: ngrokの公開URLを取得できませんでした。", file=sys.stderr)
        print("  ngrok status: http://localhost:4040/ を確認してください。", file=sys.stderr)
        print(f"  ローカルアクセス: http://localhost:{port}/visualize", file=sys.stderr)

    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve JSONL stream for live visualization.")
    parser.add_argument("--file", default="hivc_sim/results/turn_game/experiment/stream.jsonl")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ngrok", action="store_true", default=False,
                        help="ngrokトンネルを同時起動し、インターネット経由でアクセス可能にする。"
                             "リモートGPUサーバーで実験する場合に便利。"
                             "事前に ngrok install + ngrok config add-authtoken が必要。")
    args = parser.parse_args()

    jsonl_path = Path(args.file)
    if not jsonl_path.is_absolute():
        jsonl_path = REPO_ROOT / jsonl_path
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if not jsonl_path.exists():
        jsonl_path.touch()

    # ngrokトンネル起動（--ngrok指定時）
    ngrok_proc = None
    if args.ngrok:
        ngrok_proc = _start_ngrok(args.port)
        if ngrok_proc is None:
            print("ngrokなしで続行します（ローカルアクセスのみ）。", file=sys.stderr)

    CORSJSONLHandler.jsonl_path = str(jsonl_path)
    server = ReusableTCPServer((args.host, args.port), CORSJSONLHandler)
    if not args.ngrok or ngrok_proc is None:
        print(f"Serving {jsonl_path} at http://{args.host}:{args.port}/stream.jsonl")
        print(f"Visualizer: http://{args.host}:{args.port}/visualize")
        print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.shutdown()
    finally:
        if ngrok_proc is not None:
            print("Stopping ngrok...")
            ngrok_proc.terminate()
            try:
                ngrok_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ngrok_proc.kill()


if __name__ == "__main__":
    main()
