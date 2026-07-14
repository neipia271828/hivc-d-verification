"""可視化用 HTTP サーバー（再設計版）。

結果ディレクトリを自動スキャンしてファイル一覧を提供し、
JSONL ライブストリームと CSV リプレイを配信する。

主な改善点:
  - --file 指定不要: 結果ディレクトリを自動スキャンしてファイル一覧を返す
  - /api/files: 利用可能な JSONL / CSV ファイル一覧（メタデータ付き）
  - /api/file?path=...: 任意のファイルを配信（CSV リプレイ用）
  - /api/status: ヘルスチェック
  - /stream.jsonl: 後方互換（デフォルトファイルまたは最新JSONL）
  - /visualize: ビジュアライザーHTML

使い方:
  # 最もシンプル: オプションなしで起動
  python3 scripts/live_server.py --port 8765

  # リモートGPUサーバーの場合: ngrokトンネルを同時起動
  python3 scripts/live_server.py --port 8765 --ngrok

  # スキャンルートを明示的に指定
  python3 scripts/live_server.py --root hivc_sim/results --port 8765

  # 従来通り --file でデフォルトストリームを固定
  python3 scripts/live_server.py --file hivc_sim/results/turn_game/experiment/stream.jsonl

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
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# スキャン対象の拡張子
DATA_EXTENSIONS = {".jsonl", ".csv"}

# スキャンから除外するディレクトリ名
EXCLUDE_DIRS = {"__pycache__", ".git", "node_modules"}


def _scan_data_files(root: Path) -> list[dict]:
    """root 配下のデータファイルをスキャンしてメタデータ付きリストを返す。"""
    files = []
    if not root.exists():
        return files
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in DATA_EXTENSIONS:
            continue
        # 除外ディレクトリ内のファイルはスキップ
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        rel = p.relative_to(root)
        files.append({
            "path": str(rel),
            "name": p.name,
            "ext": p.suffix.lower(),
            "type": "jsonl" if p.suffix.lower() == ".jsonl" else "csv",
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "mtime_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        })
    # mtime 降順（新しいものが先頭）
    files.sort(key=lambda f: f["mtime"], reverse=True)
    return files


class CORSJSONLHandler(http.server.BaseHTTPRequestHandler):
    jsonl_path: str = ""
    scan_root: Path = REPO_ROOT / "hivc_sim" / "results"

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

    def _send_json(self, obj: dict, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text: str, status: int = 200, ctype: str = "text/plain; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self._send_cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path in ("/", "/visualize"):
            self._serve_html()
        elif path == "/stream.jsonl":
            self._serve_default_jsonl()
        elif path == "/api/files":
            self._serve_file_list()
        elif path == "/api/file":
            self._serve_file(params)
        elif path == "/api/status":
            self._serve_status()
        else:
            self._send_text("Not Found", status=404)

    def _serve_html(self) -> None:
        html_path = Path(__file__).resolve().parent / "visualize_game.html"
        if not html_path.exists():
            self._send_text("visualize_game.html not found", status=404)
            return
        data = html_path.read_bytes()
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _resolve_safe_path(self, rel_path: str) -> Path | None:
        """ユーザー指定の相対パスを scan_root 内に安全に解決する。
        ディレクトリトラバーサルを防止。
        """
        if not rel_path:
            return None
        root = self.scan_root.resolve()
        try:
            candidate = (root / rel_path).resolve()
        except (ValueError, RuntimeError):
            return None
        # scan_root 配下かチェック
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return candidate

    def _serve_file_list(self) -> None:
        files = _scan_data_files(self.scan_root)
        self._send_json({"files": files, "root": str(self.scan_root.relative_to(REPO_ROOT)) if self.scan_root.is_relative_to(REPO_ROOT) else str(self.scan_root)})

    def _serve_file(self, params: dict) -> None:
        rel_path = params.get("path", [""])[0]
        p = self._resolve_safe_path(rel_path)
        if p is None:
            self._send_json({"error": "file not found or outside allowed directory", "path": rel_path}, status=404)
            return
        file_size = p.stat().st_size
        ctype = "application/jsonl; charset=utf-8" if p.suffix == ".jsonl" else "text/csv; charset=utf-8"

        # Range リクエスト対応（ライブポーリングの効率化）
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                range_spec = range_header[6:].split("-")
                start = int(range_spec[0]) if range_spec[0] else 0
                end = int(range_spec[1]) if len(range_spec) > 1 and range_spec[1] else file_size - 1
                if start >= file_size:
                    # Range Not Satisfiable
                    self.send_response(416)
                    self._send_cors()
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                end = min(end, file_size - 1)
                with open(p, "rb") as f:
                    f.seek(start)
                    chunk = f.read(end - start + 1)
                self.send_response(206)
                self._send_cors()
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(chunk)
                return
            except (ValueError, OSError):
                pass  # Range パース失敗時は全体を返す

        data = p.read_bytes()
        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    def _serve_default_jsonl(self) -> None:
        """デフォルトの JSONL ファイルを配信。
        --file が指定されていればそれを使い、なければ最新の .jsonl を自動選択。
        """
        p = None
        if self.jsonl_path:
            p = Path(self.jsonl_path)
        else:
            files = _scan_data_files(self.scan_root)
            jsonl_files = [f for f in files if f["type"] == "jsonl"]
            if jsonl_files:
                p = self.scan_root / jsonl_files[0]["path"]
        if p is None or not p.exists():
            # 空レスポンス
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

    def _serve_status(self) -> None:
        files = _scan_data_files(self.scan_root)
        jsonl_count = sum(1 for f in files if f["type"] == "jsonl")
        csv_count = sum(1 for f in files if f["type"] == "csv")
        self._send_json({
            "status": "ok",
            "root": str(self.scan_root),
            "jsonl_count": jsonl_count,
            "csv_count": csv_count,
            "default_jsonl": self.jsonl_path or "",
        })

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
        print(f"{'='*60}\n")
        print("ブラウザで上記URLを開くだけで、ファイル一覧が自動表示されます。")
        print("\nPress Ctrl+C to stop.\n")
    else:
        print("WARNING: ngrokの公開URLを取得できませんでした。", file=sys.stderr)
        print("  ngrok status: http://localhost:4040/ を確認してください。", file=sys.stderr)
        print(f"  ローカルアクセス: http://localhost:{port}/visualize", file=sys.stderr)

    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve data files for live visualization.")
    parser.add_argument("--file", default=None,
                        help="デフォルトのJSONLファイルパス（省略時は最新の.jsonlを自動選択）")
    parser.add_argument("--root", default=None,
                        help="データファイルのスキャンルートディレクトリ（省略時は hivc_sim/results）")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ngrok", action="store_true", default=False,
                        help="ngrokトンネルを同時起動し、インターネット経由でアクセス可能にする。"
                             "リモートGPUサーバーで実験する場合に便利。"
                             "事前に ngrok install + ngrok config add-authtoken が必要。")
    args = parser.parse_args()

    # スキャンルートの決定
    if args.root:
        scan_root = Path(args.root)
        if not scan_root.is_absolute():
            scan_root = REPO_ROOT / scan_root
    else:
        scan_root = REPO_ROOT / "hivc_sim" / "results"
    scan_root.mkdir(parents=True, exist_ok=True)

    # デフォルトJSONLパスの決定
    jsonl_path = ""
    if args.file:
        p = Path(args.file)
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.touch()
        jsonl_path = str(p)

    # ngrokトンネル起動（--ngrok指定時）
    ngrok_proc = None
    if args.ngrok:
        ngrok_proc = _start_ngrok(args.port)
        if ngrok_proc is None:
            print("ngrokなしで続行します（ローカルアクセスのみ）。", file=sys.stderr)

    CORSJSONLHandler.jsonl_path = jsonl_path
    CORSJSONLHandler.scan_root = scan_root
    server = ReusableTCPServer((args.host, args.port), CORSJSONLHandler)

    # 起動時のファイル一覧を表示
    files = _scan_data_files(scan_root)
    jsonl_files = [f for f in files if f["type"] == "jsonl"]
    csv_files = [f for f in files if f["type"] == "csv"]

    if not args.ngrok or ngrok_proc is None:
        print(f"Scan root: {scan_root}")
        print(f"Found {len(jsonl_files)} JSONL file(s), {len(csv_files)} CSV file(s)")
        if jsonl_files:
            print(f"  Latest JSONL: {jsonl_files[0]['path']} ({jsonl_files[0]['mtime_iso']})")
        print(f"Visualizer: http://{args.host}:{args.port}/visualize")
        print(f"File list API: http://{args.host}:{args.port}/api/files")
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
