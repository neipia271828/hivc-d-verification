"""ローカル実験ログプレビューサーバー。

`hivc_sim/results/turn_game/downloads/` 以下のrunディレクトリをスキャンし、
取得済みCSV（all_games.csv / {condition}_games.csv / summary.csv）を閲覧する。
外部ネットワーク・GPU接続は不要。既定の待受は 127.0.0.1 のみ。

使い方:
  # 既定ポート（8765）で起動
  python3 scripts/local_preview.py

  # ポート指定
  python3 scripts/local_preview.py --port 8080

  # 保存先を変更
  python3 scripts/local_preview.py --downloads-dir /path/to/downloads
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOADS_DIR = "hivc_sim/results/turn_game/downloads"


class PreviewServer:
    def __init__(self, downloads_dir: Path, port: int, host: str):
        self.downloads_dir = downloads_dir
        self.port = port
        self.host = host

    def _load_html(self) -> bytes:
        html_path = SCRIPT_DIR / "local_preview.html"
        if html_path.exists():
            return html_path.read_bytes()
        # バックアップ: シンプルなメッセージを返す
        return b"<html><body><p>local_preview.html not found</p></body></html>"

    def _list_runs(self) -> list[dict]:
        runs = []
        if self.downloads_dir.exists():
            for p in sorted(self.downloads_dir.iterdir()):
                if p.is_dir():
                    manifest = p / "manifest.json"
                    acquired_at = None
                    if manifest.exists():
                        try:
                            data = json.loads(manifest.read_text(encoding="utf-8"))
                            acquired_at = data.get("acquired_at")
                        except Exception:
                            pass
                    files = [f.name for f in sorted(p.iterdir()) if f.is_file()]
                    runs.append({
                        "run_id": p.name,
                        "path": str(p),
                        "acquired_at": acquired_at,
                        "files": files,
                    })
        return runs

    def _is_valid_run_id(self, run_id: str) -> bool:
        if not run_id or run_id in (".", ".."):
            return False
        if os.path.sep in run_id or (os.path.altsep and os.path.altsep in run_id) or "\\" in run_id:
            return False
        if "/" in run_id or ".." in run_id.split(os.path.sep):
            return False
        return True

    def _safe_path(self, *parts: str) -> Path | None:
        try:
            target = self.downloads_dir.joinpath(*parts).resolve()
            base = self.downloads_dir.resolve()
            if base not in target.parents and target != base:
                return None
            return target
        except (OSError, ValueError):
            return None

    def _run_dir(self, run_id: str) -> Path | None:
        if not self._is_valid_run_id(run_id):
            return None
        run_dir = self._safe_path(run_id)
        if run_dir is None or not run_dir.exists() or not run_dir.is_dir():
            return None
        return run_dir

    def _list_run_files(self, run_id: str) -> list[dict] | None:
        run_dir = self._run_dir(run_id)
        if not run_dir:
            return None
        files = []
        for p in sorted(run_dir.iterdir()):
            if p.is_file():
                files.append({
                    "name": p.name,
                    "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                })
        return files

    def _read_file(self, run_id: str, filename: str) -> bytes | None:
        if not self._is_valid_run_id(run_id):
            return None
        if os.path.sep in filename or (os.path.altsep and os.path.altsep in filename) or "\\" in filename:
            return None
        if filename in (".", "..") or "/" in filename:
            return None
        target = self._safe_path(run_id, filename)
        if target is None or not target.is_file():
            return None
        try:
            return target.read_bytes()
        except Exception:
            return None

    def _make_handler(self):
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # noqa: D401
                print(fmt % args)

            def _json_response(self, status: int, data: object) -> None:
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _text_response(self, status: int, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _not_found(self, message: str = "Not found") -> None:
                body = message.encode("utf-8")
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)

                if path == "/" or path == "/visualize":
                    html = server._load_html()
                    self._text_response(200, html, "text/html; charset=utf-8")
                    return

                if path == "/api/runs":
                    self._json_response(200, server._list_runs())
                    return

                if path.startswith("/api/runs/"):
                    parts = path.split("/")
                    if len(parts) >= 4:
                        run_id = parts[3]
                        rest = "/".join(parts[4:])
                        if rest == "files":
                            files = server._list_run_files(run_id)
                            if files is None:
                                self._not_found("Run not found")
                            else:
                                self._json_response(200, files)
                            return
                        if rest == "manifest":
                            data = server._read_file(run_id, "manifest.json")
                            if data is None:
                                self._not_found("Manifest not found")
                            else:
                                self._text_response(200, data, "application/json; charset=utf-8")
                            return
                        if rest == "summary":
                            data = server._read_file(run_id, "summary.csv")
                            if data is None:
                                self._not_found("Summary not found")
                            else:
                                self._text_response(200, data, "text/csv; charset=utf-8")
                            return
                        if rest.startswith("file/"):
                            filename = rest[5:]
                            if not filename or filename in ("..", ".") or "/" in filename:
                                self._not_found("Invalid filename")
                                return
                            data = server._read_file(run_id, filename)
                            if data is None:
                                self._not_found("File not found")
                            else:
                                content_type, _ = mimetypes.guess_type(filename)
                                content_type = content_type or "application/octet-stream"
                                self._text_response(200, data, content_type)
                            return

                self._not_found()

        return _Handler

    def run(self) -> None:
        handler = self._make_handler()
        server = HTTPServer((self.host, self.port), handler)
        print(f"Local preview server running at http://{self.host}:{self.port}")
        print(f"Downloads directory: {self.downloads_dir}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ローカル実験ログプレビューサーバーを起動する。"
    )
    parser.add_argument("--port", type=int, default=8765, help="待受ポート")
    parser.add_argument("--host", default="127.0.0.1", help="待受ホスト（既定: 127.0.0.1）")
    parser.add_argument("--downloads-dir", default=DEFAULT_DOWNLOADS_DIR,
                        help="runディレクトリの親ディレクトリ")
    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir)
    if not downloads_dir.is_absolute():
        downloads_dir = REPO_ROOT / downloads_dir

    if not downloads_dir.exists():
        downloads_dir.mkdir(parents=True, exist_ok=True)

    server = PreviewServer(downloads_dir, args.port, args.host)
    server.run()


if __name__ == "__main__":
    main()
