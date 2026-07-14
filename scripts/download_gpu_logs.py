"""GPUサーバーで完了した実験ログをローカルMacに取得し、run単位で保存する。

使い方:
  # 既定の実験出力（configs/gpu_server.yaml + 既定output_dir）を取得
  python3 scripts/download_gpu_logs.py

  # 明示的にリモート出力ディレクトリとrun名を指定
  python3 scripts/download_gpu_logs.py \
    --remote-output-dir hivc_sim/results/turn_game/experiment \
    --run-id 2026-07-13-run1

  # ローカル保存先を変更
  python3 scripts/download_gpu_logs.py --local-dir /path/to/downloads

  # 既存runへの上書きを無条件に許可
  python3 scripts/download_gpu_logs.py --overwrite

取得内容:
  - {control,consulting,hivc_d}_games.csv
  - all_games.csv
  - summary.csv
  - manifest.json（取得メタデータ）
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_yaml  # noqa: E402


DEFAULT_REMOTE_OUTPUT_DIR = "hivc_sim/results/turn_game/experiment"
DEFAULT_LOCAL_DOWNLOADS_DIR = "hivc_sim/results/turn_game/downloads"
REQUIRED_FILES = ["all_games.csv", "summary.csv"]
CONDITION_FILES = ["control_games.csv", "consulting_games.csv", "hivc_d_games.csv"]


def load_gpu_config(config_path: str = "configs/gpu_server.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        print(f"ERROR: GPU server config not found: {path}", file=sys.stderr)
        print("  configs/gpu_server.yaml を作成してください。", file=sys.stderr)
        sys.exit(1)
    return load_yaml(path)


def generate_run_id() -> str:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    return now.strftime("%Y%m%d-%H%M%S")


def file_list_with_sizes(dir_path: Path) -> list[dict]:
    files = []
    for p in sorted(dir_path.iterdir()):
        if p.is_file():
            files.append({"name": p.name, "size": p.stat().st_size, "path": str(p)})
    return files


def run_rsync(
    cfg: dict,
    remote_output_dir: str,
    local_dir: Path,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """rsyncでGPUサーバーの実験出力をローカルrunディレクトリへコピーする。"""
    remote_project_dir = cfg.get("remote_project_dir", "~/projects/hivc-d-verification")
    remote_path = f"{remote_project_dir}/{remote_output_dir}/"
    remote_path = remote_path.replace("//", "/")

    ssh_key = os.path.expanduser(cfg["ssh_key"])
    ssh_opts = f"-p {cfg['ssh_port']} -i {ssh_key} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10"
    target = f"{cfg['ssh_user']}@{cfg['ssh_host']}:{remote_path}"

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e", f"ssh {ssh_opts}",
        target,
        str(local_dir) + "/",
    ]
    if dry_run:
        cmd.append("--dry-run")

    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, text=True, encoding="utf-8")


def validate_run_dir(local_dir: Path) -> list[str]:
    """runディレクトリに必要なCSVが含まれているか確認する。"""
    missing = []
    for name in REQUIRED_FILES:
        if not (local_dir / name).exists():
            missing.append(name)
    if not missing:
        return missing
    # all_games.csv が無い場合、少なくとも1つの condition_games.csv があれば許容する
    if "all_games.csv" in missing:
        if any((local_dir / name).exists() for name in CONDITION_FILES):
            missing.remove("all_games.csv")
    return missing


def write_manifest(
    local_dir: Path,
    cfg: dict,
    remote_output_dir: str,
    files: list[dict],
    run_id: str,
) -> None:
    manifest = {
        "run_id": run_id,
        "acquired_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
        "gpu_host": cfg.get("ssh_host"),
        "gpu_user": cfg.get("ssh_user"),
        "gpu_port": cfg.get("ssh_port"),
        "remote_project_dir": cfg.get("remote_project_dir"),
        "remote_output_dir": remote_output_dir,
        "local_destination": str(local_dir),
        "tool_version": "1.0.0",
        "tool_path": "scripts/download_gpu_logs.py",
        "files": files,
    }
    (local_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def confirm_overwrite(local_dir: Path) -> bool:
    print(f"WARNING: 既存のrunディレクトリが存在します: {local_dir}")
    print("  既存ディレクトリを削除して上書きしますか？ [y/N]", end=" ")
    response = input().strip().lower()
    return response in ("y", "yes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPUサーバーの完了済み実験ログをローカルに取得する。"
    )
    parser.add_argument("--gpu-config", default="configs/gpu_server.yaml",
                        help="GPUサーバーSSH接続設定ファイル")
    parser.add_argument("--remote-output-dir", default=DEFAULT_REMOTE_OUTPUT_DIR,
                        help="GPUサーバー上の実験出力ディレクトリ（remote_project_dirからの相対パス）")
    parser.add_argument("--run-id", default=None,
                        help="ローカル保存するrun名（既定: 取得時刻）")
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DOWNLOADS_DIR,
                        help="ローカル保存先の親ディレクトリ")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存runディレクトリを上書きする")
    parser.add_argument("--dry-run", action="store_true",
                        help="rsyncのドライランを実行する（実際にはコピーしない）")
    args = parser.parse_args()

    cfg = load_gpu_config(args.gpu_config)
    run_id = args.run_id or generate_run_id()

    # run_id は単一のパス要素でなければならない
    if (
        os.path.sep in run_id
        or (os.path.altsep and os.path.altsep in run_id)
        or "\\" in run_id
        or run_id in (".", "..")
    ):
        print(f"ERROR: invalid run_id: {run_id}", file=sys.stderr)
        sys.exit(1)

    local_dir = Path(args.local_dir)
    if not local_dir.is_absolute():
        local_dir = REPO_ROOT / local_dir
    run_dir = local_dir / run_id

    # local_dir 配下に run_dir が解決されることを検証
    try:
        run_dir = run_dir.resolve()
        local_dir = local_dir.resolve()
        if local_dir not in run_dir.parents and run_dir != local_dir:
            print(f"ERROR: run_dir {run_dir} is outside local_dir {local_dir}", file=sys.stderr)
            sys.exit(1)
    except OSError:
        print(f"ERROR: failed to resolve run_dir {run_dir}", file=sys.stderr)
        sys.exit(1)

    if run_dir.exists() and not args.dry_run:
        if not args.overwrite:
            if not confirm_overwrite(run_dir):
                print("Cancelled.", file=sys.stderr)
                sys.exit(0)
        import shutil
        shutil.rmtree(run_dir)

    if args.dry_run:
        print(f"[dry-run] run directory would be: {run_dir}")
        result = run_rsync(cfg, args.remote_output_dir, run_dir, dry_run=True)
        if result.returncode != 0:
            print(f"ERROR: rsync dry-run failed (exit {result.returncode})", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            sys.exit(1)
        print("Dry-run completed. No files were copied.")
        sys.exit(0)

    run_dir.mkdir(parents=True, exist_ok=True)

    result = run_rsync(cfg, args.remote_output_dir, run_dir, dry_run=False)
    if result.returncode != 0:
        print(f"ERROR: rsync failed (exit {result.returncode})", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    missing = validate_run_dir(run_dir)
    if missing:
        print(f"ERROR: 必要なCSVが取得されませんでした: {missing}", file=sys.stderr)
        print(f"  取得先: {run_dir}", file=sys.stderr)
        print("  リモートに実験が完了しているか、remote-output-dirが正しいか確認してください。", file=sys.stderr)
        sys.exit(1)

    files = file_list_with_sizes(run_dir)
    write_manifest(run_dir, cfg, args.remote_output_dir, files, run_id)

    print(f"\nSaved to: {run_dir}")
    for f in files:
        print(f"  {f['name']}: {f['size']} bytes")


if __name__ == "__main__":
    main()
