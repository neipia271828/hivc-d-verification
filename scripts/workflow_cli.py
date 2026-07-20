"""`uv run` から使うGPU実験ワークフローCLI。"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from .config_loader import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ORIGIN = "git@github-neipia:neipia271828/hivc-d-verification.git"
DEFAULT_GPU_CONFIG = "configs/gpu_server.yaml"
DEFAULT_EXPERIMENT_CONFIG = "configs/experiment.yaml"
DEFAULT_REMOTE_RUNS_ROOT = "hivc_sim/results/turn_game/experiment/runs"
DEFAULT_DOWNLOADS_DIR = "hivc_sim/results/turn_game/downloads"
STATE_PATH = REPO_ROOT / ".hivc-workflow.json"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
NEIPIA_PUBLIC_KEY_PATH = Path("~/.ssh/id_ed25519_neipia.pub").expanduser()


class WorkflowError(RuntimeError):
    """利用者にそのまま表示できるワークフローエラー。"""


def _load_gpu_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise WorkflowError(f"GPU設定が見つかりません: {path}")
    cfg = load_yaml(path)
    required = ("ssh_host", "ssh_port", "ssh_user", "ssh_key", "remote_project_dir")
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise WorkflowError(f"GPU設定に必須項目がありません: {', '.join(missing)}")
    return cfg


def _ssh_command(
    cfg: dict[str, Any],
    remote_command: str,
    *,
    forward_agent: bool = False,
) -> list[str]:
    command = [
        "ssh",
        "-p",
        str(cfg["ssh_port"]),
        "-i",
        os.path.expanduser(str(cfg["ssh_key"])),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
    ]
    if forward_agent:
        command.append("-A")
    command.extend([f"{cfg['ssh_user']}@{cfg['ssh_host']}", remote_command])
    return command


def _remote_project_shell(cfg: dict[str, Any]) -> str:
    path = str(cfg["remote_project_dir"])
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def _run(
    command: list[str],
    *,
    capture_output: bool = False,
    check: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=capture_output,
        check=check,
        timeout=timeout,
    )


def _remote(
    cfg: dict[str, Any],
    remote_command: str,
    *,
    capture_output: bool = False,
    timeout: int | None = None,
    forward_agent: bool = False,
) -> subprocess.CompletedProcess[str]:
    return _run(
        _ssh_command(cfg, remote_command, forward_agent=forward_agent),
        capture_output=capture_output,
        timeout=timeout,
    )


def _git_output(*args: str) -> str:
    result = _run(["git", *args], capture_output=True)
    if result.returncode != 0:
        raise WorkflowError(result.stderr.strip() or f"git {' '.join(args)} に失敗しました")
    return result.stdout.strip()


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise WorkflowError(
            "run IDには英数字、ピリオド、アンダースコア、ハイフンだけを使用してください"
        )
    return run_id


def _new_run_id() -> str:
    return dt.datetime.now().astimezone().strftime("episode-%Y%m%d-%H%M%S")


def _read_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {}
    try:
        loaded = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_state(**updates: Any) -> None:
    state = _read_state()
    state.update(updates)
    state["updated_at"] = dt.datetime.now().astimezone().isoformat()
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_run_id(explicit: str | None) -> str:
    if explicit:
        return _validate_run_id(explicit)
    run_id = _read_state().get("last_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise WorkflowError("run IDがありません。先に `uv run experiment` を実行してください")
    return _validate_run_id(run_id)


def _print_command(command: list[str]) -> None:
    print("$ " + shlex.join(command))


def _neipia_public_key() -> str:
    if not NEIPIA_PUBLIC_KEY_PATH.is_file():
        raise WorkflowError(f"GitHub公開鍵が見つかりません: {NEIPIA_PUBLIC_KEY_PATH}")
    parts = NEIPIA_PUBLIC_KEY_PATH.read_text(encoding="utf-8").split()
    if len(parts) < 2:
        raise WorkflowError(f"GitHub公開鍵の形式が不正です: {NEIPIA_PUBLIC_KEY_PATH}")
    agent = _run(["ssh-add", "-L"], capture_output=True)
    if agent.returncode != 0 or not any(
        line.split()[1:2] == parts[1:2] for line in agent.stdout.splitlines()
    ):
        raise WorkflowError(
            "github-neipia の鍵がssh-agentにありません。"
            "`ssh-add ~/.ssh/id_ed25519_neipia` を実行してください。"
        )
    # コメントは同期に不要で、改行を含む古い公開鍵ファイルもあるため鍵種別と本体だけを使う。
    return " ".join(parts[:2])


def _sync_remote_command(
    cfg: dict[str, Any],
    branch: str,
    local_head: str,
    public_key: str,
) -> str:
    project = _remote_project_shell(cfg)
    expected = shlex.quote(EXPECTED_ORIGIN)
    github_url = "git@github.com:neipia271828/hivc-d-verification.git"
    return "\n".join(
        [
            "set -eu",
            f"cd {project}",
            "key_file=$(mktemp)",
            "ssh_config=$(mktemp)",
            "bootstrap_dir=''",
            "cleanup() { rm -f \"$key_file\" \"$ssh_config\"; [ -z \"$bootstrap_dir\" ] || rm -rf \"$bootstrap_dir\"; }",
            "trap cleanup EXIT HUP INT TERM",
            "umask 077",
            f"printf '%s\\n' {shlex.quote(public_key)} > \"$key_file\"",
            "cat > \"$ssh_config\" <<EOF",
            "Host github-neipia github.com",
            "  HostName github.com",
            "  User git",
            "  IdentityFile $key_file",
            "  IdentitiesOnly yes",
            "  StrictHostKeyChecking accept-new",
            "EOF",
            "if [ ! -d .git ]; then",
            "  echo 'GPU側に .git がないためGit管理を初期化します'",
            "  bootstrap_dir=$(mktemp -d)",
            f"  GIT_SSH_COMMAND=\"ssh -F $ssh_config\" git clone --quiet --branch {shlex.quote(branch)} --single-branch {shlex.quote(github_url)} \"$bootstrap_dir/repo\"",
            "  mv \"$bootstrap_dir/repo/.git\" .git",
            f"  git remote set-url origin {expected}",
            f"  git reset --hard origin/{shlex.quote(branch)}",
            "fi",
            "remote_url=$(git remote get-url origin)",
            f"if [ \"$remote_url\" != {expected} ]; then",
            "  echo \"ERROR: GPU側originが不正です: $remote_url\" >&2",
            "  exit 21",
            "fi",
            f"GIT_SSH_COMMAND=\"ssh -F $ssh_config\" git pull --ff-only origin {shlex.quote(branch)}",
            "remote_head=$(git rev-parse HEAD)",
            f"if [ \"$remote_head\" != {shlex.quote(local_head)} ]; then",
            "  echo \"ERROR: 同期後HEADが一致しません: $remote_head\" >&2",
            "  exit 22",
            "fi",
            "echo \"Synced HEAD: $remote_head\"",
        ]
    )


def _local_commit_commands(
    dirty: str,
    *,
    allow_dirty: bool,
    message: str,
) -> list[list[str]]:
    """sync前に必要なstage/commitコマンドを返す。"""
    if not dirty or allow_dirty:
        return []
    return [
        ["git", "add", "-A"],
        ["git", "commit", "-m", message],
    ]


def sync_main() -> None:
    parser = argparse.ArgumentParser(description="変更を自動commitし、GPUサーバーへGit同期する。")
    parser.add_argument("--gpu-config", default=DEFAULT_GPU_CONFIG)
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--message",
        default="chore: sync experiment workflow",
        help="自動commitのメッセージ",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="自動commitせず、未コミット変更を同期対象外として現在のHEADだけを同期する",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        cfg = _load_gpu_config(args.gpu_config)
        public_key = _neipia_public_key()
        branch = _git_output("branch", "--show-current")
        if branch != args.branch:
            raise WorkflowError(f"現在のブランチは {branch!r} です。{args.branch!r} で実行してください")

        origin = _git_output("remote", "get-url", "origin")
        if origin != EXPECTED_ORIGIN:
            raise WorkflowError(f"originが規定値と異なります: {origin}")

        commit_message = args.message.strip()
        if not commit_message:
            raise WorkflowError("--message を空にはできません")

        dirty = _git_output("status", "--porcelain=v1")
        commit_commands = _local_commit_commands(
            dirty,
            allow_dirty=args.allow_dirty,
            message=commit_message,
        )
        current_head = _git_output("rev-parse", "HEAD")
        target_head = "NEW_COMMIT_HEAD" if commit_commands else current_head
        push_command = ["git", "push", "origin", args.branch]

        if args.dry_run:
            for command in commit_commands:
                _print_command(command)
            _print_command(push_command)
            remote_command = _sync_remote_command(cfg, args.branch, target_head, public_key)
            _print_command(_ssh_command(cfg, remote_command, forward_agent=True))
            return

        for command in commit_commands:
            _print_command(command)
            result = _run(command)
            if result.returncode != 0:
                raise WorkflowError(f"{shlex.join(command)} に失敗しました")

        if commit_commands:
            remaining = _git_output("status", "--porcelain=v1")
            if remaining:
                raise WorkflowError(
                    "自動commit後にも未コミット変更が残っています。"
                    "commit hook等による変更を確認してください。"
                )
            print(f"自動commit完了: {commit_message}")
        elif dirty and args.allow_dirty:
            print("WARNING: 未コミット変更は同期せず、現在のHEADだけを同期します")

        local_head = _git_output("rev-parse", "HEAD")
        remote_command = _sync_remote_command(cfg, args.branch, local_head, public_key)
        _print_command(push_command)
        push = _run(push_command)
        if push.returncode != 0:
            raise WorkflowError("GitHubへのpushに失敗しました")

        result = _remote(cfg, remote_command, timeout=120, forward_agent=True)
        if result.returncode != 0:
            raise WorkflowError(f"GPU側git pullに失敗しました (exit={result.returncode})")
        print(f"同期完了: {local_head}")
    except WorkflowError as exc:
        parser.error(str(exc))


def _experiment_runner_args(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    run_dir: str,
) -> list[str]:
    venv = str(cfg.get("remote_venv", ".venv"))
    command = [
        f"{venv}/bin/python",
        "-u",
        "scripts/qwen_two_agent_experiment.py",
        "--config",
        args.experiment_config,
        "--conditions",
        *args.conditions,
        "--games",
        str(args.games),
        "--output-dir",
        run_dir,
        "--live-jsonl",
        f"{run_dir}/stream.jsonl",
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if getattr(args, "role_value_mode", None) is not None:
        command.extend(["--role-value-mode", args.role_value_mode])
    if getattr(args, "role_file", None) is not None:
        command.extend(["--role-file", args.role_file])
    return command


def _parallel_runner_args(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    run_dir: str,
) -> list[str]:
    venv = str(cfg.get("remote_venv", ".venv"))
    command = [
        f"{venv}/bin/python",
        "-u",
        "scripts/qwen_parallel_experiment.py",
        "--config",
        args.experiment_config,
        "--conditions",
        *args.conditions,
        "--games",
        str(args.games),
        "--output-dir",
        run_dir,
        "--parallel",
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if getattr(args, "role_value_mode", None) is not None:
        command.extend(["--role-value-mode", args.role_value_mode])
    if getattr(args, "role_file", None) is not None:
        command.extend(["--role-file", args.role_file])
    gpus = getattr(args, "gpus", None)
    if gpus:
        command.extend(["--gpus", *[str(g) for g in gpus]])
    workers_per_gpu = getattr(args, "workers_per_gpu", 1)
    if workers_per_gpu != 1:
        command.extend(["--workers-per-gpu", str(workers_per_gpu)])
    temperature_warning = getattr(args, "temperature_warning", 80)
    if temperature_warning != 80:
        command.extend(["--temperature-warning", str(temperature_warning)])
    temperature_stop = getattr(args, "temperature_stop_scheduling", 83)
    if temperature_stop != 83:
        command.extend(["--temperature-stop-scheduling", str(temperature_stop)])
    if getattr(args, "thermal_duty_cycle", False):
        command.append("--thermal-duty-cycle")
        suspend_temperature = getattr(args, "thermal_suspend_temperature", 78)
        resume_temperature = getattr(args, "thermal_resume_temperature", 70)
        if suspend_temperature != 78:
            command.extend(["--thermal-suspend-temperature", str(suspend_temperature)])
        if resume_temperature != 70:
            command.extend(["--thermal-resume-temperature", str(resume_temperature)])
    power_limit_w = getattr(args, "power_limit_w", None)
    if power_limit_w is not None:
        command.extend(["--power-limit-w", str(power_limit_w)])
    if getattr(args, "resume", False):
        command.append("--resume")
    return command


def _start_experiment_remote_command(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
) -> tuple[str, str]:
    project = _remote_project_shell(cfg)
    run_dir = f"{DEFAULT_REMOTE_RUNS_ROOT}/{run_id}"
    scientific_gate = getattr(args, "scientific_gate", "required")
    smoke_run_id = getattr(args, "smoke_run_id", None)
    needs_smoke_gate = args.games > 1 and not getattr(args, "resume", False) and scientific_gate != "not-applicable"
    if needs_smoke_gate and not smoke_run_id:
        raise WorkflowError(
            "2ゲーム以上の本実験には --smoke-run-id が必要です。"
            "Vゲート対象外のlegacy実験だけ --scientific-gate not-applicable を明示してください。"
        )
    if getattr(args, "parallel", False):
        runner = shlex.join(_parallel_runner_args(cfg, args, run_dir))
    else:
        runner = shlex.join(_experiment_runner_args(cfg, args, run_dir))
    inner = "\n".join(
        [
            "set +e",
            f"run_dir={shlex.quote(run_dir)}",
            "trap 'printf \"%s\\n\" 1 > \"$run_dir/exit_code\"; date -Iseconds > \"$run_dir/finished_at\"; exit 1' TERM",
            runner,
            "code=$?",
            "printf '%s\\n' \"$code\" > \"$run_dir/exit_code\"",
            "date -Iseconds > \"$run_dir/finished_at\"",
            "exit \"$code\"",
        ]
    )
    if getattr(args, "resume", False):
        run_setup = [
            f"test -d {shlex.quote(run_dir)} || {{ echo 'ERROR: resume対象runが存在しません' >&2; exit 24; }}",
            f"for f in exit_code finished_at run.log; do [ ! -e {shlex.quote(run_dir)}/\"$f\" ] || mv {shlex.quote(run_dir)}/\"$f\" {shlex.quote(run_dir)}/\"$f.pre-resume\"; done",
            f"date -Iseconds > {shlex.quote(run_dir + '/resumed_at')}",
        ]
    else:
        run_setup = [
            f"test ! -e {shlex.quote(run_dir)} || {{ echo 'ERROR: run IDが既に存在します' >&2; exit 24; }}",
            f"mkdir -p {shlex.quote(run_dir)}",
            f"printf '%s\\n' {shlex.quote(run_id)} > {shlex.quote(run_dir + '/run_id')}",
            f"date -Iseconds > {shlex.quote(run_dir + '/started_at')}",
        ]
    remote = "\n".join(
        [
            "set -eu",
            f"cd {project}",
            "test -d .git || { echo 'ERROR: 先に uv run sync を完了してください' >&2; exit 20; }",
            "grep -q '_discussion_json_contract' scripts/llm_turn_game_common.py || { echo 'ERROR: JSONスキーマ修正版が未同期です' >&2; exit 21; }",
            *(
                [
                    f"test -d {shlex.quote(DEFAULT_REMOTE_RUNS_ROOT + '/' + _validate_run_id(smoke_run_id))} || {{ echo 'ERROR: smoke runが存在しません' >&2; exit 25; }}",
                    f"{shlex.quote(str(cfg.get('remote_venv', '.venv')) + '/bin/python')} scripts/validate_experiment_preflight.py "
                    f"{shlex.quote(DEFAULT_REMOTE_RUNS_ROOT + '/' + _validate_run_id(smoke_run_id))} --applicability required",
                ]
                if needs_smoke_gate
                else []
            ),
            "for pid_file in " + shlex.quote(DEFAULT_REMOTE_RUNS_ROOT) + "/*/pid; do",
            "  [ -f \"$pid_file\" ] || continue",
            "  active_run_dir=${pid_file%/pid}",
            "  [ ! -f \"$active_run_dir/exit_code\" ] || continue",
            "  active_pid=$(cat \"$pid_file\")",
            "  case \"$active_pid\" in (*[!0-9]*|'') continue ;; esac",
            "  [ \"$active_pid\" != \"$$\" ] || continue",
            "  [ \"$active_pid\" != \"$PPID\" ] || continue",
            "  kill -0 \"$active_pid\" 2>/dev/null || continue",
            "  [ -r \"/proc/$active_pid/cmdline\" ] || continue",
            "  if tr '\\000' ' ' < \"/proc/$active_pid/cmdline\" | grep -q -E 'scripts/(qwen_two_agent_experiment|qwen_parallel_experiment|qwen_parallel_worker)\\.py'; then",
            "    echo \"ERROR: 別の実験が実行中です: $(basename \"$active_run_dir\") (pid=$active_pid)\" >&2",
            "    exit 23",
            "  fi",
            "done",
            *run_setup,
            f"printf '%s\\n' {shlex.quote(runner)} > {shlex.quote(run_dir + '/command.txt')}",
            f"nohup sh -c {shlex.quote(inner)} > {shlex.quote(run_dir + '/run.log')} 2>&1 < /dev/null &",
            "pid=$!",
            f"printf '%s\\n' \"$pid\" > {shlex.quote(run_dir + '/pid')}",
            "echo \"experiment PID: $pid\"",
            f"echo {shlex.quote('run ID: ' + run_id)}",
            f"echo {shlex.quote('remote output: ' + run_dir)}",
        ]
    )
    return remote, run_dir


def _run_status_command(cfg: dict[str, Any], run_id: str) -> str:
    project = _remote_project_shell(cfg)
    run_dir = f"{DEFAULT_REMOTE_RUNS_ROOT}/{run_id}"
    return "\n".join(
        [
            "set -eu",
            f"cd {project}",
            f"test -d {shlex.quote(run_dir)} || {{ echo 'run not found' >&2; exit 4; }}",
            f"if [ -f {shlex.quote(run_dir + '/exit_code')} ]; then",
            f"  code=$(cat {shlex.quote(run_dir + '/exit_code')})",
            "  echo \"completed exit_code=$code\"",
            f"elif [ -f {shlex.quote(run_dir + '/pid')} ] && kill -0 $(cat {shlex.quote(run_dir + '/pid')}) 2>/dev/null; then",
            f"  echo \"running pid=$(cat {shlex.quote(run_dir + '/pid')})\"",
            "else",
            "  echo 'stopped without exit_code'",
            "fi",
            f"tail -n 12 {shlex.quote(run_dir + '/run.log')} 2>/dev/null || true",
        ]
    )


def _build_experiment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPUサーバーでrun単位の実験を開始・確認する。")
    parser.add_argument("--gpu-config", default=DEFAULT_GPU_CONFIG)
    parser.add_argument("--experiment-config", default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["hivc_d"],
        choices=["control", "consulting", "hivc_d", "hivc_d_prescribed_v1"],
    )
    parser.add_argument(
        "--role-value-mode",
        choices=["legacy_hard", "soft_value", "expertise_only"],
        default=None,
    )
    parser.add_argument(
        "--role-file",
        default=None,
        help="使用するRoleプロファイルファイル（未指定時は--role-value-modeに応じたデフォルトを使用）",
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument(
        "--smoke-run-id",
        help="2ゲーム以上の本実験を許可する、合格済みsmoke run ID",
    )
    parser.add_argument(
        "--scientific-gate",
        choices=["required", "not-applicable"],
        default="required",
        help="V固有preflightを要求する。対象外legacyモードのみnot-applicableを明示",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--parallel", action="store_true", help="shard並列モードを有効化")
    parser.add_argument("--gpus", nargs="+", type=int, default=None, help="使用する物理GPU ID（未指定時は自動検出）")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="GPUあたりの最大worker数（通常運用では1）")
    parser.add_argument("--temperature-warning", type=int, default=80, help="警告温度（℃）")
    parser.add_argument("--temperature-stop-scheduling", type=int, default=83, help="新規shard起動を止める温度（℃）")
    parser.add_argument(
        "--thermal-duty-cycle",
        action="store_true",
        help="78℃でworkerを一時停止し、70℃まで冷えたら再開する（sudo不要）",
    )
    parser.add_argument("--thermal-suspend-temperature", type=int, default=78)
    parser.add_argument("--thermal-resume-temperature", type=int, default=70)
    parser.add_argument(
        "--power-limit-w",
        type=int,
        default=None,
        help="並列実験中だけ各GPUへ適用する電力上限(W)。終了時に元へ復元",
    )
    parser.add_argument("--resume", action="store_true", help="成功済みshardを再利用して再開")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--logs", action="store_true")
    actions.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def experiment_main() -> None:
    parser = _build_experiment_parser()
    args = parser.parse_args()

    try:
        cfg = _load_gpu_config(args.gpu_config)
        if args.games < 1:
            raise WorkflowError("--games は1以上にしてください")
        if args.power_limit_w is not None and not args.parallel:
            raise WorkflowError("--power-limit-w は --parallel と併用してください")
        if args.power_limit_w is not None and args.power_limit_w < 1:
            raise WorkflowError("--power-limit-w は1以上にしてください")
        if args.thermal_duty_cycle and not args.parallel:
            raise WorkflowError("--thermal-duty-cycle は --parallel と併用してください")
        if args.thermal_duty_cycle and not (
            args.thermal_resume_temperature
            < args.thermal_suspend_temperature
            < args.temperature_stop_scheduling
        ):
            raise WorkflowError(
                "温度は thermal-resume < thermal-suspend < temperature-stop-scheduling の順にしてください"
            )

        if args.status or args.logs or args.stop:
            run_id = _resolve_run_id(args.run_id)
            if args.status:
                result = _remote(cfg, _run_status_command(cfg, run_id), timeout=30)
                if result.returncode != 0:
                    raise WorkflowError(f"状態確認に失敗しました (exit={result.returncode})")
                return
            project = _remote_project_shell(cfg)
            run_dir = f"{DEFAULT_REMOTE_RUNS_ROOT}/{run_id}"
            if args.logs:
                command = f"cd {project} && tail -f {shlex.quote(run_dir + '/run.log')}"
                raise SystemExit(_remote(cfg, command).returncode)
            command = "\n".join(
                [
                    "set -eu",
                    f"cd {project}",
                    f"run_dir={shlex.quote(run_dir)}",
                    "pid=$(cat \"$run_dir/pid\")",
                    "for worker_pid_file in \"$run_dir\"/shards/*/pid; do",
                    "  [ -f \"$worker_pid_file\" ] || continue",
                    "  worker_pid=$(cat \"$worker_pid_file\")",
                    "  case \"$worker_pid\" in (*[!0-9]*|'') continue ;; esac",
                    "  kill -TERM -\"$worker_pid\" 2>/dev/null || kill \"$worker_pid\" 2>/dev/null || true",
                    "  kill -CONT -\"$worker_pid\" 2>/dev/null || kill -CONT \"$worker_pid\" 2>/dev/null || true",
                    "done",
                    "for _ in 1 2 3 4 5; do",
                    "  workers_alive=0",
                    "  for worker_pid_file in \"$run_dir\"/shards/*/pid; do",
                    "    [ -f \"$worker_pid_file\" ] || continue",
                    "    worker_pid=$(cat \"$worker_pid_file\")",
                    "    kill -0 \"$worker_pid\" 2>/dev/null && workers_alive=1",
                    "  done",
                    "  [ \"$workers_alive\" = 1 ] || break",
                    "  sleep 1",
                    "done",
                    "if [ -f \"$run_dir/orchestrator_pid\" ]; then",
                    "  orchestrator_pid=$(cat \"$run_dir/orchestrator_pid\")",
                    "  case \"$orchestrator_pid\" in (*[!0-9]*|'') ;; (*) kill -TERM \"$orchestrator_pid\" 2>/dev/null || true ;; esac",
                    "fi",
                    "for _ in 1 2 3 4 5; do kill -0 \"$pid\" 2>/dev/null || break; sleep 1; done",
                    "kill \"$pid\" 2>/dev/null || true",
                    "date -Iseconds > \"$run_dir/stopped_at\"",
                    "echo \"stopped pid=$pid\"",
                ]
            )
            result = _remote(cfg, command, timeout=30)
            if result.returncode != 0:
                raise WorkflowError(f"停止に失敗しました (exit={result.returncode})")
            return

        run_id = _validate_run_id(
            args.run_id or (_resolve_run_id(None) if args.resume else _new_run_id())
        )
        remote_command, run_dir = _start_experiment_remote_command(cfg, args, run_id)
        if args.dry_run:
            _print_command(_ssh_command(cfg, remote_command))
            return

        result = _remote(cfg, remote_command, timeout=30)
        if result.returncode != 0:
            raise WorkflowError(f"実験開始に失敗しました (exit={result.returncode})")
        _write_state(last_run_id=run_id, remote_output_dir=run_dir)
        print(f"\n開始しました: {run_id}")
        print("状態確認: uv run experiment --status")
        print("ログ追跡: uv run experiment --logs")
        print("完了後取得: uv run download")
    except WorkflowError as exc:
        parser.error(str(exc))


def _latest_remote_run_id(cfg: dict[str, Any]) -> str:
    project = _remote_project_shell(cfg)
    command = (
        f"cd {project} && "
        f"find {shlex.quote(DEFAULT_REMOTE_RUNS_ROOT)} -mindepth 1 -maxdepth 1 -type d "
        "-printf '%T@ %f\\n' 2>/dev/null | sort -nr | head -1"
    )
    result = _remote(cfg, command, capture_output=True, timeout=30)
    if result.returncode != 0 or not result.stdout.strip():
        raise WorkflowError("GPU側にrunが見つかりません")
    return _validate_run_id(result.stdout.strip().split(maxsplit=1)[1])


def download_main() -> None:
    parser = argparse.ArgumentParser(description="GPUサーバーの最新実験ログをMacへ取得する。")
    parser.add_argument("--gpu-config", default=DEFAULT_GPU_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--local-dir", default=DEFAULT_DOWNLOADS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        cfg = _load_gpu_config(args.gpu_config)
        if args.run_id:
            run_id = _validate_run_id(args.run_id)
        else:
            state_run = _read_state().get("last_run_id")
            run_id = _validate_run_id(state_run) if isinstance(state_run, str) and state_run else _latest_remote_run_id(cfg)

        remote_output_dir = f"{DEFAULT_REMOTE_RUNS_ROOT}/{run_id}"
        if not args.dry_run:
            project = _remote_project_shell(cfg)
            check_command = "\n".join(
                [
                    "set -eu",
                    f"cd {project}",
                    f"test -f {shlex.quote(remote_output_dir + '/exit_code')} || {{ echo 'ERROR: 実験はまだ完了していません' >&2; exit 30; }}",
                    f"code=$(cat {shlex.quote(remote_output_dir + '/exit_code')})",
                    "test \"$code\" = 0 || { echo \"ERROR: 実験が失敗しました exit_code=$code\" >&2; exit 31; }",
                ]
            )
            check = _remote(cfg, check_command, timeout=30)
            if check.returncode == 30:
                raise WorkflowError("実験はまだ完了していません。`uv run experiment --status` で確認してください")
            if check.returncode != 0:
                raise WorkflowError("実験が失敗しています。`uv run experiment --logs` で確認してください")

        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "download_gpu_logs.py"),
            "--gpu-config",
            args.gpu_config,
            "--remote-output-dir",
            remote_output_dir,
            "--run-id",
            run_id,
            "--local-dir",
            args.local_dir,
        ]
        if args.overwrite:
            command.append("--overwrite")
        if args.dry_run:
            command.append("--dry-run")
        result = _run(command)
        if result.returncode != 0:
            raise WorkflowError(f"ログ取得に失敗しました (exit={result.returncode})")
        if not args.dry_run:
            _write_state(last_run_id=run_id, last_downloaded_run_id=run_id)
            print("可視化: uv run visualize")
    except WorkflowError as exc:
        parser.error(str(exc))


def visualize_main() -> None:
    parser = argparse.ArgumentParser(description="取得済み実験ログのローカルGUIを起動する。")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--downloads-dir", default=DEFAULT_DOWNLOADS_DIR)
    parser.add_argument("--no-open", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("--port は1から65535の範囲にしてください")
    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_open:
        threading.Timer(0.7, webbrowser.open, args=(url,)).start()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "local_preview.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--downloads-dir",
        args.downloads_dir,
    ]
    print(f"GUI: {url}")
    try:
        result = _run(command)
    except KeyboardInterrupt:
        print("\nGUIを終了しました。")
        return
    raise SystemExit(result.returncode)
