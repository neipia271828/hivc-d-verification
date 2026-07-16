"""GPU並列実験の1 shard（1 GPU・1 condition・連続seed範囲）を実行するworker。

orchestrator (`qwen_parallel_experiment.py`) から `CUDA_VISIBLE_DEVICES` で1枚のGPUだけを
見える状態にして起動される。モデルはそのGPUに単一配置され、condition 内の games 回を
連続実行して `condition_games.csv` を出力する。

終了時に `shard_manifest.json` を更新し、orchestrator が最終結合を行う。
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
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "hivc_sim"))
sys.path.insert(0, str(WORKER_DIR))

from config_loader import merge_config_and_cli  # noqa: E402
from llm_turn_game_common import (  # noqa: E402
    build_value_manifest,
    append_profile_assignment,
    condition_order_for_seed,
    load_model,
    load_personas,
    resolve_role_file_path,
    run_one_game,
    write_value_manifest,
)
from qwen_two_agent_experiment import ARG_TYPES, CLI_DEFAULTS  # noqa: E402


# pause要求を検知した際の終了コード。orchestratorはこれを paused_thermal と扱う。
PAUSED_EXIT_CODE = 2


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_of_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
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
    return _sha256_of_file(p)


def _query_gpu_info(gpu_id: int) -> dict[str, object]:
    """nvidia-smi から指定物理GPUのUUID・名前・VRAM使用量を取得する。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=uuid,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            return {
                "gpu_uuid": parts[0],
                "gpu_name": parts[1],
                "vram_used_mb": int(parts[2]),
                "vram_total_mb": int(parts[3]),
            }
    except Exception:
        pass
    return {}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _update_shard_manifest(path: Path, updates: dict[str, object]) -> None:
    data: dict[str, object] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    data.update(updates)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU並列実験の1 shard worker。"
    )
    parser.add_argument("--config", default=None, help="YAML設定ファイル")
    parser.add_argument("--condition", default=None, choices=list(CLI_DEFAULTS["conditions"]), help="単一条件を実行する場合の条件（未指定時はconfigのconditionsを使用）")
    parser.add_argument("--seed", type=int, required=True, help="shardの開始seed")
    parser.add_argument("--games", type=int, required=True, help="shardのseed数")
    parser.add_argument("--output-dir", required=True, help="shard出力ディレクトリ")
    parser.add_argument("--gpu-id", type=int, default=0, help="物理GPU ID")
    parser.add_argument("--pause-file", default=None, help="pause要求ファイルパス")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # config 読み込み。condition / seed / games / output_dir は CLI で上書き
    cli_overrides: dict[str, object] = {
        "seed": args.seed,
        "games": args.games,
        "output_dir": args.output_dir,
        "live_jsonl": None,
    }
    if args.condition is not None:
        cli_overrides["conditions"] = [args.condition]
    cfg = merge_config_and_cli(args.config, cli_overrides, CLI_DEFAULTS, ARG_TYPES)
    cfg["model_path"] = str(Path(str(cfg["model_path"])).expanduser())
    # shard worker は独自の live stream を使わない
    cfg["live_jsonl"] = None

    # role_value_mode と role_file の整合性を取る（--role-value-mode 単独指定時の自動選択）
    cfg["role_file"] = resolve_role_file_path(cfg.get("role_file"), cfg.get("role_value_mode"))

    # load_personas 用の args 風 namespace を構築
    persona_args = argparse.Namespace()
    for key, value in cfg.items():
        setattr(persona_args, key, value)

    # 固定情報を manifest へ事前書き込み
    shard_manifest_path = output_dir / "shard_manifest.json"
    git_sha = _git_sha(REPO_ROOT)
    config_hash = _sha256_of_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=str))
    framework_info = _framework_info()
    persona_hash = _persona_file_hash(cfg)

    started_at = _now_iso()
    _update_shard_manifest(
        shard_manifest_path,
        {
            "shard_id": output_dir.name,
            "condition": args.condition if args.condition is not None else ",".join(cfg.get("conditions", [])),
            "conditions": list(cfg.get("conditions", [])),
            "seed_start": args.seed,
            "seed_count": args.games,
            "gpu_id": args.gpu_id,
            "pid": os.getpid(),
            "config_path": args.config,
            "config_hash": config_hash,
            "git_sha": git_sha,
            "framework_info": framework_info,
            "persona_hash": persona_hash,
            "model_path": cfg["model_path"],
            "max_new_tokens": cfg.get("max_new_tokens"),
            "max_discussion_turns": cfg.get("max_discussion_turns"),
            "discussion_token_budget": cfg.get("discussion_token_budget"),
            "evaluator_rollouts": cfg.get("evaluator_rollouts"),
            "decision_schedule_seed": cfg.get("decision_schedule_seed"),
            "max_decision_opportunities": cfg.get("max_decision_opportunities"),
            "random_persona": cfg.get("random_persona"),
            "started_at": started_at,
            "status": "running",
        },
    )

    try:
        # Resolve and snapshot profiles before model loading so failed starts are reproducible.
        personas, persona_params, role_keys = load_personas(persona_args)
        random_persona = cfg["random_persona"]
        conditions: list[str] = list(cfg["conditions"])
        value_manifest_path = output_dir / "value_manifest.json"
        value_manifest = build_value_manifest(
                cfg,
                personas,
                persona_params,
                role_keys,
                role_value_mode=str(cfg["role_value_mode"]),
                framework_ids=conditions,
                runner_version="qwen_parallel_worker-v2",
        )
        write_value_manifest(value_manifest_path, value_manifest)

        print(f"[worker {output_dir.name}] Loading model: {cfg['model_path']} on GPU {args.gpu_id}")
        model, tokenizer = load_model(cfg["model_path"])

        # モデル読込み後のGPU情報を追記
        gpu_info = _query_gpu_info(args.gpu_id)
        _update_shard_manifest(shard_manifest_path, gpu_info)

        condition_rows: dict[str, list[dict[str, object]]] = {c: [] for c in conditions}
        pause_file = Path(args.pause_file) if args.pause_file else None

        for game_index in range(cfg["games"]):
            game_seed = cfg["seed"] + game_index
            seed_conditions = condition_order_for_seed(conditions, game_seed)
            if random_persona:
                persona_args.random_seed = cfg["random_seed"] if cfg["random_seed"] is not None else game_seed
                personas, persona_params, role_keys = load_personas(persona_args)
                print(f"[worker {output_dir.name}] random persona alpha={role_keys['alpha']} beta={role_keys['beta']}")
                append_profile_assignment(value_manifest, game_seed, personas, persona_params, role_keys)
                write_value_manifest(value_manifest_path, value_manifest)

            for cond_index, condition in enumerate(seed_conditions):
                if pause_file is not None and pause_file.exists():
                    print(f"[worker {output_dir.name}] pause requested after seed {game_seed} condition {condition}")
                    for c, c_rows in condition_rows.items():
                        if c_rows:
                            _write_csv(output_dir / f"{c}_games.csv", c_rows)
                    _update_shard_manifest(
                        shard_manifest_path,
                        {
                            "status": "paused_thermal",
                            "exit_code": PAUSED_EXIT_CODE,
                            "row_count": sum(len(r) for r in condition_rows.values()),
                            "finished_at": _now_iso(),
                            "paused_after_game_index": game_index,
                        },
                    )
                    sys.exit(PAUSED_EXIT_CODE)

                game_rows = run_one_game(
                    model,
                    tokenizer,
                    condition,
                    game_seed,
                    personas,
                    persona_params,
                    role_keys,
                    max_new_tokens=cfg["max_new_tokens"],
                    max_discussion_turns=cfg["max_discussion_turns"],
                    discussion_token_budget=cfg["discussion_token_budget"],
                    evaluator_rollouts=cfg["evaluator_rollouts"],
                    enable_thinking=cfg["enable_thinking"],
                    thinking_budget=cfg["thinking_budget"],
                    decision_schedule_seed=cfg["decision_schedule_seed"],
                    max_decision_opportunities=cfg["max_decision_opportunities"],
                    role_value_mode=cfg["role_value_mode"],
                )
                condition_rows[condition].extend(game_rows)

        output_csvs: dict[str, Path] = {}
        for condition, rows in condition_rows.items():
            output_csv = output_dir / f"{condition}_games.csv"
            _write_csv(output_csv, rows)
            output_csvs[condition] = output_csv
        print(f"[worker {output_dir.name}] Wrote {sum(len(r) for r in condition_rows.values())} rows to {output_dir}")

        _update_shard_manifest(
            shard_manifest_path,
            {
                "status": "completed",
                "exit_code": 0,
                "row_count": sum(len(r) for r in condition_rows.values()),
                "finished_at": _now_iso(),
            },
        )
        sys.exit(0)

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[worker {output_dir.name}] failed: {exc}\n{tb}", file=sys.stderr)
        # 失敗しても既存の出力を上書きせず、エラー情報のみを manifest に記録
        _update_shard_manifest(
            shard_manifest_path,
            {
                "status": "failed",
                "exit_code": 1,
                "error": str(exc),
                "traceback": tb,
                "finished_at": _now_iso(),
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
