"""REQUIREMENTS §7 の3条件（control / consulting / hivc_d）バッチ実験ランナー。

全条件に同一のゲームルール・初期状態・行動一覧・過去プレイ統計を与え、条件間で
差をつけるのは合意形成時に与える手順知識のみとする（REQUIREMENTS §7）。

モデルは1回だけロードし、条件 × N ゲームを反復する。出力:
  - {output_dir}/{condition}_games.csv   ターン別生ログ（§6 記録項目込み）
  - {output_dir}/all_games.csv           全条件結合
  - {output_dir}/summary.csv             条件別 §6 主要評価指標

使い方:
  # configファイルで実行（推奨）
  python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml

  # CLI引数で個別上書き
  python3 scripts/qwen_two_agent_experiment.py --config configs/experiment.yaml --games 50

  # configなし（従来通り全引数をCLIで指定）
  python3 scripts/qwen_two_agent_experiment.py --model-path ~/models/Qwen3-14B --games 30
"""
from __future__ import annotations

import argparse
import atexit
import csv
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hivc_sim"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import merge_config_and_cli, resolve_path  # noqa: E402
from llm_turn_game_common import (  # noqa: E402
    CONDITIONS,
    add_persona_args,
    append_profile_assignment,
    build_value_manifest,
    condition_order_for_seed,
    _git_commit,
    load_model,
    load_personas,
    resolve_role_file_path,
    run_one_game,
    write_value_manifest,
)
from turn_game_metrics import compute_summary_metrics  # noqa: E402


# 引数の型定義（config読み込み時の型変換に使用）
ARG_TYPES: dict[str, type] = {
    "model_path": str,
    "conditions": list,
    "games": int,
    "seed": int,
    "max_new_tokens": int,
    "max_discussion_turns": int,
    "discussion_token_budget": int,
    "evaluator_rollouts": int,
    "output_dir": str,
    "live_jsonl": str,
    "role_file": str,
    "alpha_role_key": str,
    "beta_role_key": str,
    "personas_file": str,
    "persona_params_file": str,
    "alpha_persona": str,
    "beta_persona": str,
    "random_persona": bool,
    "random_seed": int,
    "role_value_mode": str,
    "enable_thinking": bool,
    "thinking_budget": int,
    "decision_schedule_seed": int,
    "max_decision_opportunities": int,
}

# CLI引数未指定時の既定値（configなし実行時のフォールバック）
CLI_DEFAULTS: dict[str, object] = {
    "model_path": "/home/student222/models/Qwen3-14B",
    "conditions": list(CONDITIONS),
    "games": 30,
    "seed": 42,
    "max_new_tokens": 96,
    "max_discussion_turns": 6,
    "discussion_token_budget": 768,
    "evaluator_rollouts": 24,
    "output_dir": "hivc_sim/results/turn_game/experiment",
    "live_jsonl": None,
    "role_file": None,
    "alpha_role_key": None,
    "beta_role_key": None,
    "personas_file": None,
    "persona_params_file": None,
    "alpha_persona": None,
    "beta_persona": None,
    "random_persona": False,
    "random_seed": None,
    "role_value_mode": "legacy_hard",
    "enable_thinking": False,
    "thinking_budget": None,
    "decision_schedule_seed": 0,
    "max_decision_opportunities": 3,
}

RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
WORKFLOW_BOOTSTRAP_FILES = {"run_id", "started_at", "command.txt", "run.log", "pid"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_direct_run(
    output_dir: Path,
    run_id: str | None,
    live_jsonl: str | None,
) -> tuple[str, str | None, dict[str, object], Path]:
    """Reserve an isolated run directory and an exclusive stream path."""
    resolved_run_id = run_id or output_dir.name
    if not RUN_ID_RE.fullmatch(resolved_run_id):
        raise ValueError(f"invalid run_id: {resolved_run_id!r}")
    if output_dir.exists():
        unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in WORKFLOW_BOOTSTRAP_FILES)
        if unexpected:
            raise FileExistsError(
                f"output_dir already contains run artifacts: {output_dir} ({', '.join(unexpected)})"
            )
    else:
        output_dir.mkdir(parents=True)
    bootstrap_run_id = output_dir / "run_id"
    if bootstrap_run_id.exists() and bootstrap_run_id.read_text(encoding="utf-8").strip() != resolved_run_id:
        raise ValueError("output_dir run_id does not match --run-id")
    if not bootstrap_run_id.exists():
        bootstrap_run_id.write_text(resolved_run_id + "\n", encoding="utf-8")

    live_path: Path | None = None
    if live_jsonl:
        live_path = Path(live_jsonl)
        if not live_path.is_absolute():
            live_path = REPO_ROOT / live_path
        live_path = live_path.resolve()
        if live_path.parent != output_dir.resolve():
            raise ValueError("--live-jsonl must be inside the isolated output_dir")
        try:
            live_path.touch(exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to reuse existing stream: {live_path}") from exc

    metadata_path = output_dir / "run_metadata.json"
    metadata = {
        "run_id": resolved_run_id,
        "runner": "qwen_two_agent_experiment-v3",
        "status": "running",
        "started_at": _now_iso(),
        "completed_at": None,
        "git_commit": _git_commit(),
        "output_dir": str(output_dir),
        "stream": live_path.name if live_path else None,
        "value_manifest": "value_manifest.json",
        "artifacts": {},
    }
    _write_run_metadata(metadata_path, metadata)
    return resolved_run_id, str(live_path) if live_path else None, metadata, metadata_path


def _complete_direct_run(
    output_dir: Path,
    metadata: dict[str, object],
    metadata_path: Path,
    artifact_names: list[str],
) -> None:
    artifacts = {}
    for name in artifact_names:
        path = output_dir / name
        if path.is_file():
            artifacts[name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    metadata.update({"status": "completed", "completed_at": _now_iso(), "artifacts": artifacts})
    _write_run_metadata(metadata_path, metadata)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 3-condition HIVC-D turn-game batch experiment."
    )
    parser.add_argument("--config", default=None,
                        help="YAML設定ファイルパス（指定時はその値が既定値になる）")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--conditions", nargs="*", default=None,
                        choices=list(CONDITIONS) + ["all"])
    parser.add_argument("--games", type=int, default=None, help="1条件あたりのゲーム数")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-discussion-turns", type=int, default=None)
    parser.add_argument("--discussion-token-budget", type=int, default=None)
    parser.add_argument("--evaluator-rollouts", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None, help="run metadataに保存する識別子（既定: output-dir名）")
    parser.add_argument("--live-jsonl", default=None,
                        help="各ターン終了時にJSONLを追記するパス（visualize_game.html ライブモード用）")
    parser.add_argument("--enable-thinking", default=None, type=str,
                        help="Qwen3 thinkingモードを有効化 (true/false)。config未指定時は false。")
    parser.add_argument("--thinking-budget", default=None, type=int,
                        help="thinkingモードの推論トークン上限（Qwen3 thinking_budget）。未指定時は制限なし。")
    add_persona_args(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # CLI引数で明示的に指定されたもの（None以外）を抽出
    cli_overrides: dict[str, object] = {}
    for key in ARG_TYPES:
        value = getattr(args, key, None)
        if value is not None:
            cli_overrides[key] = value

    # config → CLI の順で優先される設定dictを構築
    cfg = merge_config_and_cli(args.config, cli_overrides, CLI_DEFAULTS, ARG_TYPES)

    # YAML 内の ~/... はシェルを経由しないため自動展開されない。
    # ローカルモデルを指定する設定でも Transformers に正しい絶対パスを渡す。
    cfg["model_path"] = str(Path(str(cfg["model_path"])).expanduser())

    # role_value_mode と role_file の整合性を取る（--role-value-mode 単独指定時の自動選択）
    cfg["role_file"] = resolve_role_file_path(cfg.get("role_file"), cfg.get("role_value_mode"))

    # conditions の "all" 処理
    conditions = cfg["conditions"]
    if "all" in conditions:
        conditions = list(CONDITIONS)
    else:
        # conditions を正規化（list化）
        if isinstance(conditions, str):
            conditions = [c.strip() for c in conditions.split(",") if c.strip()]

    # personas 読み込み用に args を cfg の値で再構築
    for key in ("role_file", "alpha_role_key", "beta_role_key",
                "personas_file", "persona_params_file",
                "alpha_persona", "beta_persona",
                "random_persona", "random_seed", "role_value_mode"):
        setattr(args, key, cfg[key])
    personas, persona_params, role_keys = load_personas(args)
    random_persona = cfg["random_persona"]

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    run_id, live_jsonl_path, run_metadata, run_metadata_path = _prepare_direct_run(
        output_dir, args.run_id, cfg["live_jsonl"]
    )
    print(f"Run ID: {run_id}")
    completion_state = {"completed": False}

    def mark_failed_at_exit() -> None:
        if completion_state["completed"]:
            return
        run_metadata.update({"status": "failed", "completed_at": _now_iso()})
        _write_run_metadata(run_metadata_path, run_metadata)

    atexit.register(mark_failed_at_exit)
    value_manifest_path = output_dir / "value_manifest.json"
    value_manifest = build_value_manifest(
            cfg,
            personas,
            persona_params,
            role_keys,
            role_value_mode=str(cfg["role_value_mode"]),
            framework_ids=list(conditions),
            runner_version="qwen_two_agent_experiment-v2",
    )
    write_value_manifest(value_manifest_path, value_manifest)

    if live_jsonl_path:
        print(f"Live JSONL stream: {live_jsonl_path}")

    print(f"Loading model: {cfg['model_path']}")
    model, tokenizer = load_model(cfg["model_path"])

    thinking_mode = cfg["enable_thinking"]
    thinking_budget = cfg["thinking_budget"]
    print(f"Thinking mode: {thinking_mode}" + (f" (budget={thinking_budget})" if thinking_budget is not None else ""))

    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    condition_rows: dict[str, list[dict[str, object]]] = {condition: [] for condition in conditions}

    for game_index in range(cfg["games"]):
        game_seed = cfg["seed"] + game_index
        # random_persona の場合、同一seedの全条件で同じ割付けを使う。
        if random_persona:
            args.random_seed = cfg["random_seed"] if cfg["random_seed"] is not None else game_seed
            personas, persona_params, role_keys = load_personas(args)
            print(f"  [random_persona seed={game_seed}] alpha={role_keys['alpha']} beta={role_keys['beta']}")
        seed_conditions = condition_order_for_seed(list(conditions), game_seed)
        print(f"\n=== seed: {game_seed}; condition order: {seed_conditions} ===")
        for condition in seed_conditions:
            rows = run_one_game(
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
                live_jsonl_path=live_jsonl_path,
                enable_thinking=cfg["enable_thinking"],
                thinking_budget=cfg["thinking_budget"],
                decision_schedule_seed=cfg["decision_schedule_seed"],
                max_decision_opportunities=cfg["max_decision_opportunities"],
                role_value_mode=cfg["role_value_mode"],
            )
            # seed・condition ごとの割当レコードを権威ソースとして保存する
            append_profile_assignment(value_manifest, game_seed, personas, persona_params, role_keys, condition=condition)
            condition_rows[condition].extend(rows)
            all_rows.extend(rows)
        write_value_manifest(value_manifest_path, value_manifest)

    for condition in conditions:
        cond_rows = condition_rows[condition]
        _write_csv(output_dir / f"{condition}_games.csv", cond_rows)
        metrics = compute_summary_metrics(cond_rows)
        summary_rows.append({"condition": condition, "games": cfg["games"], **metrics})
        print(f"[{condition}] summary: {metrics}")

    _write_csv(output_dir / "all_games.csv", all_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    artifact_names = [f"{condition}_games.csv" for condition in conditions]
    artifact_names.extend(["all_games.csv", "summary.csv", "value_manifest.json"])
    if live_jsonl_path:
        artifact_names.append(Path(live_jsonl_path).name)
    _complete_direct_run(output_dir, run_metadata, run_metadata_path, artifact_names)
    completion_state["completed"] = True
    print(f"\nSaved {len(all_rows)} turn rows to {output_dir}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
