"""REQUIREMENTS §10: LLM 実験ログと探索ベース評価ログを結合するパイプライン。

LLM 実験のターン別ログ（qwen_two_agent_experiment.py の all_games.csv 等）と、
policy ロールアウトの評価ログ（turn_game_cli.py の *_games.csv）を (seed, turn) で
結合し、各 LLM 判断に対する参照方策の行動・best_action・regret を横並びにする。

これにより「LLM の group_action が参照方策とどこで乖離したか」「LLM regret と
参照方策 regret の差」を 1 行で比較できる。

使い方:
  python3 hivc_sim/join_llm_eval.py \
    --llm-csv hivc_sim/results/turn_game/experiment/all_games.csv \
    --eval-csv hivc_sim/results/turn_game/heuristic_games.csv \
    --eval-label heuristic \
    --output hivc_sim/results/turn_game/experiment/joined_heuristic.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from turn_game_metrics import compute_summary_metrics  # noqa: E402


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _key(row: dict) -> tuple[str, str]:
    seed = str(row.get("seed", row.get("game_id", "")))
    turn = str(row.get("turn", ""))
    return (seed, turn)


def join(llm_rows: list[dict], eval_rows: list[dict], label: str) -> list[dict]:
    eval_by_key: dict[tuple[str, str], dict] = {}
    for row in eval_rows:
        eval_by_key[_key(row)] = row

    joined: list[dict] = []
    for row in llm_rows:
        key = _key(row)
        ev = eval_by_key.get(key, {})
        merged = dict(row)
        for col in ("action", "best_action", "acceptable_actions", "regret", "q_values", "terminal_score"):
            if col in ev:
                merged[f"{label}_{col}"] = ev[col]
        merged[f"{label}_matched"] = str(int(ev != {}))
        joined.append(merged)
    return joined


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join LLM experiment logs with rollout-based eval logs.")
    parser.add_argument("--llm-csv", required=True)
    parser.add_argument("--eval-csv", required=True, help="policy rollout games.csv (e.g. heuristic_games.csv)")
    parser.add_argument("--eval-label", default="eval", help="prefix for eval columns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None, help="optional joined summary CSV path")
    args = parser.parse_args()

    llm_path = Path(args.llm_csv)
    eval_path = Path(args.eval_csv)
    out_path = Path(args.output)
    if not llm_path.is_absolute():
        llm_path = Path(__file__).resolve().parent.parent / llm_path
    if not eval_path.is_absolute():
        eval_path = Path(__file__).resolve().parent.parent / eval_path
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent.parent / out_path

    llm_rows = _read_csv(llm_path)
    eval_rows = _read_csv(eval_path)
    joined = join(llm_rows, eval_rows, args.eval_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, joined)
    print(f"Joined {len(joined)} rows (llm={len(llm_rows)}, eval={len(eval_rows)}) -> {out_path}")

    if args.summary:
        summary = compute_summary_metrics(joined)
        summary_row = {"eval_label": args.eval_label, "rows": len(joined), **summary}
        summary_path = Path(args.summary)
        if not summary_path.is_absolute():
            summary_path = Path(__file__).resolve().parent.parent / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)
        print(f"Saved joined summary -> {summary_path}")
        print(summary_row)


if __name__ == "__main__":
    main()
