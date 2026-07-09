from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from turn_game import (  # noqa: E402
    mcts_policy,
    play_policy_game,
    random_policy,
    heuristic_policy,
    summarize_games,
)
from turn_game_metrics import compute_summary_metrics  # noqa: E402


RESULTS_DIR = Path(__file__).parent / "results" / "turn_game"


POLICIES = {
    "random": random_policy,
    "heuristic": heuristic_policy,
    "mcts": mcts_policy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn-based HIVC-D evaluation game")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="heuristic")
    parser.add_argument("--evaluator-rollouts", type=int, default=120)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    policy = POLICIES[args.policy]
    for game_index in range(args.games):
        game_seed = args.seed + game_index
        rows.extend(
            play_policy_game(
                policy=policy,
                seed=game_seed,
                evaluator_rollouts=args.evaluator_rollouts,
            )
        )

    raw_path = RESULTS_DIR / f"{args.policy}_games.csv"
    serializable_rows = []
    for row in rows:
        serialized = dict(row)
        for column in ("state_before", "state_after", "q_values"):
            serialized[column] = json.dumps(serialized[column], ensure_ascii=False, sort_keys=True)
        serializable_rows.append(serialized)

    if serializable_rows:
        with raw_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(serializable_rows[0].keys()))
            writer.writeheader()
            writer.writerows(serializable_rows)
    else:
        raw_path.write_text("", encoding="utf-8")

    summary = summarize_games(rows)
    extended = compute_summary_metrics(rows)
    summary_row = {
        **{"policy": args.policy, "games_requested": args.games},
        **summary,
        **{k: v for k, v in extended.items() if k not in summary},
    }
    summary_path = RESULTS_DIR / f"{args.policy}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved turn-game logs to {raw_path}")
    print(f"Saved summary to {summary_path}")
    print(summary_row)


if __name__ == "__main__":
    main()
