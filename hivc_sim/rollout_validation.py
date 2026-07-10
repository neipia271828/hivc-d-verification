"""REQUIREMENTS §5.2 重み調整 + §8 事前ロールアウト検証レポート。

事前に N ゲーム以上を自動プレイし、以下を検証する（§8）。
- 難易度が極端でないこと（勝率が 0 でも 1 でもない）。
- ランダム方策より heuristic 方策が明確に良いこと。
- MCTS が heuristic 方策を平均的に上回ること。
- 全行動が少なくとも一定割合で最適または許容行動になること。
- 初期状態から勝利可能であること（少なくとも1勝）。
- イベントによって最適行動が変化すること。

§5.2 重み調整は、上記を満たすように win/loss 重みを倍率で探索する簡易ヒルクライム
を行い、満たした重みを stdout と JSON に出力する。

使い方:
  # 検証レポートのみ（現行重みで §8 チェック）
  python3 hivc_sim/rollout_validation.py --games 100 --seed 42

  # 重み調整も実行
  python3 hivc_sim/rollout_validation.py --games 100 --tune --output hivc_sim/results/turn_game/validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from turn_game import (  # noqa: E402
    ALL_ACTIONS,
    Action,
    Event,
    SCORE_WEIGHTS,
    estimate_q_values,
    heuristic_policy,
    initial_state,
    mcts_policy,
    play_policy_game,
    random_policy,
    set_score_weights,
)
from turn_game_metrics import compute_summary_metrics  # noqa: E402

Policy = Callable


def _run_policy_games(policy: Policy, games: int, seed: int, evaluator_rollouts: int, evaluator_policy: str = "mcts") -> list[dict]:
    rows: list[dict] = []
    for i in range(games):
        rows.extend(play_policy_game(policy, seed=seed + i, evaluator_rollouts=evaluator_rollouts, evaluator_policy=evaluator_policy))
    return rows


def _action_coverage(rows: list[dict]) -> dict[str, float]:
    """best_action として各行動が現れる割合。"""
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row["best_action"])] += 1
    total = sum(counter.values()) or 1
    return {a.value: counter.get(a.value, 0) / total for a in ALL_ACTIONS}


def _event_changes_best(rows: list[dict]) -> bool:
    """イベントの有無で best_action の分布が変化するか。"""
    by_event: dict[str, Counter[str]] = {}
    for row in rows:
        ev = str(row.get("event", "none"))
        by_event.setdefault(ev, Counter())[str(row["best_action"])] += 1
    if len(by_event) < 2:
        return False
    distributions = []
    for ev, counter in by_event.items():
        total = sum(counter.values()) or 1
        distributions.append(tuple(counter.get(a.value, 0) / total for a in ALL_ACTIONS))
    return len(set(distributions)) > 1


def evaluate_weights(games: int, seed: int, evaluator_rollouts: int, evaluator_policy: str = "mcts") -> dict:
    """現行の SCORE_WEIGHTS で §8 チェックを実施し、レポート dict を返す。"""
    results: dict[str, dict] = {}
    for name, policy in (("random", random_policy), ("heuristic", heuristic_policy), ("mcts", mcts_policy)):
        rows = _run_policy_games(policy, games, seed, evaluator_rollouts, evaluator_policy)
        results[name] = {
            "rows": rows,
            "summary": compute_summary_metrics(rows),
            "coverage": _action_coverage(rows),
        }

    rand = results["random"]["summary"]
    heur = results["heuristic"]["summary"]
    mcts = results["mcts"]["summary"]

    coverage = results["heuristic"]["coverage"]
    min_coverage = min(coverage.values()) if coverage else 0.0
    any_win = max(
        results["random"]["summary"]["win_rate"],
        results["heuristic"]["summary"]["win_rate"],
        results["mcts"]["summary"]["win_rate"],
    ) > 0.0

    checks = {
        "difficulty_not_extreme": 0.0 < heur["win_rate"] < 1.0,
        "heuristic_beats_random": heur["mean_return"] > rand["mean_return"],
        "mcts_beats_heuristic": mcts["mean_return"] >= heur["mean_return"],
        "all_actions_sometimes_optimal": min_coverage > 0.0,
        "win_possible": any_win,
        "event_changes_best_action": _event_changes_best(results["heuristic"]["rows"]),
    }
    return {
        "weights": dict(SCORE_WEIGHTS),
        "checks": checks,
        "all_pass": all(checks.values()),
        "summaries": {name: results[name]["summary"] for name in results},
        "best_action_coverage": {name: results[name]["coverage"] for name in results},
    }


def tune_weights(games: int, seed: int, evaluator_rollouts: int, evaluator_policy: str = "mcts", base_win: float = 1000.0) -> dict:
    """§5.2: win/loss 倍率を探索して §8 制約を満たす重みを探す。

    win と loss の比率を保ちつつ、win 重みを大きくして「勝利が資源温存より常に優先」
    を満たしやすくする。heuristic 勝率が 0 なら win 重みを増加、1 なら loss 重みを
    増加して難易度を非極端化する。
    """
    best_report = None
    win_candidates = [1000.0, 1500.0, 2000.0, 3000.0]
    loss_ratios = [0.2, 0.3, 0.5, 0.8]  # loss / win

    for win in win_candidates:
        for ratio in loss_ratios:
            set_score_weights({"win": win, "loss": win * ratio})
            report = evaluate_weights(games, seed, evaluator_rollouts, evaluator_policy)
            if best_report is None or (
                sum(report["checks"].values()) > sum(best_report["checks"].values())
                or (sum(report["checks"].values()) == sum(best_report["checks"].values())
                    and report["all_pass"] and not best_report["all_pass"])
            ):
                best_report = report
            if report["all_pass"]:
                return report
    # 全候補で all_pass にならなければ、最もチェック数の多いものを返す
    if best_report is not None:
        best_report["all_pass"] = all(best_report["checks"].values())
    return best_report or evaluate_weights(games, seed, evaluator_rollouts, evaluator_policy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout-based validation and weight tuning (§5.2/§8).")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluator-rollouts", type=int, default=80)
    parser.add_argument("--evaluator-policy", choices=["heuristic", "mcts"], default="mcts",
                        help="探索ベース評価に使う方策。MCTS を使うと best_action カバレッジが向上。")
    parser.add_argument("--tune", action="store_true", help="§5.2 重み調整を実行する。")
    parser.add_argument("--output", default=None, help="JSON レポート出力先。")
    args = parser.parse_args()

    if args.tune:
        print("=== §5.2 weight tuning ===")
        report = tune_weights(args.games, args.seed, args.evaluator_rollouts, args.evaluator_policy)
    else:
        report = evaluate_weights(args.games, args.seed, args.evaluator_rollouts, args.evaluator_policy)

    print("\n=== §8 validation report ===")
    print(f"weights: {report['weights']}")
    print(f"all_pass: {report['all_pass']}")
    for check, ok in report["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {check}")
    print("\nsummaries:")
    for name, summary in report["summaries"].items():
        print(f"  {name}: win={summary['win_rate']:.3f} "
              f"survival={summary['survival_rate']:.3f} "
              f"mean_return={summary['mean_return']:.1f} "
              f"mean_regret={summary['mean_regret']:.1f} "
              f"expert_match={summary.get('expert_match_rate', float('nan')):.3f}")
    print("\nbest_action_coverage (heuristic):")
    for action, share in report["best_action_coverage"]["heuristic"].items():
        print(f"  {action}: {share:.3f}")

    if args.output:
        path = Path(args.output)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved report to {path}")


if __name__ == "__main__":
    main()
