from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rollout_validation import evaluate_weights  # noqa: E402
from turn_game import set_score_weights  # noqa: E402


def test_evaluate_weights_returns_checks() -> None:
    set_score_weights({"win": 1000.0, "loss": 200.0})
    report = evaluate_weights(games=4, seed=42, evaluator_rollouts=6)
    assert set(report["checks"]) == {
        "difficulty_not_extreme",
        "heuristic_beats_random",
        "mcts_beats_heuristic",
        "all_actions_sometimes_optimal",
        "win_possible",
        "event_changes_best_action",
    }
    assert "summaries" in report
    assert set(report["summaries"]) == {"random", "heuristic", "mcts"}
    assert "best_action_coverage" in report
