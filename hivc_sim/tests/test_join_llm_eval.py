from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from join_llm_eval import join  # noqa: E402


def test_join_on_seed_turn() -> None:
    llm_rows = [
        {"seed": "42", "turn": "0", "group_action": "A", "regret": "10"},
        {"seed": "42", "turn": "1", "group_action": "C", "regret": "5"},
    ]
    eval_rows = [
        {"seed": "42", "turn": "0", "action": "B", "best_action": "B", "regret": "0"},
        {"seed": "42", "turn": "1", "action": "C", "best_action": "C", "regret": "0"},
    ]
    joined = join(llm_rows, eval_rows, "heur")
    assert len(joined) == 2
    assert joined[0]["heur_action"] == "B"
    assert joined[0]["heur_best_action"] == "B"
    assert joined[1]["heur_matched"] == "1"
    assert joined[0]["group_action"] == "A"


def test_join_missing_eval_row() -> None:
    llm_rows = [{"seed": "42", "turn": "0", "group_action": "A"}]
    eval_rows = [{"seed": "42", "turn": "1", "action": "B"}]
    joined = join(llm_rows, eval_rows, "heur")
    assert joined[0]["heur_matched"] == "0"
    assert "heur_action" not in joined[0]
