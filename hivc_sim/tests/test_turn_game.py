from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from turn_game import (  # noqa: E402
    ACCEPTABLE_DELTA,
    Action,
    Event,
    GameState,
    acceptable_actions,
    best_action,
    estimate_q_values,
    heuristic_policy,
    initial_state,
    play_policy_game,
    random_policy,
    step,
    summarize_games,
    terminal_score,
)


def test_initial_state_is_reproducible() -> None:
    assert initial_state(123) == initial_state(123)
    assert initial_state(123).turn == 0


def test_communication_win_condition() -> None:
    rng = np.random.default_rng(0)
    state = GameState(
        turn=0,
        oxygen=8,
        power=6,
        hull_damage=2,
        flooding=1,
        communication=2,
        morale=80,
        current_event=Event.SIGNAL_WINDOW,
    )
    result = step(state, Action.REPAIR_COMMUNICATION, rng)
    assert result.state_after.done
    assert result.outcome == "win"
    assert terminal_score(result.state_after) > 1000


def test_flooding_action_reduces_flooding() -> None:
    rng = np.random.default_rng(1)
    state = GameState(flooding=4, current_event=Event.NONE)
    result = step(state, Action.SEAL_FLOODING, rng)
    assert result.state_after.flooding < state.flooding


def test_q_values_include_all_actions() -> None:
    state = GameState(current_event=Event.NONE)
    q_values = estimate_q_values(state, n_rollouts=8, policy=random_policy, seed=5)
    assert set(q_values) == set(Action)
    assert all(isinstance(value, float) for value in q_values.values())


def test_acceptable_actions_uses_delta() -> None:
    q_values = {
        Action.STABILIZE_OXYGEN: 100.0,
        Action.REPAIR_POWER: 100.0 - ACCEPTABLE_DELTA + 0.1,
        Action.REPAIR_COMMUNICATION: 50.0,
        Action.SEAL_FLOODING: 10.0,
    }
    allowed = acceptable_actions(q_values)
    assert Action.STABILIZE_OXYGEN in allowed
    assert Action.REPAIR_POWER in allowed
    assert best_action(q_values) == Action.STABILIZE_OXYGEN


def test_play_policy_game_outputs_evaluation_rows() -> None:
    rows = play_policy_game(heuristic_policy, seed=42, evaluator_rollouts=6)
    assert rows
    first = rows[0]
    assert "q_values" in first
    assert "regret" in first
    assert "best_action" in first
    assert "acceptable_actions" in first

    summary = summarize_games(rows)
    assert summary["games"] == 1.0
    assert 0.0 <= summary["win_rate"] <= 1.0
