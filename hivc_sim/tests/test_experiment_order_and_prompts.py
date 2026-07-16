from __future__ import annotations

from scripts.llm_turn_game_common import format_persona
from scripts.qwen_two_agent_experiment import condition_order_for_seed


def test_condition_order_is_deterministic_per_seed_and_not_globally_fixed() -> None:
    conditions = ["control", "consulting", "hivc_d"]
    orders = [condition_order_for_seed(conditions, seed) for seed in range(42, 50)]
    assert orders == [condition_order_for_seed(conditions, seed) for seed in range(42, 50)]
    assert all(set(order) == set(conditions) for order in orders)
    assert len({tuple(order) for order in orders}) > 1


def test_legacy_prompt_marks_weights_fixed_while_soft_value_is_updatable() -> None:
    base = {
        "role": {"id": "role", "label": "担当"},
        "persona": {"id": "persona"},
        "value": {"initial_priority_weights": {"oxygen": 1.0}},
    }
    legacy = format_persona(
        "alpha", "legacy", {"_resolved_profile": {**base, "role_value_mode": "legacy_hard"}}
    )
    soft = format_persona(
        "alpha", "soft", {"_resolved_profile": {**base, "role_value_mode": "soft_value"}}
    )
    assert "固定された意思決定基準" in legacy
    assert "更新可能" not in legacy
    assert "更新可能" in soft
