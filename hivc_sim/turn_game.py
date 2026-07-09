from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import log, sqrt
from typing import Callable, Iterable

import numpy as np


MAX_TURNS = 5
ACCEPTABLE_DELTA = 25.0

# REQUIREMENTS §5.2 終端スコア重み。set_score_weights() で上書き可能。
SCORE_WEIGHTS: dict[str, float] = {
    "win": 1000.0,
    "loss": 200.0,
    "oxygen": 30.0,
    "power": 20.0,
    "hull_damage": 80.0,
    "flooding": 70.0,
    "communication": 50.0,
    "morale": 5.0,
    "turn": 10.0,
    "severe_risk": 25.0,
}


def set_score_weights(weights: dict[str, float]) -> None:
    """終端スコア重みを一部上書きする（§5.2 重み調整用）。"""
    SCORE_WEIGHTS.update(weights)


class Action(str, Enum):
    STABILIZE_OXYGEN = "A"
    REPAIR_POWER = "B"
    REPAIR_COMMUNICATION = "C"
    SEAL_FLOODING = "D"


ACTION_LABELS: dict[Action, str] = {
    Action.STABILIZE_OXYGEN: "酸素供給を安定化",
    Action.REPAIR_POWER: "発電系を修理",
    Action.REPAIR_COMMUNICATION: "通信アンテナを修理",
    Action.SEAL_FLOODING: "浸水区画を封鎖",
}


class Event(str, Enum):
    NONE = "none"
    PRESSURE_SPIKE = "pressure_spike"
    LEAK_SURGE = "leak_surge"
    SIGNAL_WINDOW = "signal_window"
    CREW_PANIC = "crew_panic"


EVENT_LABELS: dict[Event, str] = {
    Event.NONE: "異常なし",
    Event.PRESSURE_SPIKE: "外圧上昇",
    Event.LEAK_SURGE: "浸水増加",
    Event.SIGNAL_WINDOW: "短時間の通信窓",
    Event.CREW_PANIC: "乗員の動揺",
}


ALL_ACTIONS: tuple[Action, ...] = tuple(Action)
EVENT_PROBS: tuple[float, ...] = (0.35, 0.18, 0.18, 0.17, 0.12)


@dataclass(frozen=True)
class GameState:
    turn: int = 0
    oxygen: int = 10
    power: int = 8
    hull_damage: int = 2
    flooding: int = 1
    communication: int = 0
    morale: int = 80
    severe_risk_count: int = 0
    current_event: Event = Event.NONE
    done: bool = False
    outcome: str = "running"

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            "turn": self.turn,
            "oxygen": self.oxygen,
            "power": self.power,
            "hull_damage": self.hull_damage,
            "flooding": self.flooding,
            "communication": self.communication,
            "morale": self.morale,
            "severe_risk_count": self.severe_risk_count,
            "current_event": self.current_event.value,
            "done": self.done,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class StepResult:
    state_before: GameState
    action: Action
    state_after: GameState
    event: Event
    outcome: str
    reward_score: float


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def initial_state(seed: int | None = None) -> GameState:
    rng = np.random.default_rng(seed)
    return replace(GameState(), current_event=sample_event(rng))


def sample_event(rng: np.random.Generator) -> Event:
    return Event(rng.choice([event.value for event in Event], p=EVENT_PROBS))


def terminalize(state: GameState) -> GameState:
    if state.communication >= 3:
        return replace(state, done=True, outcome="win")
    if state.oxygen <= 0:
        return replace(state, done=True, outcome="loss_oxygen")
    if state.power <= 0:
        return replace(state, done=True, outcome="loss_power")
    if state.hull_damage >= 5:
        return replace(state, done=True, outcome="loss_hull")
    if state.flooding >= 5:
        return replace(state, done=True, outcome="loss_flooding")
    if state.turn >= MAX_TURNS:
        return replace(state, done=True, outcome="survived_timeout")
    return replace(state, done=False, outcome="running")


def terminal_score(state: GameState) -> float:
    win = 1 if state.outcome == "win" else 0
    loss = 1 if state.outcome.startswith("loss_") else 0
    w = SCORE_WEIGHTS
    return (
        w["win"] * win
        - w["loss"] * loss
        + w["oxygen"] * max(state.oxygen, 0)
        + w["power"] * max(state.power, 0)
        - w["hull_damage"] * state.hull_damage
        - w["flooding"] * state.flooding
        + w["communication"] * state.communication
        + w["morale"] * state.morale
        - w["turn"] * state.turn
        - w["severe_risk"] * state.severe_risk_count
    )


def step(state: GameState, action: Action, rng: np.random.Generator) -> StepResult:
    if state.done:
        return StepResult(state, action, state, state.current_event, state.outcome, terminal_score(state))

    event = state.current_event
    oxygen = state.oxygen - 1
    power = state.power - 1
    hull_damage = state.hull_damage
    flooding = state.flooding
    communication = state.communication
    morale = state.morale
    severe_risk_count = state.severe_risk_count

    if event == Event.PRESSURE_SPIKE:
        hull_damage += 1
    elif event == Event.LEAK_SURGE:
        flooding += 1
    elif event == Event.CREW_PANIC:
        morale -= 10

    if action == Action.STABILIZE_OXYGEN:
        oxygen += 3
        power -= 1
        morale += 2
    elif action == Action.REPAIR_POWER:
        power += 3
        morale -= 2
        if rng.random() < 0.25:
            hull_damage += 1
            severe_risk_count += 1
    elif action == Action.REPAIR_COMMUNICATION:
        communication += 1
        power -= 1
        if event == Event.SIGNAL_WINDOW:
            communication += 1
        if rng.random() < 0.25:
            flooding += 1
            severe_risk_count += 1
    elif action == Action.SEAL_FLOODING:
        flooding -= 2
        oxygen -= 1
        if event == Event.LEAK_SURGE:
            flooding -= 1

    if oxygen <= 2:
        morale -= 8
        severe_risk_count += 1
    if power <= 2:
        morale -= 6
        severe_risk_count += 1
    if flooding >= 4 or hull_damage >= 4:
        morale -= 6
        severe_risk_count += 1

    next_state = GameState(
        turn=state.turn + 1,
        oxygen=clamp(oxygen, -5, 12),
        power=clamp(power, -5, 12),
        hull_damage=clamp(hull_damage, 0, 6),
        flooding=clamp(flooding, 0, 6),
        communication=clamp(communication, 0, 4),
        morale=clamp(morale, 0, 100),
        severe_risk_count=severe_risk_count,
        current_event=sample_event(rng),
    )
    next_state = terminalize(next_state)
    return StepResult(state, action, next_state, event, next_state.outcome, terminal_score(next_state))


Policy = Callable[[GameState, np.random.Generator], Action]


def random_policy(state: GameState, rng: np.random.Generator) -> Action:
    return Action(rng.choice([action.value for action in ALL_ACTIONS]))


def heuristic_policy(state: GameState, rng: np.random.Generator) -> Action:
    if state.current_event == Event.SIGNAL_WINDOW and state.power > 2:
        return Action.REPAIR_COMMUNICATION
    if state.flooding >= 3 or state.current_event == Event.LEAK_SURGE:
        return Action.SEAL_FLOODING
    if state.oxygen <= 4:
        return Action.STABILIZE_OXYGEN
    if state.power <= 4:
        return Action.REPAIR_POWER
    if state.communication < 3 and state.power > 2 and state.flooding < 4:
        return Action.REPAIR_COMMUNICATION
    return Action.STABILIZE_OXYGEN


def rollout(
    state: GameState,
    first_action: Action | None = None,
    policy: Policy = heuristic_policy,
    seed: int | None = None,
) -> tuple[float, list[StepResult]]:
    rng = np.random.default_rng(seed)
    current = state
    history: list[StepResult] = []
    if first_action is not None and not current.done:
        result = step(current, first_action, rng)
        history.append(result)
        current = result.state_after
    while not current.done:
        action = policy(current, rng)
        result = step(current, action, rng)
        history.append(result)
        current = result.state_after
    return terminal_score(current), history


def estimate_q_values(
    state: GameState,
    n_rollouts: int = 200,
    policy: Policy = heuristic_policy,
    seed: int = 0,
) -> dict[Action, float]:
    q_values: dict[Action, float] = {}
    for action in ALL_ACTIONS:
        scores = []
        for i in range(n_rollouts):
            action_seed = seed + 10_000 * list(ALL_ACTIONS).index(action) + i
            score, _ = rollout(state, first_action=action, policy=policy, seed=action_seed)
            scores.append(score)
        q_values[action] = float(np.mean(scores))
    return q_values


@dataclass
class SearchNode:
    state: GameState
    parent: "SearchNode | None" = None
    action_from_parent: Action | None = None
    visits: int = 0
    total_score: float = 0.0
    children: dict[Action, "SearchNode"] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = {}

    @property
    def untried_actions(self) -> list[Action]:
        assert self.children is not None
        return [action for action in ALL_ACTIONS if action not in self.children]

    @property
    def mean_score(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits


def mcts_policy(
    state: GameState,
    rng: np.random.Generator,
    simulations: int = 160,
    exploration: float = 1.4,
) -> Action:
    q_values = mcts_q_values(state, seed=int(rng.integers(0, 1_000_000_000)), simulations=simulations, exploration=exploration)
    return max(q_values, key=q_values.get)


def mcts_q_values(
    state: GameState,
    seed: int = 0,
    simulations: int = 400,
    exploration: float = 1.4,
) -> dict[Action, float]:
    rng = np.random.default_rng(seed)
    root = SearchNode(state=state)

    for _ in range(simulations):
        node = root
        current = state

        while not current.done and not node.untried_actions and node.children:
            node = select_child(node, exploration, rng)
            assert node.action_from_parent is not None
            sim_result = step(current, node.action_from_parent, rng)
            current = sim_result.state_after

        if not current.done and node.untried_actions:
            action = Action(rng.choice([a.value for a in node.untried_actions]))
            sim_result = step(current, action, rng)
            current = sim_result.state_after
            child = SearchNode(state=current, parent=node, action_from_parent=action)
            assert node.children is not None
            node.children[action] = child
            node = child

        score, _ = rollout(current, policy=heuristic_policy, seed=int(rng.integers(0, 1_000_000_000)))

        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    q_values = {action: float("-inf") for action in ALL_ACTIONS}
    assert root.children is not None
    for action, child in root.children.items():
        q_values[action] = child.mean_score
    fallback = estimate_q_values(state, n_rollouts=20, seed=seed)
    for action, value in q_values.items():
        if value == float("-inf"):
            q_values[action] = fallback[action]
    return q_values


def select_child(node: SearchNode, exploration: float, rng: np.random.Generator) -> SearchNode:
    assert node.children
    log_parent = log(max(node.visits, 1))

    def uct(child: SearchNode) -> float:
        if child.visits == 0:
            return float("inf")
        jitter = float(rng.random() * 1e-9)
        return child.mean_score + exploration * sqrt(log_parent / child.visits) + jitter

    return max(node.children.values(), key=uct)


def acceptable_actions(q_values: dict[Action, float], delta: float = ACCEPTABLE_DELTA) -> set[Action]:
    ordered = sorted(q_values.items(), key=lambda item: item[1], reverse=True)
    best_value = ordered[0][1]
    return {action for action, value in ordered if best_value - value < delta}


def best_action(q_values: dict[Action, float]) -> Action:
    return max(q_values, key=q_values.get)


def play_policy_game(
    policy: Policy,
    seed: int,
    evaluator_rollouts: int = 120,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    state = initial_state(seed)
    rows: list[dict[str, object]] = []
    while not state.done:
        q_values = estimate_q_values(state, n_rollouts=evaluator_rollouts, policy=heuristic_policy, seed=seed + state.turn * 1000)
        allowed = acceptable_actions(q_values)
        selected = policy(state, rng)
        result = step(state, selected, rng)
        optimal = best_action(q_values)
        rows.append(
            {
                "seed": seed,
                "turn": state.turn,
                "event": result.event.value,
                "state_before": state.as_dict(),
                "action": selected.value,
                "action_label": ACTION_LABELS[selected],
                "best_action": optimal.value,
                "acceptable_actions": ",".join(sorted(action.value for action in allowed)),
                "regret": q_values[optimal] - q_values[selected],
                "q_values": {action.value: value for action, value in q_values.items()},
                "state_after": result.state_after.as_dict(),
                "outcome": result.outcome,
                "terminal_score": terminal_score(result.state_after),
            }
        )
        state = result.state_after
    return rows


def summarize_games(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"games": 0.0, "win_rate": 0.0, "survival_rate": 0.0, "mean_return": 0.0, "mean_regret": 0.0}
    terminal_rows_by_seed: dict[int, dict[str, object]] = {}
    for row in rows:
        terminal_rows_by_seed[int(row["seed"])] = row
    terminal_rows = list(terminal_rows_by_seed.values())
    return {
        "games": float(len(terminal_rows)),
        "win_rate": float(np.mean([row["outcome"] == "win" for row in terminal_rows])),
        "survival_rate": float(np.mean([not str(row["outcome"]).startswith("loss_") for row in terminal_rows])),
        "mean_return": float(np.mean([float(row["terminal_score"]) for row in terminal_rows])),
        "mean_regret": float(np.mean([float(row["regret"]) for row in rows])),
    }
