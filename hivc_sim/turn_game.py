from __future__ import annotations

import json
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
    "pod_readiness": 80.0,
    "pod_integrity": 60.0,
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
    PREP_POD = "E"
    EXECUTE_ESCAPE = "F"


ACTION_LABELS: dict[Action, str] = {
    Action.STABILIZE_OXYGEN: "酸素供給を安定化",
    Action.REPAIR_POWER: "発電系を修理",
    Action.REPAIR_COMMUNICATION: "通信アンテナを修理",
    Action.SEAL_FLOODING: "浸水区画を封鎖",
    Action.PREP_POD: "脱出艇を整備",
    Action.EXECUTE_ESCAPE: "自力脱出を実行",
}


class Event(str, Enum):
    NONE = "none"
    PRESSURE_SPIKE = "pressure_spike"
    LEAK_SURGE = "leak_surge"
    SIGNAL_WINDOW = "signal_window"
    CREW_PANIC = "crew_panic"
    RELAY_SHORT = "relay_short"
    POD_FLOODING = "pod_flooding"
    CURRENT_CHANGE = "current_change"
    BACKUP_POWER_FOUND = "backup_power_found"
    HULL_FRACTURE = "hull_fracture"


EVENT_LABELS: dict[Event, str] = {
    Event.NONE: "異常なし",
    Event.PRESSURE_SPIKE: "外圧上昇",
    Event.LEAK_SURGE: "浸水増加",
    Event.SIGNAL_WINDOW: "短時間の通信窓",
    Event.CREW_PANIC: "乗員の動揺",
    Event.RELAY_SHORT: "中継器短絡",
    Event.POD_FLOODING: "脱出艇区画の漏水",
    Event.CURRENT_CHANGE: "海流変化",
    Event.BACKUP_POWER_FOUND: "非常用電源発見",
    Event.HULL_FRACTURE: "船体亀裂拡大",
}


ALL_ACTIONS: tuple[Action, ...] = tuple(Action)
EVENT_VALUES: tuple[Event, ...] = (
    Event.NONE,
    Event.PRESSURE_SPIKE,
    Event.LEAK_SURGE,
    Event.SIGNAL_WINDOW,
    Event.CREW_PANIC,
    Event.RELAY_SHORT,
    Event.POD_FLOODING,
    Event.CURRENT_CHANGE,
    Event.BACKUP_POWER_FOUND,
    Event.HULL_FRACTURE,
)
EVENT_PROBS: tuple[float, ...] = (
    0.22,
    0.10,
    0.10,
    0.10,
    0.08,
    0.10,
    0.10,
    0.08,
    0.07,
    0.05,
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    base_oxygen: int
    base_power: int
    base_hull_damage: int
    base_flooding: int
    base_communication: int
    base_pod_readiness: int
    base_pod_integrity: int
    base_morale: int
    pod_priority: str  # "readiness", "integrity", "auto"
    event_sequence: tuple[Event, ...] | None = None
    jitter_scale: int = 1


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "comms_favored",
        base_oxygen=8,
        base_power=8,
        base_hull_damage=1,
        base_flooding=1,
        base_communication=1,
        base_pod_readiness=0,
        base_pod_integrity=0,
        base_morale=80,
        pod_priority="readiness",
    ),
    Scenario(
        "escape_favored",
        base_oxygen=6,
        base_power=6,
        base_hull_damage=2,
        base_flooding=1,
        base_communication=0,
        base_pod_readiness=1,
        base_pod_integrity=1,
        base_morale=70,
        pod_priority="integrity",
    ),
    Scenario(
        "ambiguous",
        base_oxygen=6,
        base_power=6,
        base_hull_damage=2,
        base_flooding=2,
        base_communication=0,
        base_pod_readiness=1,
        base_pod_integrity=1,
        base_morale=70,
        pod_priority="auto",
    ),
    Scenario(
        "route_reversal",
        base_oxygen=8,
        base_power=8,
        base_hull_damage=2,
        base_flooding=1,
        base_communication=0,
        base_pod_readiness=0,
        base_pod_integrity=0,
        base_morale=75,
        pod_priority="auto",
        event_sequence=(
            Event.POD_FLOODING,
            Event.BACKUP_POWER_FOUND,
            Event.SIGNAL_WINDOW,
            Event.BACKUP_POWER_FOUND,
            Event.NONE,
        ),
        jitter_scale=0,
    ),
)

SCENARIO_BY_ID: dict[str, Scenario] = {s.scenario_id: s for s in SCENARIOS}


def get_scenario(scenario_id: str | None, seed: int) -> Scenario:
    if scenario_id is not None and scenario_id in SCENARIO_BY_ID:
        return SCENARIO_BY_ID[scenario_id]
    ids = [s.scenario_id for s in SCENARIOS]
    return SCENARIO_BY_ID[ids[seed % len(ids)]]


@dataclass(frozen=True)
class GameState:
    turn: int = 0
    oxygen: int = 10
    power: int = 8
    hull_damage: int = 2
    flooding: int = 1
    communication: int = 0
    pod_readiness: int = 0
    pod_integrity: int = 0
    rescue_eta: int | None = None
    morale: int = 80
    severe_risk_count: int = 0
    current_event: Event = Event.NONE
    scenario_id: str = "ambiguous"
    done: bool = False
    outcome: str = "running"

    def as_dict(self) -> dict[str, int | str | bool | None]:
        return {
            "turn": self.turn,
            "oxygen": self.oxygen,
            "power": self.power,
            "hull_damage": self.hull_damage,
            "flooding": self.flooding,
            "communication": self.communication,
            "pod_readiness": self.pod_readiness,
            "pod_integrity": self.pod_integrity,
            "rescue_eta": self.rescue_eta,
            "morale": self.morale,
            "severe_risk_count": self.severe_risk_count,
            "current_event": self.current_event.value,
            "scenario_id": self.scenario_id,
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
    premature: bool


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def initial_state(seed: int | None = None, scenario_id: str | None = None) -> GameState:
    rng = np.random.default_rng(seed)
    scenario = get_scenario(scenario_id, int(seed) if seed is not None else 0)

    jitter_scale = getattr(scenario, "jitter_scale", 1)

    def jitter(base: int, low: int, high: int, scale: int = jitter_scale) -> int:
        return clamp(base + int(rng.integers(-scale, scale + 1)), low, high)

    state = GameState(
        turn=0,
        oxygen=jitter(scenario.base_oxygen, 1, 10),
        power=jitter(scenario.base_power, 1, 10),
        hull_damage=jitter(scenario.base_hull_damage, 0, 4),
        flooding=jitter(scenario.base_flooding, 0, 4),
        communication=jitter(scenario.base_communication, 0, 2),
        pod_readiness=jitter(scenario.base_pod_readiness, 0, 2),
        pod_integrity=jitter(scenario.base_pod_integrity, 0, 2),
        rescue_eta=None,
        morale=jitter(scenario.base_morale, 50, 100, 5),
        severe_risk_count=0,
        current_event=sample_event(rng, scenario_id=scenario.scenario_id, turn=0),
        scenario_id=scenario.scenario_id,
    )
    return state


def sample_event(rng: np.random.Generator, scenario_id: str | None = None, turn: int | None = None) -> Event:
    scenario = SCENARIO_BY_ID.get(scenario_id) if scenario_id is not None else None
    if scenario is not None and scenario.event_sequence is not None:
        index = turn if turn is not None else 0
        if index is not None and 0 <= index < len(scenario.event_sequence):
            return scenario.event_sequence[index]
    return Event(rng.choice([event.value for event in EVENT_VALUES], p=EVENT_PROBS))


def terminalize(state: GameState) -> GameState:
    # 敗北条件を先にチェック
    if state.oxygen <= 0:
        return replace(state, done=True, outcome="loss_oxygen")
    if state.power <= 0:
        return replace(state, done=True, outcome="loss_power")
    if state.hull_damage >= 5:
        return replace(state, done=True, outcome="loss_hull")
    if state.flooding >= 5:
        return replace(state, done=True, outcome="loss_flooding")

    # 通信救助: 通信復旧後、rescue_eta ターン生存する必要がある
    if state.rescue_eta is not None:
        eta = state.rescue_eta - 1
        if eta <= 0:
            return replace(state, done=True, outcome="win", rescue_eta=0)
        if state.turn >= MAX_TURNS:
            return replace(state, done=True, outcome="survived_timeout")
        return replace(state, done=False, outcome="running", rescue_eta=eta)

    if state.communication >= 3:
        if state.turn >= MAX_TURNS:
            return replace(state, done=True, outcome="survived_timeout")
        return replace(state, done=False, outcome="running", rescue_eta=2)

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
        + w["pod_readiness"] * state.pod_readiness
        + w["pod_integrity"] * state.pod_integrity
        + w["morale"] * state.morale
        - w["turn"] * state.turn
        - w["severe_risk"] * state.severe_risk_count
    )


def _choose_pod_target(state: GameState) -> str:
    scenario = SCENARIO_BY_ID.get(state.scenario_id, SCENARIOS[-1])
    priority = scenario.pod_priority
    if priority == "readiness":
        return "readiness"
    if priority == "integrity":
        return "integrity"
    # auto: 目標値 2 に対する不足度が大きい方を優先
    readiness_deficit = 2 - state.pod_readiness
    integrity_deficit = 2 - state.pod_integrity
    if readiness_deficit > integrity_deficit:
        return "readiness"
    if integrity_deficit > readiness_deficit:
        return "integrity"
    return "readiness"


def _post_start_state(state: GameState) -> GameState:
    """ターン開始時の基本消費とイベント効果を反映した状態を返す。"""
    event = state.current_event
    oxygen = state.oxygen - 1
    power = state.power - 1
    hull_damage = state.hull_damage
    flooding = state.flooding
    communication = state.communication
    pod_readiness = state.pod_readiness
    pod_integrity = state.pod_integrity
    morale = state.morale

    if event == Event.PRESSURE_SPIKE:
        hull_damage += 1
    elif event == Event.LEAK_SURGE:
        flooding += 1
    elif event == Event.CREW_PANIC:
        morale -= 10
    elif event == Event.RELAY_SHORT:
        power -= 1
        communication -= 1
    elif event == Event.POD_FLOODING:
        flooding += 1
        pod_integrity -= 1
    elif event == Event.CURRENT_CHANGE:
        hull_damage += 1
        flooding += 1
    elif event == Event.BACKUP_POWER_FOUND:
        power += 2
    elif event == Event.HULL_FRACTURE:
        hull_damage += 2

    return GameState(
        turn=state.turn + 1,
        oxygen=clamp(oxygen, -5, 12),
        power=clamp(power, -5, 12),
        hull_damage=clamp(hull_damage, 0, 6),
        flooding=clamp(flooding, 0, 6),
        communication=clamp(communication, 0, 4),
        pod_readiness=clamp(pod_readiness, 0, 3),
        pod_integrity=clamp(pod_integrity, 0, 4),
        rescue_eta=state.rescue_eta,
        morale=clamp(morale, 0, 100),
        severe_risk_count=state.severe_risk_count,
        current_event=state.current_event,
        scenario_id=state.scenario_id,
    )


def _escape_conditions_met(state: GameState) -> bool:
    """発進直前の状態（基本消費・イベント効果反映済み）で脱出条件を満たすか。

    呼び出し側は step() 内であれば _post_start_state() 済みの state を、
    それ以外では _escape_conditions_met_current() を使うこと。
    """
    return (
        state.pod_readiness >= 2
        and state.pod_integrity >= 2
        and state.oxygen >= 3
        and state.power >= 2
        and state.flooding <= 3
    )


def _escape_conditions_met_current(state: GameState) -> bool:
    """現在の state から基本消費・イベント効果を反映して脱出条件を満たすか。"""
    return _escape_conditions_met(_post_start_state(state))


def step(state: GameState, action: Action, rng: np.random.Generator) -> StepResult:
    if state.done:
        return StepResult(state, action, state, state.current_event, state.outcome, terminal_score(state), False)

    event = state.current_event
    post_state = _post_start_state(state)
    # 基本消費・イベント効果反映後の値を作業変数に展開
    oxygen = post_state.oxygen
    power = post_state.power
    hull_damage = post_state.hull_damage
    flooding = post_state.flooding
    communication = post_state.communication
    pod_readiness = post_state.pod_readiness
    pod_integrity = post_state.pod_integrity
    morale = post_state.morale
    severe_risk_count = post_state.severe_risk_count
    premature = False

    # 行動効果
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
    elif action == Action.PREP_POD:
        target = _choose_pod_target(state)
        if target == "integrity":
            pod_integrity += 1
        else:
            pod_readiness += 1
        power -= 1
    elif action == Action.EXECUTE_ESCAPE:
        if not _escape_conditions_met(post_state):
            premature = True
            # 発進失敗: 未充足条件に応じて重大損傷
            if post_state.pod_readiness < 2:
                hull_damage += 2 - post_state.pod_readiness
            if post_state.pod_integrity < 2:
                hull_damage += 2 - post_state.pod_integrity
            if post_state.oxygen < 3:
                oxygen -= 3
            if post_state.power < 2:
                power -= 2
            if post_state.flooding > 3:
                flooding += 1
            severe_risk_count += 2

    # リソース危機による士気・リスクカウント
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
        turn=post_state.turn,
        oxygen=clamp(oxygen, -5, 12),
        power=clamp(power, -5, 12),
        hull_damage=clamp(hull_damage, 0, 6),
        flooding=clamp(flooding, 0, 6),
        communication=clamp(communication, 0, 4),
        pod_readiness=clamp(pod_readiness, 0, 3),
        pod_integrity=clamp(pod_integrity, 0, 4),
        rescue_eta=post_state.rescue_eta,
        morale=clamp(morale, 0, 100),
        severe_risk_count=severe_risk_count,
        current_event=sample_event(rng, scenario_id=post_state.scenario_id, turn=post_state.turn),
        scenario_id=post_state.scenario_id,
    )

    if action == Action.EXECUTE_ESCAPE:
        if _escape_conditions_met(post_state):
            # 自力脱出成功
            next_state = replace(next_state, done=True, outcome="win")
        else:
            next_state = terminalize(next_state)
            if not next_state.done:
                # 未達発進は重大リスク行動として敗北扱い
                next_state = replace(next_state, done=True, outcome="loss_escape_failed")
    else:
        next_state = terminalize(next_state)

    return StepResult(state, action, next_state, event, next_state.outcome, terminal_score(next_state), premature)


Policy = Callable[[GameState, np.random.Generator], Action]


def random_policy(state: GameState, rng: np.random.Generator) -> Action:
    return Action(rng.choice([action.value for action in ALL_ACTIONS]))


def heuristic_policy(state: GameState, rng: np.random.Generator) -> Action:
    if state.current_event == Event.SIGNAL_WINDOW and state.power > 2:
        return Action.REPAIR_COMMUNICATION
    if state.flooding >= 3 or state.current_event == Event.LEAK_SURGE:
        return Action.SEAL_FLOODING
    if _escape_conditions_met_current(state):
        return Action.EXECUTE_ESCAPE
    if state.oxygen <= 4:
        return Action.STABILIZE_OXYGEN
    if state.power <= 4:
        return Action.REPAIR_POWER
    if state.communication >= 3 and state.rescue_eta is None:
        return Action.REPAIR_COMMUNICATION
    if state.pod_readiness < 2 or state.pod_integrity < 2:
        return Action.PREP_POD
    if state.communication < 3 and state.power > 2 and state.flooding < 4:
        return Action.REPAIR_COMMUNICATION
    return Action.STABILIZE_OXYGEN


def comms_forced_policy(state: GameState, rng: np.random.Generator) -> Action:
    """通信救助ルートを強制する方策。"""
    if state.rescue_eta is not None:
        if state.oxygen <= 3:
            return Action.STABILIZE_OXYGEN
        if state.flooding >= 4:
            return Action.SEAL_FLOODING
        if state.power <= 2:
            return Action.REPAIR_POWER
        return Action.STABILIZE_OXYGEN
    if state.communication >= 3:
        if state.power <= 2:
            return Action.REPAIR_POWER
        return Action.STABILIZE_OXYGEN
    if state.oxygen <= 2:
        return Action.STABILIZE_OXYGEN
    if state.flooding >= 4:
        return Action.SEAL_FLOODING
    if state.power <= 1:
        return Action.REPAIR_POWER
    return Action.REPAIR_COMMUNICATION


def escape_forced_policy(state: GameState, rng: np.random.Generator) -> Action:
    """自力脱出ルートを強制する方策。"""
    if _escape_conditions_met_current(state):
        return Action.EXECUTE_ESCAPE
    if state.pod_readiness < 2 or state.pod_integrity < 2:
        return Action.PREP_POD
    if state.oxygen <= 3:
        return Action.STABILIZE_OXYGEN
    if state.flooding >= 4:
        return Action.SEAL_FLOODING
    if state.power <= 2:
        return Action.REPAIR_POWER
    return Action.PREP_POD


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


def _route_value(
    state: GameState,
    policy: Policy,
    seed: int,
    n_rollouts: int,
) -> float:
    """指定方策で n_rollouts 回ロールアウトした終端スコアの平均を返す。"""
    scores = []
    for i in range(n_rollouts):
        score, _ = rollout(state, policy=policy, seed=seed + i * 1000)
        scores.append(score)
    return float(np.mean(scores)) if scores else float("-inf")


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


def route_of_action(action: Action) -> str:
    if action in (Action.REPAIR_COMMUNICATION,):
        return "comms"
    if action in (Action.PREP_POD, Action.EXECUTE_ESCAPE):
        return "escape"
    return "support"


def optimal_route(
    state: GameState,
    q_values: dict[Action, float] | None = None,
    seed: int = 0,
    n_rollouts: int = 20,
) -> str:
    """通信救助・自力脱出の強制ルートを別々にロールアウトし、期待値最大の勝ち筋を返す。

    q_values 引数は後方互換のため残すが、state があれば強制ルートロールアウトを優先する。
    """
    if q_values is not None and state is None:
        comms_value = q_values[Action.REPAIR_COMMUNICATION]
        escape_value = max(q_values[Action.PREP_POD], q_values[Action.EXECUTE_ESCAPE])
    else:
        comms_value = _route_value(state, comms_forced_policy, seed, n_rollouts)
        escape_value = _route_value(state, escape_forced_policy, seed + 100_000, n_rollouts)
    if abs(comms_value - escape_value) < ACCEPTABLE_DELTA:
        return "undecided"
    return "comms" if comms_value > escape_value else "escape"


def role_specific_evidence(role: str, state: GameState) -> str:
    """エージェントごとの役割固有情報（診断情報）を返す。"""
    if role == "alpha":
        return (
            f"安全管理診断: 船体損傷={state.hull_damage}, 浸水度={state.flooding}, "
            f"救助待機中の生存リスク={_rescue_wait_risk(state)}"
        )
    if role == "beta":
        return (
            f"通信・脱出艇診断: 通信修理見込み={_communication_repair_estimate(state)}, "
            f"脱出艇健全性={state.pod_integrity}, beta確認済みの発進準備="
            f"{pod_ready_status(state)}"
        )
    return ""


def _rescue_wait_risk(state: GameState) -> str:
    if state.rescue_eta is None:
        return "未送信"
    if state.oxygen <= 2 or state.power <= 2 or state.flooding >= 4 or state.hull_damage >= 4:
        return "高"
    if state.oxygen <= 4 or state.power <= 4:
        return "中"
    return "低"


def _communication_repair_estimate(state: GameState) -> str:
    if state.communication >= 3:
        return "完了"
    need = 3 - state.communication
    if state.power <= 2:
        return "難（電力不足）"
    if state.current_event == Event.SIGNAL_WINDOW:
        return f"容易（あと{need}回）"
    if state.current_event == Event.RELAY_SHORT:
        return f"遅延（あと{need + 1}回程度）"
    return f"通常（あと{need}回）"


def pod_ready_status(state: GameState) -> str:
    # beta 可視情報（pod_readiness, pod_integrity, oxygen, power）のみを診断文に反映。
    # flooding は alpha 担当のため、ここでは触れない。
    post = _post_start_state(state)
    missing = []
    if post.pod_readiness < 2:
        missing.append("整備不足")
    if post.pod_integrity < 2:
        missing.append("艇損傷")
    if post.oxygen < 3:
        missing.append("酸素不足")
    if post.power < 2:
        missing.append("電力不足")
    if missing:
        return ",".join(missing)
    return "艇・資源の準備完了（浸水・船体は安全担当に確認）"


def play_policy_game(
    policy: Policy,
    seed: int,
    evaluator_rollouts: int = 120,
    evaluator_policy: str = "heuristic",
    scenario_id: str | None = None,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    state = initial_state(seed, scenario_id)
    rows: list[dict[str, object]] = []
    planned_route = "undecided"
    while not state.done:
        if evaluator_policy == "mcts":
            q_values = mcts_q_values(state, seed=seed + state.turn * 1000, simulations=evaluator_rollouts)
        else:
            q_values = estimate_q_values(state, n_rollouts=evaluator_rollouts, policy=heuristic_policy, seed=seed + state.turn * 1000)
        allowed = acceptable_actions(q_values)
        selected = policy(state, rng)
        result = step(state, selected, rng)
        optimal = best_action(q_values)
        optimal_route_value = optimal_route(state, seed=seed + state.turn * 1000, n_rollouts=20)
        route = route_of_action(selected)
        if route in ("comms", "escape"):
            planned_route = route
        elif planned_route == "undecided":
            planned_route = optimal_route_value
        route_switch = False
        # 前ターンの planned_route は前の行から復元できないため、ここでは同ゲーム内の turn 連続性を利用
        if rows:
            prev_route = str(rows[-1].get("planned_route", "undecided"))
            route_switch = (prev_route != planned_route) and route in ("comms", "escape")
        row = {
            "seed": seed,
            "scenario_id": state.scenario_id,
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
            "planned_route": planned_route,
            "optimal_route": optimal_route_value,
            "route_switch": route_switch,
            "premature": result.premature,
        }
        rows.append(row)
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
