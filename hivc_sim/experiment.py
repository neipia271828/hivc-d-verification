from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from config import (
    CONDITIONS, MECHANISMS, N_TRIALS, T_MAX, RANDOM_SEED_BASE, K_SIGNS,
    FIELD_SIZE, MOVE_STEP,
)
from environment import Field
from agent import Agent
from agreement import agree_baseline, agree_hivc


@dataclass
class TrialResult:
    cumulative_reward: float
    reward_per_turn: list[float]
    trajectory: list[np.ndarray]
    agreement_costs: list[int]
    rho_history: list[float]
    ct_similarity: list[float]
    v_distance: float


class Trial:
    def __init__(self, v_condition: str, mechanism: str, seed: int) -> None:
        self.v_condition = v_condition
        self.mechanism = mechanism
        self.seed = seed
        self.cond = CONDITIONS[v_condition]

    def run(self) -> TrialResult:
        rng = np.random.default_rng(self.seed)
        # 観測ノイズ用に各探索者へ独立なRNGを割り当てる（I不一致=観測ノイズ）
        rng_obs1 = np.random.default_rng(self.seed + 1_000_000)
        rng_obs2 = np.random.default_rng(self.seed + 2_000_000)
        field = Field(self.seed)
        reset_count = 0

        v1 = self.cond["v1"]
        v2 = self.cond["v2"]
        v_distance = float(np.linalg.norm(v1 - v2))

        # 単一チーム位置: 二人の探索者は同じ位置から環境を観測し、合意ベクトルで移動する。
        # これにより「常に同一方向へ移動する二体」という擬似的二体問題を排し、
        # I不一致を観測ノイズとして純粋に操作できる。
        agent1 = Agent(v=v1, position=np.zeros(2))
        agent2 = Agent(v=v2, position=np.zeros(2))
        team_pos: np.ndarray = rng.uniform(0.0, FIELD_SIZE, size=2)

        current_dir = float(rng.uniform(0, 2 * np.pi))
        cumulative = 0.0
        reward_per_turn: list[float] = []
        trajectory: list[np.ndarray] = [team_pos.copy()]
        agreement_costs: list[int] = []
        rho_history: list[float] = []
        ct_similarity: list[float] = []

        for t in range(T_MAX):
            if field.all_eaten():
                reset_count += 1
                # リセット後フィールドを決定論的に: 試行内のタイミングに依存させない
                field.reset(self.seed + reset_count * 100_003)

            # 同一のK枚の標識を両探索者が独立ノイズで観測する
            nearby = field.get_nearby_signs(team_pos, K_SIGNS)
            if nearby:
                signs_obs1 = [s.observe(rng_obs1) for s in nearby]
                signs_obs2 = [s.observe(rng_obs2) for s in nearby]
            else:
                signs_obs1 = [{"direction": current_dir, "distance": 1.0, "reward": 0.0}]
                signs_obs2 = [{"direction": current_dir, "distance": 1.0, "reward": 0.0}]

            # C_t: 選択肢集合C(=可視標識の方向)の中で報酬効率が真に最大の方向。
            # ノイズなしの真値で評価する理想解(REQUIREMENTS.md: C_t=真の最適選択肢)。
            # エージェントが実際に選べる選択肢の中から定義することで、
            # 合意ベクトルとの比較が意味を持つ。
            if nearby:
                best_sign = max(
                    nearby,
                    key=lambda s: s.true_reward / (s.true_distance + 1e-9),
                )
                ang = best_sign.true_direction
                ct_dir = np.array([np.cos(ang), np.sin(ang)])
            else:
                ct_dir = np.array([1.0, 0.0])

            if self.mechanism == "baseline":
                agreed = agree_baseline(agent1, agent2, signs_obs1, signs_obs2, current_dir)
                cost = 0
                rho = float("nan")
            else:
                agreed, log = agree_hivc(agent1, agent2, signs_obs1, signs_obs2, current_dir)
                cost = log["cost_turns"]
                rho = log["rho_before"]

            # C_t similarity (合意ベクトルと理想方向の内積; agreedは単位ベクトル)
            sim = float(np.dot(agreed, ct_dir))
            ct_similarity.append(sim)
            agreement_costs.append(cost)
            rho_history.append(rho)

            current_dir = float(np.arctan2(agreed[1], agreed[0]))

            # チームを合意方向へ移動（境界でクランプ）
            team_pos = np.clip(team_pos + agreed * MOVE_STEP, 0.0, FIELD_SIZE)

            # 食料獲得
            turn_reward = 0.0
            f = field.check_food(team_pos)
            if f is not None:
                turn_reward += f.reward
                f.eaten = True
                cumulative += f.reward

            reward_per_turn.append(turn_reward)
            trajectory.append(team_pos.copy())

        return TrialResult(
            cumulative_reward=cumulative,
            reward_per_turn=reward_per_turn,
            trajectory=trajectory,
            agreement_costs=agreement_costs,
            rho_history=rho_history,
            ct_similarity=ct_similarity,
            v_distance=v_distance,
        )


def run_all_experiments(
    n_trials: Optional[int] = None,
    seed_base: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    from tqdm import tqdm

    if n_trials is None:
        n_trials = N_TRIALS
    if seed_base is None:
        seed_base = RANDOM_SEED_BASE

    records = []
    trial_id = 0
    tasks = [(vc, mech) for vc in CONDITIONS for mech in MECHANISMS]

    outer = tqdm(tasks, desc="Conditions") if verbose else tasks
    for v_condition, mechanism in outer:
        inner_iter = range(n_trials)
        if verbose:
            inner_iter = tqdm(inner_iter, desc=f"{v_condition}/{mechanism}", leave=False)
        for i in inner_iter:
            seed = seed_base + trial_id
            trial = Trial(v_condition, mechanism, seed)
            result = trial.run()
            records.append({
                "trial_id": trial_id,
                "v_condition": v_condition,
                "mechanism": mechanism,
                "v_distance": result.v_distance,
                "cumulative_reward": result.cumulative_reward,
                "mean_agreement_cost": float(np.mean(result.agreement_costs)),
                "mean_ct_similarity": float(np.mean(result.ct_similarity)),
                "rho_history": result.rho_history,
                "ct_similarity": result.ct_similarity,
                "reward_per_turn": result.reward_per_turn,
                "trajectory": result.trajectory,
            })
            trial_id += 1

    return pd.DataFrame(records)
