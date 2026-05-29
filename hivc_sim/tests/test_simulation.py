from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SIGMA_THETA, SIGMA_D, SIGMA_P, REWARD_MIN, REWARD_MAX
from environment import Field, Food, Sign
from agent import Agent
from agreement import agree_baseline, agree_hivc


def _make_food() -> Food:
    return Food(position=np.array([50.0, 50.0]), reward=5.0)


def _make_sign() -> Sign:
    food = _make_food()
    return Sign(position=np.array([40.0, 40.0]), target_food=food)


def test_sign_noise() -> None:
    """標識のノイズが指定分布に従うか（同一標識を多数回観測し統計検定）"""
    food = _make_food()
    pos = np.array([40.0, 40.0])
    true_dir = float(np.arctan2(food.position[1] - pos[1], food.position[0] - pos[0]))
    true_dist = float(np.linalg.norm(food.position - pos))

    sign = Sign(position=pos.copy(), target_food=food)
    rng = np.random.default_rng(0)
    n = 2000
    dirs, dists, rewards = [], [], []
    for _ in range(n):
        obs = sign.observe(rng)
        dirs.append(obs["direction"] - true_dir)
        dists.append((obs["distance"] / true_dist) - 1.0)
        rewards.append(obs["reward"] - food.reward)

    # Direction noise ~ N(0, SIGMA_THETA)
    assert abs(np.mean(dirs)) < 0.1, "Direction noise mean too large"
    assert abs(np.std(dirs) - SIGMA_THETA) < 0.05, "Direction noise std wrong"

    # Distance noise ~ N(0, SIGMA_D) multiplicative
    assert abs(np.mean(dists)) < 0.05, "Distance noise mean too large"
    assert abs(np.std(dists) - SIGMA_D) < 0.05, "Distance noise std wrong"

    # Reward noise ~ N(0, SIGMA_P)
    assert abs(np.mean(rewards)) < 0.2, "Reward noise mean too large"
    assert abs(np.std(rewards) - SIGMA_P) < 0.15, "Reward noise std wrong"


def test_score_range() -> None:
    """スコアが[0,1]に収まるか"""
    v = np.array([0.33, 0.33, 0.34])
    agent = Agent(v=v, position=np.array([50.0, 50.0]))
    rng = np.random.default_rng(42)
    field = Field(seed=42)
    signs = field.get_nearby_signs(agent.position, k=10)
    for sign in signs:
        obs = sign.observe(rng)
        score = agent.compute_score(obs, current_dir=0.0)
        assert 0.0 <= score <= 1.0 + 1e-9, f"Score {score} out of [0, 1]"


def test_agreement_baseline() -> None:
    """条件Aの合意ベクトルが両希望ベクトルの平均であるか"""
    v1 = np.array([0.5, 0.25, 0.25])
    v2 = np.array([0.25, 0.25, 0.5])
    agent1 = Agent(v=v1, position=np.array([20.0, 20.0]))
    agent2 = Agent(v=v2, position=np.array([80.0, 80.0]))
    field = Field(seed=7)

    rng1 = np.random.default_rng(11)
    rng2 = np.random.default_rng(22)
    obs1 = [s.observe(rng1) for s in field.get_nearby_signs(agent1.position, 5)]
    obs2 = [s.observe(rng2) for s in field.get_nearby_signs(agent2.position, 5)]
    current_dir = 0.0

    wish1 = agent1.compute_wish_vector(obs1, current_dir)
    wish2 = agent2.compute_wish_vector(obs2, current_dir)
    expected_raw = (wish1 + wish2) / 2.0
    norm = np.linalg.norm(expected_raw)
    expected = expected_raw / norm if norm > 1e-9 else expected_raw

    agreed = agree_baseline(agent1, agent2, obs1, obs2, current_dir)
    assert np.allclose(agreed, expected, atol=1e-9), "Baseline agreement != mean of wishes"


def test_agreement_hivc_low_v() -> None:
    """V-Low条件でV交渉が発生しないか（ρ >= threshold なら cost=0）"""
    # V-Low: very similar v values, should produce high ρ
    v1 = np.array([0.33, 0.33, 0.34])
    v2 = np.array([0.34, 0.33, 0.33])
    agent1 = Agent(v=v1, position=np.array([50.0, 50.0]))
    agent2 = Agent(v=v2, position=np.array([55.0, 55.0]))
    field = Field(seed=1)
    rng1 = np.random.default_rng(11)
    rng2 = np.random.default_rng(22)
    # 同一位置から同一標識集合を独立ノイズで観測（新モデルの想定使用法）
    nearby = field.get_nearby_signs(agent1.position, 5)
    obs1 = [s.observe(rng1) for s in nearby]
    obs2 = [s.observe(rng2) for s in nearby]
    current_dir = 0.0

    _, log = agree_hivc(agent1, agent2, obs1, obs2, current_dir)
    assert log["cost_turns"] == 0, f"Expected cost=0 for V-Low, got {log['cost_turns']}"
    assert log["v_negotiated"] is None, "V_negotiated should be None when ρ is sufficient"
