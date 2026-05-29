from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from agent import Agent
from config import RHO_AGREE_THRESHOLD


def _circular_mean(a: float, b: float) -> float:
    """2つの角度の循環平均（±πの折り返しに対応）。"""
    s = (np.sin(a) + np.sin(b)) / 2.0
    c = (np.cos(a) + np.cos(b)) / 2.0
    return float(np.arctan2(s, c))


def _merge_obs(obs1: list[dict], obs2: list[dict]) -> list[dict]:
    """I共有: 両探索者は同一標識集合を独立ノイズで観測しているため、
    インデックス対応で観測値を平均する。これにより観測ノイズが約1/√2に低減し、
    かつ標識集合サイズはK枚のまま（baselineと情報量が対称）に保たれる。
    """
    n = min(len(obs1), len(obs2))
    merged: list[dict] = []
    for i in range(n):
        merged.append({
            "direction": _circular_mean(obs1[i]["direction"], obs2[i]["direction"]),
            "distance": (obs1[i]["distance"] + obs2[i]["distance"]) / 2.0,
            "reward": (obs1[i]["reward"] + obs2[i]["reward"]) / 2.0,
        })
    # 長さが異なる場合の余剰分はそのまま付与（通常は発生しない）
    merged.extend(obs1[n:])
    merged.extend(obs2[n:])
    return merged


def _random_unit() -> np.ndarray:
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)])


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return _random_unit()
    return v / norm


def agree_baseline(
    agent1: Agent,
    agent2: Agent,
    signs_obs1: list[dict],
    signs_obs2: list[dict],
    current_dir: float,
) -> np.ndarray:
    wish1 = agent1.compute_wish_vector(signs_obs1, current_dir)
    wish2 = agent2.compute_wish_vector(signs_obs2, current_dir)
    agreed = (wish1 + wish2) / 2.0
    return _normalize(agreed)


def agree_hivc(
    agent1: Agent,
    agent2: Agent,
    signs_obs1: list[dict],
    signs_obs2: list[dict],
    current_dir: float,
) -> tuple[np.ndarray, dict]:
    # Step 1: share observations
    shared_obs = _merge_obs(signs_obs1, signs_obs2)

    if len(shared_obs) < 2:
        wish1 = agent1.compute_wish_vector(shared_obs, current_dir)
        wish2 = agent2.compute_wish_vector(shared_obs, current_dir)
        agreed = _normalize((wish1 + wish2) / 2.0)
        log = {"rho_before": 1.0, "rho_after": 1.0, "v_negotiated": None, "cost_turns": 0}
        return agreed, log

    # Step 2: V alignment check
    scores1 = [agent1.compute_score(s, current_dir) for s in shared_obs]
    scores2 = [agent2.compute_score(s, current_dir) for s in shared_obs]
    rank1 = np.argsort(np.argsort(scores1))
    rank2 = np.argsort(np.argsort(scores2))
    rho_result = spearmanr(rank1, rank2)
    rho_before = float(rho_result.statistic)  # type: ignore[union-attr]

    if rho_before >= RHO_AGREE_THRESHOLD:
        wish1 = agent1.compute_wish_vector(shared_obs, current_dir)
        wish2 = agent2.compute_wish_vector(shared_obs, current_dir)
        agreed = _normalize((wish1 + wish2) / 2.0)
        log = {
            "rho_before": rho_before,
            "rho_after": rho_before,
            "v_negotiated": None,
            "cost_turns": 0,
        }
        return agreed, log

    # Step 2b: V negotiation
    v_star = (agent1.v + agent2.v) / 2.0
    v_star = v_star / v_star.sum()

    tmp_agent = Agent(v=v_star, position=agent1.position)
    agreed = _normalize(tmp_agent.compute_wish_vector(shared_obs, current_dir))

    scores_star = [tmp_agent.compute_score(s, current_dir) for s in shared_obs]
    rank_star = np.argsort(np.argsort(scores_star))
    rho_after_result = spearmanr(rank1, rank_star)
    rho_after = float(rho_after_result.statistic)  # type: ignore[union-attr]

    log = {
        "rho_before": rho_before,
        "rho_after": rho_after,
        "v_negotiated": v_star,
        "cost_turns": 1,
    }
    return agreed, log
