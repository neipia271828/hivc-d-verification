from __future__ import annotations
import numpy as np
from config import MOVE_STEP, FIELD_SIZE, REWARD_MAX, FOOD_RADIUS


class Agent:
    def __init__(self, v: np.ndarray, position: np.ndarray) -> None:
        self.v: np.ndarray = v.copy()
        self.position: np.ndarray = position.copy()
        self.total_reward: float = 0.0

    def compute_score(self, sign_obs: dict, current_dir: float) -> float:
        w_c, w_d, w_p = float(self.v[0]), float(self.v[1]), float(self.v[2])

        c = float(np.cos(current_dir - sign_obs["direction"]))
        c = (c + 1.0) / 2.0  # normalize to [0, 1]

        # FOOD_RADIUSを単位として正規化: d=0.5 @ distance==FOOD_RADIUS, 遠ざかるほど急激に減衰
        d = 1.0 / (1.0 + sign_obs["distance"] / FOOD_RADIUS)

        p = float(np.clip(sign_obs["reward"], 0.0, REWARD_MAX)) / REWARD_MAX

        return w_c * c + w_d * d + w_p * p

    def compute_wish_vector(self, signs_obs: list[dict], current_dir: float) -> np.ndarray:
        wish = np.zeros(2)
        for obs in signs_obs:
            score = self.compute_score(obs, current_dir)
            direction = obs["direction"]
            distance = max(obs["distance"], 0.01)
            unit = np.array([np.cos(direction), np.sin(direction)])
            wish += score * unit / distance

        norm = float(np.linalg.norm(wish))
        if norm < 1e-9:
            angle = np.random.uniform(0, 2 * np.pi)
            return np.array([np.cos(angle), np.sin(angle)])
        return wish / norm

    def move(self, direction_vector: np.ndarray) -> None:
        norm = float(np.linalg.norm(direction_vector))
        if norm < 1e-9:
            return
        self.position = self.position + direction_vector / norm * MOVE_STEP
        self.position = np.clip(self.position, 0.0, FIELD_SIZE)
