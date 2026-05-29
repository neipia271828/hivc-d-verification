from __future__ import annotations
import numpy as np
from numpy.random import Generator
from typing import Optional
from config import (
    FIELD_SIZE, N_FOOD, N_SIGNS, REWARD_MIN, REWARD_MAX,
    SIGMA_THETA, SIGMA_D, SIGMA_P, FOOD_RADIUS,
)


class Food:
    def __init__(self, position: np.ndarray, reward: float) -> None:
        self.position: np.ndarray = position
        self.reward: float = reward
        self.eaten: bool = False


class Sign:
    def __init__(self, position: np.ndarray, target_food: Food) -> None:
        self.position: np.ndarray = position
        self.target_food: Food = target_food

        diff = target_food.position - position
        self.true_direction: float = float(np.arctan2(diff[1], diff[0]))
        self.true_distance: float = float(np.linalg.norm(diff))
        self.true_reward: float = float(target_food.reward)

    def observe(self, rng: Generator) -> dict:
        """観測ごとに独立なノイズを付与する。

        理論上I不一致は「観測ノイズ」として扱う（REQUIREMENTS.md）。
        同一標識でも観測者が異なれば独立な観測値を得る。
        """
        noisy_distance = self.true_distance * (1.0 + rng.normal(0.0, SIGMA_D))
        return {
            "direction": self.true_direction + rng.normal(0.0, SIGMA_THETA),
            "distance": max(0.01, noisy_distance),
            "reward": self.true_reward + rng.normal(0.0, SIGMA_P),
        }


class Field:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.foods: list[Food] = []
        self.signs: list[Sign] = []
        self._init(seed)

    def _init(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.foods = []
        self.signs = []

        positions = rng.uniform(0.0, FIELD_SIZE, size=(N_FOOD, 2))
        rewards = rng.uniform(REWARD_MIN, REWARD_MAX, size=N_FOOD)
        for pos, rew in zip(positions, rewards):
            self.foods.append(Food(pos.copy(), float(rew)))

        food_positions = np.array([f.position for f in self.foods])
        sign_positions = rng.uniform(0.0, FIELD_SIZE, size=(N_SIGNS, 2))
        for spos in sign_positions:
            dists = np.linalg.norm(food_positions - spos, axis=1)
            nearest_food = self.foods[int(np.argmin(dists))]
            self.signs.append(Sign(spos.copy(), nearest_food))

    def get_nearby_signs(self, pos: np.ndarray, k: int) -> list[Sign]:
        sign_positions = np.array([s.position for s in self.signs])
        dists = np.linalg.norm(sign_positions - pos, axis=1)
        indices = np.argsort(dists)[:k]
        return [self.signs[i] for i in indices]

    def check_food(self, pos: np.ndarray) -> Optional[Food]:
        for food in self.foods:
            if not food.eaten:
                if np.linalg.norm(food.position - pos) <= FOOD_RADIUS:
                    return food
        return None

    def reset(self, seed: int) -> None:
        self._init(seed)

    def all_eaten(self) -> bool:
        return all(f.eaten for f in self.foods)
