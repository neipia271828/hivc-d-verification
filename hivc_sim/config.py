import numpy as np

# Field
FIELD_SIZE: float = 100.0
N_FOOD: int = 20
N_SIGNS: int = 50
REWARD_MIN: float = 1.0
REWARD_MAX: float = 10.0

# Sign noise
SIGMA_THETA: float = 0.3
SIGMA_D: float = 0.2
SIGMA_P: float = 1.0

# Agent
K_SIGNS: int = 5
MOVE_STEP: float = 2.0
FOOD_RADIUS: float = 2.0

# Experiment
N_TRIALS: int = 100
T_MAX: int = 200
RANDOM_SEED_BASE: int = 42

# Agreement parameters (Condition B)
RHO_AGREE_THRESHOLD: float = 0.6
V_NEGOTIATION_MODE: str = "average"

# V conditions
CONDITIONS: dict = {
    "V-Low": {
        "v1": np.array([0.33, 0.33, 0.33]),
        "v2": np.array([0.40, 0.30, 0.30]),
    },
    "V-Mid": {
        "v1": np.array([0.60, 0.20, 0.20]),
        "v2": np.array([0.20, 0.20, 0.60]),
    },
    "V-High": {
        "v1": np.array([0.80, 0.10, 0.10]),
        "v2": np.array([0.10, 0.10, 0.80]),
    },
}

MECHANISMS: list = ["baseline", "hivc"]
