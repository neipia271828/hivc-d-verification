from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

V_CONDITIONS = ["V-Low", "V-Mid", "V-High"]
MECHANISMS = ["baseline", "hivc"]
COLORS = {
    ("V-Low", "baseline"): "#1f77b4",
    ("V-Low", "hivc"): "#aec7e8",
    ("V-Mid", "baseline"): "#ff7f0e",
    ("V-Mid", "hivc"): "#ffbb78",
    ("V-High", "baseline"): "#2ca02c",
    ("V-High", "hivc"): "#98df8a",
}


def plot_reward_by_condition(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = []
    labels = []
    data_groups = []
    pos = 1
    for vc in V_CONDITIONS:
        for mech in MECHANISMS:
            sub = df[(df["v_condition"] == vc) & (df["mechanism"] == mech)]["cumulative_reward"].values
            data_groups.append(sub)
            positions.append(pos)
            labels.append(f"{vc}\n{mech}")
            pos += 1
        pos += 0.5

    bp = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True)
    for patch, (vc, mech) in zip(bp["boxes"], [(vc, mech) for vc in V_CONDITIONS for mech in MECHANISMS]):
        patch.set_facecolor(COLORS[(vc, mech)])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward by Condition × Mechanism")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reward_by_condition.png", dpi=150)
    plt.close(fig)


def plot_reward_curve(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for vc in V_CONDITIONS:
        for mech in MECHANISMS:
            sub = df[(df["v_condition"] == vc) & (df["mechanism"] == mech)]
            curves = sub["reward_per_turn"].tolist()
            max_len = max(len(c) for c in curves)
            arr = np.array([c + [0.0] * (max_len - len(c)) for c in curves])
            cumsum = np.cumsum(arr, axis=1)
            mean_curve = cumsum.mean(axis=0)
            ax.plot(mean_curve, label=f"{vc}/{mech}", color=COLORS[(vc, mech)],
                    linestyle="--" if mech == "hivc" else "-")

    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean Cumulative Reward")
    ax.set_title("Cumulative Reward Curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reward_curve.png", dpi=150)
    plt.close(fig)


def plot_trajectory_sample(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, (vc, mech) in enumerate([(vc, mech) for vc in V_CONDITIONS for mech in MECHANISMS]):
        ax = axes[idx // 3][idx % 3]
        sub = df[(df["v_condition"] == vc) & (df["mechanism"] == mech)]
        if len(sub) == 0:
            continue
        traj = sub.iloc[0]["trajectory"]
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], "-o", markersize=2, color=COLORS[(vc, mech)])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(f"{vc} / {mech}")
        ax.set_aspect("equal")
    fig.suptitle("Sample Trajectories")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "trajectory_sample.png", dpi=150)
    plt.close(fig)


def plot_ct_similarity(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for vc in V_CONDITIONS:
        for mech in MECHANISMS:
            sub = df[(df["v_condition"] == vc) & (df["mechanism"] == mech)]
            curves = sub["ct_similarity"].tolist()
            max_len = max(len(c) for c in curves)
            arr = np.array([c + [float("nan")] * (max_len - len(c)) for c in curves])
            mean_curve = np.nanmean(arr, axis=0)
            ax.plot(mean_curve, label=f"{vc}/{mech}", color=COLORS[(vc, mech)],
                    linestyle="--" if mech == "hivc" else "-")
    ax.set_xlabel("Turn")
    ax.set_ylabel("C_t Similarity (cos)")
    ax.set_title("C_t Approximation Rate Over Time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ct_similarity.png", dpi=150)
    plt.close(fig)


def plot_rho_history(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = df[(df["v_condition"] == "V-High") & (df["mechanism"] == "hivc")]
    if len(sub) == 0:
        plt.close(fig)
        return
    rho = sub.iloc[0]["rho_history"]
    rho_clean = [x if not np.isnan(x) else None for x in rho]
    turns = list(range(len(rho_clean)))
    vals = [v for v in rho_clean if v is not None]
    idxs = [i for i, v in enumerate(rho_clean) if v is not None]
    ax.plot(idxs, vals, "-", color=COLORS[("V-High", "hivc")])
    ax.axhline(y=0.6, color="red", linestyle="--", label="ρ threshold (0.6)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("ρ History — V-High / HIVC-D (sample trial)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rho_history.png", dpi=150)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame) -> None:
    plot_reward_by_condition(df)
    plot_reward_curve(df)
    plot_trajectory_sample(df)
    plot_ct_similarity(df)
    plot_rho_history(df)
