"""Grid search over HIVC-D simulation hyperparameters.

Parameter grid:
  RHO_AGREE_THRESHOLD : [0.3, 0.5, 0.7, 0.9]   – V整合判定閾値
  SIGMA_THETA         : [0.1, 0.3, 0.5]          – 方向ノイズ(rad)
  K_SIGNS             : [3, 5, 10]               – 1ターンに観測する標識数

Axes chosen because they directly determine:
  - when V-negotiation fires (RHO)
  - how noisy agent observations are (SIGMA_THETA)
  - how much info each agent can gather (K_SIGNS)

Usage:
  python3.12 grid_search.py [--trials N]   (default 30)
"""
from __future__ import annotations
import sys
import importlib
import itertools
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent / "results" / "grid_search"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PARAM_GRID: dict[str, list] = {
    "RHO_AGREE_THRESHOLD": [0.3, 0.5, 0.7, 0.9],
    "SIGMA_THETA": [0.1, 0.3, 0.5],
    "K_SIGNS": [3, 5, 10],
}

# Values to restore after each grid point
DEFAULTS: dict[str, float | int] = {
    "RHO_AGREE_THRESHOLD": 0.6,
    "SIGMA_THETA": 0.3,
    "SIGMA_D": 0.2,
    "SIGMA_P": 1.0,
    "K_SIGNS": 5,
}

SEED_BASE = 42


# ──────────────────────────────────────────────────────────────────────────────
# Module reload helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_config(cfg, params: dict) -> None:
    for k, v in params.items():
        setattr(cfg, k, v)


def _reload_sim_modules() -> tuple:
    """Reload environment→agent→agreement→experiment→analysis in dependency order."""
    import environment, agent, agreement, experiment, analysis
    importlib.reload(environment)
    importlib.reload(agent)
    importlib.reload(agreement)
    importlib.reload(experiment)
    importlib.reload(analysis)
    # Return freshly reloaded references
    import experiment as exp_mod
    import analysis as ana_mod
    return exp_mod, ana_mod


# ──────────────────────────────────────────────────────────────────────────────
# Grid search runner
# ──────────────────────────────────────────────────────────────────────────────

def run_grid_search(n_trials: int = 30) -> pd.DataFrame:
    import config as cfg

    keys = list(PARAM_GRID.keys())
    combinations = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    n_combos = len(combinations)
    print(f"Grid: {n_combos} combinations × {n_trials} trials/condition × 6 conditions = "
          f"{n_combos * n_trials * 6} total trials\n")

    records: list[dict] = []

    for combo in tqdm(combinations, desc="Grid point"):
        params = dict(zip(keys, combo))
        _apply_config(cfg, params)
        exp_mod, ana_mod = _reload_sim_modules()

        df = exp_mod.run_all_experiments(n_trials=n_trials, seed_base=SEED_BASE, verbose=False)

        h1 = ana_mod.test_H1(df)
        h2 = ana_mod.test_H2(df)
        h3 = ana_mod.test_H3(df)
        h4 = ana_mod.test_H4(df)

        reward_diffs: dict[str, float] = {}
        for vc in ["V-Low", "V-Mid", "V-High"]:
            base_r = df[(df["v_condition"] == vc) & (df["mechanism"] == "baseline")]["cumulative_reward"].mean()
            hivc_r = df[(df["v_condition"] == vc) & (df["mechanism"] == "hivc")]["cumulative_reward"].mean()
            label = vc.lower().replace("-", "")  # "vlow", "vmid", "vhigh"
            reward_diffs[f"rdiff_{label}"] = float(hivc_r - base_r)

        # V交渉発生率（hivc条件でcost>0の割合）
        hivc_df = df[df["mechanism"] == "hivc"]
        neg_rate = float(hivc_df["mean_agreement_cost"].mean())

        record: dict = {
            **params,
            **reward_diffs,
            "neg_rate": neg_rate,
            "H1_F": h1["F_statistic"],
            "H1_p": h1["p_value"],
            "H1_eta2": h1["eta_squared"],
            "H2_vlow_t": h2["V-Low"]["t"],
            "H2_vlow_p": h2["V-Low"]["p"],
            "H2_vmid_t": h2["V-Mid"]["t"],
            "H2_vmid_p": h2["V-Mid"]["p"],
            "H2_vhigh_t": h2["V-High"]["t"],
            "H2_vhigh_p": h2["V-High"]["p"],
            "H3_F": h3["F_interaction"],
            "H3_p": h3["p_interaction"],
            "H4_slope": h4["slope"],
            "H4_r2": h4["r_squared"],
            "H4_p": h4["p_value"],
        }
        records.append(record)

    # Restore defaults
    _apply_config(cfg, DEFAULTS)
    _reload_sim_modules()

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis & visualization
# ──────────────────────────────────────────────────────────────────────────────

def analyze_grid(df: pd.DataFrame) -> None:
    df = df.copy()
    df["n_sig_H2"] = (
        (df["H2_vlow_p"] < 0.05).astype(int)
        + (df["H2_vmid_p"] < 0.05).astype(int)
        + (df["H2_vhigh_p"] < 0.05).astype(int)
    )
    df["H1_sig"] = df["H1_p"] < 0.05
    df["H2_vhigh_sig"] = df["H2_vhigh_p"] < 0.05

    rho_vals = sorted(df["RHO_AGREE_THRESHOLD"].unique())
    sig_vals = sorted(df["SIGMA_THETA"].unique())
    k_vals = sorted(df["K_SIGNS"].unique())

    # ── Figure 1: Heatmap (RHO × SIGMA_THETA) of rdiff_vhigh, faceted by K_SIGNS ──
    fig, axes = plt.subplots(1, len(k_vals), figsize=(5 * len(k_vals), 4))
    vmin = df["rdiff_vhigh"].min()
    vmax = df["rdiff_vhigh"].max()
    absmax = max(abs(vmin), abs(vmax))
    for ax, k in zip(axes, k_vals):
        sub = df[df["K_SIGNS"] == k]
        pivot = sub.pivot(index="SIGMA_THETA", columns="RHO_AGREE_THRESHOLD", values="rdiff_vhigh")
        im = ax.imshow(
            pivot.values, aspect="auto", cmap="RdYlGn",
            vmin=-absmax, vmax=absmax,
            origin="lower",
        )
        ax.set_xticks(range(len(rho_vals)))
        ax.set_xticklabels([str(v) for v in rho_vals])
        ax.set_yticks(range(len(sig_vals)))
        ax.set_yticklabels([str(v) for v in sig_vals])
        ax.set_xlabel("RHO_AGREE_THRESHOLD")
        ax.set_ylabel("SIGMA_THETA")
        ax.set_title(f"K_SIGNS={int(k)}")
        for r, sig in enumerate(sig_vals):
            for c, rho in enumerate(rho_vals):
                val = pivot.loc[sig, rho]
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(val) < absmax * 0.6 else "white")
        plt.colorbar(im, ax=ax, label="Δreward (HIVC-D−Base)")
    fig.suptitle("V-High条件のHIVC-D報酬優位性\n(green=HIVC-D有利, red=baseline有利)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "heatmap_rdiff_vhigh.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: H2有意数 vs RHO, 色分け K_SIGNS ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for ax, metric, ylabel in [
        (axes[0], "n_sig_H2", "H2 有意検定数 (0–3)"),
        (axes[1], "H1_F", "H1 F統計量"),
    ]:
        for k, col in zip(k_vals, colors):
            sub = df[df["K_SIGNS"] == k].groupby("RHO_AGREE_THRESHOLD")[metric].mean()
            ax.plot(sub.index, sub.values, marker="o", label=f"K={int(k)}", color=col)
        if metric == "n_sig_H2":
            ax.axhline(1, color="gray", linestyle="--", alpha=0.5, label="有意1件")
        else:
            ax.axhline(3.0, color="red", linestyle="--", alpha=0.5, label="F≈3 (p<0.05目安)")
        ax.set_xlabel("RHO_AGREE_THRESHOLD")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_title("H2有意数 vs RHO閾値")
    axes[1].set_title("H1 F統計量 vs RHO閾値")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "h1_h2_vs_rho.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: SIGMA_THETA × K_SIGNS の2D map – H4 slope と neg_rate ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, cmap, label in [
        (axes[0], "H4_slope", "RdBu_r", "H4 slope (ρ vs V距離)"),
        (axes[1], "neg_rate", "YlOrRd", "平均V交渉発生率"),
    ]:
        pivot = df.groupby(["SIGMA_THETA", "K_SIGNS"])[metric].mean().unstack()
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(k_vals)))
        ax.set_xticklabels([str(int(k)) for k in k_vals])
        ax.set_yticks(range(len(sig_vals)))
        ax.set_yticklabels([str(v) for v in sig_vals])
        ax.set_xlabel("K_SIGNS")
        ax.set_ylabel("SIGMA_THETA")
        ax.set_title(label)
        for r in range(len(sig_vals)):
            for c in range(len(k_vals)):
                ax.text(c, r, f"{pivot.values[r, c]:.3f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "h4_negrate_map.png", dpi=150)
    plt.close(fig)

    # ── Figure 4: 全36点の rdiff_vhigh 散布図（点ラベル付き） ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, col in zip(k_vals, colors):
        sub = df[df["K_SIGNS"] == k]
        sc = ax.scatter(
            sub["RHO_AGREE_THRESHOLD"],
            sub["rdiff_vhigh"],
            c=sub["SIGMA_THETA"],
            cmap="cool",
            vmin=0.0, vmax=0.6,
            s=120, zorder=3, label=f"K={int(k)}", marker=["o", "s", "^"][k_vals.index(k)],
            edgecolors=col, linewidths=1.5,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("RHO_AGREE_THRESHOLD")
    ax.set_ylabel("Δreward V-High (HIVC-D − Baseline)")
    ax.set_title("グリッド全点: HIVC-D優位性 (V-High)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="SIGMA_THETA")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "scatter_rdiff_all.png", dpi=150)
    plt.close(fig)

    # ── Figure 5: 上位・下位10設定のバーチャート ──
    df_sorted = df.sort_values("rdiff_vhigh", ascending=False).reset_index(drop=True)
    top10 = df_sorted.head(10)
    bot10 = df_sorted.tail(10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sub, title in [(axes[0], top10, "上位10設定"), (axes[1], bot10, "下位10設定")]:
        labels = [
            f"RHO={r}\nσθ={s}\nK={int(k)}"
            for r, s, k in zip(sub["RHO_AGREE_THRESHOLD"], sub["SIGMA_THETA"], sub["K_SIGNS"])
        ]
        bars = ax.barh(labels, sub["rdiff_vhigh"],
                       color=["#2ca02c" if v >= 0 else "#d62728" for v in sub["rdiff_vhigh"]])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Δreward V-High")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("HIVC-D優位性 Top / Bottom 10 (V-High)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "top_bottom10.png", dpi=150)
    plt.close(fig)

    # ── テキスト出力 ──
    print("\n" + "=" * 60)
    print(f"グリッドサーチ結果サマリ ({len(df)} 設定)")
    print("=" * 60)

    print(f"\nH1有意 (p<0.05):         {df['H1_sig'].sum():2d} / {len(df)}")
    print(f"H2 ≥1条件有意:           {(df['n_sig_H2'] >= 1).sum():2d} / {len(df)}")
    print(f"H2 全3条件有意:          {(df['n_sig_H2'] == 3).sum():2d} / {len(df)}")
    print(f"H2 V-High有意:           {df['H2_vhigh_sig'].sum():2d} / {len(df)}")

    print("\n── HIVC-D優位性 (rdiff_vhigh) 統計 ──")
    print(df["rdiff_vhigh"].describe().to_string())

    print("\n── Top 10設定 (rdiff_vhigh 降順) ──")
    cols = ["RHO_AGREE_THRESHOLD", "SIGMA_THETA", "K_SIGNS",
            "rdiff_vhigh", "H2_vhigh_t", "H2_vhigh_p",
            "H1_F", "H1_p", "neg_rate", "H4_slope", "H4_r2"]
    print(df_sorted.head(10)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n── パラメータ別 rdiff_vhigh 平均 ──")
    for param in ["RHO_AGREE_THRESHOLD", "SIGMA_THETA", "K_SIGNS"]:
        print(f"\n  {param}:")
        g = df.groupby(param)["rdiff_vhigh"].agg(["mean", "std", "max"])
        print(g.to_string(float_format=lambda x: f"{x:.3f}"))

    best = df_sorted.iloc[0]
    print(f"\n── 最良設定 ──")
    print(f"  RHO={best['RHO_AGREE_THRESHOLD']}, σθ={best['SIGMA_THETA']}, K={int(best['K_SIGNS'])}")
    print(f"  Δreward V-High : {best['rdiff_vhigh']:.3f}")
    print(f"  H2 V-High      : t={best['H2_vhigh_t']:.3f}, p={best['H2_vhigh_p']:.4f}")
    print(f"  H1             : F={best['H1_F']:.3f}, p={best['H1_p']:.4f}")
    print(f"  H4             : slope={best['H4_slope']:.4f}, r²={best['H4_r2']:.3f}")
    print(f"  V交渉発生率    : {best['neg_rate']:.3f}")

    df_sorted[cols].to_csv(RESULTS_DIR / "top10_configs.csv", index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HIVC-D hyperparameter grid search")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    results_df = run_grid_search(n_trials=args.trials)
    results_df.to_csv(RESULTS_DIR / "grid_results.csv", index=False)
    print(f"\n結果CSV: {RESULTS_DIR / 'grid_results.csv'}")

    analyze_grid(results_df)
    print(f"\n図出力先: {RESULTS_DIR}")
