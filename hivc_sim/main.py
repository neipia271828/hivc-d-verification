from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Allow imports from this directory when running as script
sys.path.insert(0, str(Path(__file__).parent))

from config import N_TRIALS, RANDOM_SEED_BASE
from experiment import run_all_experiments
from analysis import test_H1, test_H2, test_H3, test_H4
from visualize import generate_all_plots

RAW_DIR = Path(__file__).parent / "results" / "raw"
SUMMARY_DIR = Path(__file__).parent / "results" / "summary"
FIGURES_DIR = Path(__file__).parent / "results" / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description="HIVC-D Verification Simulation")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_BASE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.trials} trials per condition (seed={args.seed})...")
    df = run_all_experiments(n_trials=args.trials, seed_base=args.seed)

    # Save raw results (drop non-serializable columns)
    raw_df = df[["trial_id", "v_condition", "mechanism", "v_distance",
                 "cumulative_reward", "mean_agreement_cost", "mean_ct_similarity"]].copy()
    raw_df.to_csv(RAW_DIR / "trials.csv", index=False)
    print(f"Saved raw results to {RAW_DIR / 'trials.csv'}")

    # Statistical tests
    h1 = test_H1(df)
    h2 = test_H2(df)
    h3 = test_H3(df)
    h4 = test_H4(df)

    stats_rows = []
    stats_rows.append({"hypothesis": "H1", **h1})
    for vc, res in h2.items():
        stats_rows.append({"hypothesis": f"H2_{vc}", **res})
    stats_rows.append({"hypothesis": "H3", **h3})
    stats_rows.append({"hypothesis": "H4", **h4})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(SUMMARY_DIR / "stats.csv", index=False)
    print(f"Saved stats to {SUMMARY_DIR / 'stats.csv'}")

    # Print summary
    print("\n=== Statistical Results ===")
    print("\nH1 (V不一致→報酬低下, baseline):")
    print(f"  ANOVA : F={h1['F_statistic']:.3f}, p={h1['p_value']:.4f}, η²={h1['eta_squared']:.3f}")
    print(f"  線形回帰: slope={h1['reg_slope']:.3f}, p={h1['reg_p']:.4f}, r²={h1['reg_r2']:.3f}")

    print("\nH2 (HIVC-D vs baseline, Welch t-test; Bonferroni α=0.0167):")
    for vc, res in h2.items():
        mark = " *" if res["sig_bonferroni"] else ""
        print(f"  {vc}: t={res['t']:.3f}, p={res['p']:.4f}{mark}")

    print("\nH3 (交互作用, two-way ANOVA):")
    print(f"  F_interaction={h3['F_interaction']:.3f}, p={h3['p_interaction']:.4f}")

    print("\nH4 (ρ vs V不一致度, 線形回帰):")
    print(f"  slope={h4['slope']:.4f}, r²={h4['r_squared']:.3f}, p={h4['p_value']:.4f}")

    # Condition means
    print("\n=== Mean Cumulative Reward by Condition ===")
    pivot = df.groupby(["v_condition", "mechanism"])["cumulative_reward"].agg(["mean", "std"])
    print(pivot.to_string())

    if not args.no_plot:
        print("\nGenerating plots...")
        generate_all_plots(df)
        print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
