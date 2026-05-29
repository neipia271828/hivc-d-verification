from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def test_H1(df: pd.DataFrame) -> dict:
    """H1: V不一致度↑ → 累積報酬↓（条件A）

    一元配置ANOVA（カテゴリ扱い）に加え、V距離は順序変数のため
    cumulative_reward ~ v_distance の線形回帰も併記する（検出力が高い）。
    """
    baseline = df[df["mechanism"] == "baseline"]
    groups = [
        baseline[baseline["v_condition"] == vc]["cumulative_reward"].values
        for vc in ["V-Low", "V-Mid", "V-High"]
    ]
    f_stat, p_value = stats.f_oneway(*groups)
    grand_mean = baseline["cumulative_reward"].mean()
    ss_total = ((baseline["cumulative_reward"] - grand_mean) ** 2).sum()
    group_means = [g.mean() for g in groups]
    group_ns = [len(g) for g in groups]
    ss_between = sum(n * (gm - grand_mean) ** 2 for n, gm in zip(group_ns, group_means))
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    # 線形トレンド検定: V距離を連続変数として回帰
    vd = baseline["v_distance"].values
    rew = baseline["cumulative_reward"].values
    slope, intercept, r_value, reg_p, _ = stats.linregress(vd, rew)

    return {
        "F_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_squared),
        "reg_slope": float(slope),
        "reg_p": float(reg_p),
        "reg_r2": float(r_value ** 2),
    }


def test_H2(df: pd.DataFrame) -> dict:
    """H2: 条件B（HIVC-D）は条件Aより累積報酬が高い

    各V条件でWelchのt検定。3条件の多重比較に対しBonferroni補正
    （有意水準 α=0.05/3≈0.0167）の判定も付与する。
    """
    results = {}
    bonferroni_alpha = 0.05 / 3
    for vc in ["V-Low", "V-Mid", "V-High"]:
        sub = df[df["v_condition"] == vc]
        base = sub[sub["mechanism"] == "baseline"]["cumulative_reward"].values
        hivc = sub[sub["mechanism"] == "hivc"]["cumulative_reward"].values
        t_stat, p_value = stats.ttest_ind(hivc, base, equal_var=False)
        results[vc] = {
            "t": float(t_stat),
            "p": float(p_value),
            "sig_bonferroni": bool(p_value < bonferroni_alpha),
        }
    return results


def test_H3(df: pd.DataFrame) -> dict:
    """H3: V-Highでの条件A-B差が最大（二元配置ANOVA交互作用）。scipy only."""
    v_levels = ["V-Low", "V-Mid", "V-High"]
    m_levels = ["baseline", "hivc"]
    y = df["cumulative_reward"].values
    grand_mean = y.mean()
    n = len(y)

    # cell means
    cells: dict[tuple[str, str], np.ndarray] = {}
    for vc in v_levels:
        for mech in m_levels:
            cells[(vc, mech)] = df[(df["v_condition"] == vc) & (df["mechanism"] == mech)]["cumulative_reward"].values

    n_per_cell = min(len(v) for v in cells.values())

    # marginal means
    v_means = {vc: np.mean([cells[(vc, mech)] for mech in m_levels]) for vc in v_levels}
    m_means = {mech: np.mean([cells[(vc, mech)] for vc in v_levels]) for mech in m_levels}

    # SS interaction
    ss_ab = 0.0
    for vc in v_levels:
        for mech in m_levels:
            cell_mean = cells[(vc, mech)].mean()
            n_cell = len(cells[(vc, mech)])
            ss_ab += n_cell * (cell_mean - v_means[vc] - m_means[mech] + grand_mean) ** 2

    # SS error
    ss_e = sum(((cells[(vc, mech)] - cells[(vc, mech)].mean()) ** 2).sum()
               for vc in v_levels for mech in m_levels)

    df_ab = (len(v_levels) - 1) * (len(m_levels) - 1)
    df_e = n - len(v_levels) * len(m_levels)

    if df_e <= 0 or ss_e == 0:
        return {"F_interaction": float("nan"), "p_interaction": float("nan")}

    ms_ab = ss_ab / df_ab
    ms_e = ss_e / df_e
    f_interaction = ms_ab / ms_e
    p_interaction = float(1.0 - stats.f.cdf(f_interaction, df_ab, df_e))
    return {"F_interaction": float(f_interaction), "p_interaction": p_interaction}


def test_H4(df: pd.DataFrame) -> dict:
    """H4: 共通IはVに依存せずρの改善を促す (ρ_before vs V不一致度の線形回帰)"""
    hivc = df[df["mechanism"] == "hivc"].copy()

    rho_means = hivc["rho_history"].apply(
        lambda lst: float(np.nanmean([x for x in lst if not np.isnan(x)])) if lst else float("nan")
    )
    v_dist = hivc["v_distance"].values
    rho_vals = rho_means.values

    mask = ~np.isnan(rho_vals)
    if mask.sum() < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r_squared": float("nan"), "p_value": float("nan")}

    slope, intercept, r_value, p_value, _ = stats.linregress(v_dist[mask], rho_vals[mask])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
    }
