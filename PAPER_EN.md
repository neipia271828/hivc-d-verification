# Does Disagreement in Judgment Criteria (V) Degrade Consensus Quality Under a Shared Goal?
## A Multi-Agent Simulation Test of the HIVC-D Framework

**Author**: pai314
**Date**: May 2026

---

## Abstract

The conflict-resolution framework "HIVC-D" claims that even when multiple decision-makers share a common goal, a mismatch in their judgment-criteria vectors **V** (value weightings) drives the consensus away from the true optimal choice **C_t**, thereby degrading outcomes. This study quantitatively tests that core hypothesis using a multi-agent simulation in which two foragers jointly explore a field of rewards (food). We manipulated the degree of V-disagreement at three levels (low, mid, high) and the agreement mechanism at two levels (simple averaging = baseline / the HIVC-D intervention flow = information sharing → V-alignment check → V-negotiation when needed), running 100 trials per cell for 600 trials total.

In the initial implementation, hypotheses H1–H3 were all non-significant, but we discovered several confounds in internal validity (both agents moving along identical trajectories, an information-quantity asymmetry between baseline and HIVC-D, and an inconsistent definition of C_t). After a redesign that removed these confounds (a single-team-position model, symmetrization of information quantity via observation averaging, and a choice-set-relative definition of C_t), **measurement reliability improved but hypotheses H1–H3 remained non-significant**. Decomposing the causal chain revealed that the theory's first link, "V-disagreement → C_t deviation," structurally does not hold (linear regression slope = −0.033, p = 0.49, r² = 0.0016). Meanwhile, hypothesis H4 ("after information sharing, the rank correlation ρ is inversely proportional to V-disagreement") held strongly and consistently (r² = 0.733, p < 0.001), but this is a quasi-tautological relationship that closes within the score computation alone.

In conclusion, **in a regime where all options are aligned with the shared goal, agreement is a symmetric average, and the goal is shared, V-disagreement is self-canceled by the averaging operation and does not degrade consensus quality**. For the HIVC-D intervention to demonstrate superiority, a structure in which V systematically biases toward inferior choices (misleading information, asymmetric agreement, or V components orthogonal to the goal) appears to be required.

**Keywords**: consensus formation, value disagreement, multi-agent simulation, collective decision-making, null result

---

## 1. Introduction

### 1.1 Background and the Originating Idea

In organizational and team decision-making, the question often arises: "Everyone is aiming at the same goal, so why does reaching agreement fail?" The HIVC-D framework explains this phenomenon by decomposing it into the following elements.

- **Holding a common goal G** is not a sufficient condition for agreement.
- If the **judgment criteria V (what to value and by how much)** used by each decision-maker differ, the same information and the same goal can still lead to different conclusions.
- Furthermore, the **environmental information I** each decision-maker references may also differ.

The starting idea of this study can be distilled to a single point:

> **Despite the existence of a common goal G, could disagreement in judgment criteria V itself be an independent factor that drives a group's consensus away from the true optimal choice and harms outcomes?**

If this is correct, it implies the practical lesson that conflict-resolution intervention should prioritize "**diagnosing and aligning judgment criteria (V-alignment)**" over "reconciling goals." HIVC-D codifies this intervention order as a diagnostic flow: "share I (information) → align V (criteria) → confirm A (ability)."

### 1.2 Objectives

This study aims to operationalize the abstract theoretical claims into a falsifiable form and to test the following four hypotheses via simulation.

| Hypothesis | Statement |
|------|------|
| **H1** | The higher the V-disagreement, the lower the consensus outcome (cumulative reward). |
| **H2** | The HIVC-D intervention flow (Condition B) yields higher outcomes than simple-average agreement (Condition A). |
| **H3** | The greater the V-disagreement, the larger the room for improvement from intervention (the A–B gap). |
| **H4** | Sharing common information I promotes convergence of the rank correlation ρ independent of V. |

---

## 2. The HIVC-D Framework

This section provides a self-contained explanation of the framework's purpose, components, and claims for researchers encountering HIVC-D for the first time. Readers already familiar may skip to the verification scope in 2.7.

### 2.1 The Problem the Framework Addresses

HIVC-D is **a descriptive framework for decomposing and diagnosing the process by which multiple decision-makers reach "agreement."** Its motivating empirical observation distills to a single point.

> Even though all team members hold the same goal, agreement often fails to form, and when it does form, it can reach an erroneous conclusion.

In everyday terms, this failure tends to be lumped together as "interpersonal problems" or "lack of communication." HIVC-D's claim is that such agreement failures can in fact be reduced to **a mismatch in one of a small number of identifiable factors (information, judgment criteria, ability)**, and that identifying the factor lets one select the corresponding intervention. In other words, the framework's goal is to treat conflict not as a matter of mindset but as **a diagnosable structure**.

### 2.2 Name and Basic Elements

The framework name **HIVC-D** combines the initials of its core components (judge **H**, information **I**, judgment criterion **V**, choice **C**) with the extension **D (Dynamic)** denoting iterativeness. A decision situation is described by the following elements.

```
E   : Environment (including external constraints and resource limits)
H   : Set of judges (Human) {h1, h2, ...}
G   : Common goal (assumed already agreed upon)
P   : Proposition to be solved (what to decide)
C   : Set of choices (available moves)
C_t : The true optimal choice w.r.t. G (Choice True; existence assumed but not necessarily unique)
I   : Environmental information (data/facts each judge references)
V   : Judgment criterion (Value; the value weighting each judge uses to evaluate choices)
A   : Each h's estimation ability (the precision ceiling of information processing/judgment)
```

Intuitively, each judge h evaluates choices **C** by weighting their own information **I** with their own values **V**, and selects the move deemed best. If everyone selects the true optimum **C_t**, agreement is trivial; but in reality, because one of I, V, or A differs across people, evaluations diverge and conflict arises.

**Key premise**: HIVC-D addresses situations where the common goal G is *already agreed upon*. That is, the trivial case of "fighting because goals differ" is out of scope; the focus is sharpened on **"why we clash even with the same goal."** This is what makes the framework's problem setting incisive.

### 2.3 Structure and Weaknesses of the Original Version

The early HIVC-D was a static model asserting that "if all judges reason ideally, they necessarily converge to C_t and agree." This formulation, however, has the following theoretical weaknesses. This study takes as its object the revised version (Dynamic) that corrects them.

| Problem | Concrete defect |
|------|-----------|
| Independence assumption of V and I | Ignores the interdependence whereby information changes V and V changes the direction of I collection |
| Tautology assumption | "If everyone is ideal, they agree" is a definition with no predictive power |
| Static model | Treats decisions as one-shot, ignoring the iterativeness (feedback) of practice |
| Coarse ability term | Discards "lack of ability" as untreatable |
| Agreement ≠ execution | Post-agreement commitment/execution is undefined |

### 2.4 Revised Assumptions of HIVC-D (Dynamic)

| Old assumption | Revision |
|--------|------|
| C_t exists uniquely | C_t exists as a **Pareto-optimal set**; uniqueness is not guaranteed |
| I and V are independent | I(t) and V(t) are **interdependent**: I updates V, and V determines the direction of I collection |
| All ideal → agreement (tautology) | **Removed**, replaced by an explicit definition of agreement conditions (2.5) |
| Ability is fixed | A is a **manipulable parameter** via training, delegation, and division of labor |

### 2.5 Explicit Definition of Agreement Conditions (Replacing the Tautology)

The revised version discards "ideal implies agreement" and defines agreement by three verifiable conditions.

```
For any pair of judges hi, hj ∈ H, agreement is reached when all of the following hold:

  1. I-match condition  : the difference between the I that hi and hj reference is within tolerance ε
  2. V-alignment cond.   : the priority orders of Vi and Vj match, or both converge to a mutually acceptable V*
  3. Ability condition   : each h's ability A is sufficient to evaluate C_t
```

These three conditions are also the starting point of the diagnosis described later (2.6). When agreement fails, asking which of the three conditions is broken isolates the cause.

### 2.6 The Dynamic Model and Diagnosis/Intervention

The essence of the revised version is to treat decisions **not as one-shot but as an iterative process with environmental feedback**.

```
t=0: Agree on the common goal G
      |
Step 1: Collect I(t)     -- V governs the direction of I collection
      |
Step 2: Update V(t)      -- the collected I may revise V
      |
Step 3: Evaluate C(t)    -- estimate C_t with current I(t), V(t), A(t)
      |
Step 4: Agreement check  -- are the 3 conditions (I-match, V-align, ability) met?
      |- YES -> execute/commit -> environmental feedback E(t+1), I(t+1) -> next cycle
      `- NO  -> diagnose -> intervene (table below)
```

When agreement fails, one first diagnoses the cause by asking **"would they agree if given the same I(t)?"**

- **YES (they agree given the same information)** → the cause is **V-disagreement**.
- **NO (they disagree even with the same information)** → the cause is **I-disagreement** or **lack of ability**.

Depending on the diagnosis, interventions are applied in order of lowest cost.

| Intervention | Target cause | Cost | Notes |
|---------|---------|--------|------|
| Share I | I-disagreement | Low | Try first |
| Negotiate V | V-disagreement | Mid | Seek a common V* anchored to consistency with G |
| Enumerate C | I·V compound | Mid | Search for new choices that partially satisfy both sides' V |
| Strengthen A | Lack of ability | High | Training, delegation, bringing in outside experts |
| Random C | Time-out | — | The decision to **explicitly give up** converging to C_t |

This "**diagnostic order I → V → A**" is the practical core of HIVC-D. Information mismatches (I) can be resolved cheaply, so they are eliminated first; if agreement still fails, value mismatches (V) are negotiated; and finally, ability (A) issues are addressed.

### 2.7 The Claims This Study Tests and Their Scope

Of the entire framework above, this study quantitatively tests only the following two points.

1. **Core hypothesis (the harm of V-disagreement)**: Even under a common goal G, V-disagreement drives consensus away from the true optimum C_t and lowers outcomes.
2. **Superiority of the diagnostic order (HIVC-D intervention)**: The intervention flow "share I → check V-alignment → negotiate V when needed" yields better outcomes than a simple compromise (averaging).

Ability A heterogeneity, scaling to many participants, interventions such as C-enumeration and A-strengthening, and post-agreement execution/feedback are out of scope; A is held uniform.

The theory predicts the core hypothesis as the following **two-stage causal chain**. The methodological contribution of this study is to decompose this chain and measure each link individually.

```
V-disagreement --(Link 1)--> C_t deviation of the agreed vector --(Link 2)--> drop in cumulative reward
```

If Link 1 does not hold, then even if Link 2 (deviation lowers outcomes) is true, V does not affect outcomes. As shown in Section 4, the main finding of this study is precisely the failure of Link 1.

### 2.8 Correspondence of Framework Elements to the Simulation

The abstract elements were operationalized into a foraging simulation as follows (details in Section 3).

| Symbol | Name | Correspondence in the simulation |
|------|------|------------------------|
| E    | Environment | The field (placement of food and signs) |
| H    | Set of judges | Two foragers |
| G    | Common goal | Maximizing cumulative reward |
| C    | Set of choices | Direction of travel (a vector in continuous space) |
| C_t  | True optimal choice | The direction of maximal reward efficiency (the noise-free ideal solution) |
| I    | Environmental information | The set of signs each forager observes |
| V    | Judgment criterion | A weight vector (w_c, w_d, w_p) |
| A    | Estimation ability | Held uniform in this experiment (control variable) |

The judgment criterion V = (w_c, w_d, w_p) is the weight each forager uses to evaluate a sign, expressing preferences for "consistency with the direction of travel (c)," "distance efficiency (d)," and "reward magnitude (p)," respectively (normalized to sum to 1). The three agreement conditions are operationalized in this simulation as follows.

- **I-match condition**: In the HIVC-D condition, the two foragers' observations are shared (`shared_obs`), forcibly satisfying it.
- **V-alignment condition**: The Spearman rank correlation ρ of the score rankings over the common information is at or above a threshold (RHO_AGREE_THRESHOLD = 0.6). When ρ < 0.6, "V-misalignment" is declared and V-negotiation (`v* = (v1 + v2) / 2`, both sides conceding equally) is triggered.
- **Ability condition**: Held uniform in this experiment and kept out of scope.

---

## 3. Methods

### 3.1 Environment

In a 100×100 continuous 2D field, 20 food items with reward magnitude in [1.0, 10.0] and 50 signs are randomly placed. Each sign points to the nearest food, and on observation independently adds direction noise (σθ = 0.3 rad), distance noise (σd = 0.2, multiplicative), and reward noise (σp = 1.0, additive). Each turn, a forager observes the K = 5 nearest signs and moves MOVE_STEP = 2.0 in the agreed direction. The food-acquisition radius is 2.0. One trial is T_MAX = 200 turns.

### 3.2 Independent Variables

**(a) Degree of V-disagreement (3 levels)**

| Condition | V1 | V2 | ‖V1−V2‖₂ |
|------|------|------|-----------|
| V-Low  | (0.33, 0.33, 0.33) | (0.40, 0.30, 0.30) | 0.082 |
| V-Mid  | (0.60, 0.20, 0.20) | (0.20, 0.20, 0.60) | 0.566 |
| V-High | (0.80, 0.10, 0.10) | (0.10, 0.10, 0.80) | 0.990 |

In V-Mid and V-High, w_d (the distance weight) is identical for both foragers, so the disagreement appears mainly as a conflict between "emphasis on directional consistency (w_c)" and "emphasis on reward magnitude (w_p)."

**(b) Agreement mechanism (2 levels)**

- **Condition A (baseline)**: Each forager independently computes a desired vector, and their simple average is the direction of travel. Neither I-sharing nor V-negotiation is performed.
- **Condition B (HIVC-D)**: ① Share I (integrate both foragers' observations) → ② V-alignment check (compute ρ) → ③ if ρ ≥ 0.6, average desired vectors over the integrated information; if ρ < 0.6, V-negotiation (recompute desired vectors with v*).

3 × 2 = 6 cells, 100 trials each, 600 trials total. Randomness was controlled with `seed = 42 + trial_id` for reproducibility.

### 3.3 Dependent Variables

- **Cumulative reward** (primary outcome measure)
- **C_t approximation rate** (the inner product of the agreed vector and C_t; a direct measure of Link 1)
- **Rank correlation ρ** (V-alignment; Condition B only)
- **Mean agreement cost** (V-negotiation occurrence rate)

### 3.4 Statistical Methods

- H1: One-way ANOVA over V-Low/Mid/High on baseline only, plus a linear trend regression treating V-distance as a continuous variable.
- H2: Welch's t-test (baseline vs hivc) per V condition, with Bonferroni correction for the three multiple comparisons (α = 0.0167).
- H3: The interaction term of a two-way ANOVA (V condition × mechanism).
- H4: Linear regression of ρ_before (pre-negotiation) on V-disagreement.

### 3.5 Iterative Refinement of the Implementation and Securing Internal Validity

Because all hypotheses were non-significant in the initial implementation, we conducted a systematic review of internal validity and identified and removed the following confounds. This process itself is a methodological contribution of the study.

| Confound found | Problem | Fix |
|---|---|---|
| **Identical-trajectory movement** | Both agents move by the same vector every turn, so the initial coordinate gap is conserved as a constant—effectively a pseudo two-body problem | Move to a single-team-position model; purify I-disagreement as observation noise |
| **Information-quantity asymmetry** | Baseline uses K signs while HIVC-D uses up to 2K, so the comparison measures a "difference in information quantity" rather than a "difference in mechanism" | Average observations of the same signs index-wise, symmetrizing both to K |
| **Inconsistent C_t definition** | C_t was the best food over the whole field (including distant ones), diverging from the forager's visible choices; the C_t approximation rate was systematically negative (−0.27) | Redefine C_t as the noise-free best direction within the visible sign set C |
| **Scale of the score's distance term** | Normalization by an arbitrary constant | Physically meaningful normalization with FOOD_RADIUS as the unit |
| **Field reset** | `seed+t` made it depend on the reset timing | Made deterministic with `seed + reset_count × 100003` |
| **Direction duplicate detection** | The 0.1-rad threshold was arbitrary and did not handle ±π wraparound | Replaced with a circular mean |

After these fixes, the sign of the C_t approximation rate normalized from **−0.27 → +0.53**, enabling a reliable decomposed measurement of the causal chain.

---

## 4. Results

### 4.1 Tests of the Main Hypotheses (final model, n = 100/cell)

| Hypothesis | Statistic | p-value | Verdict |
|------|--------|------|------|
| H1 (ANOVA) | F = 0.196, η² = 0.001 | 0.822 | n.s. |
| H1 (linear regression) | slope = 0.350, r² = 0.000 | 0.736 | n.s. |
| H2 V-Low | t = −0.923 | 0.357 | n.s. |
| H2 V-Mid | t = −0.987 | 0.325 | n.s. |
| H2 V-High | t = −0.383 | 0.702 | n.s. |
| H3 (interaction) | F = 0.091 | 0.914 | n.s. |
| H4 (ρ vs V-distance) | slope = −0.661, r² = 0.733 | <0.001 | **significant** |

### 4.2 Cumulative Reward by Condition

| V condition | Mechanism | Mean | Std. dev. |
|------|------|------|----------|
| V-Low  | baseline | 7.06 | 6.70 |
| V-Low  | hivc | 6.21 | 6.30 |
| V-Mid  | baseline | 6.80 | 6.20 |
| V-Mid  | hivc | 5.98 | 5.65 |
| V-High | baseline | 7.39 | 7.08 |
| V-High | hivc | 7.03 | 6.44 |

Neither a systematic decline in reward with increasing V-disagreement (H1) nor a superiority of HIVC-D (H2) was observed. If anything, HIVC-D showed slightly lower means across all conditions.

### 4.3 Decomposition of the Causal Chain

Measuring each stage of the theory's two-stage chain individually (baseline condition):

```
Link 1: V-disagreement -> C_t deviation   slope=-0.033, p=0.49, r^2=0.0016   -- does not hold
Link 2: C_t deviation -> cumulative reward slope=-1.58,  p=0.21               -- weak
```

The C_t approximation rate was nearly constant across all V conditions (V-Low: 0.558, V-Mid: 0.530, V-High: 0.528). That is, **increasing V-disagreement roughly 12-fold (0.082 → 0.990) did not change the deviation of the agreed vector from C_t**.

### 4.4 Direct Measurement of the Intervention Effect

Directly testing whether HIVC-D improves the C_t approximation rate revealed no significant improvement in any V condition (V-Low: Δ = −0.001, p = 0.98; V-Mid: Δ = +0.022, p = 0.62; V-High: Δ = −0.012, p = 0.78).

### 4.5 Hyperparameter Grid Search

Searching 36 settings of RHO_AGREE_THRESHOLD (4 levels) × SIGMA_THETA (3 levels) × K_SIGNS (3 levels) × 30 trials each, H1 was significant in 0/36 and H2 in at least one condition in only 1/36. There was a tendency for HIVC-D's superiority to be maximized when the observation noise σθ was smallest (0.1) and the number of signs K_SIGNS was largest (10), but none reached a statistically significant level.

---

## 5. Discussion

### 5.1 Why Was V-Disagreement Harmless?

The most important finding is that the theory's first link, "V-disagreement → C_t deviation," structurally does not hold in this experimental regime. Its mechanism reduces to the combination of the following three conditions.

1. **All options align with the common goal**: Every sign points (with noise) to a real food item; there are no misleading choices.
2. **Symmetric average agreement**: Agreement is the simple average of the two desired vectors.
3. **Shared goal**: Both foragers pursue the same maximization of cumulative reward.

Under these conditions, the two desired vectors weighted by different V values each bias in different directions, but **the averaging operation cancels the biases**, and the result converges to an unbiased average. For example, in the V-High condition one side emphasizes "directional consistency" and the other "reward magnitude," but averaging the two yields a moderate direction that reasonably satisfies both preferences, and the closeness to C_t is no different from the single-criterion case. In other words, because **there is no conflict to solve**, the HIVC-D intervention has no effect to exert.

### 5.2 Interpretation of H4's Significance

Only H4 (ρ vs V-distance) held strongly and consistently, but this is a **quasi-tautological relationship** rather than a test of the theory. ρ is the degree of agreement of score rankings over the common information, determined by score computation alone without passing through any agent movement or reward. That rankings change when V differs is nearly inevitable by definition, and the holding of this relationship does not support the claim that "V affects C_t deviation or outcomes." Treating H4 as core evidence requires caution.

### 5.3 The Value of a Null Result and Its Methodological Implications

This study ended in a result that does not support the hypotheses, but this is a **reliable null result**. It is a conclusion reached after removing confounds through iterative internal-validity review, and we did not engage in "arbitrary parameter tuning to make the hypotheses significant (p-hacking)." Rather, in the process of removing confounds, we revealed that the apparent behavior of the initial implementation stemmed from several artifacts—this is where the methodological value lies.

### 5.4 Regimes Where the Hypotheses Could Hold

The conditions that could rescue the theory amount to breaking one of the premises listed in 5.1.

1. **Introducing misleading/adversarial information**: An environment where some signs present false food information. A forager strongly biased toward "reward magnitude" in V would be lured to false high-reward signs, and the bias could persist even under average agreement.
2. **Asymmetric agreement**: Agreement where one forager carries a dominant weight. Cancellation does not occur, and the dominant side's V bias is reflected directly in the outcome.
3. **A V component orthogonal to the goal**: A forager with a value unrelated to reward maximization (e.g., fixation on a particular direction). This component is not canceled by averaging and produces systematic C_t deviation.

In these settings, room theoretically arises for HIVC-D's "I → V diagnostic order" to outperform simple averaging.

---

## 6. Limitations and Future Work

- **Single-regime test**: This experiment treated only one specific regime—"cooperative, aligned, symmetric." Testing the alternative regimes in 5.4 is the next task.
- **Simplicity of the V-negotiation model**: Only equal-concession `v* = (v1 + v2) / 2` was implemented. The effects of other negotiation algorithms, such as the Nash bargaining solution, are untested.
- **Two parties, fixed ability**: There are two judges and estimation ability A is held uniform. The effects of scaling to many participants or of A heterogeneity are out of scope.
- **Noise in the outcome measure**: Cumulative reward is highly path-dependent and high-variance. From a statistical-power standpoint, a design using a more direct measure such as the C_t approximation rate as the primary dependent variable is preferable.

---

## 7. Conclusion

We tested HIVC-D's core hypothesis—that under a shared goal, disagreement in judgment criteria V degrades consensus quality—via a multi-agent simulation. In a final model that secured internal validity, **V-disagreement did not cause C_t deviation of the agreed vector (the theory's first link does not hold) and did not affect cumulative reward**. This null result stems from the fact that, in a regime where all options align with the goal and agreement is a symmetric average, V-disagreement is self-canceled by the averaging operation. For the HIVC-D intervention to exhibit superiority, an environmental structure in which V systematically biases toward inferior choices (misleading information, asymmetric agreement, or V orthogonal to the goal) is required. The contribution of this study lies in delimiting the theory's scope of application and making its boundary conditions explicit.

---

## Code and Data Availability

All simulation code, statistical tests, the grid search, and the generated results (figures and summary statistics) used in this study are publicly available under the MIT License at the following repository.

> **GitHub**: https://github.com/neipia271828/hivc-d-verification

The repository includes this paper (`PAPER.md` / `PAPER.tex`, with English versions `PAPER_EN.md` / `PAPER_EN.tex`), reproduction instructions (`README.md`), dependencies (`requirements.txt`), and the full unit-test suite. All randomness is seed-controlled, so the results are fully reproducible.

---

## Appendix: Reproduction

Clone the repository, install dependencies, and run.

```bash
git clone https://github.com/neipia271828/hivc-d-verification.git
cd hivc-d-verification
pip install -r requirements.txt
cd hivc_sim

# Unit tests
python -m pytest tests/ -v

# Main experiment (all 600 trials)
python main.py --trials 100

# Hyperparameter grid search
python grid_search.py --trials 30
```

Outputs: `results/raw/trials.csv` (per-trial raw data), `results/summary/stats.csv` (test statistics), `results/figures/` (figures), and `results/grid_search/` (search results).
