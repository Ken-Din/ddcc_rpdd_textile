"""
External benchmark: TOPSIS-based disposition heuristic vs. RPDD (min-cost flow).

A per-unit multi-criteria disposition heuristic representative of MCDM
approaches in the reverse logistics disposition literature (cf. Barker &
Zabinsky 2011): for each return unit, feasible recovery positions are
ranked by TOPSIS closeness over two criteria — residual value Vij
(benefit) and recovery cost Cij (cost) — using the same weights as RPDD
(w1 = 0.7, w2 = 0.3). Units are processed in arrival order and assigned
to their highest-ranked feasible position with remaining capacity and
positive net utility; otherwise the unit is disposed of.

Both methods consume identical inputs (same data, utility, feasibility,
capacities, seed), so the comparison isolates the assignment logic.

Usage: python experiments/external_benchmark.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.data_generator import generate_data
from model.utility import compute_utility, BASE_VALUE, BASE_COST
from model.feasibility import compute_feasibility
from model.optimization import solve
from model.config import (POSITIONS as P, DEFAULT_W1, DEFAULT_W2,
                          DEFAULT_EPSILON, BASE_CAPACITIES,
                          AGE_DECAY_RATE, BRAND_MULTIPLIER)


def vc_matrices(df):
    base_values = BASE_VALUE.loc[df["condition"].values, P].to_numpy(dtype=float)
    base_costs  = BASE_COST.loc[df["condition"].values, P].to_numpy(dtype=float)
    age_decay   = np.exp(-AGE_DECAY_RATE * df["age_days"].to_numpy(dtype=float))
    brand_mult  = df["brand_tier"].map(BRAND_MULTIPLIER).to_numpy(dtype=float)
    V = base_values * age_decay[:, None] * brand_mult[:, None]
    return V, base_costs


def topsis_order(V_row, C_row, feas_idx, w1, w2):
    M = np.column_stack([V_row[feas_idx], C_row[feas_idx]]).astype(float)
    norms = np.linalg.norm(M, axis=0); norms[norms == 0] = 1.0
    Wt = (M / norms) * np.array([w1, w2])
    ideal = np.array([Wt[:, 0].max(), Wt[:, 1].min()])
    anti  = np.array([Wt[:, 0].min(), Wt[:, 1].max()])
    d_pos = np.linalg.norm(Wt - ideal, axis=1)
    d_neg = np.linalg.norm(Wt - anti, axis=1)
    denom = d_pos + d_neg
    closeness = np.where(denom > 0, d_neg / denom, 0.0)
    return feas_idx[np.argsort(-closeness)]


def run_topsis(U, V, C, F, caps, w1=DEFAULT_W1, w2=DEFAULT_W2):
    I = U.shape[0]
    remaining = dict(caps)
    assign = np.full(I, -1, dtype=int)
    for i in range(I):
        feas_idx = np.where(F[i].astype(bool))[0]
        if len(feas_idx) == 0:
            continue
        for j in topsis_order(V[i], C[i], feas_idx, w1, w2):
            if remaining[P[j]] > 0 and U[i, j] > 0:
                assign[i] = j
                remaining[P[j]] -= 1
                break
    total = float(sum(U[i, assign[i]] for i in range(I) if assign[i] >= 0))
    return total, int((assign == -1).sum())


def main(I=1000, seed=42):
    df = generate_data(I=I, seed=seed)
    U, _ = compute_utility(df, w1=DEFAULT_W1, w2=DEFAULT_W2)
    F = compute_feasibility(df)
    V, C = vc_matrices(df)
    scale = I / 100
    caps = {pos: max(1, int(BASE_CAPACITIES[pos] * scale)) for pos in P}

    t0 = time.time()
    res = solve(U, F, I, epsilon=DEFAULT_EPSILON, scale_capacities=True)
    t_opt = time.time() - t0

    t0 = time.time()
    tot_h, disp_h = run_topsis(U, V, C, F, caps)
    t_h = time.time() - t0

    opt = res["total_utility"]
    disp_o = res["assignments"].count(len(P))
    gap = opt - tot_h
    print(f"I={I}, seed={seed}")
    print(f"RPDD (min-cost flow): utility={opt:.2f}  disposal={disp_o}  time={t_opt:.3f}s")
    print(f"TOPSIS heuristic:     utility={tot_h:.2f}  disposal={disp_h}  time={t_h:.3f}s")
    print(f"Gap: {gap:.2f} ({100*gap/opt:.2f}% below optimal)")


if __name__ == "__main__":
    for I in (100, 1000, 10000):
        main(I=I)
        print("-" * 50)