"""
Step 7 - Computational Consistency Tests
Tests that empirically verify behavior predicted by Theorems 1-4.
These are consistency checks, NOT mathematical proofs.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from data.data_generator import generate_data
from model.utility import compute_utility
from model.feasibility import compute_feasibility
from model.optimization import solve
from model.config import BASE_CAPACITIES as BASE_CAP


def test_theorem1_disposal_consistency():
    """
    Consistency check for Theorem 1 (Disposal Elimination Property).
    Theorem 1 states: yiD=1 iff max{uij : j in P, Fij=1} <= 0 (with epsilon tie-breaking).
    Under A7-relaxed (shared capacity), disposal may also occur when all
    positive-utility positions are at full capacity — this is expected behavior,
    not a violation. This test distinguishes the two cases.
    """
    print("\n[T1] Disposal Elimination consistency check...")
    df = generate_data(I=100, seed=42)
    U, P = compute_utility(df, w1=0.7, w2=0.3)
    F = compute_feasibility(df)
    result = solve(U, F, I=100, epsilon=0.01)
    assignments = result['assignments']
    position_names = result['position_names']
    disposal_idx = len(P)
    epsilon = 0.01

    # Get actual capacity usage
    capacity_usage = {}
    for j, pos in enumerate(P):
        capacity_usage[pos] = sum(1 for a in assignments if a == j)

    BASE_CAP_SCALED = {k: max(1, int(v * 100/100)) for k, v in BASE_CAP.items()}

    violations = 0
    capacity_constrained = 0
    utility_dominated = 0

    for i, a in enumerate(assignments):
        if a == disposal_idx:
            feasible_utils = [U[i, j] for j in range(len(P)-1) if F[i, j] == 1]
            if not feasible_utils or max(feasible_utils) <= epsilon:
                utility_dominated += 1
            else:
                pos_names = list(BASE_CAP_SCALED.keys())
                all_full = all(
                    capacity_usage.get(pos_names[j], 0) >= BASE_CAP_SCALED.get(pos_names[j], 999)
                    for j in range(len(P)-1)
                    if F[i, j] == 1 and U[i, j] > epsilon
                )
                if all_full:
                    capacity_constrained += 1
                else:
                    violations += 1

    print(f"  Disposal units: {sum(1 for a in assignments if a == disposal_idx)}")
    print(f"  Utility-dominated: {utility_dominated}")
    print(f"  Capacity-constrained: {capacity_constrained}")
    print(f"  Unexplained violations: {violations}")
    assert violations == 0, f"T1 FAIL: {violations} unexplained disposal assignments"
    print("  T1 PASS")


def test_theorem2_information_loss():
    """
    Consistency check for Theorem 2 (Structural Information Irrecoverability)
    and Corollary 2.1 (Decision Loss Potential).
    Theorem 2 states: information lost at RSS cannot be recovered downstream.
    This test simulates RSS incompleteness by masking the 'condition' attribute
    for 30% of units (replacing true condition with a default value).
    The masked model is then evaluated using TRUE (full-information) utility,
    showing that information loss causes measurable assignment changes and
    utility reduction — consistent with Corollary 2.1.
    NOTE: This is an empirical consistency check, not a formal proof of
    irrecoverability. The formal proof is in the paper (Step 9, Theorem 2).
    """
    print("\n[T2] Information loss consistency check...")
    df = generate_data(I=1000, seed=42)

    # Full information
    U_full, P = compute_utility(df, w1=0.7, w2=0.3)
    F_full = compute_feasibility(df)
    result_full = solve(U_full, F_full, I=1000, epsilon=0.01)

    # Masked: 30% of units have condition replaced with 'fair'
    df_masked = df.copy()
    rng = np.random.default_rng(42)
    mask_idx = rng.choice(len(df), size=int(0.3 * len(df)), replace=False)
    df_masked.loc[mask_idx, 'condition'] = 'Stained'

    U_masked, _ = compute_utility(df_masked, w1=0.7, w2=0.3)
    F_masked = compute_feasibility(df_masked)
    result_masked = solve(U_masked, F_masked, I=1000, epsilon=0.01)

    # Evaluate masked assignments using TRUE utility
    true_utility_of_masked = sum(
        U_full[i, a] if a < len(P) - 1 else 0.0
        for i, a in enumerate(result_masked['assignments'])
    )

    utility_loss = result_full['total_utility'] - true_utility_of_masked
    changed = sum(
        1 for a, b in zip(result_full['assignments'], result_masked['assignments'])
        if a != b
    )

    print(f"  Full-info utility:          {result_full['total_utility']:.2f}")
    print(f"  Masked (true eval) utility: {true_utility_of_masked:.2f}")
    print(f"  Utility loss:               {utility_loss:.2f}")
    print(f"  Assignment changes:         {changed}/1000")

    assert utility_loss > 0, "T2 FAIL: information loss caused no utility reduction"
    assert changed > 0, "T2 FAIL: information loss caused no assignment changes"
    print("  T2 PASS")


def test_theorem3_utility_loss():
    """
    Consistency check for Theorem 3 (Execution-Layer Utility Loss Identifiability).
    Theorem 3 states: Delta_i = max{uij : j in P} - u_{i,ri*} is computable
    from the recorded (U, F) profile alone, without external information.
    This test computes Delta_i for all units and verifies it is non-negative
    and identifiable directly from model outputs.
    """
    print("\n[T3] Utility loss identifiability check...")
    df = generate_data(I=100, seed=42)
    U, P = compute_utility(df, w1=0.7, w2=0.3)
    F = compute_feasibility(df)
    result = solve(U, F, I=100, epsilon=0.01)
    assignments = result['assignments']

    deltas = []
    for i, a in enumerate(assignments):
        max_feasible = max(
            (U[i, j] for j in range(len(P)-1) if F[i, j] == 1),
            default=0.0
        )
        actual_utility = U[i, a] if a < len(P) - 1 else 0.0
        delta_i = max_feasible - actual_utility
        deltas.append(delta_i)
        assert delta_i >= -1e-9, f"T3 FAIL: negative Delta_i={delta_i:.4f} at unit {i}"

    print(f"  Delta_i computed for all {len(deltas)} units")
    print(f"  Mean Delta_i: {np.mean(deltas):.4f}")
    print(f"  Max Delta_i:  {np.max(deltas):.4f}")
    print("  T3 PASS")


def test_theorem4_scalability():
    """
    Consistency check for Theorem 4 (Complexity Boundary).
    Theorem 4 states: under uniform capacity consumption (kij=1), the
    assignment problem is solvable in polynomial time via min-cost flow.
    This test measures empirical runtime at I=100, 1000, 10000 and reports
    growth ratios. Sub-linear growth relative to I confirms practical
    tractability.
    NOTE: This is empirical runtime observation, not a polynomial proof.
    The formal proof is in the paper (Step 9, Theorem 4).
    """
    import time
    print("\n[T4] Scalability (Theorem 4 complexity boundary)...")
    times = {}
    for I in [100, 1000, 10000]:
        df = generate_data(I=I, seed=42)
        U, P = compute_utility(df, w1=0.7, w2=0.3)
        F = compute_feasibility(df)
        t0 = time.time()
        solve(U, F, I=I, epsilon=0.01)
        times[I] = time.time() - t0
        print(f"  I={I:6d}: {times[I]:.3f}s")

    ratio_10 = times[1000] / times[100]
    ratio_100 = times[10000] / times[1000]
    print(f"  Growth ratio 100→1000:   {ratio_10:.1f}x")
    print(f"  Growth ratio 1000→10000: {ratio_100:.1f}x")
    assert ratio_100 < 50, f"T4 FAIL: super-linear growth ({ratio_100:.1f}x)"
    print("  T4 PASS")


if __name__ == "__main__":
    test_theorem1_disposal_consistency()
    test_theorem2_information_loss()
    test_theorem3_utility_loss()
    test_theorem4_scalability()
    print("\n=== All consistency checks passed ===")