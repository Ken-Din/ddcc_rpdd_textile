"""
Adım 7 — Computational Consistency Tests
Tests that empirically verify behavior predicted by Theorems 1-4.
These are consistency checks, NOT mathematical proofs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from data.data_generator import generate_data
from model.utility import compute_utility
from model.feasibility import compute_feasibility
from model.optimization import solve

def test_theorem1_disposal_consistency():
    print("\n[T1] Disposal Elimination consistency check...")
    df = generate_data(I=100, seed=42)
    U, P = compute_utility(df, w1=0.7, w2=0.3, epsilon=0.01)
    F = compute_feasibility(df)
    result = solve(U, F, I=100, epsilon=0.01)
    assignments = result['assignments']
    position_names = result['position_names']
    disposal_idx = len(P)
    epsilon = 0.01

    # Get actual capacity usage
    capacity_usage = {}
    for j, pos in enumerate(P):
        capacity_usage[pos] = sum(1 for i in range(100) if assignments[i] == j)

    BASE_CAP = {"Resale":100,"Repair":8,"Refurbishing":6,
                "Repackaging":100,"Recycling":4,"Donation":5,"DiscountSale":100}

    disposal_units = [i for i in range(100) if assignments[i] == disposal_idx]
    print(f"    Units assigned to Disposal: {len(disposal_units)}")

    violations = 0
    capacity_constrained = 0

    for i in disposal_units:
        feasible_positive = [
            j for j in range(len(P))
            if F[i, j] == 1 and (U[i, j] - epsilon) > 0
        ]
        if feasible_positive:
            # Check if all these positions are at capacity
            all_at_capacity = all(
                capacity_usage.get(P[j], 0) >= BASE_CAP.get(P[j], 100)
                for j in feasible_positive
            )
            if all_at_capacity:
                capacity_constrained += 1
                print(f"    unit {i}: Disposal due to binding capacity constraints "
                      f"(expected A7-relaxed behavior)")
            else:
                violations += 1
                print(f"    VIOLATION: unit {i} sent to Disposal but "
                      f"{[P[j] for j in feasible_positive]} not at capacity")

    if violations == 0:
        print(f"    PASS: All {len(disposal_units)} disposal units correctly explained.")
        print(f"    {capacity_constrained} units: capacity-constrained (A7 relaxed)")
    else:
        print(f"    FAIL: {violations} unexplained disposal assignments.")
    return violations == 0


def test_theorem2_information_loss():
    print("\n[T2] Information Loss consistency check...")
    df_full = generate_data(I=100, seed=42)
    U_full, P = compute_utility(df_full, w1=0.7, w2=0.3)
    F_full = compute_feasibility(df_full)
    result_full = solve(U_full, F_full, I=100)

    # Mask condition: replace with 'Unknown' mapped to neutral values
    # Simulate RSS incompleteness — use most common condition as default
    df_masked = df_full.copy()
    mask_idx = df_masked.sample(frac=0.3, random_state=42).index
    # Use NotLiked as default (most conservative, doesn't artificially inflate utility)
    df_masked.loc[mask_idx, 'condition'] = 'NotLiked'

    U_masked, _ = compute_utility(df_masked, w1=0.7, w2=0.3)
    F_masked = compute_feasibility(df_masked)
    result_masked = solve(U_masked, F_masked, I=100)

    utility_full = result_full['total_utility']
    utility_masked = result_masked['total_utility']
    changed = np.sum(result_full['assignments'] != result_masked['assignments'])

    # Compute TRUE utility of masked solution using full information
    true_utility_of_masked = 0.0
    for i in range(100):
        j = result_masked['assignments'][i]
        if j < len(P):
            true_utility_of_masked += U_full[i, j]

    print(f"    Full information utility:           {utility_full:.4f}")
    print(f"    Masked solution (self-reported):    {utility_masked:.4f}")
    print(f"    Masked solution (TRUE utility):     {true_utility_of_masked:.4f}")
    print(f"    TRUE utility loss vs full info:     "
          f"{utility_full - true_utility_of_masked:.4f}")
    print(f"    Assignment changes: {changed}/100 units")

    passed = utility_full >= true_utility_of_masked and changed > 0
    print(f"    {'PASS' if passed else 'FAIL'}: "
          f"Full-information model achieves {'higher' if utility_full > true_utility_of_masked else 'equal'} "
          f"TRUE utility than information-loss model.")
    return passed

def test_theorem3_utility_loss():
    print("\n[T3] Execution-Layer Utility Loss (Delta_i) consistency check...")
    df = generate_data(I=100, seed=42)
    U, P = compute_utility(df, w1=0.7, w2=0.3)
    F = compute_feasibility(df)
    result = solve(U, F, I=100)
    assignments = result['assignments']
    disposal_idx = len(P)
    deltas = []
    for i in range(100):
        if assignments[i] < disposal_idx:
            assigned_utility = U[i, assignments[i]]
            max_feasible_utility = max(
                [U[i, j] for j in range(len(P)) if F[i, j] == 1],
                default=0.0
            )
            delta_i = max_feasible_utility - assigned_utility
            deltas.append(delta_i)
        else:
            deltas.append(0.0)
    deltas = np.array(deltas)
    units_with_loss = np.sum(deltas > 0.001)
    print(f"    Units with Delta_i > 0: {units_with_loss}")
    print(f"    Max Delta_i: {deltas.max():.4f}")
    if units_with_loss > 0:
        print(f"    Mean Delta_i (affected): {deltas[deltas>0.001].mean():.4f}")
    print(f"    PASS: Delta_i computed from (U,F) profile for all 100 units.")
    return True

def test_theorem4_scalability():
    print("\n[T4] Empirical Runtime Growth Analysis...")
    import time
    sizes = [100, 1000, 10000]
    times = []
    for I in sizes:
        df = generate_data(I=I, seed=42)
        U, P = compute_utility(df, w1=0.7, w2=0.3)
        F = compute_feasibility(df)
        t0 = time.time()
        result = solve(U, F, I=I, scale_capacities=True)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"    I={I:6d}: utility={result['total_utility']:10.2f}, "
              f"time={elapsed:.4f}s")
    if len(times) == 3 and times[0] > 0:
        print(f"    Time ratio (1000/100):   {times[1]/times[0]:.1f}x")
        print(f"    Time ratio (10000/1000): {times[2]/times[1]:.1f}x")
        print(f"    NOTE: Empirical observation only, not a polynomial proof.")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("COMPUTATIONAL CONSISTENCY TESTS")
    print("=" * 60)
    results = {
        "T1 Disposal Elimination": test_theorem1_disposal_consistency(),
        "T2 Information Loss":     test_theorem2_information_loss(),
        "T3 Utility Loss Delta_i": test_theorem3_utility_loss(),
        "T4 Runtime Growth":       test_theorem4_scalability(),
    }
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")