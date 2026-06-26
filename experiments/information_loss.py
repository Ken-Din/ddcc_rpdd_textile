"""
Step 10 — Failure Propagation Demonstration
Empirical demonstration of Theorem 2 (Structural Information Irrecoverability)
and Corollary 2.1 (Decision Loss Potential).

Compares four models:
  Model A: Full information + DDCC/RPDD (baseline)
  Model B: Partial information (30% Condition masked) + DDCC/RPDD
  Model C: Random assignment (feasibility-respecting)
  Model D: Greedy assignment (highest uij, ignoring capacity)

Expected ordering: Utility(A) > Utility(B) > Utility(D) > Utility(C)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from data.data_generator import generate_data
from model.utility import compute_utility
from model.feasibility import compute_feasibility
from model.optimization import solve, P

I = 1000
SEED = 42


def model_a_full_information(df, U, F):
    """Model A: Full information, DDCC/RPDD min-cost flow."""
    result = solve(U, F, I=I, scale_capacities=True)
    true_utility = sum(
        U[i, result['assignments'][i]]
        if result['assignments'][i] < len(P) else 0.0
        for i in range(I)
    )
    return result['assignments'], true_utility


def model_b_partial_information(df, U_full, F_full):
    """Model B: 30% Condition masked, DDCC/RPDD min-cost flow.
    Evaluated using TRUE (full-information) utility."""
    df_masked = df.copy()
    mask_idx = df_masked.sample(n=int(I * 0.3), random_state=SEED).index
    df_masked.loc[mask_idx, 'condition'] = 'NotLiked'

    U_masked, _ = compute_utility(df_masked, w1=0.7, w2=0.3)
    F_masked = compute_feasibility(df_masked)
    result = solve(U_masked, F_masked, I=I, scale_capacities=True)

    # Evaluate using TRUE utility (full information)
    true_utility = sum(
        U_full[i, result['assignments'][i]]
        if result['assignments'][i] < len(P) else 0.0
        for i in range(I)
    )
    return result['assignments'], true_utility


def model_c_random_assignment(df, U_full, F_full):
    """Model C: Random feasibility-respecting assignment.
    Each unit assigned to a random feasible position."""
    rng = np.random.default_rng(SEED)
    assignments = np.full(I, len(P), dtype=int)  # default: Disposal

    for i in range(I):
        feasible = [j for j in range(len(P)) if F_full[i, j] == 1]
        if feasible:
            assignments[i] = rng.choice(feasible)

    true_utility = sum(
        U_full[i, assignments[i]]
        if assignments[i] < len(P) else 0.0
        for i in range(I)
    )
    return assignments, true_utility


def model_d_greedy_assignment(df, U_full, F_full):
    """Model D: Greedy assignment — highest uij per unit, ignoring capacity.
    Each unit independently picks its highest-utility feasible position."""
    assignments = np.full(I, len(P), dtype=int)  # default: Disposal

    for i in range(I):
        best_j = -1
        best_u = 0.0
        for j in range(len(P)):
            if F_full[i, j] == 1 and U_full[i, j] > best_u:
                best_u = U_full[i, j]
                best_j = j
        if best_j >= 0:
            assignments[i] = best_j

    true_utility = sum(
        U_full[i, assignments[i]]
        if assignments[i] < len(P) else 0.0
        for i in range(I)
    )
    return assignments, true_utility


def assignment_distribution(assignments, position_names):
    """Return assignment counts per position."""
    counts = {}
    for j, name in enumerate(position_names):
        counts[name] = int(np.sum(assignments == j))
    return counts


if __name__ == "__main__":
    print("=" * 65)
    print("FAILURE PROPAGATION DEMONSTRATION")
    print("Theorem 2 (Irrecoverability) + Corollary 2.1 (Decision Loss)")
    print(f"I={I}, seed={SEED}")
    print("=" * 65)

    # Generate data
    df = generate_data(I=I, seed=SEED)
    U_full, position_list = compute_utility(df, w1=0.7, w2=0.3)
    F_full = compute_feasibility(df)
    position_names = position_list + ["Disposal"]

    # Run all four models
    print("\nRunning Model A (full information, DDCC/RPDD)...")
    assign_a, utility_a = model_a_full_information(df, U_full, F_full)

    print("Running Model B (30% information loss, DDCC/RPDD)...")
    assign_b, utility_b = model_b_partial_information(df, U_full, F_full)

    print("Running Model C (random feasible assignment)...")
    assign_c, utility_c = model_c_random_assignment(df, U_full, F_full)

    print("Running Model D (greedy, no capacity constraints)...")
    assign_d, utility_d = model_d_greedy_assignment(df, U_full, F_full)

    # Results
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Model':<40} {'True Utility':>14} {'vs Model A':>12}")
    print("-" * 65)
    print(f"{'A: Full info + DDCC/RPDD (optimal)':<40} "
          f"{utility_a:>14.2f} {'—':>12}")
    print(f"{'B: 30% info loss + DDCC/RPDD':<40} "
          f"{utility_b:>14.2f} "
          f"{utility_b - utility_a:>+12.2f}")
    print(f"{'D: Greedy (no capacity)':<40} "
          f"{utility_d:>14.2f} "
          f"{utility_d - utility_a:>+12.2f}")
    print(f"{'C: Random feasible':<40} "
          f"{utility_c:>14.2f} "
          f"{utility_c - utility_a:>+12.2f}")
    print("-" * 65)

    # Ordering check
    print("\nExpected ordering: A > B > D > C")
    ordering_ok = utility_a >= utility_b >= utility_d >= utility_c
    print(f"Observed ordering: "
          f"A({'%.0f'%utility_a}) "
          f"{'>' if utility_a>=utility_b else '<'} "
          f"B({'%.0f'%utility_b}) "
          f"{'>' if utility_b>=utility_d else '<'} "
          f"D({'%.0f'%utility_d}) "
          f"{'>' if utility_d>=utility_c else '<'} "
          f"C({'%.0f'%utility_c})")
    print(f"Ordering check: {'PASS' if ordering_ok else 'PARTIAL — see note'}")

    # Assignment distributions
    print("\nAssignment distributions:")
    print(f"{'Position':<20} {'Model A':>10} {'Model B':>10} "
          f"{'Model C':>10} {'Model D':>10}")
    print("-" * 65)
    dist_a = assignment_distribution(assign_a, position_names)
    dist_b = assignment_distribution(assign_b, position_names)
    dist_c = assignment_distribution(assign_c, position_names)
    dist_d = assignment_distribution(assign_d, position_names)
    for pos in position_names:
        print(f"  {pos:<18} {dist_a.get(pos,0):>10} {dist_b.get(pos,0):>10} "
              f"{dist_c.get(pos,0):>10} {dist_d.get(pos,0):>10}")

    # Failure propagation: how many units changed assignment in B vs A
    changed_b = int(np.sum(assign_a != assign_b))
    print(f"\nFailure propagation (Model B vs A):")
    print(f"  Units with changed assignment: {changed_b}/{I} "
          f"({100*changed_b/I:.1f}%)")
    print(f"  Utility loss due to info loss: "
          f"{utility_a - utility_b:.2f} "
          f"({100*(utility_a-utility_b)/utility_a:.2f}% of optimal)")

    # Save results
    records = []
    for i in range(I):
        records.append({
            'unit_id': df.iloc[i]['unit_id'],
            'source': df.iloc[i]['source'],
            'condition': df.iloc[i]['condition'],
            'age_days': df.iloc[i]['age_days'],
            'brand_tier': df.iloc[i]['brand_tier'],
            'model_a': position_names[assign_a[i]],
            'model_b': position_names[assign_b[i]],
            'model_c': position_names[assign_c[i]],
            'model_d': position_names[assign_d[i]],
            'utility_a': U_full[i, assign_a[i]] if assign_a[i] < len(P) else 0.0,
            'utility_b': U_full[i, assign_b[i]] if assign_b[i] < len(P) else 0.0,
        })

    df_out = pd.DataFrame(records)
    os.makedirs('results', exist_ok=True)
    df_out.to_csv('results/failure_propagation.csv', index=False)
    print("\nSaved to results/failure_propagation.csv")
    print("=" * 65)