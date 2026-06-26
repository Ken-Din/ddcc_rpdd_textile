"""
Step 9 — Sensitivity Analysis
Fixed I=1000, four parameter groups:
1. w1/w2 utility weights
2. epsilon (tie-breaking penalty)
3. Capacity limits (Repair position)
4. Information loss ratio (RSS incompleteness simulation)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from data.data_generator import generate_data
from model.utility import compute_utility
from model.feasibility import compute_feasibility
from model.optimization import solve

I = 1000
SEED = 42
df_base = generate_data(I=I, seed=SEED)
F_base = compute_feasibility(df_base)

def run_weight_sensitivity():
    print("\n[S1] Weight sensitivity (w1/w2)...")
    combos = [(0.9,0.1),(0.7,0.3),(0.5,0.5),(0.3,0.7),(0.1,0.9)]
    records = []
    for w1, w2 in combos:
        U, P = compute_utility(df_base, w1=w1, w2=w2)
        result = solve(U, F_base, I=I, scale_capacities=True)
        disposal = sum(1 for a in result['assignments'] if a == len(P))
        records.append({
            'w1': w1, 'w2': w2,
            'total_utility': result['total_utility'],
            'disposal_count': disposal
        })
        print(f"  w1={w1}, w2={w2}: utility={result['total_utility']:.2f}, "
              f"disposal={disposal}")
    return pd.DataFrame(records)

def run_epsilon_sensitivity():
    print("\n[S2] Epsilon (tie-breaking penalty) sensitivity...")
    epsilons = [0.001, 0.01, 0.1, 1.0, 5.0]
    records = []
    U, P = compute_utility(df_base, w1=0.7, w2=0.3)
    for eps in epsilons:
        result = solve(U, F_base, I=I, epsilon=eps, scale_capacities=True)
        disposal = sum(1 for a in result['assignments'] if a == len(P))
        records.append({
            'epsilon': eps,
            'total_utility': result['total_utility'],
            'disposal_count': disposal
        })
        print(f"  epsilon={eps}: utility={result['total_utility']:.2f}, "
              f"disposal={disposal}")
    return pd.DataFrame(records)

def run_capacity_sensitivity():
    print("\n[S3] Capacity sensitivity (Repair position)...")
    repair_caps = [5, 8, 15, 20, 40]
    records = []
    U, P = compute_utility(df_base, w1=0.7, w2=0.3)

    # Base capacities scaled for I=1000
    BASE_CAPS = {
        "Resale": 1000, "Repair": 80, "Refurbishing": 60,
        "Repackaging": 1000, "Recycling": 40,
        "Donation": 50, "DiscountSale": 1000
    }

    for repair_cap in repair_caps:
        caps = BASE_CAPS.copy()
        caps["Repair"] = repair_cap
        result = solve(U, F_base, I=I, epsilon=0.01,
                      scale_capacities=False, custom_caps=caps)
        disposal = sum(1 for a in result['assignments'] if a == len(P))
        repair_used = sum(1 for a in result['assignments']
                         if a == P.index("Repair") if "Repair" in P)
        records.append({
            'repair_capacity': repair_cap,
            'total_utility': result['total_utility'],
            'disposal_count': disposal,
            'repair_used': repair_used
        })
        print(f"  Repair cap={repair_cap}: utility={result['total_utility']:.2f}, "
              f"repair_used={repair_used}, disposal={disposal}")
    return pd.DataFrame(records)

def run_information_loss_sensitivity():
    print("\n[S4] Information loss ratio sensitivity (RSS incompleteness)...")
    loss_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
    records = []
    U_full, P = compute_utility(df_base, w1=0.7, w2=0.3)
    result_full = solve(U_full, F_base, I=I, scale_capacities=True)
    utility_full = result_full['total_utility']

    for ratio in loss_ratios:
        df_masked = df_base.copy()
        n_mask = int(I * ratio)
        if n_mask > 0:
            # Simulate RSS incompleteness: replace true condition with default
            mask_idx = df_masked.sample(n=n_mask, random_state=42).index
            df_masked.loc[mask_idx, 'condition'] = 'NotLiked'
        U_masked, _ = compute_utility(df_masked, w1=0.7, w2=0.3)
        F_masked = compute_feasibility(df_masked)
        result_masked = solve(U_masked, F_masked, I=I, scale_capacities=True)

        # Evaluate masked solution using TRUE (full-information) utility
        true_utility = sum(
            U_full[i, result_masked['assignments'][i]]
            if result_masked['assignments'][i] < len(P) else 0.0
            for i in range(I)
        )
        utility_loss = utility_full - true_utility
        changed = np.sum(result_full['assignments'] != result_masked['assignments'])

        records.append({
            'loss_ratio': ratio,
            'n_masked': n_mask,
            'true_utility': true_utility,
            'utility_loss': utility_loss,
            'assignment_changes': changed
        })
        print(f"  loss={ratio:.0%}: true_utility={true_utility:.2f}, "
              f"utility_loss={utility_loss:.2f}, assignment_changes={changed}")
    return pd.DataFrame(records)

if __name__ == "__main__":
    print("=" * 60)
    print("SENSITIVITY ANALYSIS  (I=1000, seed=42)")
    print("=" * 60)

    df_w  = run_weight_sensitivity()
    df_e  = run_epsilon_sensitivity()
    df_c  = run_capacity_sensitivity()
    df_il = run_information_loss_sensitivity()

    os.makedirs('results', exist_ok=True)
    df_w.to_csv('results/sensitivity_weights.csv', index=False)
    df_e.to_csv('results/sensitivity_epsilon.csv', index=False)
    df_c.to_csv('results/sensitivity_capacity.csv', index=False)
    df_il.to_csv('results/sensitivity_info_loss.csv', index=False)

    print("\n" + "=" * 60)
    print("Results saved to results/sensitivity_*.csv")
    print("=" * 60)