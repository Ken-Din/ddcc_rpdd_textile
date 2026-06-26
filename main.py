"""
DDCC/RPDD Textile Return Optimization — Main Pipeline
I=100, seed=42
"""
import time
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_generator import generate_data
from model.utility import compute_utility
from model.feasibility import compute_feasibility, feasibility_summary
from model.optimization import solve, results_summary

def main(I=100, seed=42, w1=0.7, w2=0.3, epsilon=0.01):
    print("=" * 60)
    print("DDCC/RPDD TEXTILE RETURN OPTIMIZATION")
    print(f"I={I}, seed={seed}, w1={w1}, w2={w2}, epsilon={epsilon}")
    print("=" * 60)

    # STEP 1: Generate data
    print("\n[1] Generating synthetic data...")
    df = generate_data(I=I, seed=seed)
    print(f"    Generated {len(df)} units")
    print(f"    Source distribution:\n{df['source'].value_counts()}")
    print(f"    Condition distribution:\n{df['condition'].value_counts()}")

    # STEP 2: Compute utility matrix
    print("\n[2] Computing utility matrix...")
    U, P = compute_utility(df, w1=w1, w2=w2, epsilon=epsilon)
    print(f"    U shape: {U.shape}")
    print(f"    Positions: {P}")
    print(f"    U stats: min={U.min():.2f}, max={U.max():.2f}, mean={U.mean():.2f}")

    # STEP 3: Compute feasibility matrix
    print("\n[3] Computing feasibility matrix...")
    F = compute_feasibility(df)
    print(f"    F shape: {F.shape}")
    feasibility_summary(F, df)

    # STEP 4: Solve optimization
    print("\n[4] Solving min-cost flow assignment...")
    result = solve(U, F, I=I, epsilon=epsilon, scale_capacities=True)
    print(f"    Solve time: {result['solve_time']:.4f}s")
    print(f"    Total utility: {result['total_utility']:.4f}")
    print(f"    Binding constraints: {result['binding_constraints']}")

    # STEP 5: Results summary
    print("\n[5] Assignment results:")
    results_summary(result, df)

    # STEP 6: Save results
    print("\n[6] Saving results...")
    assignments = result['assignments']
    position_names = result['position_names']
    df['assigned_position'] = [position_names[a] for a in assignments]
    df['utility_score'] = [U[i, assignments[i]] 
                           if assignments[i] < len(P) else 0.0 
                           for i in range(I)]

    os.makedirs('results', exist_ok=True)
    output_path = 'results/main_results.csv'
    df.to_csv(output_path, index=False)
    print(f"    Saved to {output_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return df, result

if __name__ == "__main__":
    main()