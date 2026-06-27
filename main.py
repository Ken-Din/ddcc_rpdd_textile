"""
Main pipeline for the DDCC/RPDD textile return optimization.

Runs the full RUS-CEDE optimization layer:
    1. Generate synthetic return data
    2. Compute utility matrix (RUS)
    3. Compute feasibility matrix
    4. Solve min-cost flow assignment (CEDE)
    5. Report results
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data.data_generator import generate_data
from model.utility import compute_utility
from model.feasibility import compute_feasibility
from model.optimization import solve
from model.config import POSITIONS, DISPOSAL, DEFAULT_W1, DEFAULT_W2, DEFAULT_EPSILON
POSITION_NAMES = POSITIONS + [DISPOSAL]


def run_pipeline(I: int = 100, seed: int = 42, w1: float = DEFAULT_W1,
                 w2: float = DEFAULT_W2, epsilon: float = DEFAULT_EPSILON,
                 scale_capacities: bool = True) -> dict:
    """
    Run the full RUS-CEDE optimization pipeline.

    Parameters
    ----------
    I : int
        Number of return units.
    seed : int
        Random seed for reproducibility.
    w1 : float
        Weight for residual value in utility function.
    w2 : float
        Weight for recovery cost in utility function.
    epsilon : float
        Tie-breaking penalty for disposal edges.
    scale_capacities : bool
        If True, scale capacities proportionally with I.

    Returns
    -------
    dict with pipeline results.
    """
    print(f"\n{'='*55}")
    print(f"  DDCC/RPDD Textile Return Optimization")
    print(f"  I={I}, seed={seed}, w1={w1}, w2={w2}")
    print(f"{'='*55}")

    # Step 1: Generate data
    df = generate_data(I=I, seed=seed)
    print(f"\n[1] Generated {I} return units")
    print(f"    Conditions: {df['condition'].value_counts().to_dict()}")

    # Step 2: Compute utility
    U, P = compute_utility(df, w1=w1, w2=w2)
    print(f"\n[2] Utility matrix computed: shape {U.shape}")
    print(f"    Mean utility: {U.mean():.4f}, Std: {U.std():.4f}")

    # Step 3: Compute feasibility
    F = compute_feasibility(df)
    print(f"\n[3] Feasibility matrix computed: shape {F.shape}")
    print(f"    Feasible pairs: {F.sum()} / {F.size}")

    # Step 4: Solve
    result = solve(U, F, I=I, epsilon=epsilon,
                   scale_capacities=scale_capacities)
    assignments = result['assignments']
    total_utility = result['total_utility']
    solver = result.get('solver', 'unknown')
    print(f"\n[4] Optimization solved via {solver}")
    print(f"    Total utility: {total_utility:.2f}")

    # Step 5: Report
    position_names = result['position_names']
    counts = {name: 0 for name in position_names}
    for a in assignments:
        counts[position_names[a]] += 1

    print(f"\n[5] Assignment summary:")
    for pos, cnt in counts.items():
        print(f"    {pos:<15}: {cnt:4d} units")

    from model.config import BASE_CAPACITIES
    scale = (I / 100) if scale_capacities else 1.0
    binding = [
        pos for pos in BASE_CAPACITIES
        if counts.get(pos, 0) >= max(1, int(BASE_CAPACITIES[pos] * scale))
        and BASE_CAPACITIES[pos] < I
    ]
    print(f"\n    Binding constraints: {binding}")
    print(f"    Disposal units:      {counts.get('Disposal', 0)}")

    return {
        'df': df,
        'U': U,
        'F': F,
        'assignments': assignments,
        'total_utility': total_utility,
        'counts': counts,
        'solver': solver,
    }


if __name__ == "__main__":
    run_pipeline(I=100, seed=42)