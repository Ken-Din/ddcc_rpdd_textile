"""Feasibility matrix for textile return positions."""

import numpy as np
import pandas as pd

from model.utility import P

CONDITIONS = [
    "NotLiked",
    "ColorMismatch",
    "Undamaged",
    "Torn",
    "SeamDefect",
    "Stained",
    "MissingPart",
]

FEASIBILITY = pd.DataFrame(
    {
        "Resale": [1, 1, 1, 0, 0, 0, 0],
        "Repair": [0, 0, 0, 1, 1, 0, 1],
        "Refurbishing": [1, 1, 0, 1, 1, 1, 1],
        "Repackaging": [1, 1, 1, 0, 0, 0, 0],
        "Recycling": [1, 1, 1, 1, 1, 1, 1],
        "Donation": [1, 1, 1, 1, 1, 1, 1],
        "DiscountSale": [1, 1, 1, 0, 0, 0, 0],
    },
    index=CONDITIONS,
)


def compute_feasibility(df: pd.DataFrame) -> np.ndarray:
    """
    Compute feasibility matrix F for positions in P.

    F[i, j] = 1 if unit i can be assigned to position j, else 0.
    Disposal is always feasible but is not included in F.
    """
    return FEASIBILITY.loc[df["condition"].values, P].to_numpy(dtype=int)


def feasibility_summary(F: np.ndarray, df: pd.DataFrame) -> None:
    """Print how many units are feasible for each position."""
    n_units = F.shape[0]
    print(f"Feasibility summary ({n_units} units):")
    for j, position in enumerate(P):
        count = int(F[:, j].sum())
        pct = 100.0 * count / n_units if n_units else 0.0
        print(f"  {position}: {count}/{n_units} ({pct:.1f}%)")
