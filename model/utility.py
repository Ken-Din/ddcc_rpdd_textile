"""
Utility computation for the DDCC/RPDD textile optimization model.

Implements the Recovery Utility Scoring (RUS) component:
    uij = w1 * Vij - w2 * Cij

This is a reference instantiation of the general form uij = f(Vij, Cij)
defined in the theoretical model (Step 9, Section 4.1).

Feasibility (Fij) is deliberately excluded from uij — capacity and
institutional constraints are enforced separately in the optimization
model (model/optimization.py, constraints C4-C6).

Disposal utility uiD = 0 for all units (not included in U).
"""

import numpy as np
import pandas as pd

from model.config import POSITIONS as P, DISPOSAL as D
from model.config import BRAND_MULTIPLIER, AGE_DECAY_RATE

CONDITIONS = [
    "NotLiked",
    "ColorMismatch",
    "Undamaged",
    "Torn",
    "SeamDefect",
    "Stained",
    "MissingPart",
]

BASE_VALUE = pd.DataFrame(
    {
        "Resale":       [90, 50, 95,  5, 10,  8, 15],
        "Repair":       [20, 15, 10, 40, 45, 30, 50],
        "Refurbishing": [40, 35, 30, 25, 30, 50, 35],
        "Repackaging":  [85, 45, 90, 10, 15, 10, 20],
        "Recycling":    [10,  8, 10, 15, 12, 10, 10],
        "Donation":     [30, 25, 35, 20, 22, 20, 25],
        "DiscountSale": [60, 40, 70, 15, 20, 18, 25],
    },
    index=CONDITIONS,
)

BASE_COST = pd.DataFrame(
    {
        "Resale":       [ 5, 10,  3, 20, 18, 15, 12],
        "Repair":       [60, 50, 70, 40, 35, 45, 30],
        "Refurbishing": [30, 25, 35, 30, 28, 20, 25],
        "Repackaging":  [ 5, 10,  3, 25, 22, 20, 15],
        "Recycling":    [20, 18, 20, 15, 15, 12, 12],
        "Donation":     [10, 10, 10, 10, 10, 10, 10],
        "DiscountSale": [10, 12,  8, 25, 22, 18, 15],
    },
    index=CONDITIONS,
)


def compute_utility(
    df: pd.DataFrame, w1: float = 0.7, w2: float = 0.3
) -> tuple[np.ndarray, list[str]]:
    """
    Compute per-unit utility matrix U (shape I x |P|) for each position in P.

    This is a reference instantiation of the general form uij = f(Vij, Cij):
        uij = w1 * Vij - w2 * Cij
    where:
        Vij = BaseValue(condition, position) * exp(-lambda * age_days) * BrandMultiplier(brand)
        Cij = BaseCost(condition, position)

    Feasibility (Fij) is deliberately excluded — capacity and institutional
    constraints are enforced separately in the optimization model (C4-C6).
    Disposal utility uiD = 0 for all units (not included in U).

    Parameters
    ----------
    df : pd.DataFrame
        Return unit data with columns: condition, age_days, brand_tier.
    w1 : float
        Weight for residual value Vij (default 0.7).
    w2 : float
        Weight for recovery cost Cij (default 0.3).

    Returns
    -------
    U : np.ndarray, shape (I, |P|)
        Utility matrix.
    position_names : list[str]
        Ordered list of position labels corresponding to columns of U.
    """
    base_values = BASE_VALUE.loc[df["condition"].values, P].to_numpy(dtype=float)
    base_costs  = BASE_COST.loc[df["condition"].values, P].to_numpy(dtype=float)
    age_decay   = np.exp(-AGE_DECAY_RATE * df["age_days"].to_numpy(dtype=float))
    brand_mult  = df["brand_tier"].map(BRAND_MULTIPLIER).to_numpy(dtype=float)

    value = base_values * age_decay[:, np.newaxis] * brand_mult[:, np.newaxis]
    U = w1 * value - w2 * base_costs

    return U, list(P)


def objective_with_epsilon(
    U: np.ndarray, y: np.ndarray, epsilon: float = 0.01
) -> float:
    """
    Compute sum over i,j of (uij - epsilon * 1[j in P]) * yij.
    U covers positions in P only. Disposal column in y is ignored
    because uiD = 0 and Disposal is not in P.
    """
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n_pos = U.shape[1]
    return float(np.sum((U - epsilon) * y[..., :n_pos]))