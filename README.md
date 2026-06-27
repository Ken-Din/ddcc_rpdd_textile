# DDCC/RPDD Textile Return Optimization

Computational implementation of the **RUS–CEDE optimization layer**
of the DDCC/RPDD framework for textile product return disposition.

## Scope
This repository implements the Recovery Utility Scoring (RUS) and
Constraint-Embedded Decision Execution (CEDE) components of the
DDCC/RPDD theoretical framework using min-cost flow optimization.

The Return Signal Structuring (RSS) component is represented here
by a synthetic data generator (`data/data_generator.py`) that
simulates structured return signals si = σ(zi) for textile products.
The full DDCC capability measurement framework is described in the
accompanying paper.

## Requirements
pip install -r requirements.txt

## Quick Start
python main.py

## Experiments
python experiments/scalability.py
python experiments/sensitivity.py
python experiments/information_loss.py

## Tests
python tests/test_consistency.py

## Consistency Tests and Theorems
The tests in `tests/test_consistency.py` are empirical consistency
checks — they verify that the implementation exhibits behavior
consistent with the theoretical results (Theorems 1-4), but do not
constitute formal proofs. The formal proofs are in the paper.

| Test | Theorem | What it checks |
|---|---|---|
| T1 | Theorem 1 (Disposal Elimination) | Disposal assignments are capacity-constrained or utility-dominated |
| T2 | Theorem 2 + Corollary 2.1 | Information loss reduces TRUE utility |
| T3 | Theorem 3 (Utility Loss Identifiability) | Delta_i computable from recorded profile |
| T4 | Theorem 4 (Complexity Boundary) | Sub-linear empirical runtime growth |

## Utility Function
The implementation uses the weighted linear instantiation of the
general form uij = f(Vij, Cij):

    uij = w1 * Vij - w2 * Cij  (default: w1=0.7, w2=0.3)

This is one reference instantiation. Other functional forms
(TOPSIS-type, fuzzy utility, stochastic expected value) are
compatible with the theoretical framework without altering Theorems 1-4.

## Expected Results (I=100, seed=42)
- Total utility: 5136.94
- Binding: Repair, Refurbishing, Recycling, Donation
- Disposal: 3 units (capacity-constrained)
- Consistency tests: 4/4 PASS