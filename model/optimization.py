"""
Optimization layer for the DDCC/RPDD textile optimization model.

Implements the Constraint-Embedded Decision Execution (CEDE) component
using min-cost flow (OR-Tools SimpleMinCostFlow, NetworkX fallback).

Under uniform capacity consumption (kij=1), the assignment problem is
solvable in polynomial time via min-cost flow (Theorem 4).
"""

import time
import numpy as np

from model.config import (
    POSITIONS as P, DISPOSAL as D,
    BASE_CAPACITIES, COST_SCALE,
)

POSITION_NAMES = P + [D]


def solve(
    U: np.ndarray,
    F: np.ndarray,
    I: int,
    epsilon: float = 0.01,
    scale_capacities: bool = True,
    custom_caps: dict = None,
) -> dict:
    if custom_caps is not None:
        # Explicit per-position capacities override scale_capacities entirely.
        # Used by experiments/sensitivity.py (S3) to test specific capacity
        # levels (e.g. Repair capacity in {5, 8, 15, 20, 40}) at a fixed I.
        capacities = {pos: max(1, int(custom_caps[pos])) for pos in P}
    else:
        scale = (I / 100) if scale_capacities else 1.0
        capacities = {pos: max(1, int(BASE_CAPACITIES[pos] * scale)) for pos in P}

    try:
        result = _solve_ortools(U, F, I, epsilon, capacities)
    except Exception:
        result = _solve_networkx(U, F, I, epsilon, capacities)

    return result


def _solve_ortools(U, F, I, epsilon, capacities):
    from ortools.graph.python import min_cost_flow

    mcf = min_cost_flow.SimpleMinCostFlow()

    n_positions = len(P)
    source = 0
    unit_nodes = list(range(1, I + 1))
    pos_nodes = list(range(I + 1, I + n_positions + 1))
    disposal_node = I + n_positions + 1
    sink = I + n_positions + 2

    for i in range(I):
        mcf.add_arc_with_capacity_and_unit_cost(source, unit_nodes[i], 1, 0)

    for i in range(I):
        for j in range(n_positions):
            if F[i, j] == 1:
                cost = int(-U[i, j] * COST_SCALE)
                mcf.add_arc_with_capacity_and_unit_cost(
                    unit_nodes[i], pos_nodes[j], 1, cost
                )
        disposal_cost = int(epsilon * COST_SCALE)
        mcf.add_arc_with_capacity_and_unit_cost(
            unit_nodes[i], disposal_node, 1, disposal_cost
        )

    for j, pos in enumerate(P):
        mcf.add_arc_with_capacity_and_unit_cost(
            pos_nodes[j], sink, capacities[pos], 0
        )

    mcf.add_arc_with_capacity_and_unit_cost(disposal_node, sink, I, 0)

    mcf.set_node_supply(source, I)
    mcf.set_node_supply(sink, -I)

    t0 = time.perf_counter()
    status = mcf.solve()
    elapsed = time.perf_counter() - t0

    if status != mcf.OPTIMAL:
        raise RuntimeError(f"OR-Tools status: {status}")

    assignments = [None] * I
    total_utility = 0.0

    for arc in range(mcf.num_arcs()):
        if mcf.flow(arc) == 1:
            tail = mcf.tail(arc)
            head = mcf.head(arc)
            if tail in unit_nodes:
                i = tail - 1
                if head == disposal_node:
                    assignments[i] = len(P)
                elif head in pos_nodes:
                    j = head - (I + 1)
                    assignments[i] = j
                    total_utility += U[i, j]

    binding = [
        P[j] for j in range(len(P))
        if sum(1 for a in assignments if a == j) >= capacities[P[j]]
    ]

    return {
        "assignments": assignments,
        "position_names": POSITION_NAMES,
        "total_utility": total_utility,
        "binding_constraints": binding,
        "solver": "ortools",
        "solve_time": elapsed,
        "status": "optimal",
    }


def _solve_networkx(U, F, I, epsilon, capacities):
    import networkx as nx

    G = nx.DiGraph()
    n_positions = len(P)

    source = "source"
    sink = "sink"
    disposal = "disposal"

    G.add_node(source, demand=-I)
    G.add_node(sink, demand=I)

    for i in range(I):
        unit = f"u{i}"
        G.add_node(unit, demand=0)
        G.add_edge(source, unit, capacity=1, weight=0)

        for j in range(n_positions):
            if F[i, j] == 1:
                pos = f"p{j}"
                cost = int(-U[i, j] * COST_SCALE)
                G.add_edge(unit, pos, capacity=1, weight=cost)

        G.add_edge(unit, disposal, capacity=1,
                   weight=int(epsilon * COST_SCALE))

    for j, pos_name in enumerate(P):
        pos = f"p{j}"
        G.add_node(pos, demand=0)
        G.add_edge(pos, sink, capacity=capacities[pos_name], weight=0)

    G.add_node(disposal, demand=0)
    G.add_edge(disposal, sink, capacity=I, weight=0)

    t0 = time.perf_counter()
    flow_dict = nx.min_cost_flow(G)
    elapsed = time.perf_counter() - t0

    assignments = [None] * I
    total_utility = 0.0

    for i in range(I):
        unit = f"u{i}"
        for j in range(n_positions):
            pos = f"p{j}"
            if pos in flow_dict.get(unit, {}) and flow_dict[unit][pos] == 1:
                assignments[i] = j
                total_utility += U[i, j]
                break
        if assignments[i] is None:
            assignments[i] = len(P)

    binding = [
        P[j] for j in range(len(P))
        if sum(1 for a in assignments if a == j) >= capacities[P[j]]
    ]

    return {
        "assignments": assignments,
        "position_names": POSITION_NAMES,
        "total_utility": total_utility,
        "binding_constraints": binding,
        "solver": "networkx",
        "solve_time": elapsed,
        "status": "optimal",
    }
