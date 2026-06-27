"""
Optimization layer for the DDCC/RPDD textile optimization model.

Implements the Constraint-Embedded Decision Execution (CEDE) component
using min-cost flow (OR-Tools SimpleMinCostFlow, NetworkX fallback).

Under uniform capacity consumption (kij=1), the assignment problem is
solvable in polynomial time via min-cost flow (Theorem 4).
"""

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
) -> dict:
    """
    Solve the min-cost flow assignment problem.

    Parameters
    ----------
    U : np.ndarray, shape (I, |P|)
        Utility matrix from compute_utility().
    F : np.ndarray, shape (I, |P|)
        Feasibility matrix from compute_feasibility().
    I : int
        Number of return units.
    epsilon : float
        Tie-breaking penalty added to disposal edge cost.
    scale_capacities : bool
        If True, scale BASE_CAPACITIES by I/100.

    Returns
    -------
    dict with keys:
        assignments      : list[int], length I — position index per unit
        position_names   : list[str]
        total_utility    : float
        solver           : str — 'ortools' or 'networkx'
    """
    scale = (I / 100) if scale_capacities else 1.0
    capacities = {pos: max(1, int(BASE_CAPACITIES[pos] * scale)) for pos in P}

    try:
        result = _solve_ortools(U, F, I, epsilon, capacities)
        result["solver"] = "ortools"
    except Exception:
        result = _solve_networkx(U, F, I, epsilon, capacities)
        result["solver"] = "networkx"

    return result


def _solve_ortools(U, F, I, epsilon, capacities):
    from ortools.graph.python import min_cost_flow

    mcf = min_cost_flow.SimpleMinCostFlow()

    n_positions = len(P)
    # Node layout:
    # 0         : source
    # 1..I      : unit nodes
    # I+1..I+|P|: position nodes
    # I+|P|+1  : disposal node
    # I+|P|+2  : sink

    source = 0
    unit_nodes = list(range(1, I + 1))
    pos_nodes = list(range(I + 1, I + n_positions + 1))
    disposal_node = I + n_positions + 1
    sink = I + n_positions + 2

    # Source → each unit (supply=1)
    for i in range(I):
        mcf.add_arc_with_capacity_and_unit_cost(source, unit_nodes[i], 1, 0)

    # Unit → position edges (feasible only)
    for i in range(I):
        for j in range(n_positions):
            if F[i, j] == 1:
                cost = int(-U[i, j] * COST_SCALE)
                mcf.add_arc_with_capacity_and_unit_cost(
                    unit_nodes[i], pos_nodes[j], 1, cost
                )
        # Unit → disposal (always feasible, cost = epsilon)
        disposal_cost = int(epsilon * COST_SCALE)
        mcf.add_arc_with_capacity_and_unit_cost(
            unit_nodes[i], disposal_node, 1, disposal_cost
        )

    # Position → sink (capacity-constrained)
    for j, pos in enumerate(P):
        mcf.add_arc_with_capacity_and_unit_cost(
            pos_nodes[j], sink, capacities[pos], 0
        )

    # Disposal → sink (unlimited)
    mcf.add_arc_with_capacity_and_unit_cost(disposal_node, sink, I, 0)

    # Supply/demand
    mcf.set_node_supply(source, I)
    mcf.set_node_supply(sink, -I)

    status = mcf.solve()
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
                    assignments[i] = len(P)  # disposal index
                elif head in pos_nodes:
                    j = head - (I + 1)
                    assignments[i] = j
                    total_utility += U[i, j]

    return {
        "assignments": assignments,
        "position_names": POSITION_NAMES,
        "total_utility": total_utility,
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

    flow_dict = nx.min_cost_flow(G)

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

    return {
        "assignments": assignments,
        "position_names": POSITION_NAMES,
        "total_utility": total_utility,
    }