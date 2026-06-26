"""Min-cost flow solver for DDCC/RPDD unit-position assignment."""

import time
from typing import Any

import numpy as np
import pandas as pd

from model.utility import D, P

POSITION_NAMES = P + [D]

BASE_CAPACITIES = {
    "Resale": 100,
    "Repair": 8,
    "Refurbishing": 6,
    "Repackaging": 100,
    "Recycling": 4,
    "Donation": 5,
    "DiscountSale": 100,
}

COST_SCALE = 10_000


def _position_capacities(I: int, scale_capacities: bool) -> list[int]:
    scale = I / 100.0 if scale_capacities else 1.0
    return [int(round(BASE_CAPACITIES[name] * scale)) for name in P]


def _edge_cost(utility: float, epsilon: float) -> int:
    """Min-cost flow edge cost for a position in P (integer-scaled)."""
    return int(round(-(utility - epsilon) * COST_SCALE))


def _build_network(
    U: np.ndarray,
    F: np.ndarray,
    I: int,
    epsilon: float,
    scale_capacities: bool,
    custom_caps=None,
) -> dict[str, Any]:
    if custom_caps is not None:
        capacities = [custom_caps.get(name, BASE_CAPACITIES[name]) for name in P]
    else:
        capacities = _position_capacities(I, scale_capacities)

    source = 0
    unit_base = 1
    pos_base = unit_base + I
    sink = pos_base + len(POSITION_NAMES)

    edges: list[tuple[int, int, int, int]] = []

    for i in range(I):
        unit_node = unit_base + i
        edges.append((source, unit_node, 1, 0))

        for j in range(len(P)):
            if F[i, j] == 1:
                cost = _edge_cost(U[i, j], epsilon)
                edges.append((unit_node, pos_base + j, 1, cost))

        edges.append((unit_node, pos_base + len(P), 1, 0))

    for j, capacity in enumerate(capacities):
        edges.append((pos_base + j, sink, capacity, 0))

    edges.append((pos_base + len(P), sink, I, 0))

    return {
        "source": source,
        "sink": sink,
        "unit_base": unit_base,
        "pos_base": pos_base,
        "capacities": capacities,
        "edges": edges,
        "I": I,
    }


def _extract_assignments(
    network: dict[str, Any], unit_to_position_flow: dict[tuple[int, int], int]
) -> np.ndarray:
    I = network["I"]
    unit_base = network["unit_base"]
    pos_base = network["pos_base"]
    assignments = np.full(I, len(P), dtype=int)

    for i in range(I):
        unit_node = unit_base + i
        for j in range(len(POSITION_NAMES)):
            pos_node = pos_base + j
            if unit_to_position_flow.get((unit_node, pos_node), 0) > 0:
                assignments[i] = j
                break

    return assignments


def _binding_constraints(
    network: dict[str, Any], position_to_sink_flow: dict[tuple[int, int], int]
) -> list[str]:
    pos_base = network["pos_base"]
    sink = network["sink"]
    binding = []

    for j, capacity in enumerate(network["capacities"]):
        flow = position_to_sink_flow.get((pos_base + j, sink), 0)
        if flow == capacity:
            binding.append(P[j])

    return binding


def _solve_ortools(network: dict[str, Any]) -> tuple[dict[tuple[int, int], int], float]:
    from ortools.graph.python import min_cost_flow

    smcf = min_cost_flow.SimpleMinCostFlow()
    arc_indices: list[tuple[int, int]] = []

    for tail, head, capacity, cost in network["edges"]:
        arc_indices.append((tail, head))
        smcf.add_arcs_with_capacity_and_unit_cost([tail], [head], [capacity], [cost])

    smcf.set_node_supply(network["source"], network["I"])
    smcf.set_node_supply(network["sink"], -network["I"])

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"OR-Tools min-cost flow failed with status {status}")

    unit_flow: dict[tuple[int, int], int] = {}
    position_flow: dict[tuple[int, int], int] = {}
    unit_base = network["unit_base"]
    pos_base = network["pos_base"]
    sink = network["sink"]

    for arc_id, (tail, head) in enumerate(arc_indices):
        flow = smcf.flow(arc_id)
        if flow == 0:
            continue
        if unit_base <= tail < pos_base:
            unit_flow[(tail, head)] = flow
        elif pos_base <= tail < sink and head == sink:
            position_flow[(tail, head)] = flow

    total_utility = -smcf.optimal_cost() / COST_SCALE
    return {**unit_flow, **position_flow}, total_utility


def _solve_networkx(network: dict[str, Any]) -> tuple[dict[tuple[int, int], int], float]:
    import networkx as nx

    graph = nx.DiGraph()
    for tail, head, capacity, cost in network["edges"]:
        graph.add_edge(tail, head, capacity=capacity, weight=cost)

    for node in graph.nodes:
        graph.nodes[node]["demand"] = 0
    graph.nodes[network["source"]]["demand"] = -network["I"]
    graph.nodes[network["sink"]]["demand"] = network["I"]

    total_cost, flow_dict = nx.network_simplex(graph)

    unit_flow: dict[tuple[int, int], int] = {}
    position_flow: dict[tuple[int, int], int] = {}
    unit_base = network["unit_base"]
    pos_base = network["pos_base"]
    sink = network["sink"]

    for tail, neighbors in flow_dict.items():
        for head, flow in neighbors.items():
            if flow == 0:
                continue
            if unit_base <= tail < pos_base:
                unit_flow[(tail, head)] = flow
            elif pos_base <= tail < sink and head == sink:
                position_flow[(tail, head)] = flow

    total_utility = -total_cost / COST_SCALE
    return {**unit_flow, **position_flow}, total_utility


def solve(
    U: np.ndarray,
    F: np.ndarray,
    I: int,
    epsilon: float = 0.01,
    scale_capacities: bool = True,
    custom_caps=None
) -> dict[str, Any]:
    """
    Solve the DDCC/RPDD assignment problem as a min-cost flow.

    Returns assignment indices (0-6 for P, 7 for Disposal), objective value,
    binding capacity constraints, and solve time.
    """
    U = np.asarray(U, dtype=float)
    F = np.asarray(F, dtype=int)

    if U.shape != (I, len(P)):
        raise ValueError(f"U must have shape ({I}, {len(P)}), got {U.shape}")
    if F.shape != (I, len(P)):
        raise ValueError(f"F must have shape ({I}, {len(P)}), got {F.shape}")

    network = _build_network(U, F, I, epsilon, scale_capacities, custom_caps)
    start = time.perf_counter()

    try:
        flows, total_utility = _solve_ortools(network)
    except ImportError:
        flows, total_utility = _solve_networkx(network)
    except RuntimeError:
        flows, total_utility = _solve_networkx(network)

    solve_time = time.perf_counter() - start

    unit_base = network["unit_base"]
    pos_base = network["pos_base"]
    unit_flow = {
        edge: flow
        for edge, flow in flows.items()
        if unit_base <= edge[0] < pos_base
    }
    position_flow = {
        edge: flow
        for edge, flow in flows.items()
        if pos_base <= edge[0] < network["sink"]
    }

    assignments = _extract_assignments(network, unit_flow)

    return {
        "assignments": assignments,
        "position_names": list(POSITION_NAMES),
        "total_utility": float(total_utility),
        "binding_constraints": _binding_constraints(network, position_flow),
        "solve_time": solve_time,
        "U": U,
        "F": F,
    }


def results_summary(result: dict[str, Any], df: pd.DataFrame) -> None:
    """Print assignment counts, utility, binding constraints, and disposal reasons."""
    assignments = result["assignments"]
    position_names = result["position_names"]
    U = result["U"]
    F = result["F"]
    I = len(assignments)

    print(f"Optimization results ({I} units)")
    print(f"  Solve time: {result['solve_time']:.4f}s")
    print(f"  Total utility: {result['total_utility']:.4f}")
    print("\nAssignment counts:")
    for j, name in enumerate(position_names):
        count = int(np.sum(assignments == j))
        print(f"  {name}: {count}")

    if result["binding_constraints"]:
        print("\nBinding capacity constraints:")
        for name in result["binding_constraints"]:
            print(f"  {name}")
    else:
        print("\nBinding capacity constraints: none")

    disposal_idx = len(P)
    disposal_units = np.where(assignments == disposal_idx)[0]
    if len(disposal_units) == 0:
        print("\nDisposal: no units assigned")
        return

    print(f"\nDisposal: {len(disposal_units)} unit(s)")
    for i in disposal_units:
        unit_id = df.iloc[i]["unit_id"] if "unit_id" in df.columns else i + 1
        if F[i].sum() == 0:
            reason = "no feasible position in P (Fij=0 for all j)"
        elif np.max(U[i][F[i] == 1]) <= 0:
            reason = "all feasible utilities <= 0 (uij<=0)"
        else:
            reason = "capacity constraints or network optimum"
        print(f"  unit {unit_id}: {reason}")
