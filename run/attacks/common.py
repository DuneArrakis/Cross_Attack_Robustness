"""Shared full-order evaluation for the revision's unweighted topology attacks.

An order contains the ORIGINAL igraph vertex indices, exactly once each.
curve[q] = LCC after q explicit removals / N, q=0..N.
R = sum(curve[1:]) / N, as in the submitted paper (not mean(curve)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, List

import igraph as ig
import numpy as np


def validate_graph(graph: ig.Graph) -> None:
    if graph.is_directed() or not graph.is_simple():
        raise ValueError("Attacks require a simple undirected graph.")


def lcc_curve_from_order(graph: ig.Graph, order: List[int]) -> List[float]:
    """Evaluate every removal using reverse union-find; retain isolated nodes."""
    validate_graph(graph)
    n = graph.vcount()
    if (len(order) != n or any(not isinstance(v, Integral) for v in order)
            or set(order) != set(range(n))):
        raise ValueError("Attack order must be a permutation of original vertex indices 0..N-1.")
    if n == 0:
        return [0.0]

    parent = list(range(n))
    size = [0] * n
    active = [False] * n
    adjacency = graph.get_adjlist()
    curve = [0.0] * (n + 1)
    largest = 0

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for q in range(n - 1, -1, -1):
        v = order[q]
        active[v] = True
        size[v] = 1
        largest = max(largest, 1)
        for u in adjacency[v]:
            if active[u]:
                a, b = find(v), find(u)
                if a == b:
                    continue
                if size[a] < size[b]:
                    a, b = b, a
                parent[b] = a
                size[a] += size[b]
                largest = max(largest, size[a])
        curve[q] = largest / n
    return curve


def robustness_from_curve(curve, n=None) -> float:
    """Discrete paper R from a FULL q=0..N curve, never resampled data."""
    values = np.asarray(curve, dtype=float)
    if n is None:
        n = len(values) - 1
    if (n < 0 or values.ndim != 1 or len(values) != n + 1
            or not np.all(np.isfinite(values))):
        raise ValueError("Expected a finite full curve with N+1 values.")
    if (np.any(values < 0) or np.any(values > 1)
            or values[-1] != 0 or np.any(np.diff(values) > 1e-12)):
        raise ValueError("Invalid LCC/N curve (range, monotonicity or endpoint).")
    return float(values[1:].sum() / n) if n else 0.0


@dataclass
class AttackResult:
    order: List[int]
    curve: List[float]
    robustness: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def result_from_order(graph, order, **metadata):
    order = list(order)
    curve = lcc_curve_from_order(graph, order)
    return AttackResult(order, curve, robustness_from_curve(curve, graph.vcount()), metadata)
