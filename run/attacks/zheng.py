"""Adaptive version of Zheng et al.'s topological SR formula.

Default: recompute d_i**2 / mean_neighbor_degree_i on the remaining graph
after every removal. This is our adaptive variant, not the author's default
oneshot protocol. update='static' retains the previous audited implementation.
Both versions give degree <= 1 score zero, break ties by original vertex ID,
omit reinsertion, and return all N removals for the full-curve R.

Source and static protocol: ../../revision_r1/attack_protocol.md
"""
from __future__ import annotations

import heapq

import numpy as np

from .common import result_from_order, validate_graph


def zheng_scores(graph):
    """Compute scores on the input topology, in original igraph index order."""
    validate_graph(graph)
    degree = graph.degree()
    scores = np.zeros(graph.vcount(), dtype=float)
    for v, neighbors in enumerate(graph.get_adjlist()):
        if degree[v] > 1:
            mean_neighbor_degree = sum(degree[u] for u in neighbors) / degree[v]
            scores[v] = degree[v] ** 2 / mean_neighbor_degree
    return scores


def _adaptive_order(graph):
    """Exact residual scores, updating only nodes within two original hops.

Removing v changes the degrees of its live neighbors. Their neighbors'
denominators can also change; no other score depends on v. Heap versions
discard stale entries without renumbering or modifying the input graph.
    """
    validate_graph(graph)
    adjacency = graph.get_adjlist()
    degree = graph.degree()
    alive = [True] * graph.vcount()
    versions = [0] * graph.vcount()

    def score(v):
        if degree[v] <= 1:
            return 0.0
        return degree[v] ** 3 / sum(degree[u] for u in adjacency[v] if alive[u])

    heap = [(-score(v), v, 0) for v in range(graph.vcount())]
    heapq.heapify(heap)
    order = []
    while heap:
        _, v, version = heapq.heappop(heap)
        if not alive[v] or version != versions[v]:
            continue
        order.append(v)
        alive[v] = False
        neighbors = [u for u in adjacency[v] if alive[u]]
        for u in neighbors:
            degree[u] -= 1
        affected = set(neighbors)
        for u in neighbors:
            affected.update(w for w in adjacency[u] if alive[w])
        for u in affected:
            versions[u] += 1
            heapq.heappush(heap, (-score(u), u, versions[u]))
    return order


def zheng_deletion_order(graph, update="adaptive"):
    if update == "adaptive":
        return _adaptive_order(graph)
    if update == "static":
        scores = zheng_scores(graph)
        return sorted(range(graph.vcount()), key=lambda v: (-scores[v], v))
    raise ValueError("Zheng update must be 'adaptive' or 'static'.")


def zheng_attack(graph, update="adaptive"):
    return result_from_order(
        graph, zheng_deletion_order(graph, update=update),
        method="Zheng", update=update, tie_break="min_original_index",
        degree_le_one_score=0.0, reinsertion=False, n_attack_rollouts=1,
        tail="continue_{}_order_to_N".format(update),
        score_update="residual_two_hop" if update == "adaptive" else "initial_only",
        variant="project_adaptive_SR" if update == "adaptive" else "static_SR",
    )


def zheng_robustness_curve(graph, **kwargs):
    return zheng_attack(graph, **kwargs).curve


def zheng_robustness_total(graph, **kwargs):
    return zheng_attack(graph, **kwargs).robustness
