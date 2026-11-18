"""Static DomiRank with per-topology sigma search, for igraph inputs.

Solve (I + sigma*A) Gamma = sigma*A*1 with theta=1 and
0 < sigma < -1/lambda_min(A). Negative scores are valid, not clipped.
For each sigma, rank the input topology once, then delete in that fixed order.
Select the smallest full-curve paper R; equal R selects the larger sigma.
Every SA candidate topology gets its own sigma search and ranking.
The optional update='adaptive' project variant recomputes scores after every
deletion, holding sigma fixed within each rollout. It is not the main protocol.

Paper: https://doi.org/10.1038/s41467-023-44257-0
Reference: https://github.com/mengsig/DomiRank
We use the paper's minimum eigenvalue, not upstream's -lambda_max shortcut.
No multiprocessing inside the attack (safe in an existing SA worker).
"""
from __future__ import annotations

from numbers import Integral
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, spsolve

from .common import result_from_order, validate_graph

DEFAULT_SIGMA_POINTS = 100
DEFAULT_FRACTION_MIN = 1e-6
DEFAULT_FRACTION_MAX = 0.9999


def _adjacency(graph):
    validate_graph(graph)
    edges = graph.get_edgelist()
    if not edges:
        return sparse.csr_matrix((graph.vcount(), graph.vcount()), dtype=float)
    u, v = np.asarray(edges, dtype=int).T
    return sparse.csr_matrix(
        (np.ones(2 * len(edges)), (np.r_[u, v], np.r_[v, u])),
        shape=(graph.vcount(), graph.vcount()),
    )


def _smallest_eigenvalue(adjacency):
    n = adjacency.shape[0]
    if not adjacency.nnz:
        return 0.0
    if n <= 64:
        return float(np.linalg.eigvalsh(adjacency.toarray())[0])
    # A fixed, nonsymmetric start avoids dependence on process/global RNG state.
    initial = np.random.RandomState(0).normal(size=n)
    value = float(eigsh(adjacency, k=1, which="SA", v0=initial,
                        return_eigenvectors=False, tol=1e-10)[0])
    if value >= 0 or not np.isfinite(value):
        raise RuntimeError("Could not determine DomiRank's negative spectral bound.")
    return value


def _solve(adjacency, sigma):
    if not adjacency.nnz:
        return np.zeros(adjacency.shape[0])
    system = sparse.eye(adjacency.shape[0], format="csc") + sigma * adjacency
    rhs = sigma * np.asarray(adjacency.sum(axis=1)).ravel()
    scores = np.asarray(spsolve(system.tocsc(), rhs)).ravel()
    residual = np.linalg.norm(system.dot(scores) - rhs, ord=np.inf)
    if not np.all(np.isfinite(scores)) or residual > 1e-8 * (1 + np.linalg.norm(rhs, ord=np.inf)):
        raise RuntimeError("DomiRank linear solve did not pass the residual check.")
    return scores


def domirank_scores(graph, sigma):
    """Scores at an explicit sigma, for verification/analysis, without search."""
    adjacency = _adjacency(graph)
    eigenvalue = _smallest_eigenvalue(adjacency)
    if not np.isfinite(sigma) or sigma <= 0 or (eigenvalue < 0 and sigma >= -1 / eigenvalue):
        raise ValueError("sigma must lie strictly inside (0, -1/lambda_min).")
    return _solve(adjacency, sigma)


def _adaptive_order(adjacency, sigma):
    """Re-solve the residual system at every step; preserve original node IDs."""
    remaining = np.arange(adjacency.shape[0])
    order = []
    n_solves = 0
    while adjacency.nnz:
        scores = _solve(adjacency, sigma)
        n_solves += 1
        # remaining stays sorted, so argmax resolves exact ties by original ID.
        chosen = int(np.argmax(scores))
        order.append(int(remaining[chosen]))
        keep = np.ones(len(remaining), dtype=bool)
        keep[chosen] = False
        remaining = remaining[keep]
        adjacency = adjacency[keep, :][:, keep]
    # All remaining scores are exactly zero; retain every isolated vertex.
    order.extend(remaining.tolist())
    return order, n_solves


def domirank_attack(graph, sigma_points=DEFAULT_SIGMA_POINTS,
                    fraction_min=DEFAULT_FRACTION_MIN,
                    fraction_max=DEFAULT_FRACTION_MAX, update='static'):
    """Search an equally spaced grid of sigma / sigma_limit; return one result.

    The default is 100 full attack rollouts. A smaller grid is for explicitly
    labelled smoke/pilot runs, not silently substituted in an SA objective.
    All candidates, solve times and the selected sigma are retained in metadata.
    """
    if update not in ('adaptive', 'static'):
        raise ValueError("update must be 'adaptive' or 'static'.")
    if not isinstance(sigma_points, Integral) or sigma_points < 2:
        raise ValueError("sigma_points must be an integer >= 2.")
    if not (0 < fraction_min < fraction_max < 1):
        raise ValueError("Require 0 < fraction_min < fraction_max < 1.")
    start = perf_counter()
    adjacency = _adjacency(graph)
    spectral_start = perf_counter()
    eigenvalue = _smallest_eigenvalue(adjacency)
    spectral_seconds = perf_counter() - spectral_start
    metadata = dict(
        method="DomiRank", update=update, tie_break="min_original_index",
        sigma_points_requested=int(sigma_points), fraction_min=float(fraction_min),
        fraction_max=float(fraction_max), lambda_min=eigenvalue,
        spectral_seconds=spectral_seconds, search_objective="paper_R_min",
        sigma_tie_break="largest_sigma",
        sigma_policy="initial_spectral_grid_fixed_within_rollout",
        sigma_search_update=update,
    )
    if not adjacency.nnz:
        # All scores are zero and all orders are equivalent; no spectral search.
        result = result_from_order(graph, range(graph.vcount()))
        metadata.update(sigma=None, sigma_fraction=None, sigma_limit=None,
                        n_attack_rollouts=1, n_linear_solves=0,
                        solve_seconds=0.0, sigma_search=[], degenerate="edgeless")
    else:
        sigma_limit = -1.0 / eigenvalue
        trials = []
        result = None
        for fraction in np.linspace(fraction_min, fraction_max, sigma_points):
            sigma = float(fraction * sigma_limit)
            solve_start = perf_counter()
            if update == 'adaptive':
                order, n_solves = _adaptive_order(adjacency, sigma)
            else:
                scores = _solve(adjacency, sigma)
                order = sorted(range(graph.vcount()), key=lambda v: (-scores[v], v))
                n_solves = 1
            solve_seconds = perf_counter() - solve_start
            curve_start = perf_counter()
            candidate = result_from_order(graph, order)
            trials.append(dict(sigma_fraction=float(fraction), sigma=sigma,
                               R=candidate.robustness, solve_seconds=solve_seconds,
                               n_linear_solves=n_solves,
                               curve_seconds=perf_counter() - curve_start))
            # '<=' deliberately gives the last (largest sigma) tied minimum.
            if result is None or candidate.robustness <= result.robustness:
                result = candidate
                metadata.update(sigma=sigma, sigma_fraction=float(fraction))
        metadata.update(sigma_limit=sigma_limit, n_attack_rollouts=len(trials),
                        n_linear_solves=sum(t['n_linear_solves'] for t in trials), sigma_search=trials,
                        solve_seconds=sum(t['solve_seconds'] for t in trials))
    metadata['search_seconds'] = perf_counter() - start
    result.metadata.update(metadata)
    return result


def domirank_deletion_order(graph, **kwargs):
    return domirank_attack(graph, **kwargs).order


def domirank_robustness_curve(graph, **kwargs):
    return domirank_attack(graph, **kwargs).curve


def domirank_robustness_total(graph, **kwargs):
    return domirank_attack(graph, **kwargs).robustness
