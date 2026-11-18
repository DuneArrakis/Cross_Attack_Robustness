"""Existing batched-candidate SA with optional edge-scaled candidate counts.

Candidates at one temperature are generated from the same current graph, then
accepted sequentially in pool.map order. Cooling, thresholds and acceptance are
unchanged. New runs sample both swap pairings; legacy is available explicitly.
Batch size is resolved once from the initial edge count (preserved by swaps).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from time import perf_counter

import igraph as ig

from ..config import SA_DEFAULTS
from ..experiments.timing import measure_attack, require_success
from ..reproducibility import derive_seed


@dataclass
class SAResult:
    graph: ig.Graph
    initial_R: float
    best_R: float
    wall_seconds: float
    objective_calls: int
    n_attack_rollouts: int
    attack_wall_seconds_sum: float
    attack_cpu_seconds_sum: float
    iterations: int
    accepted: int
    N_per_iter: int
    stop_reason: str


def random_edge_reconnect(G: ig.Graph, rng=None, pairing='legacy') -> bool:
    """A simple degree-preserving swap; both nontrivial pairings are equiprobable."""
    rng = random if rng is None else rng
    edges = G.get_edgelist()
    if len(edges) < 2:
        return False
    for _ in range(100):
        e1, e2 = rng.sample(edges, 2)
        a, b = e1
        c, d = e2
        if len({a, b, c, d}) < 4:
            continue
        if pairing == 'both' and rng.random() < .5:
            c, d = d, c
        if not (G.are_connected(a, c) or G.are_connected(b, d)):
            G.delete_edges([e1, e2])
            G.add_edges([(a, c), (b, d)])
            return True
    return False


def _sa_worker(args):
    graph, target, options, context, pairing, evaluator = args
    seed = context.get('proposal_seed')
    changed = random_edge_reconnect(graph, None if seed is None else random.Random(seed), pairing)
    result, timing = (measure_attack if evaluator is None else evaluator)(graph, target, options, **context)
    timing['proposal_changed'] = changed
    return graph, None if result is None else result.robustness, timing


def optimize_sa(*, G_init, target, pool, parameters=None, attack_options=None,
                record_timing=None, instance_id='', on_iteration=None, verbose=True,
                optimization_seed=None, attack_seed=None, evaluator=None):
    """Optional evaluator permits exact precomputed objectives for tiny diagnostics.

    The default evaluator, proposals, acceptance, and cooling remain unchanged.
    An injected evaluator must return the same (AttackResult, timing) contract.
    """
    parameters = dict(SA_DEFAULTS if parameters is None else parameters)
    T_max, T_min = parameters['T_max'], parameters['T_min']
    N_per_iter, alpha, max_iter = (parameters['N_per_iter'], parameters['alpha'], parameters['max_iter'])
    reference = parameters.get('batch_reference_edges')
    if reference is not None:
        # Integer ceiling; null/missing reference retains the fixed legacy batch.
        N_per_iter = max(1, (N_per_iter * G_init.ecount() + reference - 1) // reference)
    pairing = parameters.get('swap_pairing', 'legacy')
    rng = random if optimization_seed is None else random.Random(derive_seed(optimization_seed, 'acceptance'))
    context = dict(instance_id=instance_id, design=target,
                   optimization_seed=optimization_seed, attack_seed=attack_seed)
    wall_start = perf_counter()
    totals = dict(calls=0, rollouts=0, wall=0.0, cpu=0.0)

    def record(row):
        totals['calls'] += 1
        totals['rollouts'] += row.get('n_attack_rollouts', 0)
        totals['wall'] += row['wall_seconds']
        totals['cpu'] += row['cpu_seconds']
        if record_timing is not None:
            record_timing(row)

    best_G = G_init.copy()
    initial, timing = (measure_attack if evaluator is None else evaluator)(best_G, target, attack_options, stage='sa_initial',
                                    iteration=0, candidate=0, **context)
    record(timing)
    best_R = require_success(initial, timing).robustness
    initial_R = best_R
    G_cur = best_G.copy()
    cur_R = best_R
    T = T_max
    iterations, accepted_total = 0, 0
    stop_reason = 'max_iter'

    for it in range(1, max_iter + 1):
        if T_min - T > 1e-20:
            stop_reason = 'temperature'
            break
        tasks = [(G_cur.copy(), target, attack_options,
                  dict(context, stage='sa_candidate', iteration=it, candidate=j + 1,
                       proposal_seed=None if optimization_seed is None else derive_seed(optimization_seed, it, j + 1)),
                  pairing, evaluator)
                 for j in range(N_per_iter)]
        results = pool.map(_sa_worker, tasks)
        # Record completed worker calls in the parent, including failed calls.
        for _, _, timing in results:
            record(timing)
        for _, score, timing in results:
            if score is None:
                require_success(None, timing)

        accepted = 0
        improved_flag = False
        for cand_G, R, _ in results:
            if cand_G is None:
                continue
            delta = R - cur_R
            if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-300)):
                G_cur = cand_G
                cur_R = R
                accepted += 1
                if R - best_R > 1e-10:
                    best_G, best_R = cand_G.copy(), R
                    improved_flag = True

        iterations = it
        accepted_total += accepted
        if on_iteration is not None:
            on_iteration(dict(iteration=it, temperature=T, best_G=best_G, cur_G=G_cur,
                              best_R=best_R, cur_R=cur_R, accepted=accepted,
                              objective_calls=totals['calls'], n_attack_rollouts=totals['rollouts'],
                              wall_seconds=perf_counter() - wall_start,
                              attack_wall_seconds_sum=totals['wall'],
                              valid_proposals=sum(int(t['proposal_changed']) for _, _, t in results),
                              improved=improved_flag,
                              candidates=[(g, r) for g, r, _ in results]))
        T *= alpha
        if verbose and (it == 1 or it % 10 == 0 or it == max_iter):
            print('[SA {}] it={} T={:.3e} cur_R={:.6f} best_R={:.6f} accept={}/{}'.format(
                target, it, T, cur_R, best_R, accepted, N_per_iter), flush=True)

    return SAResult(best_G, initial_R, best_R, perf_counter() - wall_start,
                    totals['calls'], totals['rollouts'], totals['wall'], totals['cpu'],
                    iterations, accepted_total, N_per_iter, stop_reason)
