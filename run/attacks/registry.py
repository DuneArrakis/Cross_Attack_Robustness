"""Single registry for optimization and cross-attack evaluation.

Lazy imports let Zheng/DomiRank run without loading FINDER/TensorFlow.
Use evaluate_attack once when both order, curve and R are needed.
Legacy algorithms keep their deletion policies; the metric is shared.
"""
from __future__ import annotations

from functools import partial
from time import perf_counter

from .common import result_from_order, validate_graph

ATTACK_NAMES = ("HDA", "CI", "CoreHD", "EI", "FINDER", "Zheng", "DomiRank")


def evaluate_attack(graph, method, seed=None, **kwargs):
    validate_graph(graph)
    start = perf_counter()
    if method == "Zheng":
        from .zheng import zheng_attack
        result = zheng_attack(graph, **kwargs)
    elif method == "DomiRank":
        from .domirank import domirank_attack
        result = domirank_attack(graph, **kwargs)
    elif method in ATTACK_NAMES:
        if method == "HDA":
            from .hda import hda_attack_sequence
            order = hda_attack_sequence(graph, **kwargs)
        elif method == "CI":
            from .ci import ci_deletion_order
            order = ci_deletion_order(graph, **kwargs)
        elif method == "CoreHD":
            from .corehd import corehd_robustness_curve_igraph_with_marker
            _, _, order = corehd_robustness_curve_igraph_with_marker(graph, **kwargs)
        elif method == "EI":
            from .ei import _explosive_immunization_with_curve
            order, _ = _explosive_immunization_with_curve(graph, seed=seed, **kwargs)
        else:
            from .finder import FinderWrapperND
            order = FinderWrapperND().attack_order(graph, **kwargs)
        result = result_from_order(graph, order, method=method, n_attack_rollouts=1)
    else:
        raise ValueError("Unknown attack {!r}; choose from {}".format(method, ATTACK_NAMES))
    result.metadata['elapsed_seconds'] = perf_counter() - start
    result.metadata['metric'] = 'sum_LCC_q1_to_N_over_N_squared'
    result.metadata['attack_seed'] = seed
    result.metadata['seed_used'] = method == 'EI'
    return result


def attack_R(graph, method, **kwargs):
    return evaluate_attack(graph, method, **kwargs).robustness


def attack_curve(graph, method, **kwargs):
    return evaluate_attack(graph, method, **kwargs).curve


def attack_order(graph, method, **kwargs):
    return evaluate_attack(graph, method, **kwargs).order


def build_attack_methods(names=ATTACK_NAMES):
    """Picklable partials for the existing multiprocessing/SA interface."""
    unknown = set(names) - set(ATTACK_NAMES)
    if unknown:
        raise ValueError("Unknown attacks: {}".format(sorted(unknown)))
    return ({name: partial(attack_R, method=name) for name in names},
            {name: partial(attack_curve, method=name) for name in names})
