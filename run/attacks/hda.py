# HDA.py
# ------------------------------------------------------------
# High Degree Adaptive (HDA) attack
# - Remove highest-degree node sequentially
# - Recompute degrees after each removal
# - If multiple nodes share the max degree:
#     * default: pick one uniformly at random
#     * optional: pick the one with smallest current index (deterministic)
# - Robustness curve includes k=0..N (N+1 points): GCC/N
# - Provides full attack sequence in ORIGINAL vertex IDs
# ------------------------------------------------------------

from __future__ import annotations

import random
from typing import List, Tuple, Optional

import igraph as ig
import numpy as np
from .common import robustness_from_curve


def largest_connected_component_size(G: ig.Graph) -> int:
    """Size of the largest connected component."""
    if G.vcount() == 0:
        return 0
    return max(G.connected_components().sizes(), default=0)


def simulate_attack_once_curve_optimized(
    G_original: ig.Graph,
    seed: Optional[int] = None,
    tie_break: str = "min",  # "random" or "min"
) -> Tuple[List[int], List[float]]:
    """
    Return:
      - attack_list: [orig_id_1, orig_id_2, ..., orig_id_N0]  (length N0)
      - R_list: robustness curve R[k] = GCC(after deleting k nodes)/N0, k=0..N0 (length N0+1)

    Notes:
    - Uses physical deletion (delete_vertices), so current vertex indices are dynamic.
    - We store G.vs["orig_id"] to keep original IDs stable in attack_list.
    """
    if seed is not None:
        random.seed(seed)

    G = G_original.copy()
    N0 = G_original.vcount()

    if N0 == 0:
        return [], [0.0]

    # Fix original vertex IDs
    G.vs["orig_id"] = list(range(N0))

    attack_list: List[int] = []
    R_list: List[float] = []

    # k = 0 (no deletion)
    R_list.append(largest_connected_component_size(G) / N0)

    # k = 1..N0
    for _ in range(N0):
        if G.vcount() == 0:
            R_list.append(0.0)
            continue

        degrees = G.degree()
        if not degrees:
            R_list.append(0.0)
            continue

        max_deg = max(degrees)
        candidates = [i for i, d in enumerate(degrees) if d == max_deg]

        if tie_break == "min":
            max_idx = min(candidates)
        elif tie_break == "random":
            max_idx = random.choice(candidates)
        else:
            raise ValueError("tie_break must be 'random' or 'min'")

        # Record ORIGINAL ID
        attack_list.append(int(G.vs[max_idx]["orig_id"]))

        # Delete selected vertex
        G.delete_vertices(max_idx)

        # Record robustness
        R_list.append(largest_connected_component_size(G) / N0)

    # Defensive: ensure lengths are correct
    if len(R_list) != N0 + 1:
        R_list = (R_list + [0.0] * (N0 + 1 - len(R_list)))[: N0 + 1]
    if len(attack_list) != N0:
        attack_list = (attack_list + [-1] * (N0 - len(attack_list)))[:N0]

    return attack_list, R_list


def hda_curve_only(G: ig.Graph, seed: Optional[int] = None, tie_break: str = "min") -> List[float]:
    """Return only robustness curve (length N0+1)."""
    _, R = simulate_attack_once_curve_optimized(G, seed=seed, tie_break=tie_break)
    return R


def hda_attack_list_only(G: ig.Graph, seed: Optional[int] = None, tie_break: str = "min") -> List[int]:
    """Return only attack list (length N0), original vertex IDs."""
    attack_list, _ = simulate_attack_once_curve_optimized(G, seed=seed, tie_break=tie_break)
    return attack_list


def hda_attack_sequence(G: ig.Graph, seed: Optional[int] = None, tie_break: str = "min") -> List[int]:
    """
    NEW: Return the full attack sequence in original vertex IDs.
    Alias of hda_attack_list_only, kept as a clearer API name.
    """
    return hda_attack_list_only(G, seed=seed, tie_break=tie_break)


def simulate_attack_once_optimized(G_original: ig.Graph, seed: Optional[int] = None, tie_break: str = "min") -> float:
    """
    Return paper robustness: sum of R[k] over k=1..N, divided by N.
    """
    curve = hda_curve_only(G_original, seed=seed, tie_break=tie_break)
    return robustness_from_curve(curve, G_original.vcount())


# --------------------------
# Optional: quick self-test
# --------------------------
