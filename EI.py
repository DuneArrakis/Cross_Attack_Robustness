# ei_interfaces_igraph_parametric.py
# ------------------------------------------------------------
# Explosive Immunization (EI) implementation with centralized parameters and
# safe external calls.
# Public interfaces, requiring only G:
#   ei_robustness_curve(G) -> List[float]
#   ei_robustness_total(G) -> float
# Optional global parameter setter:
#   set_ei_params(m=None, k=None, iters=None, seed=None)
# ------------------------------------------------------------

import math
import random
from typing import List, Tuple, Optional
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

__all__ = [
    "ei_robustness_curve",
    "ei_robustness_total",
    "set_ei_params",
]

# ==========================================================
# ===============  Global Configuration  ====================
# ==========================================================

class EIConfig:
    # Default EI algorithm parameters.
    M_CAND: int = 1000       # Number of candidate nodes per round.
    K_STRONG: int = 6        # Strong hub threshold K.
    KEFF_ITERS: int = 3      # Number of k_eff iterations.
    RANDOM_SEED: Optional[int] = 42  # None means no fixed seed.

def _apply_seed():
    if EIConfig.RANDOM_SEED is not None:
        random.seed(EIConfig.RANDOM_SEED)
        np.random.seed(EIConfig.RANDOM_SEED)

_apply_seed()

def set_ei_params(m: Optional[int] = None,
                  k: Optional[int] = None,
                  iters: Optional[int] = None,
                  seed: Optional[int] = None) -> None:
    """
    Optionally set EI parameters globally while keeping the public API G-only.
    Parameters left as None do not change the current configuration.
    """
    if m is not None:
        # EIConfig.M_CAND = int(m)
        EIConfig.M_CAND = 1000  # Fixed at 1000; external changes are ignored.
    if k is not None:
        EIConfig.K_STRONG = int(k)
    if iters is not None:
        EIConfig.KEFF_ITERS = int(iters)
    if seed is not None:
        EIConfig.RANDOM_SEED = seed
        _apply_seed()

# ==========================================================
# ===============  Core Algorithm  ==========================
# ==========================================================

class UF:
    """Union-find structure for dynamically tracking connected components."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size   = [0]*n
        self.active = [False]*n
        self.max_size = 0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def activate(self, x: int) -> None:
        if self.active[x]:
            return
        self.active[x] = True
        self.parent[x] = x
        self.size[x]   = 1
        if self.size[x] > self.max_size:
            self.max_size = self.size[x]

    def union_if_active(self, a: int, b: int) -> None:
        if not (self.active[a] and self.active[b]):
            return
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra]  += self.size[rb]
        if self.size[ra] > self.max_size:
            self.max_size = self.size[ra]


def _iterate_keff(G: ig.Graph) -> List[int]:
    """Compute k_eff(i) = k(i) - L_i - M_i."""
    K = EIConfig.K_STRONG
    deg   = G.degree()
    n     = G.vcount()
    keff  = deg[:]
    neigh = [G.neighbors(v) for v in range(n)]
    for _ in range(EIConfig.KEFF_ITERS):
        strong   = [1 if keff[v] >= K else 0 for v in range(n)]
        new_keff = [0]*n
        for i in range(n):
            Li = sum(1 for j in neigh[i] if deg[j] == 1)
            Mi = sum(1 for j in neigh[i] if strong[j] == 1)
            val = deg[i] - Li - Mi
            new_keff[i] = val if val > 0 else 0
        keff = new_keff
    return keff


def _neighbor_sus_clusters(i: int, UF_obj: UF, neigh: List[List[int]]) -> Tuple[List[int], int]:
    """Return restored cluster sizes around i in descending order and their count."""
    roots = {}
    for j in neigh[i]:
        if UF_obj.active[j]:
            r = UF_obj.find(j)
            roots[r] = UF_obj.size[r]
    sizes = sorted(roots.values(), reverse=True)
    return sizes, len(sizes)


def _explosive_immunization_with_curve(G: ig.Graph) -> Tuple[List[int], List[float]]:
    """Run the reverse recovery process and return deletion order plus R=GCC/N curve."""
    n     = G.vcount()
    m     = EIConfig.M_CAND
    neigh = [G.neighbors(v) for v in range(n)]
    keff  = _iterate_keff(G)

    # Use a local RNG so repeated calls on the same graph are deterministic.
    rng = random.Random(EIConfig.RANDOM_SEED)

    UF_obj = UF(n)
    deleted = set(range(n))
    restored_order: List[int] = []
    # lcc_by_restored[r] = GCC size after restoring r nodes, r=0..n.
    lcc_by_restored = [0] * (n + 1)

    def _S() -> float:
        return UF_obj.max_size / n

    while len(restored_order) < n:
        if len(deleted) <= m:
            candidates = list(deleted)
        else:
            candidates = rng.sample(list(deleted), m)        # Scoring by sigma(1) or sigma(2).
        # Paper rule: switch to sigma(2) after the giant component dominates.
        use_sigma1 = (_S() <= 1 / math.sqrt(n))
        best, best_key = None, None

        # sigma(2) stage, following the paper:
        #   - A candidate not adjacent to the giant component is invalid.
        #   - If all sampled candidates are invalid, resample until a valid
        #     candidate appears.
        #   - Exception: when len(deleted) <= m, candidates cover all deleted
        #     nodes. If they are still all invalid, no vaccinated node is
        #     adjacent to the giant component, so sigma(2) cannot proceed and
        #     the step falls back to sigma(1) to guarantee termination.
        if not use_sigma1:
            while True:
                any_valid = False
                for i in candidates:
                    sizes, _ = _neighbor_sus_clusters(i, UF_obj, neigh)
                    has_giant = (UF_obj.max_size > 0 and len(sizes) > 0 and sizes[0] == UF_obj.max_size)
                    if has_giant:
                        any_valid = True
                        break

                if any_valid:
                    break

                # If candidates cover all deleted nodes, sigma(2) cannot proceed.
                if len(deleted) <= m:
                    use_sigma1 = True
                    break

                # Otherwise keep resampling without MAX_TRIES or fallback.
                candidates = rng.sample(list(deleted), m)

        for i in candidates:
            sizes, card = _neighbor_sus_clusters(i, UF_obj, neigh)
            if use_sigma1:
                nonlocal_term = sum((math.sqrt(s) - 1.0) for s in sizes if s > 1)
                key = (keff[i] + nonlocal_term, )
            else:
                has_giant = (len(sizes) > 0 and sizes[0] == UF_obj.max_size and UF_obj.max_size > 0)
                if not has_giant:
                    continue  # Skip invalid candidates under sigma(2).
                second = sizes[1] if len(sizes) >= 2 else 0
                key = (card, second)

            if best_key is None or key < best_key:
                best, best_key = i, key

        if best is None:
            best = candidates[0]

        deleted.remove(best)
        UF_obj.activate(best)
        for j in neigh[best]:
            UF_obj.union_if_active(best, j)
        restored_order.append(best)

        # Record the GCC after restoring to the current size.
        lcc_by_restored[len(restored_order)] = UF_obj.max_size

    deletion_order = list(reversed(restored_order))

    # After deleting k nodes, r=n-k nodes remain; use the reverse recovery GCC.
    R_curve = [(lcc_by_restored[n - k] / n) for k in range(0, n + 1)]
    return deletion_order, R_curve


def _explosive_immunization(G: ig.Graph) -> List[int]:
    """Run reverse recovery and output the deletion sequence using EIConfig."""
    deletion_order, _ = _explosive_immunization_with_curve(G)
    return deletion_order


def _robustness_curve_from_order(G: ig.Graph, deletion_order: List[int]) -> List[float]:
    """Delete nodes in order and return the R=GCC/N curve."""
    n = G.vcount()
    H = G.copy()
    R: List[float] = []
    id_map = list(range(n))
    deleted_flag = [False]*n

    for v in deletion_order:
        idx = id_map[v]
        H.delete_vertices([idx])
        deleted_flag[v] = True
        for u in range(n):
            if not deleted_flag[u] and id_map[u] > idx:
                id_map[u] -= 1
        if H.vcount() == 0:
            R.append(0.0)
        else:
            comps = H.components()
            gcc = max(comps.sizes()) if len(comps) > 0 else 0
            R.append(gcc / n)
    return R

def load_graph_from_txt_to_igraph(
    path: str,
    relabel_to_0_based: bool = True,
    assume_undirected: bool = True
) -> ig.Graph:
    """
    Read a txt graph file. Each line has at least two columns: u v.
    Extra fields such as "{}" are ignored.
    Example line: 1 2 {}
    """
    edges_raw = []
    nodes_set = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            edges_raw.append((u, v))
            nodes_set.add(u)
            nodes_set.add(v)

    # Isolated nodes cannot be recovered from this edge-only format unless a
    # separate node list is provided. Build the graph from nodes seen in edges.

    if relabel_to_0_based:
        # Relabel nodes to 0..n-1 for stable igraph-to-networkx conversion.
        nodes_sorted = sorted(nodes_set)
        mapping = {old: new for new, old in enumerate(nodes_sorted)}
        edges = [(mapping[u], mapping[v]) for (u, v) in edges_raw]
        n = len(nodes_sorted)
        G_ig = ig.Graph(n=n, edges=edges, directed=not assume_undirected)
    else:
        # Keep original ids. igraph needs vertices 0..max_id, so fill gaps.
        max_id = max(nodes_set) if nodes_set else -1
        n = max_id + 1
        G_ig = ig.Graph(n=n, edges=edges_raw, directed=not assume_undirected)

    if assume_undirected:
        G_ig.to_undirected(combine_edges=None)

    # Remove self-loops and multi-edges for a cleaner simple graph.
    G_ig.simplify(multiple=True, loops=True)

    return G_ig

# ==========================================================
# ===============  Public Interfaces, G Only  ===============
# ==========================================================

def ei_robustness_curve(G: ig.Graph) -> List[float]:
    """Interface 1: pass an igraph graph G and return robustness curve R."""
    _, R_curve = _explosive_immunization_with_curve(G)
    return R_curve



def ei_robustness_total(G: ig.Graph) -> float:
    """Interface 2: pass an igraph graph G and return robustness integral sum(R)/N."""
    _, R = _explosive_immunization_with_curve(G)
    N = len(R)
    return float(np.sum(R) / N) if N > 0 else 0.0


# ==========================================================
# ===============  Main Program, Plotting Example  ==========
# ==========================================================

if __name__ == "__main__":
    # Adjust global parameters here without affecting external imports.
    set_ei_params(m=10, k=6, iters=3, seed=666)

    # Build an example graph; ER and BA alternatives are shown below.
    N_NODES = 30000
    AVG_DEG = 4
    BARABASI_M = 2

    # G = ig.Graph.Erdos_Renyi(n=N_NODES, p=AVG_DEG/(N_NODES-1), directed=False)
    # G = ig.Graph.Barabasi(n=N_NODES, m=BARABASI_M, directed=False)
    graph_txt_path = "generate_real/graph_USAir97_0.txt"
    G = load_graph_from_txt_to_igraph(
        graph_txt_path,
        relabel_to_0_based=True,
        assume_undirected=True
    )

    R_curve = ei_robustness_curve(G)
    R_total = ei_robustness_total(G)

    print(f"Number of nodes: {G.vcount()}")
    print(f"Candidate count m = {EIConfig.M_CAND}, strong hub threshold K = {EIConfig.K_STRONG}, iters = {EIConfig.KEFF_ITERS}")
    print(f"Robustness integral R_total = {R_total:.6f}")

    plt.figure(figsize=(7,5))
    plt.plot(range(0, len(R_curve)), R_curve, linewidth=1.5)
    plt.xlabel("Number of deleted nodes (k)")
    plt.ylabel("R = GCC / N")
    plt.title(f"EI Robustness Curve (N={G.vcount()}, m={EIConfig.M_CAND}, K={EIConfig.K_STRONG})\nIntegral R_total = {R_total:.4f}")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
