# CI.py
# ------------------------------------------------------------
# Collective Influence (CI) Algorithm - optimized implementation
# - Adaptive CI removal with LOCAL updates only (default)
# - Max-heap maintenance via heapq + lazy update (stamps[])
# - Fast CI computation for l=2 using 2-hop enumeration + timestamp arrays
# - Robustness curve R(k)=GCC/N for k=0..N via reverse Union-Find (N+1 values)
#
# Requirements:
#   pip install python-igraph numpy matplotlib
# ------------------------------------------------------------

from __future__ import annotations

import heapq
from collections import deque
from typing import List, Tuple, Set, Optional

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "ci_deletion_order",
    "ci_robustness_curve",
    "ci_robustness_total",
    "load_graph_from_txt_to_igraph",
]


# =========================
# Union-Find for reverse GCC
# =========================
class UF:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [0] * n
        self.active = [False] * n
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
        self.size[x] = 1
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
        self.size[ra] += self.size[rb]
        if self.size[ra] > self.max_size:
            self.max_size = self.size[ra]


# =========================
# Helper: adjacency cache
# =========================
def _build_adj_list(G: ig.Graph) -> List[List[int]]:
    """Cache adjacency list once (undirected graph assumed; use ALL for safety)."""
    n = G.vcount()
    return [G.neighbors(v, mode="ALL") for v in range(n)]


# =========================
# Stamp marker (fast "set")
# =========================
class _Stamp:
    """Use timestamp arrays to avoid Python set allocations."""
    __slots__ = ("mark", "t")

    def __init__(self, n: int):
        self.mark = [0] * n
        self.t = 1

    def next(self) -> None:
        self.t += 1
        # Extremely unlikely overflow; reset if it happens.
        if self.t >= 2**31:
            self.mark = [0] * len(self.mark)
            self.t = 1

    def seen(self, x: int) -> bool:
        return self.mark[x] == self.t

    def set(self, x: int) -> None:
        self.mark[x] = self.t


# =========================
# CI core routines
# =========================
def _compute_ci(
    v: int,
    l: int,
    adj: List[List[int]],
    alive: List[bool],
    deg: List[int],
    stamp1: _Stamp,
    stamp2: _Stamp,
) -> float:
    """
    CI_l(v) = (k_v - 1) * sum_{j in boundary B(v,l)} (k_j - 1)
    Degrees are dynamic residual degrees (deg[*]) under alive mask.

    Fast path for l=2: two-hop enumeration with timestamp arrays.
    """
    if not alive[v]:
        return float("-inf")

    kv = deg[v]
    if kv <= 1:
        return 0.0

    # ---------- fast path: l == 2 ----------
    if l == 2:
        # stamp1 marks {v} union N(v) to exclude distance < 2
        stamp1.next()
        stamp1.set(v)
        for nb in adj[v]:
            if alive[nb]:
                stamp1.set(nb)

        # stamp2 for unique frontier nodes at exactly distance 2
        stamp2.next()
        s = 0.0

        for nb in adj[v]:
            if not alive[nb]:
                continue
            for w in adj[nb]:
                if not alive[w]:
                    continue
                if stamp1.seen(w):  # excludes v and 1-hop neighbors
                    continue
                if stamp2.seen(w):  # de-duplicate
                    continue
                stamp2.set(w)
                kw = deg[w]
                if kw > 1:
                    s += (kw - 1)

        return (kv - 1) * s

    # ---------- generic path: BFS to exact distance l ----------
    if l <= 0:
        return 0.0  # CI_0 usually not used; keep safe

    seen: Set[int] = {v}
    q = deque([(v, 0)])
    frontier: List[int] = []

    while q:
        x, d = q.popleft()
        if d == l:
            frontier.append(x)
            continue
        for y in adj[x]:
            if not alive[y] or y in seen:
                continue
            seen.add(y)
            q.append((y, d + 1))

    s = 0.0
    for j in frontier:
        kj = deg[j]
        if kj > 1:
            s += (kj - 1)

    return (kv - 1) * s


def _nodes_within_radius_from_removed(
    u: int,
    radius: int,
    adj: List[List[int]],
    alive: List[bool],
) -> Set[int]:
    """
    After removing u (alive[u]=False), return all alive nodes with dist(u, x) <= radius.
    BFS starting from alive neighbors of u.
    """
    if radius <= 0:
        return set()

    res: Set[int] = set()
    seen: Set[int] = set()
    q = deque()

    for nb in adj[u]:
        if alive[nb]:
            seen.add(nb)
            q.append((nb, 1))
            res.add(nb)

    while q:
        v, d = q.popleft()
        if d == radius:
            continue
        for w in adj[v]:
            if not alive[w] or w in seen:
                continue
            seen.add(w)
            q.append((w, d + 1))
            res.add(w)

    return res


# =========================
# Public API: deletion order
# =========================
def ci_deletion_order(
    G: ig.Graph,
    l: int = 2,
    refresh_every: int = 0,
) -> List[int]:
    """
    Compute CI adaptive removal order using:
      - local update within radius (l+1)
      - max-heap (via heapq min-heap of negative scores)
      - lazy update via stamp[] for heap entries

    refresh_every:
      - 0: no global refresh (FAST; recommended; local update only)
      - K>0: every K deletions, globally recompute CI for all alive nodes and rebuild heap
             (use only if you suspect local update drift in your own variant)

    Returns:
      removal_order: list of original vertex indices (0..N-1)
    """
    n = G.vcount()
    if n == 0:
        return []

    adj = _build_adj_list(G)

    alive = [True] * n
    deg = [len(adj[v]) for v in range(n)]  # dynamic degrees in residual graph

    ci_now = [0.0] * n
    heap_stamp = [0] * n

    # stamps for fast CI(l=2)
    stamp1 = _Stamp(n)
    stamp2 = _Stamp(n)

    # heap entry: (-ci, v, heap_stamp[v])
    # NOTE (tie-breaking): when CI values are equal, smaller vertex id v is popped first
    # because Python compares tuples lexicographically: (-ci, v, stamp).
    # This is deterministic and acceptable (the paper does not specify tie-breaking).
    heap: List[Tuple[float, int, int]] = []
    for v in range(n):
        ci = _compute_ci(v, l, adj, alive, deg, stamp1, stamp2)
        ci_now[v] = ci
        heapq.heappush(heap, (-ci, v, heap_stamp[v]))

    removal_order: List[int] = []

    for step in range(1, n + 1):
        # pop until valid
        while heap:
            neg_ci, v, s = heapq.heappop(heap)
            if not alive[v]:
                continue
            if s != heap_stamp[v]:
                continue
            break
        else:
            # fallback (should not happen): pick any alive node
            v = next(i for i in range(n) if alive[i])

        # remove v (paper Step3: remove node and decrement neighbors' degrees)
        alive[v] = False
        deg[v] = 0  # keep residual-degree array consistent for removed nodes (safe, doesn't change results)
        removal_order.append(v)

        # update degrees of alive neighbors
        for nb in adj[v]:
            if alive[nb]:
                deg[nb] -= 1

        # local update: nodes within dist <= l+1 from v
        affected = _nodes_within_radius_from_removed(v, l + 1, adj, alive)
        for x in affected:
            new_ci = _compute_ci(x, l, adj, alive, deg, stamp1, stamp2)
            ci_now[x] = new_ci
            heap_stamp[x] += 1
            heapq.heappush(heap, (-new_ci, x, heap_stamp[x]))

        # optional global refresh
        if refresh_every and (step % refresh_every == 0) and (step < n):
            heap.clear()
            for u in range(n):
                if not alive[u]:
                    continue
                new_ci = _compute_ci(u, l, adj, alive, deg, stamp1, stamp2)
                ci_now[u] = new_ci
                heap_stamp[u] += 1
                heapq.heappush(heap, (-new_ci, u, heap_stamp[u]))

    return removal_order


# =========================
# Robustness curve via reverse UF
# =========================
def _robustness_curve_from_order_reverse_union_find(
    adj: List[List[int]],
    deletion_order: List[int],
) -> List[float]:
    """
    Given deletion order (length N), compute R[k] = GCC(after deleting k nodes)/N for k=0..N
    using reverse activation + Union-Find in ~O(N + E alpha).

    Returns:
      R: length N+1
         R[0] = GCC(original)/N
         R[N] = 0
    """
    n = len(adj)
    if n == 0:
        return [0.0]

    uf = UF(n)
    active = [False] * n

    restore_order = list(reversed(deletion_order))
    gcc_after_restore = [0] * (n + 1)
    gcc_after_restore[0] = 0

    restored = 0
    for v in restore_order:
        restored += 1
        active[v] = True
        uf.activate(v)
        for nb in adj[v]:
            if active[nb]:
                uf.union_if_active(v, nb)
        gcc_after_restore[restored] = uf.max_size

    # deleted k => restored r = n-k
    R = [0.0] * (n + 1)
    for k in range(n + 1):
        r = n - k
        R[k] = gcc_after_restore[r] / n

    return R


def ci_robustness_curve(G: ig.Graph, l: int = 2) -> List[float]:
    """
    Compute CI removal order then compute robustness curve R[k]=GCC/N, k=0..N (N+1 values).
    """
    n = G.vcount()
    if n == 0:
        return [0.0]
    adj = _build_adj_list(G)
    order = ci_deletion_order(G, l=l)
    return _robustness_curve_from_order_reverse_union_find(adj, order)


def ci_robustness_total(G: ig.Graph, l: int = 2) -> float:
    """
    Average robustness (mean over k=0..N).
    """
    R = ci_robustness_curve(G, l=l)
    return float(np.mean(R)) if R else 0.0


# =========================
# Utilities: load graph
# =========================
def load_graph_from_txt_to_igraph(
    path: str,
    relabel_to_0_based: bool = True,
    assume_undirected: bool = True,
) -> ig.Graph:
    """
    Read edge list txt: each line at least "u v", trailing fields ignored (e.g., "{}").
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

    if relabel_to_0_based:
        nodes_sorted = sorted(nodes_set)
        mapping = {old: new for new, old in enumerate(nodes_sorted)}
        edges = [(mapping[u], mapping[v]) for (u, v) in edges_raw]
        n = len(nodes_sorted)
        G_ig = ig.Graph(n=n, edges=edges, directed=not assume_undirected)
    else:
        max_id = max(nodes_set) if nodes_set else -1
        n = max_id + 1
        G_ig = ig.Graph(n=n, edges=edges_raw, directed=not assume_undirected)

    if assume_undirected:
        G_ig.to_undirected(combine_edges=None)

    G_ig.simplify(multiple=True, loops=True)
    return G_ig


# =========================
# Main demo
# =========================
def _demo():
    L = 2  # CI radius
    # G = ig.Graph.Barabasi(n=5000, m=2, directed=False)
    G = load_graph_from_txt_to_igraph(path="generate_real/graph_USAir97_0.txt")

    print(f"Graph: N={G.vcount()}, E={G.ecount()}, l={L}")

    order = ci_deletion_order(G, l=L)  # local update only (fast)
    print("Top-10 removed nodes:", order[:10])

    R = ci_robustness_curve(G, l=L)  # length N+1: k=0..N
    print(f"len(R)={len(R)} (should be N+1)")
    print(f"Robustness mean (mean R over k=0..N): {float(np.mean(R)):.6f}")

    plt.figure(figsize=(7, 5))
    plt.plot(range(len(R)), R)
    plt.xlabel("k (number of removed nodes)")
    plt.ylabel("R(k) = GCC/N")
    plt.title(f"CI robustness curve (l={L})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _demo()
