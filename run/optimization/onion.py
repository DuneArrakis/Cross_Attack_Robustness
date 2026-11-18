"""Original Onion constructor; data loading and batch control live in experiments/onion.py."""
from __future__ import annotations
import random
from typing import List, Set, Tuple

Edge = Tuple[int, int]

def layer_indices_from_degrees(deg: List[int]) -> List[int]:
    n = len(deg)
    nodes_sorted = sorted(range(n), key=lambda i: (deg[i], i))

    layer = [0] * n
    cur_layer = -1
    last_k = None
    for i in nodes_sorted:
        if last_k is None or deg[i] != last_k:
            cur_layer += 1
            last_k = deg[i]
        layer[i] = cur_layer
    return layer


def generate_onion_graph(
    target_deg: List[int],
    a: float,
    seed: int,
    min_pair_attempts: int,
    pair_attempts_factor: int,
    max_reshuffle_tries: int,
    restarts: int,
    verbose: bool = True,
) -> List[Edge]:
    rng = random.Random(seed)

    n = len(target_deg)
    if any(k < 0 for k in target_deg):
        raise ValueError("Negative degree is invalid.")
    if sum(target_deg) % 2 != 0:
        raise ValueError("Sum of degrees must be even.")

    m_target = sum(target_deg) // 2
    layer = layer_indices_from_degrees(target_deg)

    def canon(u: int, v: int) -> Edge:
        return (u, v) if u < v else (v, u)

    for attempt in range(1, restarts + 1):
        adj: List[Set[int]] = [set() for _ in range(n)]
        edges: List[Edge] = []

        # stubs
        stubs: List[int] = []
        for i, k in enumerate(target_deg):
            if k:
                stubs.extend([i] * k)
        rng.shuffle(stubs)

        total_stubs = len(stubs)
        pair_budget = max(min_pair_attempts, pair_attempts_factor * total_stubs)

        attempts_used = 0
        accepted_edges = 0

        # 主配对：全局 many trials
        while len(stubs) >= 2 and attempts_used < pair_budget:
            attempts_used += 1

            i = rng.randrange(len(stubs))
            j = rng.randrange(len(stubs) - 1)
            if j >= i:
                j += 1

            v = stubs[i]
            u = stubs[j]

            if u == v or (u in adj[v]):
                continue

            p = 1.0 / (1.0 + a * abs(layer[v] - layer[u]))
            if rng.random() >= p:
                continue

            # accept
            adj[v].add(u)
            adj[u].add(v)
            edges.append(canon(u, v))
            accepted_edges += 1

            # remove 2 stubs
            for idx in sorted((i, j), reverse=True):
                stubs[idx] = stubs[-1]
                stubs.pop()

        # 剩余 stubs -> unpaired
        unpaired = stubs[:]
        stubs.clear()

        if verbose:
            frac = (len(unpaired) / total_stubs) if total_stubs > 0 else 0.0
            print(
                f"  [Attempt {attempt}/{restarts}] accepted={accepted_edges}/{m_target}, "
                f"unpaired={len(unpaired)}/{total_stubs} ({frac:.2%}), budget={pair_budget}, used={attempts_used}"
            )

        # reshuffle
        def try_reshuffle_pair(v1: int, v2: int) -> bool:
            if not edges:
                return False

            def edge_ok(p: int, q: int) -> bool:
                return (p != q) and (q not in adj[p])

            for _ in range(max_reshuffle_tries):
                eidx = rng.randrange(len(edges))
                x, y = edges[eidx]

                if y not in adj[x]:
                    continue
                adj[x].remove(y)
                adj[y].remove(x)

                if rng.random() < 0.5:
                    a1, b1 = v1, x
                    a2, b2 = v2, y
                else:
                    a1, b1 = v1, y
                    a2, b2 = v2, x

                e1 = canon(a1, b1)
                e2 = canon(a2, b2)

                if (not edge_ok(a1, b1)) or (not edge_ok(a2, b2)) or (e1 == e2):
                    adj[x].add(y)
                    adj[y].add(x)
                    continue

                adj[a1].add(b1); adj[b1].add(a1)
                adj[a2].add(b2); adj[b2].add(a2)

                edges[eidx] = e1
                edges.append(e2)
                return True

            return False

        if len(unpaired) % 2 == 1:
            if verbose:
                print("  -> unpaired 为奇数，重启")
            rng.seed(rng.randrange(1 << 30))
            continue

        reshuffle_failed = False
        while len(unpaired) >= 2:
            v1 = unpaired.pop()
            v2 = unpaired.pop()
            if not try_reshuffle_pair(v1, v2):
                reshuffle_failed = True
                break

        if reshuffle_failed:
            if verbose:
                print("  -> reshuffle 失败，重启")
            rng.seed(rng.randrange(1 << 30))
            continue

        # 校验
        deg_now = [len(adj[i]) for i in range(n)]
        if deg_now != target_deg:
            if verbose:
                print("  -> 度序列校验失败，重启")
            rng.seed(rng.randrange(1 << 30))
            continue

        edges_set = set(edges)
        if len(edges_set) != m_target:
            if verbose:
                print(f"  -> 边数不对：{len(edges_set)} vs {m_target}，重启")
            rng.seed(rng.randrange(1 << 30))
            continue

        return sorted(edges_set)

    raise RuntimeError(
        "多次重启后仍未生成成功。建议：降低 A（如 1~2），或增大 MIN_PAIR_ATTEMPTS / MAX_RESHUFFLE_TRIES / RESTARTS。"
    )
