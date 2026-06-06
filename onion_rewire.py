#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import glob
import random
from typing import Dict, List, Set, Tuple, Hashable


# =============================================================================
# 0) Configuration: edit this section only
# =============================================================================
INPUT_DIR = "generate"               # Read generate/graph_*.txt
INPUT_GLOB = "graph_*.txt"

OUT_ROOT = "results"                 # Output root directory
FINAL_GRAPH_DIRNAME = "final_graphs" # Output to results/final_graphs/<gname>/Optimized_onion.txt
OUT_FILENAME = "Optimized_onion.txt"

A = 3             # Onion parameter a
SEED_BASE = 42    # Base random seed; each graph derives a different seed

MIN_PAIR_ATTEMPTS = 200_000
PAIR_ATTEMPTS_FACTOR = 4000          # Total budget = max(MIN_PAIR_ATTEMPTS, PAIR_ATTEMPTS_FACTOR * (2M))

MAX_RESHUFFLE_TRIES = 100_000
RESTARTS = 20

VERBOSE = True


# =============================================================================
# 1) Type aliases
# =============================================================================
Label = Hashable
Edge = Tuple[int, int]  # internal indices (u < v)


# =============================================================================
# 2) Edge-list I/O
# =============================================================================
def _parse_label(tok: str):
    """Parse a node label as int when possible; otherwise keep it as a string."""
    try:
        return int(tok)
    except Exception:
        return tok


def read_edge_list(path: str) -> Tuple[List[Label], List[Tuple[Label, Label]]]:
    """
    Read an undirected txt edge list:
      each line has at least two tokens: u v;
      all later tokens, such as {}, are ignored.

    Rules:
    - ignore empty lines and # comments;
    - ignore self-loops;
    - de-duplicate undirected edges, treating u v and v u as the same edge.
    """
    edges_raw: Set[Tuple[Label, Label]] = set()
    nodes: Set[Label] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            u = _parse_label(parts[0])
            v = _parse_label(parts[1])
            if u == v:
                continue

            # Normalize undirected edge orientation.
            a, b = (u, v) if str(u) < str(v) else (v, u)
            if (a, b) in edges_raw:
                continue

            edges_raw.add((a, b))
            nodes.add(a)
            nodes.add(b)

    # Node sorting: integers by value first, then strings.
    def _node_key(x):
        return (0, x) if isinstance(x, int) else (1, str(x))

    node_list = sorted(nodes, key=_node_key)
    return node_list, list(edges_raw)


def write_edge_list(path: str, edges: List[Edge], idx2lab: List[Label]) -> None:
    """Write an undirected edge list as: u v {}."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _lab_key(x):
        return (0, x) if isinstance(x, int) else (1, str(x))

    edges_sorted = sorted(
        edges,
        key=lambda e: (_lab_key(idx2lab[e[0]]), _lab_key(idx2lab[e[1]]))
    )

    with open(path, "w", encoding="utf-8") as f:
        for u, v in edges_sorted:
            f.write(f"{idx2lab[u]} {idx2lab[v]} {{}}\n")


# =============================================================================
# 3) Mappings and degree sequence
# =============================================================================
def build_mappings(nodes: List[Label]) -> Tuple[Dict[Label, int], List[Label]]:
    lab2idx = {lab: i for i, lab in enumerate(nodes)}
    idx2lab = list(nodes)
    return lab2idx, idx2lab


def degrees_from_edges(n: int, edges: List[Edge]) -> List[int]:
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


# =============================================================================
# 4) Onion layering: sort by degree, with equal-degree nodes in the same layer
# =============================================================================
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


# =============================================================================
# 5) Core onion generation with many trials plus reshuffle
# =============================================================================
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

        # Main pairing: global many-trials phase.
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

        # Remaining stubs become unpaired.
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
                print("  -> odd number of unpaired stubs; restarting")
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
                print("  -> reshuffle failed; restarting")
            rng.seed(rng.randrange(1 << 30))
            continue

        # Validate.
        deg_now = [len(adj[i]) for i in range(n)]
        if deg_now != target_deg:
            if verbose:
                print("  -> degree sequence validation failed; restarting")
            rng.seed(rng.randrange(1 << 30))
            continue

        edges_set = set(edges)
        if len(edges_set) != m_target:
            if verbose:
                print(f"  -> wrong edge count: {len(edges_set)} vs {m_target}; restarting")
            rng.seed(rng.randrange(1 << 30))
            continue

        return sorted(edges_set)

    raise RuntimeError(
        "Generation failed after multiple restarts. Consider lowering A, for "
        "example to 1 or 2, or increasing MIN_PAIR_ATTEMPTS, "
        "MAX_RESHUFFLE_TRIES, or RESTARTS."
    )


def onion_rewire_txt_to_txt(
    input_txt: str,
    output_txt: str,
    a: float,
    seed: int,
    min_pair_attempts: int,
    pair_attempts_factor: int,
    max_reshuffle_tries: int,
    restarts: int,
    verbose: bool = True,
) -> None:
    nodes, edges_lab = read_edge_list(input_txt)
    lab2idx, idx2lab = build_mappings(nodes)

    edges_idx: List[Edge] = []
    seen: Set[Edge] = set()
    for u_lab, v_lab in edges_lab:
        u = lab2idx[u_lab]
        v = lab2idx[v_lab]
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e in seen:
            continue
        seen.add(e)
        edges_idx.append(e)

    n = len(nodes)
    m_in = len(edges_idx)
    target_deg = degrees_from_edges(n, edges_idx)

    if verbose:
        print(f"Input: {os.path.basename(input_txt)} |V|={n} |E|={m_in} sum(deg)={sum(target_deg)}")

    new_edges = generate_onion_graph(
        target_deg=target_deg,
        a=a,
        seed=seed,
        min_pair_attempts=min_pair_attempts,
        pair_attempts_factor=pair_attempts_factor,
        max_reshuffle_tries=max_reshuffle_tries,
        restarts=restarts,
        verbose=verbose,
    )

    write_edge_list(output_txt, new_edges, idx2lab)

    if verbose:
        print(f"Output saved: {output_txt} (|E|={len(new_edges)})")


# =============================================================================
# 6) Batch entry: read all graph_*.txt files under generate
# =============================================================================
def main():
    in_pattern = os.path.join(INPUT_DIR, INPUT_GLOB)
    files = sorted(glob.glob(in_pattern))
    if not files:
        raise FileNotFoundError(f"No input files found: {in_pattern}")

    out_final_root = os.path.join(OUT_ROOT, FINAL_GRAPH_DIRNAME)
    os.makedirs(out_final_root, exist_ok=True)

    print(f"Found {len(files)} graphs under {INPUT_DIR}/ ({INPUT_GLOB})")
    print(f"Output root: {out_final_root}")

    for idx, in_path in enumerate(files, start=1):
        gname = os.path.splitext(os.path.basename(in_path))[0]   # graph_XXX
        out_dir = os.path.join(out_final_root, gname)
        out_path = os.path.join(out_dir, OUT_FILENAME)

        # Derive a different reproducible seed for each graph.
        seed = (SEED_BASE * 1000003 + (hash(gname) & 0x7FFFFFFF)) & 0x7FFFFFFF

        print(f"\n[{idx}/{len(files)}] {gname}  seed={seed}")
        try:
            onion_rewire_txt_to_txt(
                input_txt=in_path,
                output_txt=out_path,
                a=A,
                seed=seed,
                min_pair_attempts=MIN_PAIR_ATTEMPTS,
                pair_attempts_factor=PAIR_ATTEMPTS_FACTOR,
                max_reshuffle_tries=MAX_RESHUFFLE_TRIES,
                restarts=RESTARTS,
                verbose=VERBOSE,
            )
        except Exception as e:
            print(f"  !! Failed on {gname}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
