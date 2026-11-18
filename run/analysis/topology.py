"""Existing degree-dependent topology diagnostics, separated from file selection/plots."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List
import igraph as ig
import numpy as np
import pandas as pd
from ..datasets.io import load_graph

def compute_degree_binned_knn(g: ig.Graph) -> Dict[int, float]:
    deg = g.degree()
    n = g.vcount()
    neighs = g.get_adjlist()
    knn_i = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if deg[i] <= 0:
            continue
        nb = neighs[i]
        if not nb:
            continue
        knn_i[i] = float(np.mean([deg[j] for j in nb]))

    byk: Dict[int, List[float]] = {}
    for i, k in enumerate(deg):
        if k <= 0:
            continue
        v = knn_i[i]
        if np.isfinite(v):
            byk.setdefault(k, []).append(v)

    return {k: float(np.mean(vals)) for k, vals in byk.items()}


def compute_degree_binned_clustering(g: ig.Graph) -> Dict[int, float]:
    deg = g.degree()
    c_local = g.transitivity_local_undirected(vertices=None, mode="zero")
    byk: Dict[int, List[float]] = {}
    for i, k in enumerate(deg):
        if k <= 0:
            continue
        byk.setdefault(k, []).append(float(c_local[i]))
    return {k: float(np.mean(v)) for k, v in byk.items()}


def compute_onion_sk_over_nk(g: ig.Graph) -> Dict[int, float]:
    deg = np.array(g.degree(), dtype=int)
    ks = sorted(set(int(x) for x in deg if x > 0))
    if not ks:
        return {}

    unique, counts = np.unique(deg[deg > 0], return_counts=True)
    Nk = {int(k): int(c) for k, c in zip(unique, counts)}

    out: Dict[int, float] = {}
    for k in ks:
        active = np.where(deg <= k)[0].tolist()
        if not active:
            continue
        sub = g.subgraph(active)
        comps = sub.components()
        membership = comps.membership

        comp_counts: Dict[int, int] = {}
        for sub_idx, comp_id in enumerate(membership):
            orig_idx = active[sub_idx]
            if deg[orig_idx] == k:
                comp_counts[comp_id] = comp_counts.get(comp_id, 0) + 1

        Sk = max(comp_counts.values()) if comp_counts else 0
        denom = Nk.get(k, 0)
        if denom > 0:
            out[k] = Sk / denom

    return out

def topology_tables(observations, directory):
    graphs = observations[['instance_id', 'design', 'graph_path', 'graph_sha256']].drop_duplicates()
    if graphs.duplicated(['instance_id', 'design']).any():
        raise ValueError('Multiple graph files for the same instance/design')
    records = []
    sizes = []
    metrics = {'knn': compute_degree_binned_knn,
               'ck': compute_degree_binned_clustering, 'onion': compute_onion_sk_over_nk}
    for row in graphs.itertuples():
        if hashlib.sha256(Path(row.graph_path).read_bytes()).hexdigest() != row.graph_sha256:
            raise ValueError('Graph changed since evaluation: ' + row.graph_path)
        graph = load_graph(row.graph_path)
        sizes.append(dict(instance_id=row.instance_id, design=row.design,
                          n_nodes=graph.vcount(), n_edges=graph.ecount(), graph_path=row.graph_path))
        for metric, function in metrics.items():
            for degree, value in function(graph).items():
                records.append(dict(instance_id=row.instance_id, design=row.design,
                                    degree_k=degree, metric=metric, value=value))
    pd.DataFrame(sizes).to_csv(directory / 'graph_sizes.csv', index=False)
    values = pd.DataFrame(records, columns=['instance_id', 'design', 'degree_k', 'metric', 'value'])
    values.to_csv(directory / 'topology_per_graph.csv', index=False)
    if values.empty:
        result = pd.DataFrame(columns=['design', 'degree_k', 'metric', 'mean', 'count', 'std', 'sem'])
    else:
        result = values.groupby(['design', 'degree_k', 'metric'])['value'].agg(['mean', 'count', 'std'])
        result['sem'] = result['std'] / np.sqrt(result['count'])
        result = result.reset_index()
    result.to_csv(directory / 'topology_mean.csv', index=False)
    return result
