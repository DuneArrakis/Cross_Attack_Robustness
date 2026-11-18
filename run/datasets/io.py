"""Small shared readers/writers for revision experiments (no hidden relabeling)."""
from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

import igraph as ig

from ..attacks.common import validate_graph


def load_graph(path):
    """Read existing 'u v {}' files and optional '# n_nodes N' metadata.

    IDs are nonnegative, zero-based igraph indices, as in the old pipeline.
    For legacy files N=max_id+1; trailing isolates cannot be inferred.
    """
    edges, declared_n, max_id = [], None, -1
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].startswith('#'):
                if parts[:2] == ['#', 'n_nodes']:
                    declared_n = int(parts[2])
                continue
            u, v = int(parts[0]), int(parts[1])
            if min(u, v) < 0:
                raise ValueError("Negative vertex index in {}".format(path))
            max_id = max(max_id, u, v)
            edges.append((u, v))
    n = max_id + 1 if declared_n is None else declared_n
    if n < max_id + 1 or n < 0:
        raise ValueError("Invalid n_nodes header in {}".format(path))
    graph = ig.Graph(n=n, edges=edges, directed=False)
    validate_graph(graph)
    return graph


def save_igraph_edgelist(graph, path, with_brace=True):
    validate_graph(graph)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        f.write('# n_nodes {}\n'.format(graph.vcount()))
        for u, v in graph.get_edgelist():
            f.write('{} {}{}\n'.format(u, v, ' {}' if with_brace else ''))
    temporary.replace(path)


def save_attack_result(result, directory, method):
    """Archive the SAME rollout's order and parameter search metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('order_{}.csv'.format(method))).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'node'])
        writer.writerows(enumerate(result.order, start=1))
    data = dict(result.metadata, R=result.robustness, n_nodes=len(result.order))
    (directory / ('metadata_{}.json'.format(method))).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def save_curve_csv(curve, path):
    """Save raw q=0..N points. Analysis may resample for plotting only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(curve) - 1
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['q', 'x', 'R'])
        writer.writerows((q, q / n if n else 0.0, value) for q, value in enumerate(curve))
