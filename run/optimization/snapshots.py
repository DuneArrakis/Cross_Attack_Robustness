"""Optional snapshots, separated from SA's acceptance decisions."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from ..datasets.io import save_igraph_edgelist


def snapshot_callback(directory, config):
    directory = Path(directory)

    def append(name, row):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        new = not path.exists()
        with path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    def save(state):
        it = state['iteration']
        append('trace.csv', {k: state[k] for k in (
            'iteration', 'temperature', 'best_R', 'cur_R', 'accepted', 'improved',
            'valid_proposals', 'objective_calls', 'n_attack_rollouts', 'wall_seconds',
            'attack_wall_seconds_sum')})
        if (config['enabled'] and it % config['save_every'] == 0
                and (not config['only_if_improved'] or state['improved'])):
            path = directory / ('best_it_{:04d}.txt'.format(it))
            save_igraph_edgelist(state['best_G'], path)
            append('meta.csv', dict(it=it, best_R=state['best_R'], cur_R=state['cur_R'],
                                   accepted=state['accepted'], improved=int(state['improved']),
                                   graph_path=str(path)))
        if (config['topk_enabled'] and
                (not config['topk_only_if_improved'] or state['improved'])):
            candidates = sorted(state['candidates'], key=lambda item: item[1], reverse=True)
            count = max(1, int(math.ceil(len(candidates) * max(0, min(1, config['top_ratio'])))))
            if config['max_keep'] is not None:
                count = min(count, int(config['max_keep']))
            for rank, (graph, score) in enumerate(candidates[:count], start=1):
                path = directory / ('Top_it_{:04d}'.format(it)) / ('rank_{:02d}.txt'.format(rank))
                save_igraph_edgelist(graph, path)
                append('meta_topk.csv', dict(it=it, rank=rank, score=score, graph_path=str(path)))
    return save
