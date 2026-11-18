"""Seven objective choices, deterministic task seeds and restartable SA tasks."""
from __future__ import annotations

import multiprocessing as mp

from ..config import input_files, project_path
from ..datasets.io import load_graph, save_igraph_edgelist
from ..optimization.sa import optimize_sa
from ..optimization.snapshots import snapshot_callback
from ..reproducibility import optimization_seeds
from .tasks import TaskStore, digest
from .timing import ATTACK_FIELDS, CSVLog, StageRun, measure_attack, require_success

SUMMARY_FIELDS = ['instance_id', 'target', 'workers', 'initial_R', 'best_R',
                  'baseline_R', 'postcheck_R', 'wall_seconds', 'objective_calls',
                  'n_attack_rollouts', 'attack_wall_seconds_sum', 'attack_cpu_seconds_sum',
                  'iterations', 'accepted', 'T_max', 'T_min', 'N_per_iter', 'alpha', 'max_iter',
                  'n_edges', 'N_per_iter_base', 'batch_reference_edges', 'swap_pairing',
                  'optimization_seed', 'attack_seed', 'stop_reason']


def run(config):
    output = project_path(config['output_dir'])
    workers = config['workers'] or max(1, int(mp.cpu_count() * 0.3))
    with StageRun(output, 'optimize', config):
        store = TaskStore(output, 'optimize', config['resume'])
        try:
            with mp.get_context('spawn').Pool(workers) as pool:
                for path in input_files(config):
                    graph, graph_id = load_graph(path), path.stem
                    directory = output / 'final_graphs' / graph_id
                    original_path = directory / 'Original.txt'
                    if original_path.exists():
                        existing = load_graph(original_path)
                        if existing.vcount() != graph.vcount() or sorted(existing.get_edgelist()) != sorted(graph.get_edgelist()):
                            raise ValueError('Saved original differs from input: {}'.format(original_path))
                    else:
                        save_igraph_edgelist(graph, original_path)
                    for method in config['optimization_attacks']:
                        options = config['attack_options'].get(method, {})
                        optimization_seed, attack_seed = optimization_seeds(config, graph_id, method)
                        spec = dict(input_sha256=digest(path), sa=config['sa'], options=options,
                                    optimization_seed=optimization_seed, attack_seed=attack_seed,
                                    snapshots=config['snapshots'])
                        with store.task((graph_id, method), spec) as task:
                            if task is None:
                                print('Reuse completed SA: {} / {}'.format(graph_id, method), flush=True)
                                continue
                            context = dict(instance_id=graph_id, design=method, iteration='', candidate='',
                                           optimization_seed=optimization_seed, attack_seed=attack_seed)
                            with CSVLog(task.directory / 'timings.csv', ATTACK_FIELDS) as log:
                                baseline, row = measure_attack(graph, method, options, stage='baseline', **context)
                                log.write(row)
                                require_success(baseline, row)
                                result = optimize_sa(
                                    G_init=graph, target=method, pool=pool, parameters=config['sa'],
                                    attack_options=options, record_timing=log.write, instance_id=graph_id,
                                    optimization_seed=optimization_seed, attack_seed=attack_seed,
                                    on_iteration=snapshot_callback(task.directory / 'trajectory', config['snapshots']))
                                checked, row = measure_attack(result.graph, method, options, stage='postcheck', **context)
                                log.write(row)
                                require_success(checked, row)
                            if abs(checked.robustness - result.best_R) > 1e-12:
                                raise ValueError('Saved best graph does not reproduce its objective R')
                            if result.graph.degree() != graph.degree() or not result.graph.is_simple():
                                raise ValueError('SA violated degree/simple-graph constraints')
                            final = directory / ('Optimized_' + method + '.txt')
                            save_igraph_edgelist(result.graph, final)
                            fields = dict(config['sa'])
                            fields.update(vars(result))
                            fields.pop('graph')
                            fields.update(instance_id=graph_id, target=method, workers=workers,
                                          baseline_R=baseline.robustness, postcheck_R=checked.robustness,
                                          n_edges=graph.ecount(), N_per_iter_base=config['sa']['N_per_iter'],
                                          optimization_seed=optimization_seed, attack_seed=attack_seed)
                            task.finish(fields, [final, original_path, task.directory / 'timings.csv'] +
                                        list((task.directory / 'trajectory').rglob('*.*')))
                            store.export('sa_summary.csv', SUMMARY_FIELDS, None, ATTACK_FIELDS)
                            print('{} | {} | R={:.6f} | SA {:.3f}s'.format(
                                graph_id, method, checked.robustness, result.wall_seconds), flush=True)
        finally:
            store.export('sa_summary.csv', SUMMARY_FIELDS, 'timings/optimization_attacks.csv', ATTACK_FIELDS)
