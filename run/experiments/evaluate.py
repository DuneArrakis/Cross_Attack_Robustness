"""Evaluate input or saved graphs; store observations without plotting."""
from __future__ import annotations


from ..config import final_graph_root, input_files, project_path
from ..datasets.io import load_graph, save_attack_result, save_curve_csv
from ..reproducibility import evaluation_seed
from .tasks import TaskStore, digest
from .timing import ATTACK_FIELDS, CSVLog, StageRun, measure_attack, require_success

EVALUATION_FIELDS = ['instance_id', 'design', 'attack', 'n_nodes', 'n_edges', 'R', 'rho_05', 'rho_01',
                     'wall_seconds', 'cpu_seconds', 'n_attack_rollouts', 'graph_path', 'graph_sha256',
                     'curve_path', 'order_path', 'metadata_path', 'attack_seed']


def dismantling_thresholds(curve):
    """First discrete q/N with LCC(q)/N <= 5% or 1%; include q=0."""
    n = len(curve) - 1
    return {name: (next(q for q, value in enumerate(curve) if value <= threshold) / n if n else 0.0)
            for name, threshold in (('rho_05', .05), ('rho_01', .01))}


def find_graph_file(directory, design):
    names = {'Original': ['Original.txt', 'original.txt'],
             'onion': ['Optimized_onion.txt', 'onion.txt', 'Onion.txt', 'Optimized_Onion.txt']}
    candidates = names.get(design, ['Optimized_' + design + '.txt',
                                     'optimized_' + design + '.txt',
                                     design + '.txt', design.lower() + '.txt'])
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    return None


def graph_inputs(config):
    if config['evaluation']['source'] == 'inputs':
        if 'Original' not in config['evaluation']['graph_types']:
            raise ValueError('Input evaluation requires Original in graph_types')
        for path in input_files(config):
            yield path.stem, 'Original', path
        return
    root = final_graph_root(config)
    if not root.is_dir():
        raise FileNotFoundError('Missing final graphs: {}'.format(root))
    found = False
    for input_path in input_files(config):
        directory = root / input_path.stem
        for design in config['evaluation']['graph_types']:
            path = find_graph_file(directory, design)
            if path is not None:
                found = True
                yield directory.name, design, path
            else:
                raise FileNotFoundError('Missing graph: {} / {}'.format(directory.name, design))
    if not found:
        raise FileNotFoundError('No selected final graphs in {}'.format(root))


def run(config):
    output = project_path(config['output_dir'])
    with StageRun(output, 'evaluate', config):
        store = TaskStore(output, 'evaluate', config['resume'])
        try:
            # Validate the selected matrix before starting attack calls.
            inputs = list(graph_inputs(config))
            for graph_id, design, path in inputs:
                graph = load_graph(path)
                graph_sha256 = digest(path)
                for method in config['evaluation_attacks']:
                    seed = evaluation_seed(config, graph_id, design, method)
                    options = config['attack_options'].get(method, {})
                    spec = dict(graph_sha256=graph_sha256, options=options, attack_seed=seed)
                    with store.task((graph_id, design, method), spec) as task:
                        if task is None:
                            continue
                        result, row = measure_attack(
                            graph, method, options, instance_id=graph_id, design=design,
                            stage='evaluation', iteration='', candidate='', attack_seed=seed)
                        with CSVLog(task.directory / 'timings.csv', ATTACK_FIELDS) as log:
                            log.write(row)
                        require_success(result, row)
                        relative = 'attacks/{}/{}/'.format(graph_id, design)
                        directory = output / relative
                        save_attack_result(result, directory, method)
                        curve_path = relative + 'curve_' + method + '_raw.csv'
                        order_path = relative + 'order_' + method + '.csv'
                        metadata_path = relative + 'metadata_' + method + '.json'
                        save_curve_csv(result.curve, output / curve_path)
                        observation = dict(
                            instance_id=graph_id, design=design, attack=method,
                            n_nodes=graph.vcount(), n_edges=graph.ecount(), R=result.robustness,
                            wall_seconds=row['wall_seconds'], cpu_seconds=row['cpu_seconds'],
                            n_attack_rollouts=row['n_attack_rollouts'], graph_path=str(path.resolve()),
                            graph_sha256=graph_sha256, attack_seed=seed,
                            curve_path=curve_path, order_path=order_path, metadata_path=metadata_path,
                            **dismantling_thresholds(result.curve))
                        task.finish(observation, [output / p for p in (curve_path, order_path, metadata_path)] +
                                    [task.directory / 'timings.csv'])
                        print('{} | {} | {} | R={:.6f} | {:.4f}s'.format(
                            graph_id, design, method, result.robustness, row['wall_seconds']), flush=True)
        finally:
            store.export('evaluations.csv', EVALUATION_FIELDS, 'timings/evaluation_attacks.csv', ATTACK_FIELDS)
