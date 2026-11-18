"""Optional Onion constructor; stable seeds and task-level restart."""
from __future__ import annotations

import igraph as ig
from time import perf_counter, process_time

from ..config import input_files, project_path
from ..datasets.io import load_graph, save_igraph_edgelist
from ..optimization.onion import generate_onion_graph
from ..reproducibility import derive_seed
from .tasks import TaskStore, digest
from .timing import CSVLog, StageRun

FIELDS = ['instance_id', 'seed', 'wall_seconds', 'cpu_seconds', 'status', 'error']


def run(config):
    output = project_path(config['output_dir'])
    options = config['onion']
    with StageRun(output, 'onion', config):
        store = TaskStore(output, 'onion', config['resume'])
        try:
            for path in input_files(config):
                seed = derive_seed(options['seed_base'], path.stem, 'onion')
                with store.task((path.stem,), dict(input_sha256=digest(path), options=options, seed=seed)) as task:
                    if task is None:
                        continue
                    graph = load_graph(path)
                    start, cpu_start = perf_counter(), process_time()
                    row = dict(instance_id=path.stem, seed=seed, status='ok', error='')
                    try:
                        kwargs = {k: v for k, v in options.items() if k != 'seed_base'}
                        edges = generate_onion_graph(target_deg=graph.degree(), seed=seed, **kwargs)
                        constructed = ig.Graph(n=graph.vcount(), edges=edges, directed=False)
                        if constructed.degree() != graph.degree() or not constructed.is_simple():
                            raise ValueError('Onion violated degree/simple-graph constraints')
                        directory = output / 'final_graphs' / path.stem
                        original = directory / 'Original.txt'
                        if original.exists():
                            previous = load_graph(original)
                            if previous.vcount() != graph.vcount() or sorted(previous.get_edgelist()) != sorted(graph.get_edgelist()):
                                raise ValueError('Saved Original differs from Onion input')
                        else:
                            save_igraph_edgelist(graph, original)
                        final = directory / 'Optimized_onion.txt'
                        save_igraph_edgelist(constructed, final)
                    except Exception as exc:
                        row.update(status='error', error=str(exc))
                        raise
                    finally:
                        row.update(wall_seconds=perf_counter() - start, cpu_seconds=process_time() - cpu_start)
                        with CSVLog(task.directory / 'timings.csv', FIELDS) as log:
                            log.write(row)
                    task.finish(row, [original, final, task.directory / 'timings.csv'])
        finally:
            store.export('onion_summary.csv', FIELDS, 'timings/onion.csv', FIELDS)
