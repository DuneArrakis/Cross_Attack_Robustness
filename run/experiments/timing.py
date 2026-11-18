"""Timing records only: no budget enforcement or changes to SA decisions."""
from __future__ import annotations

import csv
import fcntl
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Existing FINDER environment uses Python 3.7.
    from importlib_metadata import version, PackageNotFoundError

ATTACK_FIELDS = [
    'instance_id', 'design', 'attack', 'stage', 'iteration', 'candidate',
    'n_nodes', 'n_edges', 'R', 'wall_seconds', 'cpu_seconds',
    'n_attack_rollouts', 'n_linear_solves', 'sigma', 'search_seconds',
    'solve_seconds', 'spectral_seconds', 'first_call_in_process', 'pid',
    'status', 'error', 'options_json',
    'optimization_seed', 'proposal_seed', 'attack_seed', 'proposal_changed',
]
_SEEN = set()


def package_versions():
    versions = {}
    for name in ('igraph', 'numpy', 'scipy', 'networkx', 'pandas', 'matplotlib', 'tensorflow'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            # Source/egg installations (such as the existing TensorFlow) may
            # have no distribution metadata. Inspect only already loaded modules.
            if name == 'tensorflow':
                try:
                    versions[name] = version('tensorflow-gpu')
                    continue
                except PackageNotFoundError:
                    pass
            versions[name] = getattr(sys.modules.get(name), '__version__', None)
    return versions


def measure_attack(graph, method, options=None, **context):
    """Return (result, timing). Failed calls are recorded and raised by parent."""
    from ..attacks import evaluate_attack
    options = options or {}
    first = (os.getpid(), method) not in _SEEN
    _SEEN.add((os.getpid(), method))
    row = dict(context, attack=method, n_nodes=graph.vcount(), n_edges=graph.ecount(),
               first_call_in_process=first, pid=os.getpid(), status='ok', error='',
               options_json=json.dumps(options, sort_keys=True))
    start, cpu_start = perf_counter(), process_time()
    result = None
    try:
        result = evaluate_attack(graph, method, seed=context.get('attack_seed'), **options)
    except Exception as exc:
        row.update(status='error', error='{}: {}'.format(type(exc).__name__, exc))
    finally:
        row.update(wall_seconds=perf_counter() - start, cpu_seconds=process_time() - cpu_start)
    if result is not None:
        row['R'] = result.robustness
        for key in ('n_attack_rollouts', 'n_linear_solves', 'sigma', 'search_seconds',
                    'solve_seconds', 'spectral_seconds'):
            row[key] = result.metadata.get(key, '' if key == 'sigma' else 0)
        result.metadata.update(wall_seconds=row['wall_seconds'], cpu_seconds=row['cpu_seconds'],
                               first_call_in_process=first, pid=row['pid'])
    return result, row


def require_success(result, row):
    if result is None:
        raise RuntimeError('{} | {} | {}: {}'.format(
            row.get('instance_id', ''), row['stage'], row['attack'], row['error']))
    return result


class CSVLog:
    """One parent-process writer per stage. Workers return records, never write."""
    def __init__(self, path, fields):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('w', encoding='utf-8', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.file.flush()

    def write(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()


class StageRun:
    """Save resolved config, status and total elapsed time for one command."""
    def __init__(self, output_dir, stage, config):
        self.root = Path(output_dir)
        self.stage = stage
        self.record = dict(stage=stage, config=config, status='running',
                           started_utc=datetime.now(timezone.utc).isoformat(),
                           python=sys.version, executable=sys.executable,
                           package_versions=package_versions(),
                           platform=platform.platform(), machine=platform.machine(),
                           processor=platform.processor(), logical_cpus=mp.cpu_count(),
                           thread_environment={key: os.environ.get(key) for key in
                               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS',
                                'TF_NUM_INTEROP_THREADS', 'TF_NUM_INTRAOP_THREADS',
                                'RESS_FINDER_THREADS', 'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED')},
                           allowed_cpus=sorted(os.sched_getaffinity(0))
                           if hasattr(os, 'sched_getaffinity') else None,
                           pid=os.getpid())

    def _save(self):
        path = self.root / 'run_metadata' / (self.stage + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def __enter__(self):
        lock_path = self.root / 'run_metadata' / (self.stage + '.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = lock_path.open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError('Stage already running: {}'.format(self.stage))
        previous = lock_path.with_suffix('.json')
        if previous.exists():
            history = previous.parent / 'history'
            history.mkdir(exist_ok=True)
            previous.replace(history / (self.stage + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'))
        self._save()
        self.start, self.cpu_start = perf_counter(), process_time()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.record.update(status='error' if exc else 'completed',
                           package_versions=package_versions(),
                           wall_seconds=perf_counter() - self.start,
                           parent_cpu_seconds=process_time() - self.cpu_start,
                           finished_utc=datetime.now(timezone.utc).isoformat())
        if exc:
            self.record['error'] = '{}: {}'.format(exc_type.__name__, exc)
        try:
            self._save()
        finally:
            self.lock.close()
