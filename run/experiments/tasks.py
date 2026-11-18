"""Task-level restart, checked artifacts and atomic aggregate CSVs."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from ..config import PROJECT_ROOT


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def protocol_hash():
    """Computation/model changes invalidate reuse; plotting changes do not."""
    paths = [PROJECT_ROOT / 'run/reproducibility.py', PROJECT_ROOT / 'run/config.py',
             PROJECT_ROOT / 'run/datasets/io.py']
    for directory in ('attacks', 'optimization', 'experiments'):
        paths += list((PROJECT_ROOT / 'run' / directory).glob('*.py'))
    vendor = PROJECT_ROOT / 'run/vendor/FINDER_ND'
    paths += list(vendor.glob('*.so'))
    paths += list((vendor / 'models').glob('nrange_30_50_iter_78000.ckpt*'))
    return hashlib.sha256(json.dumps([(str(p.relative_to(PROJECT_ROOT)), digest(p))
                                     for p in sorted(paths)]).encode()).hexdigest()


class Task:
    def __init__(self, directory, state):
        self.directory, self.state = directory, state

    def finish(self, row, artifacts):
        write_json(self.directory / 'row.json', row)
        artifacts = list(artifacts) + [self.directory / 'row.json']
        self.state.update(status='completed', artifacts={str(Path(p).resolve()): digest(p) for p in artifacts})
        write_json(self.directory / 'state.json', self.state)


class TaskStore:
    """Use while holding the stage lock. Failed tasks restart from their seed."""
    def __init__(self, output, stage, resume=True):
        self.output, self.stage, self.resume = Path(output), stage, resume
        self.root = self.output / 'tasks' / stage
        legacy = {'optimize': 'sa_summary.csv', 'evaluate': 'evaluations.csv',
                  'onion': 'timings/onion.csv'}[stage]
        if (self.output / legacy).exists() and not self.root.exists():
            raise ValueError('Results predate task checkpoints; use a new output directory: {}'.format(self.output))
        self.protocol = protocol_hash()
        from .timing import package_versions
        self.runtime = dict(python=list(sys.version_info[:3]), packages=package_versions())
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def task(self, key, specification):
        path = self.root.joinpath(*key)
        specification = dict(specification, protocol=self.protocol, runtime=self.runtime, key=list(key))
        state_file = path / 'state.json'
        if state_file.exists():
            state = json.loads(state_file.read_text())
            if state['specification'] != specification:
                raise ValueError('Task inputs/config/code changed; choose a new output: {}'.format(path))
            if not self.resume:
                raise FileExistsError('Task already exists and resume=false: {}'.format(path))
            if state['status'] == 'completed':
                for filename, checksum in state['artifacts'].items():
                    if not Path(filename).is_file() or digest(filename) != checksum:
                        raise ValueError('Completed task artifact missing/changed: ' + filename)
                yield None
                return
        if path.exists():
            history = self.root / '.attempts'
            history.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            path.rename(history / ('__'.join(key) + '_' + stamp))
        path.mkdir(parents=True)
        state = dict(status='running', specification=specification)
        write_json(path / 'state.json', state)
        task = Task(path, state)
        try:
            yield task
            if task.state['status'] != 'completed':
                raise RuntimeError('Task exited without completing: {}'.format(path))
        except BaseException as exc:
            state.update(status='error', error='{}: {}'.format(type(exc).__name__, exc))
            write_json(path / 'state.json', state)
            raise

    def export(self, row_file, row_fields, timing_file, timing_fields):
        states = sorted(p for p in self.root.rglob('state.json') if '.attempts' not in p.parts)
        rows, timing_paths = [], []
        for path in states:
            state = json.loads(path.read_text())
            if state['status'] == 'completed':
                rows.append(json.loads((path.parent / 'row.json').read_text()))
            if (path.parent / 'timings.csv').exists():
                timing_paths.append(path.parent / 'timings.csv')

        def write_csv(destination, fields, records):
            destination = self.output / destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix('.csv.tmp')
            with temporary.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            temporary.replace(destination)

        def timings():
            for path in timing_paths:
                with path.open() as stream:
                    yield from csv.DictReader(stream)

        write_csv(row_file, row_fields, rows)
        if timing_file is not None:
            write_csv(timing_file, timing_fields, timings())
