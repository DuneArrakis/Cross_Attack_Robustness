"""Aggregate saved observations. Never invoke an attack or SA from analysis."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ..artifacts import resolve_artifact


def read_observations(root):
    metadata = json.loads((root / 'run_metadata/evaluate.json').read_text())
    if metadata['status'] != 'completed':
        raise ValueError('Evaluation is incomplete; finish it before analyzing this directory')
    observations = pd.read_csv(root / 'evaluations.csv', dtype={'instance_id': str})
    if observations.empty:
        raise ValueError('No evaluation observations')
    if observations.duplicated(['instance_id', 'design', 'attack']).any():
        raise ValueError('Duplicate instance/design/attack observations')
    if not np.isfinite(observations['R']).all():
        raise ValueError('Non-finite R in evaluation observations')
    saved = metadata['config']
    if 'graph_path' in observations:
        observations['graph_path'] = observations['graph_path'].map(
            lambda value: str(resolve_artifact(value, root, saved['output_dir'])))
    designs = saved['evaluation']['graph_types']
    if saved['evaluation']['source'] == 'inputs':
        designs = ['Original']
    attacks = saved['evaluation_attacks']
    return observations, designs, attacks


def _matrix(table, value, designs, attacks, operation):
    if table.empty:
        return pd.DataFrame(np.nan, index=designs, columns=attacks)
    grouped = table.groupby(['design', 'attack'])[value]
    result = getattr(grouped, operation)().unstack('attack')
    return result.reindex(index=designs, columns=attacks)


def robustness_tables(observations, designs, attacks, directory):
    """Retain the old per-instance R/IRB/CAD formulas and denominator guard."""
    directory.mkdir(parents=True, exist_ok=True)
    mean = _matrix(observations, 'R', designs, attacks, 'mean')
    mean.to_csv(directory / 'R_mean.csv', index_label='design')
    _matrix(observations, 'R', designs, attacks, 'count').fillna(0).astype(int).to_csv(
        directory / 'R_count.csv', index_label='design')
    records = []
    lookup = {(r.instance_id, r.design, r.attack): r.R for r in observations.itertuples()}
    for row in observations.itertuples():
        if row.design == 'Original':
            continue
        original = lookup.get((row.instance_id, 'Original', row.attack), np.nan)
        matched = lookup.get((row.instance_id, row.design, row.design), np.nan)
        records.append(dict(instance_id=row.instance_id, design=row.design, attack=row.attack,
                            IRB=(row.R-original) / max(original, 1e-10),
                            CAD=(matched-row.R) / max(matched, 1e-10)))
    paired = pd.DataFrame(records, columns=['instance_id', 'design', 'attack', 'IRB', 'CAD'])
    paired.to_csv(directory / 'paired_metrics.csv', index=False)
    optimized = [design for design in designs if design != 'Original']
    for metric in ('IRB', 'CAD'):
        values = _matrix(paired, metric, optimized, attacks, 'mean')
        values.to_csv(directory / (metric + '_mean.csv'), index_label='design')
        _matrix(paired, metric, optimized, attacks, 'count').fillna(0).astype(int).to_csv(
            directory / (metric + '_count.csv'), index_label='design')
        if metric == 'CAD':
            # Old Avg_CAD includes the zero diagonal. Average per instance first.
            if paired.empty:
                avg = pd.DataFrame(index=optimized, columns=['mean', 'count'], dtype=float)
            else:
                avg = (paired.groupby(['instance_id', 'design'])['CAD'].mean()
                       .groupby('design').agg(['mean', 'count']).reindex(optimized))
            avg.rename(columns={'mean': 'Avg_CAD', 'count': 'n_instances'}).to_csv(
                directory / 'CAD_avg.csv', index_label='design')
    return mean


def curve_tables(root, observations, points, directory):
    grid = np.linspace(0, 1, points)
    groups = {}
    for row in observations.itertuples():
        raw = pd.read_csv(root / row.curve_path)
        if (len(raw) != row.n_nodes + 1 or
                not np.array_equal(raw['q'].values, np.arange(row.n_nodes + 1))):
            raise ValueError('Invalid raw curve: ' + row.curve_path)
        raw_R = raw['R'].iloc[1:].sum() / row.n_nodes if row.n_nodes else 0.0
        if not np.isclose(raw_R, row.R, rtol=0, atol=1e-12):
            raise ValueError('Curve/R mismatch: ' + row.curve_path)
        values = np.interp(grid, raw['x'].values, raw['R'].values)
        groups.setdefault((row.attack, row.design), []).append(values)
    records = []
    for (attack, design), curves in groups.items():
        mean = np.mean(curves, axis=0)
        records.extend(dict(attack=attack, design=design, x=x, lcc_fraction=y, count=len(curves))
                       for x, y in zip(grid, mean))
    result = pd.DataFrame(records)
    result.to_csv(directory / 'mean_curves.csv', index=False)
    return result


_RUNTIME_CHUNK_SIZE = 100000


def runtime_tables(root, directory):
    # Formal SA logs contain millions of rows. Keep only one chunk of the eight
    # required columns; retain wall-time floats alone for the exact median.
    read_columns = ['stage', 'attack', 'status', 'first_call_in_process',
                    'wall_seconds', 'cpu_seconds', 'n_attack_rollouts', 'n_linear_solves']
    totals = {}
    for filename in ('optimization_attacks.csv', 'evaluation_attacks.csv'):
        path = root / 'timings' / filename
        if path.is_file():
            for chunk in pd.read_csv(path, usecols=read_columns, chunksize=_RUNTIME_CHUNK_SIZE):
                for key, group in chunk.groupby(['stage', 'attack']):
                    if key not in totals:
                        totals[key] = dict(calls=0, failed_calls=0, first_calls=0,
                                           wall_seconds_sum=0.0, cpu_seconds_sum=0.0,
                                           n_attack_rollouts=0, n_linear_solves=0,
                                           wall_values=[], wall_count=0)
                    total = totals[key]
                    total['calls'] += len(group)
                    total['failed_calls'] += int((group['status'] != 'ok').sum())
                    total['first_calls'] += int(group['first_call_in_process'].astype(str)
                                                .str.lower().eq('true').sum())
                    for field in ('wall_seconds', 'cpu_seconds'):
                        total[field + '_sum'] += group[field].sum()
                    for field in ('n_attack_rollouts', 'n_linear_solves'):
                        total[field] += group[field].sum()
                    # Copy a single numeric column, so slices cannot retain the
                    # complete chunk. Missing times follow pandas skipna rules.
                    wall = group['wall_seconds'].dropna().values.astype(float, copy=True)
                    if len(wall):
                        total['wall_values'].append(wall)
                        total['wall_count'] += len(wall)
    columns = ['stage', 'attack', 'calls', 'failed_calls', 'first_calls',
               'wall_seconds_sum', 'wall_seconds_mean', 'wall_seconds_median',
               'cpu_seconds_sum', 'n_attack_rollouts', 'n_linear_solves']
    summary = []
    for stage, attack in sorted(totals):
        total = totals[(stage, attack)]
        wall_values, count = total.pop('wall_values'), total.pop('wall_count')
        summary.append(dict(
            stage=stage, attack=attack,
            wall_seconds_mean=total['wall_seconds_sum']/count if count else np.nan,
            wall_seconds_median=float(np.median(np.concatenate(wall_values))) if count else np.nan,
            **total))
    result = pd.DataFrame(summary, columns=columns)
    result.to_csv(directory / 'attack_runtime.csv', index=False)
    stages = []
    for path in sorted((root / 'run_metadata').glob('*.json')):
        if path.stem == 'analyze':
            continue
        metadata = json.loads(path.read_text())
        stages.append({key: metadata.get(key) for key in
                       ('stage', 'status', 'wall_seconds', 'parent_cpu_seconds', 'started_utc', 'finished_utc')})
    pd.DataFrame(stages).to_csv(directory / 'stage_runtime.csv', index=False)
    return result
