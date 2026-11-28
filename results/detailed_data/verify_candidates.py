"""Recompute the BA500 feedback counts and trajectory checkpoints from the recorded proposals."""
import csv
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np


HERE = Path(__file__).resolve().parent
FEEDBACK = HERE.parents[1] / 'reproduction' / 'diagnostics' / 'feedback'


def read_groups(filename):
    groups = {}
    with (FEEDBACK / filename).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            objective = 'SRA' if row['attack'] == 'Zheng' else row['attack']
            groups.setdefault((row['instance_id'], objective), []).append(row)
    return groups


def verify():
    groups = read_groups('feedback_by_instance.csv')
    checkpoints = read_groups('trajectory_O.csv')
    total = 0
    with ZipFile(HERE / 'BA500_candidates.zip') as archive:
        expected = {f'{graph}/{method}.npz' for graph, method in groups}
        if set(archive.namelist()) != expected or len(expected) != 350:
            raise ValueError('Candidate archive does not match the 350 recorded trajectories.')
        for (graph, method), rows in groups.items():
            with np.load(io.BytesIO(archive.read(f'{graph}/{method}.npz')), allow_pickle=False) as data:
                # Use signed integers before subtracting the unsigned archive values.
                layer = data['candidate_sum_S'].astype('int64') - data['base_sum_S'][:, None]
                area = data['candidate_sum_LCC'].astype('int64') - data['base_sum_LCC'][:, None]
                valid = data['valid_swap']
                total += int(valid.sum())
                masks = {'O_plus': layer > 0, 'O_zero': layer == 0, 'O_minus': layer < 0}
                for row in rows:
                    selected = valid & masks[row['stratum']]
                    values = area[selected]
                    measured = {
                        'n_candidates': selected.sum(), 'n_R_plus': (values > 0).sum(),
                        'n_R_zero': (values == 0).sum(), 'n_R_minus': (values < 0).sum(),
                        'n_accepted': data['accepted'][selected].sum()
                    }
                    for key, count in measured.items():
                        if int(row[key]) != count:
                            raise ValueError(f"{graph}/{method} {row['stratum']}: {key}")
                for row in checkpoints[(graph, method)]:
                    i = int(row['iteration'])
                    if data['best_sum_S'][i] != int(row['sum_S']):
                        raise ValueError(f'{graph}/{method} layer statistic at iteration {i}')
                    if not np.isclose(data['best_sum_LCC'][i] / 500 ** 2, float(row['R']), rtol=0, atol=1e-12):
                        raise ValueError(f'{graph}/{method} robustness at iteration {i}')
    print(f'Verified {total:,} valid candidates on {len(groups)} optimization trajectories.')


if __name__ == '__main__':
    verify()
