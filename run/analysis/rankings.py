"""Rank saved, complete paired experiments; never run attacks from analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import input_files, project_path
from ..experiments.timing import StageRun
from .summarize import read_observations
from .statistics import describe


def _numerical_ranks(values, ascending=True):
    """Competition ranks, coalescing only floating-point arithmetic noise.

    R lies in [0,1]. This tolerance is about 3.6e-15, not display rounding or
    a statistical equivalence threshold. Compare each group with its first value.
    """
    ranks = pd.Series(index=values.index, dtype=int)
    anchor, rank = None, 0
    for position, (index, value) in enumerate(values.sort_values(ascending=ascending).items(), 1):
        if anchor is None or abs(value - anchor) > 16 * np.finfo(float).eps:
            anchor, rank = value, position
        ranks.loc[index] = rank
    return ranks.astype(int)


def _rank(values, key, ascending):
    records = []
    for label, group in values.groupby(key):
        records.append(dict({key: label}, **describe(group.sort_values('instance_id')['R'].values)))
    table = pd.DataFrame(records)
    for metric in ('rho_05', 'rho_01'):
        if metric in values:
            extra = values.groupby(key)[metric].mean().rename('mean_' + metric).reset_index()
            table = table.merge(extra, on=key, validate='one_to_one')
    table['rank'] = _numerical_ranks(table['mean_R'], ascending)
    return table.sort_values(['rank', key]).reset_index(drop=True)


def ranking_tables(observations, designs, attacks, directory, expected_instances=None):
    """Equal weight per graph, attack and design, with complete blocks required.

    Original and defended attack rankings: smaller R means stronger attack.
    Defense-target ranking: larger mean R across the same attacks means better
    defense. Standard deviations/counts describe input instances; attack/design
    cells are never treated as additional replicates.
    """
    directory.mkdir(parents=True, exist_ok=True)
    instances = (list(expected_instances) if expected_instances is not None
                 else sorted(observations['instance_id'].unique()))
    if not instances or 'Original' not in designs:
        raise ValueError('Rankings require original graphs and at least one input instance')
    keys = ['instance_id', 'design', 'attack']
    expected = {(i, d, a) for i in instances for d in designs for a in attacks}
    actual = set(observations[keys].itertuples(index=False, name=None))
    if (observations.duplicated(keys).any() or actual != expected
            or not np.isfinite(observations['R']).all()):
        raise ValueError('Ranking requires a complete finite instance/design/attack matrix; '
                         'missing={}, unexpected={}'.format(len(expected-actual), len(actual-expected)))

    per_instance = observations.copy()
    per_instance['rank'] = per_instance.groupby(['instance_id', 'design'])['R'].transform(_numerical_ranks)
    original = per_instance[per_instance['design'] == 'Original'].copy()
    original_rank = _rank(original, 'attack', True)
    by_design = pd.concat([
        _rank(observations[observations['design'] == design], 'attack', True).assign(design=design)
        for design in designs], ignore_index=True)
    by_design = by_design.merge(original_rank[['attack', 'rank']].rename(
        columns={'rank': 'original_attack_rank'}), on='attack', validate='many_to_one')
    # Positive change means a stronger attack rank; ties use competition ranks.
    by_design['attack_rank_change'] = by_design['original_attack_rank'] - by_design['rank']
    tables = {'original_attack_ranking': original_rank,
              'original_attack_effectiveness': original_rank.copy(),
              'original_attack_per_instance': original,
              'attack_ranking_by_design': by_design,
              'attack_ranking_per_instance': per_instance}
    optimized = [d for d in designs if d != 'Original']
    if optimized:
        defended = observations[observations['design'].isin(optimized)]
        # Average over design/attack inside each graph before estimating the SD.
        attack_per_graph = defended.groupby(['instance_id', 'attack'])['R'].mean().reset_index()
        attack_rank = _rank(attack_per_graph, 'attack', True)
        attack_rank['n_defense_designs'] = len(optimized)
        tables['defended_attack_ranking'] = attack_rank

        defense_per_graph = defended.groupby(['instance_id', 'design'])['R'].mean().reset_index()
        baseline = original.groupby('instance_id')['R'].mean()
        defense_per_graph['original_mean_R'] = defense_per_graph['instance_id'].map(baseline)
        defense_per_graph['delta_R'] = defense_per_graph['R'] - defense_per_graph['original_mean_R']
        defense_rank = _rank(defense_per_graph, 'design', False)
        defense_rank = defense_rank.merge(
            defense_per_graph.groupby('design')['delta_R'].mean().rename('mean_delta_R').reset_index(),
            on='design', validate='one_to_one')
        defense_rank['original_mean_R'] = baseline.mean()
        defense_rank['n_evaluation_attacks'] = len(attacks)
        tables['defense_ranking'] = defense_rank
        tables['defense_per_instance'] = defense_per_graph
        tables['defended_attack_per_instance'] = attack_per_graph

        comparison = tables['original_attack_ranking'][['attack', 'rank', 'mean_R']].rename(
            columns={'attack': 'method', 'rank': 'original_attack_rank', 'mean_R': 'original_R'})
        comparison = comparison.merge(
            attack_rank[['attack', 'rank', 'mean_R']].rename(columns={
                'attack': 'method', 'rank': 'defended_attack_rank', 'mean_R': 'defended_attack_R'}),
            on='method', validate='one_to_one')
        comparison = comparison.merge(
            defense_rank[['design', 'rank', 'mean_R', 'mean_delta_R']].rename(columns={
                'design': 'method', 'rank': 'defense_rank', 'mean_R': 'defense_mean_R'}),
            on='method', how='left', validate='one_to_one')
        comparison['attack_rank_change'] = comparison['original_attack_rank'] - comparison['defended_attack_rank']
        comparison['attack_to_defense_rank_change'] = comparison['original_attack_rank'] - comparison['defense_rank']
        comparison['n_evaluation_attacks'] = len(attacks)
        comparison['n_defense_designs'] = len(optimized)
        tables['ranking_comparison'] = comparison

    for name, table in tables.items():
        table.to_csv(directory / (name + '.csv'), index=False)
    return tables


def observation_groups(observations, config):
    """Keep registered real networks separate; a synthetic batch is one setting.

    Explicit instance grouping also supports unregistered heterogeneous inputs.
    Use the configured input list to detect entirely missing instances.
    """
    instances = [p.stem for p in input_files(config)]
    mode = config['analysis'].get('group_by', 'auto')
    labels = {}
    catalog = project_path('data/catalog.csv')
    if mode == 'auto' and catalog.is_file():
        from pathlib import Path
        for row in pd.read_csv(catalog).itertuples():
            stem = Path(row.path).stem
            for instance in instances:
                if instance == stem or (instance.startswith(stem + '_') and instance[len(stem)+1:].isdigit()):
                    labels[instance] = row.dataset
        if labels and len(labels) != len(instances):
            raise ValueError('Mixed registered/unregistered inputs; set analysis.group_by="instance"')
        if not labels and 'n_nodes' in observations and observations['n_nodes'].nunique() > 1:
            labels = {i: i for i in instances}
    if mode == 'instance':
        labels = {i: i for i in instances}
    if not labels:
        return [('', observations, instances)]
    unexpected = set(observations['instance_id']) - set(instances)
    if unexpected:
        raise ValueError('Unexpected evaluation instances: {}'.format(sorted(unexpected)))
    groups = []
    for label in sorted(set(labels.values())):
        selected = [i for i in instances if labels[i] == label]
        groups.append((label, observations[observations['instance_id'].isin(selected)], selected))
    return groups


def collect_groups(groups, directory):
    """Write combined tables with an explicit network column, never pooled ranks."""
    if not groups or groups[0][0] == '':
        return
    for name in sorted(set().union(*(set(tables) for _, tables in groups))):
        records = []
        for network, tables in groups:
            if name in tables:
                table = tables[name].copy()
                table.insert(0, 'network', network)
                records.append(table)
        pd.concat(records, ignore_index=True).to_csv(directory / (name + '.csv'), index=False)


def ranking_plot(tables, directory, labels=None):
    from .plotting import plt, save_figure, _sd_bars
    from .paper_style import display_name
    specs = [('original_attack_ranking', 'attack', 'Original: attack effectiveness (lower R is stronger)')]
    if 'defense_ranking' in tables:
        specs += [('defended_attack_ranking', 'attack', 'After defense: attack effectiveness (lower is stronger)'),
                  ('defense_ranking', 'design', 'Defense target: mean robustness (higher R is better)')]
    fig, axes = plt.subplots(1, len(specs), figsize=(7*len(specs), 4.3), squeeze=False)
    for ax, (name, label, title) in zip(axes[0], specs):
        table = tables[name]
        positions = np.arange(len(table))
        _sd_bars(ax, table, positions, horizontal=True, color='#4477aa')
        ax.set_yticks(positions)
        ax.set_yticklabels(['{}. {}'.format(int(r), display_name(m, labels))
                            for r, m in zip(table['rank'], table[label])])
        ax.invert_yaxis()
        ax.set(xlabel='Mean R (error bars: SD across instances)', title=title)
        ax.grid(axis='x', alpha=.2)
        if name == 'defense_ranking':
            ax.axvline(table['original_mean_R'].iloc[0], color='#cc6677', ls='--', label='Original mean R')
            ax.legend(fontsize=8)
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    save_figure(fig, directory / 'rankings')


def run(config):
    from .paper_style import method_labels
    labels = method_labels(config.get('attack_options'))
    root = project_path(config['output_dir'])
    observations, designs, attacks = read_observations(root)
    with StageRun(root, 'rank', config):
        groups = []
        for name, subset, instances in observation_groups(observations, config):
            tables = ranking_tables(subset, designs, attacks, root / 'tables' / name, instances)
            ranking_plot(tables, root / 'figures' / name, labels=labels)
            groups.append((name, tables))
        collect_groups(groups, root / 'tables')
    print('Rankings saved to {}'.format(root / 'tables'), flush=True)


def collect_families(root, families, original_only=False):
    """Keep BA/ER/WS separate in combined CSVs; do not pool their rankings."""
    root = project_path(root)
    names = ['original_attack_ranking', 'original_attack_effectiveness', 'original_attack_per_instance']
    names += ['cross_attack_summary', 'paired_comparisons']
    if not original_only:
        names += ['defended_attack_ranking', 'defense_ranking', 'ranking_comparison',
                  'attack_ranking_by_design', 'attack_ranking_per_instance']
    defense_only = {'defended_attack_ranking', 'defense_ranking', 'ranking_comparison'}
    (root / 'tables').mkdir(parents=True, exist_ok=True)
    for name in names:
        records = []
        for family in families:
            directory = root / family / 'original' if original_only else root / family
            path = directory / 'tables' / (name + '.csv')
            if not path.is_file() and name in defense_only:
                by_design = pd.read_csv(directory / 'tables/attack_ranking_by_design.csv')
                if set(by_design['design']) == {'Original'}:
                    continue
            table = pd.read_csv(path)
            if 'network' in table:
                table = table.rename(columns={'network': 'subgroup'})
            table.insert(0, 'network', family)
            records.append(table)
        if records:
            pd.concat(records, ignore_index=True).to_csv(root / 'tables' / (name + '.csv'), index=False)
