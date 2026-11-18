"""Plots from aggregate tables; no experiment computation or fixed method count."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .paper_style import (STYLE, ROBUSTNESS_CMAP, BLUE, GREEN, METRICS,
                          panel_label, topology_panels, display_name)


def save_figure(fig, path, pdf=False):
    fig.savefig(str(path) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(path) + '.svg', bbox_inches='tight')
    if pdf:
        fig.savefig(str(path) + '.pdf', bbox_inches='tight')
    plt.close(fig)


def _robustness_heatmap(ax, matrix, labels=None):
    values = matrix.values.astype(float)
    heat = ax.imshow(np.ma.masked_invalid(values), cmap=ROBUSTNESS_CMAP,
                     aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([display_name(m, labels) for m in matrix.columns], rotation=0)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([display_name(m, labels) for m in matrix.index])
    ax.set_xlabel('Test attack')
    ax.set_ylabel('Assumed attack in optimization')
    ax.set_xticks(np.arange(-.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(matrix.index), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.4)
    ax.tick_params(which='minor', bottom=False, left=False)
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, '{:.3f}'.format(values[i, j]) if np.isfinite(values[i, j]) else '—',
                ha='center', va='center', color='black', fontsize=10)
    return heat


def robustness(matrix, directory, labels=None):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(6, .9 * len(matrix.columns)), max(3, .6 * len(matrix))))
        fig.colorbar(_robustness_heatmap(ax, matrix, labels), ax=ax, label=r'$R$')
        save_figure(fig, directory / 'R_mean')


def _sd_bars(ax, table, x, horizontal=False, **kwargs):
    """A single instance has no estimated SD; do not draw zero-length error bars."""
    bar = ax.barh if horizontal else ax.bar
    bar(x, table['mean_R'].values, edgecolor='black', linewidth=.9, zorder=3, **kwargs)
    valid = np.isfinite(table['std_R'].values) & (table['n_instances'].values > 1)
    if valid.any():
        position, value = np.asarray(x)[valid], table['mean_R'].values[valid]
        error = {'xerr' if horizontal else 'yerr': table['std_R'].values[valid]}
        ax.errorbar(value if horizontal else position, position if horizontal else value,
                    fmt='none', ecolor='#303030',
                    capsize=2, lw=.8, zorder=4, **error)


def robustness_comparison(matrix, tables, directory, labels=None):
    """Fig. 4: original R in (a), attack-to-defense rank comparison in (b).

    A ranks attacks on original graphs (lower R); D ranks optimization targets
    by their mean R under the common full evaluation set (higher R).
    """
    if 'defense_ranking' not in tables:
        return
    original = tables['original_attack_ranking'].set_index('attack')
    defense = tables['defense_ranking'].set_index('design')
    order = [d for d in defense.sort_values('mean_R', ascending=False, kind='mergesort').index
             if d in matrix.index]
    # Original is a reference, not an eighth optimization target or defense rank.
    if 'Original' in matrix.index:
        order.insert(0, 'Original')
    values = defense.reindex(order).copy()
    if 'Original' in order:
        per_graph = tables['original_attack_per_instance'].groupby('instance_id')['R'].mean()
        values.loc['Original', ['mean_R', 'std_R', 'n_instances']] = [
            per_graph.mean(), per_graph.std(ddof=1), len(per_graph)]
    baseline = original.reindex(order)
    valid = baseline['mean_R'].notna().values
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15.6, 5.8))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.08, 1], wspace=.30)
        left_grid = outer[0, 0].subgridspec(1, 2, width_ratios=[1, .045], wspace=.16)
        left = fig.add_subplot(left_grid[0, 0])
        colorbar = fig.add_subplot(left_grid[0, 1])
        right = fig.add_subplot(outer[0, 1])
        fig.subplots_adjust(left=.07, right=.99, bottom=.12, top=.96)
        heat = _robustness_heatmap(left, matrix.loc[order], labels)
        fig.colorbar(heat, cax=colorbar, label=r'$R$')
        panel_label(left, 'a', outside=True)
        panel_label(right, 'b', outside=True)
        x = np.arange(len(order), dtype=float)
        blue_x = x - .18
        blue_x[~valid] = x[~valid]
        _sd_bars(right, values, blue_x, width=.34, color=BLUE, label='All attacks (D)')
        _sd_bars(right, baseline.loc[valid], x[valid] + .18, width=.34,
                 color=GREEN, label='Original (A)')
        extent = 0.0
        for table, positions, role, shade in (
                (values, blue_x, 'D', '#dcecf8'),
                (baseline.loc[valid], x[valid]+.18, 'A', '#e9f0df')):
            for position, row in zip(positions, table.itertuples()):
                end = row.mean_R + (row.std_R if np.isfinite(row.std_R) else 0.0)
                extent = max(extent, end)
                label = '{:.3f}'.format(row.mean_R)
                if np.isfinite(row.rank):
                    label += '\n{}{}'.format(role, int(row.rank))
                right.annotate(label, (position, end), xytext=(0, 4),
                               ha='center', va='bottom', textcoords='offset points',
                               fontsize=8.5, linespacing=1.1,
                               bbox=dict(facecolor=shade, edgecolor='none', pad=1.5))
        right.set_xticks(x)
        right.set_xticklabels([display_name(m, labels) for m in order])
        right.set(ylabel=r'$R$', xlim=(-.6, len(order)-.4), ylim=(0, max(.01, extent*1.24)))
        right.set_axisbelow(True)
        right.grid(axis='y', color='#d9d9d9', ls='--', alpha=.7)
        right.legend(loc='upper right', frameon=True, fancybox=False,
                     edgecolor='black', framealpha=1, fontsize=10)
        save_figure(fig, directory / 'R_mean_combined_figure', pdf=True)


def attack_rank_changes(tables, directory, labels=None):
    """Compare the SAME attack ranking on each alternative topology, not defense ranks."""
    table = tables['attack_ranking_by_design']
    designs = list(table['design'].drop_duplicates())
    if len(designs) < 2:
        return
    methods = list(tables['original_attack_ranking']['attack'])
    ranks = table.pivot(index='attack', columns='design', values='rank').loc[methods, designs]
    changes = table.pivot(index='attack', columns='design', values='attack_rank_change').loc[methods, designs]
    fig, ax = plt.subplots(figsize=(max(6, 1.05 * len(designs)), max(4, .55 * len(methods))))
    heat = ax.imshow(ranks.values, cmap='viridis_r', vmin=1, vmax=max(2, len(methods)), aspect='auto')
    ax.set_xticks(np.arange(len(designs)))
    ax.set_xticklabels([display_name(m, labels) for m in designs], rotation=35, ha='right')
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([display_name(m, labels) for m in methods])
    for i, j in np.ndindex(ranks.shape):
        label = '#{}'.format(int(ranks.iloc[i, j]))
        if designs[j] != 'Original':
            label += '\n({:+d})'.format(int(changes.iloc[i, j]))
        ax.text(j, i, label, ha='center', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=.75, edgecolor='none', pad=1))
    ax.set(xlabel='Graph design used for attack evaluation', ylabel='Evaluation attack',
           title='Attack-effectiveness ranks across graph designs\nRank 1: lowest mean R; (+): rank improves relative to Original')
    fig.colorbar(heat, ax=ax, label='Attack rank', ticks=range(1, len(methods) + 1))
    fig.tight_layout()
    save_figure(fig, directory / 'attack_rank_changes', pdf=True)


def curves(table, designs, attacks, show_ratio, directory, labels=None):
    colors = dict(zip(designs, plt.cm.tab10(np.linspace(0, .9, max(len(designs), 1)))))
    for attack in attacks:
        subset = table[table['attack'] == attack]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for design in designs:
            rows = subset[subset['design'] == design].sort_values('x')
            if not rows.empty:
                ax.plot(rows['x'], rows['lcc_fraction'], color=colors[design],
                        label=display_name(design, labels))
        ax.set(xlim=(0, show_ratio), ylim=(0, 1.02), xlabel='Removed fraction q/N',
               ylabel='LCC / initial N', title=display_name(attack, labels))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        save_figure(fig, directory / ('curve_' + attack))


def runtime(table, directory, labels=None):
    # Evaluation calls only; SA summaries separately retain whole-task wall time.
    subset = table[table['stage'] == 'evaluation']
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(max(5, .9 * len(subset)), 4))
    ax.bar([display_name(m, labels) for m in subset['attack']], subset['wall_seconds_mean'])
    ax.set_ylabel('Mean wall seconds per evaluation call')
    ax.set_title('Measured calls (including first calls)')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=.2)
    save_figure(fig, directory / 'attack_runtime')


def topology(table, designs, directory, labels=None):
    if table.empty:
        return
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))
        topology_panels(table, designs, axes, labels=labels)
        fig.subplots_adjust(left=.055, right=.995, bottom=.14, top=.98, wspace=.28)
        save_figure(fig, directory / 'topology', pdf=True)
    topology_uncertainty(table, designs, directory, labels=labels)


def topology_uncertainty(table, designs, directory, labels=None, filename='topology_uncertainty'):
    """Separate methods to show instance SD without overlapping colored bands.

    The fourth column reports the number of networks containing each degree.
    All three metrics use the same degree support; no missing degree is zeroed.
    """
    designs = [d for d in designs if (table['design'] == d).any()]
    if not designs:
        return
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(designs), 4, figsize=(16.8, 2.2*len(designs)),
                                 squeeze=False, sharex='col', sharey='col')
        for row, design in enumerate(designs):
            data = table[table['design'] == design]
            topology_panels(table, [design], axes[row, :3], labels=labels,
                            shade_metrics=METRICS, show_legend=False)
            axes[row, 0].set_title(display_name(design, labels), loc='left', fontsize=11)
            support = data.pivot(index='degree_k', columns='metric', values='count')
            if not support.eq(support['knn'], axis=0).all().all():
                raise ValueError('Topology metrics have different degree support: ' + design)
            support = support.sort_index()
            ax = axes[row, 3]
            ax.plot(support.index, support['knn'], color='#606060', linewidth=1.2,
                    marker='o', markersize=2.5, markerfacecolor='white')
            ax.set(xlabel=r'Degree $k$', ylabel='Network count',
                   xlim=(0, max(1, table['degree_k'].max())*1.03))
            ax.grid(color='#d9d9d9', linestyle='--', linewidth=.7, alpha=.7)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
            panel_label(ax, 'd')
            if row < len(designs)-1:
                for ax in axes[row]:
                    ax.set_xlabel('')
        # Setting the clustering lower bound disables shared-axis autoscaling.
        # Compute a common range after every method has been added, so later
        # rows cannot be clipped to the first method's (usually smaller) range.
        for column, metric in enumerate(METRICS[:2]):
            data = table[(table['metric'] == metric) & table['design'].isin(designs)]
            sd = data['std'].fillna(0)
            lower, upper = (data['mean']-sd).min(), (data['mean']+sd).max()
            padding = max(upper-lower, abs(upper), .01)*.05
            axes[0, column].set_ylim(min(0, lower-padding) if metric == 'ck' else lower-padding,
                                     upper+padding)
        axes[0, 3].set_ylim(0, max(1, table['count'].max())*1.25)
        fig.suptitle('Lines: means; shading: ±1 SD across network instances; '
                     'counts exclude absent degrees', fontsize=12, y=.998)
        fig.subplots_adjust(left=.055, right=.995, bottom=.055 if len(designs) > 1 else .20,
                            top=.96 if len(designs) > 1 else .82, wspace=.32, hspace=.32)
        save_figure(fig, directory / filename, pdf=True)
