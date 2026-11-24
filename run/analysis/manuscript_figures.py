"""Figures and tables for the seven-method study. See reproduce.py."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from .mechanism_exchange_figure import load_data, make_figure
from .paper_style import (METHODS, STYLE, TOPOLOGY_STYLES, ROBUSTNESS_CMAP, METRICS, LABELS,
                          display_name, topology_panels)


DESIGNS = ['Original'] + METHODS
SETTINGS = ['BA_N100', 'BA_N300', 'BA_N500', 'BA_N1000', 'ER_N500', 'WS_N500',
            'USAir97', 'EUpower217']
NETWORKS = ['BA100', 'BA300', 'BA500', 'BA1000', 'ER500', 'WS500', 'USAir97', 'EUpower']
ASSETS = PROJECT_ROOT / 'work/figures'
INK = '#263945'
GRID = '#dde3e6'


def save(fig, name):
    fig.savefig(str(ASSETS / (name + '.pdf')), facecolor='white')
    plt.close(fig)


def tidy(ax, direction='y'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis=direction, color=GRID, linewidth=.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.6, width=.6)


def heading(ax, text):
    ax.set_title(text, loc='left', color=INK, fontsize=10, fontweight='bold', pad=5)


CONSTRUCTIONS = {'onion': 'Wu–Holme', 'backbone_rank_a3': 'Modified Onion'}
CONSTRUCTION_STYLES = {'onion': ('#986431', None, '--'),
                       'backbone_rank_a3': ('#19785c', 's', '-.')}


def legend_entries(ax, columns):
    """Keep the manuscript order when the reader scans a legend row by row."""
    handles, labels = ax.get_legend_handles_labels()
    # Matplotlib fills each column before moving to the next one.
    order = [i for col in range(columns) for i in range(col, len(labels), columns)]
    return [handles[i] for i in order], [labels[i] for i in order]


def curves(table, designs=DESIGNS, name='curves'):
    """Plot the first half of the removal sequence in the submitted curve style."""
    fig = plt.figure(figsize=(8.0, 4.08))
    grid = fig.add_gridspec(2, 12, left=.08, right=.99, bottom=.13, top=.92,
                            wspace=.54, hspace=.40)
    for i, attack in enumerate(METHODS):
        row, col, width = (0, 4*i, 4) if i < 3 else (1, 3*(i-3), 3)
        ax = fig.add_subplot(grid[row, col:col + width])
        subset = table[table.attack == attack]
        for design in designs:
            data = subset[subset.design == design].sort_values('x')
            if data.empty or data.x.iloc[0] != 0 or data.x.iloc[-1] != 1:
                raise ValueError('Incomplete curve for %s under %s' % (design, attack))
            visible = data[data.x <= .5]
            color, marker, ls = (CONSTRUCTION_STYLES[design] if design in CONSTRUCTIONS
                                 else TOPOLOGY_STYLES[design])
            ax.plot(visible.x, visible.lcc_fraction, color=color, ls=ls, lw=1.25,
                    marker=marker, ms=2.7, mew=.65, mfc='white', markevery=7,
                    label=CONSTRUCTIONS.get(design, display_name(design)))
        ax.set(xlim=(0, .5), ylim=(0, 1.13), xticks=np.arange(6) / 10.,
               yticks=np.arange(6) / 5.)
        ax.tick_params(top=True, right=True, direction='in', labelleft=(col == 0),
                       labelsize=8.1, length=2.6, pad=2)
        ax.grid(color='#dedede', lw=.45, ls=':', alpha=.85)
        ax.set_axisbelow(True)
        ax.set_title('Test: ' + display_name(attack), fontsize=9.5, pad=3)
        ax.set_title('(' + chr(97+i) + ')', loc='left', fontsize=8.6, weight='bold', pad=3)
        if i == 0:
            columns = 2 if name == 'curves' else 1
            legend = ax.legend(*legend_entries(ax, columns), loc='upper right', ncol=columns,
                               fontsize=7.0 if name == 'curves' else 8, labelspacing=.08,
                               borderaxespad=.14, borderpad=.22, handlelength=1.15,
                               handletextpad=.3, columnspacing=.55, frameon=True,
                               fancybox=False, facecolor='white', edgecolor='#778087',
                               framealpha=1)
            legend.get_frame().set_linewidth(.45)
    fig.text(.525, .02, r'Removed node fraction $q/N$', ha='center', fontsize=10)
    fig.text(.013, .52, 'Largest component fraction', va='center', rotation=90, fontsize=10)
    save(fig, name)


def matrix_figure(table, rankings):
    scores = table.set_index('design').loc[DESIGNS, METHODS]
    ranked = rankings.set_index('design').loc[METHODS]
    norm = Normalize(scores.values.min(), scores.values.max())
    fig = plt.figure(figsize=(8.0, 3.63))
    layout = fig.add_gridspec(2, 2, width_ratios=[1.47, 1], height_ratios=[1, 7],
                              hspace=.15, wspace=.28)
    band = fig.add_subplot(layout[0, 0])
    matrix = fig.add_subplot(layout[1, 0])
    means = fig.add_subplot(layout[:, 1])
    for ax, block in [(band, scores.loc[['Original']]), (matrix, scores.loc[METHODS])]:
        ax.imshow(block.values, cmap=ROBUSTNESS_CMAP, norm=norm, aspect='auto')
        for row, col in np.ndindex(block.shape):
            val = block.iloc[row, col]
            ax.text(col, row, '%.3f' % val, ha='center', va='center', fontsize=8.1,
                    color='white' if norm(val) > .65 else '#142832')
        ax.set_xticks(np.arange(-.5, len(METHODS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(block)), minor=True)
        ax.grid(which='minor', color='white', lw=.8)
        ax.tick_params(which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks(range(len(block)))
        ax.set_yticklabels([display_name(d) for d in block.index], fontsize=8.2)
    band.set_xticks([])
    heading(band, '(a)  Robustness under each test attack')
    matrix.set_xticks(range(len(METHODS)))
    matrix.set_xticklabels([display_name(m) for m in METHODS], fontsize=8.1)
    matrix.set(xlabel='Test attack', ylabel='Optimization attack')
    # The baseline row and rewired designs share the same R scale.
    scale_ax = matrix.inset_axes([.20, -.258, .64, .04])
    colorbar = fig.colorbar(matrix.images[0], cax=scale_ax, orientation='horizontal',
                            ticks=[.20, .25, .30, .35, .40], format='%.2f')
    scale_ax.text(-.035, .5, r'$R$', transform=scale_ax.transAxes,
                  ha='right', va='center', fontsize=9, color=INK)
    scale_ax.tick_params(axis='x', labelsize=8, length=2, pad=1.5)
    colorbar.outline.set_linewidth(.5)
    positions = np.arange(len(METHODS))
    for i in positions:
        matrix.add_patch(Rectangle((i-.47, i-.47), .94, .94, fill=False,
                                   edgecolor=INK, linewidth=1))
    bars = means.bar(positions, ranked.mean_R, width=.68,
                     color=[TOPOLOGY_STYLES[m][0] for m in METHODS], zorder=2)
    means.errorbar(positions, ranked.mean_R,
                   yerr=[ranked.mean_R-ranked.ci95_low_R, ranked.ci95_high_R-ranked.mean_R],
                   fmt='none', ecolor=INK, capsize=2.6, elinewidth=.9, zorder=3)
    for bar, (_, row) in zip(bars, ranked.iterrows()):
        means.text(bar.get_x()+bar.get_width()/2, row.ci95_high_R+.006, '%.3f' % row.mean_R,
                   ha='center', va='bottom', fontsize=7.8)
    means.set(xlabel='Optimization attack', ylabel=r'Mean robustness $\bar R_a$',
              xticks=positions, ylim=(0,.325), yticks=[0,.1,.2,.3])
    means.set_xticklabels([display_name(m) for m in METHODS], rotation=36, ha='right', fontsize=8.5)
    tidy(means)
    heading(means, '(b)  Mean over seven test attacks')
    fig.subplots_adjust(left=.099, right=.992, top=.91, bottom=.205)
    save(fig, 'matrix')


def topology(table):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.15))
    topology_panels(table, DESIGNS, axes, show_legend=False)
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
    fig.legend(*legend_entries(axes[0], 4), loc='lower center', ncol=4, frameon=False,
               columnspacing=1.8, handlelength=2.1, fontsize=10, labelspacing=.3)
    fig.subplots_adjust(left=.077, right=.99, bottom=.35, top=.98, wspace=.39)
    save(fig, 'topology')


def feedback(folder):
    paths = pd.read_csv(folder / 'trajectory_summary.csv')
    rates = pd.read_csv(folder / 'feedback_summary.csv').set_index(['attack', 'stratum'])
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.65, 3.18))
    for method in METHODS:
        data = paths.loc[paths.attack == method].sort_values('actual_fraction')
        color, marker, _ = TOPOLOGY_STYLES[method]
        left.vlines(100*data.actual_fraction, data.ci95_low, data.ci95_high, color=color, lw=.7)
        left.plot(100*data.actual_fraction, data['mean'], color=color, marker=marker,
                  ms=3.6, lw=1.3, mfc='white', mew=.7, label=display_name(method))
    pos = np.arange(len(METHODS))
    for key, offset, label, color in [('O_plus', -.19, r'$\Delta O>0$', '#267669'),
                                      ('O_minus', .19, r'$\Delta O<0$', '#b9c7cf')]:
        block = rates.xs(key, level='stratum').loc[METHODS]
        centers = 100 * block.mean_p
        right.bar(pos+offset, centers, width=.36, color=color, label=label, zorder=2)
        right.errorbar(pos+offset, centers,
                       yerr=[centers-100*block.ci95_low, 100*block.ci95_high-centers],
                       fmt='none', ecolor=INK, capsize=2, elinewidth=.8, zorder=3)
    heading(left, '(a)  Degree-layer connectivity')
    heading(right, '(b)  Swaps that improve robustness')
    left.set(xlabel='Evaluation budget used (%)', ylabel=r'Degree-layer connectivity $O$',
             xticks=[0, 25, 50, 75, 100], ylim=(.31, .83))
    right.set(xlabel='Optimization attack', ylabel='Improving swaps (%)',
              xticks=pos, ylim=(0,57), yticks=[0,10,20,30,40,50])
    right.set_xticklabels([display_name(m) for m in METHODS], rotation=32, ha='right')
    tidy(left)
    tidy(right)
    left.legend(*legend_entries(left, 2), loc='upper left', ncol=2, fontsize=7.4, frameon=False,
                handlelength=1.4, columnspacing=.7, handletextpad=.4)
    right.legend(loc='upper left', frameon=False, fontsize=9, ncol=2,
                 handlelength=1.0, columnspacing=1.0)
    fig.subplots_adjust(left=.075, right=.995, bottom=.24, top=.91, wspace=.30)
    save(fig, 'feedback')


def structural_variation(table):
    """Small multiples share bounds down each column and print at paper size."""
    fig, axes = plt.subplots(len(DESIGNS), len(METRICS), figsize=(7.35, 6.45),
                             sharex='col', sharey='col')
    for row, design in enumerate(DESIGNS):
        data = table[table.design == design].sort_values('degree_k')
        color = TOPOLOGY_STYLES[design][0]
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            values = data[data.metric == metric]
            if metric == 'onion':
                values = values[values.degree_k <= 10]
            x, mean, sd = (values[k].to_numpy(dtype=float) for k in ['degree_k','mean','std'])
            ax.fill_between(x, mean-sd, mean+sd, where=values['count'].to_numpy() >= 2,
                            color=color, alpha=.20, linewidth=0)
            ax.plot(x, mean, color=color, lw=.9)
        axes[row, 0].text(-.19, .5, display_name(design), transform=axes[row, 0].transAxes,
                          ha='right', va='center', fontsize=8)
        for ax in axes[row]:
            tidy(ax)
            ax.tick_params(labelsize=7, width=.5, pad=2)
    for col, name in enumerate(LABELS):
        axes[0, col].set_title(name, fontsize=9)
    for col in (0, 1):
        axes[-1,col].set_xticks([0, 40, 80])
    axes[-1,2].set_xticks([2, 4, 6, 8, 10])
    axes[-1,2].set_xlim(1.5, 10.25)
    fig.text(.54, .018, r'Degree $k$', ha='center')
    fig.subplots_adjust(left=.15, right=.993, top=.956, bottom=.072, wspace=.31, hspace=.20)
    save(fig, 'topology_uncertainty')


def onion(formal_curves, synthetic, bundle):
    """Average measured component sizes over the same 50 BA500 degree sequences."""
    designs = bundle['designs'].tolist()
    instances = bundle['instances'].tolist()
    counts = bundle['lcc_counts']
    # The array follows the archive's labeled attack axis.
    archive_methods = bundle['attacks'].tolist()
    if (designs != list(CONSTRUCTIONS) or sorted(archive_methods) != sorted(METHODS)
            or len(set(instances)) != 50 or counts.shape != (2, 50, 7, 501)):
        raise ValueError('Incomplete BA500 construction curves')
    scores = synthetic[synthetic.family == 'BA_N500']
    rows = []
    for di, design in enumerate(designs):
        measured = scores[scores.design == design].set_index('instance_id', verify_integrity=True)
        if set(measured.index) != set(instances):
            raise ValueError('Unpaired BA500 construction curves')
        # R uses every removal step after q=0, including the unseen half of the curve.
        np.testing.assert_allclose(counts[di, :, :, 1:].mean(axis=2) / 500,
                                   measured.loc[instances, archive_methods].values, rtol=0, atol=1e-12)
        means = counts[di].mean(axis=0) / 500
        for method, curve in zip(archive_methods, means):
            rows.extend(dict(attack=method, design=design, x=q/500., lcc_fraction=curve[q])
                        for q in range(0, 501, 5))
    optimized = formal_curves[formal_curves.design.isin(['HDA', 'DomiRank'])]
    curves(pd.concat([optimized, pd.DataFrame(rows)], ignore_index=True, sort=False),
           ['HDA', 'DomiRank'] + designs, name='onion')


def write_rows(filename, rows):
    # The surrounding tabular closes the last row before its bottom rule.
    separator = ' ' + r'\\' + '\n'
    (ASSETS / filename).write_text(separator.join(' & '.join(row) for row in rows) + '%\n',
                                   encoding='utf-8')


def tables(formal, synthetic, empirical):
    summary, timing, originals = {}, {}, {}
    for key, folder in formal.items():
        scores = pd.read_csv(folder / 'defense_per_instance.csv').pivot(
            index='instance_id', columns='design', values='R')
        originals_raw = pd.read_csv(folder / 'original_attack_per_instance.csv').pivot(
            index='instance_id', columns='attack', values='R')
        scores['Original'] = originals_raw.loc[scores.index, METHODS].mean(axis=1)
        scores = scores[DESIGNS]
        if scores.isnull().any().any() or len(scores) != (50 if key in SETTINGS[:6] else 1):
            raise ValueError('Incomplete paired design scores in '+key)
        for design in CONSTRUCTIONS:
            if key in SETTINGS[:6]:
                block = synthetic[(synthetic.family == key) & (synthetic.design == design)]
                block = block.set_index('instance_id', verify_integrity=True)
                if set(block.index) != set(scores.index):
                    raise ValueError('Unpaired construction scores in ' + key)
                np.testing.assert_allclose(block[METHODS].values.mean(axis=1), block.R7,
                                           atol=1e-12, rtol=0)
                scores[design] = block.R7
            else:
                block = empirical[(empirical.setting == key) & (empirical.design == design)]
                expected = 5 if key == 'USAir97' and design == 'backbone_rank_a3' else 1
                if (len(block) != expected or block.realization.nunique() != expected
                        or not np.isfinite(block[METHODS].values).all()):
                    raise ValueError('Incomplete empirical constructions in ' + key)
                scores[design] = block[METHODS].values.mean(axis=1).mean()
        summary[key] = scores.agg(['mean','std'])
        originals[key] = originals_raw[METHODS].mean()
        times = pd.read_csv(folder / 'attack_runtime.csv')
        stages = times[times.stage.isin(['sa_initial', 'sa_candidate'])]
        if stages.groupby('stage').calls.nunique().max() != 1 or stages.failed_calls.sum():
            raise ValueError('Unequal or incomplete evaluation budget in '+key)
        timing[key] = stages.groupby('attack').cpu_seconds_sum.sum()/3600
    rows = []
    for design in DESIGNS + list(CONSTRUCTIONS):
        label = CONSTRUCTIONS.get(design, display_name(design)).replace('–', '--')
        line = [(r'\midrule ' if design == 'onion' else '') + label]
        for key in SETTINGS:
            values = summary[key].loc['mean']
            val = ('%.3f' if key in SETTINGS[:6] else '%.4f') % values[design]
            if values[design] == values.max():
                val = r'\textbf{' + val + '}'
            line.append(val)
        rows.append(line)
        rows.append([''] + ['(%.3f)' % summary[key].loc['std', design]
                            for key in SETTINGS[:6]] + ['', ''])
    write_rows('design_rows.tex', rows)
    write_rows('timing_rows.tex', [[name] + ['%.2f' % timing[key][m] for m in METHODS]
                                   for key, name in zip(SETTINGS, NETWORKS)])
    write_rows('original_rows.tex', [[name] + [('%.4f' % originals[key][m]) for m in METHODS]
                                     for key, name in zip(SETTINGS, NETWORKS)])
