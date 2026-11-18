"""Figures and tables for the seven-method study. See reproduce.py."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from .mechanism_exchange_figure import load_data, make_figure
from .statistics import describe
from .paper_style import (STYLE, TOPOLOGY_STYLES, ROBUSTNESS_CMAP, METRICS, LABELS,
                          display_name, topology_panels)


METHODS = ['HDA', 'CI', 'CoreHD', 'EI', 'FINDER', 'Zheng', 'DomiRank']
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


def ci(ax, center, lower, upper, y, color, marker='o', open_marker=False):
    ax.hlines(y, lower, upper, color=color, linewidth=1.3)
    ax.plot([lower, upper], [y, y], ls='none', marker='|', ms=3.8, mew=.75, color=color)
    ax.plot(center, y, marker=marker, color=color, ms=3.4, mew=.7,
            mfc='white' if open_marker else color, ls='none')


def curves(table):
    fig, axes = plt.subplots(2, 4, figsize=(7.35, 3.85), sharex=True, sharey=True)
    for i, (attack, ax) in enumerate(zip(METHODS, axes.flat)):
        subset = table.loc[table.attack == attack]
        for design in DESIGNS:
            data = subset.loc[subset.design == design].sort_values('x')
            if len(data) == 0 or data.x.iloc[-1] != 1:
                raise ValueError('Incomplete curve for %s under %s' % (design, attack))
            color, marker, ls = TOPOLOGY_STYLES[design]
            ax.plot(data.x, data.lcc_fraction, label=display_name(design),
                    color=color, ls=ls, lw=1.0, marker=marker, ms=2.3, mew=.55,
                    markevery=(3 * DESIGNS.index(design), 17), mfc='white')
        heading(ax, '(%s)  %s' % (chr(97+i), display_name(attack)))
        ax.set(xlim=(0, 1), ylim=(-.02, 1.02), xticks=[0, .5, 1], yticks=[0, .5, 1])
        # The fourth column has no data panel below it.
        ax.tick_params(labelbottom=(i >= 3))
        tidy(ax)
    legend_ax = axes.flat[-1]
    legend_ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, title='Network design', loc='center', ncol=2,
                     frameon=False, fontsize=7.5, title_fontsize=8.5, handlelength=1.6,
                     columnspacing=.6, handletextpad=.35, labelspacing=.75)
    fig.text(.5, .028, r'Removed fraction $q/N$', ha='center')
    fig.text(.015, .51, r'Largest component fraction $S(q/N)$', va='center', rotation=90)
    fig.subplots_adjust(left=.085, right=.99, top=.925, bottom=.13, wspace=.19, hspace=.40)
    save(fig, 'curves')


def matrix_figure(table, rankings):
    scores = table.set_index('design').loc[DESIGNS, METHODS]
    ranked = rankings.set_index('design').loc[METHODS]
    norm = Normalize(scores.values.min(), scores.values.max())
    fig = plt.figure(figsize=(7.35, 3.8))
    layout = fig.add_gridspec(2, 2, width_ratios=[3.4, 1.25], height_ratios=[1, 7],
                              hspace=.13, wspace=.12)
    band = fig.add_subplot(layout[0, 0])
    matrix = fig.add_subplot(layout[1, 0])
    means = fig.add_subplot(layout[1, 1], sharey=matrix)
    for ax, block in [(band, scores.loc[['Original']]), (matrix, scores.loc[METHODS])]:
        ax.imshow(block.values, cmap=ROBUSTNESS_CMAP, norm=norm, aspect='auto')
        for row, col in np.ndindex(block.shape):
            val = block.iloc[row, col]
            ax.text(col, row, '%.3f' % val, ha='center', va='center',
                    color='white' if norm(val) > .65 else '#142832', fontsize=8)
        ax.set_xticks(np.arange(-.5, len(METHODS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(block)), minor=True)
        ax.grid(which='minor', color='white', lw=.8)
        ax.tick_params(which='both', length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks(range(len(block)))
        ax.set_yticklabels([display_name(d) for d in block.index])
    band.set_xticks([])
    heading(band, '(a)  Robustness under each test attack')
    matrix.set_xticks(range(len(METHODS)))
    matrix.set_xticklabels([display_name(m) for m in METHODS], fontsize=8)
    matrix.set_xlabel('Test attack')
    matrix.set_ylabel('Optimization attack')
    for i, method in enumerate(METHODS):
        matrix.add_patch(Rectangle((i-.47, i-.47), .94, .94, fill=False,
                                   edgecolor=INK, linewidth=1))
        row = ranked.loc[method]
        ci(means, row.mean_R, row.ci95_low_R, row.ci95_high_R, i, TOPOLOGY_STYLES[method][0])
        means.annotate('%.3f' % row.mean_R, (row.ci95_high_R, i), xytext=(5, 0),
                       textcoords='offset points', va='center', fontsize=8)
    heading(means, '(b)  Across attacks')
    means.set(xlim=(.226, .312), xticks=[.24, .26, .28, .30], xlabel=r'Mean robustness $\bar R_a$')
    means.tick_params(axis='y', which='both', left=False, labelleft=False)
    means.spines['left'].set_visible(False)
    tidy(means, 'x')
    fig.subplots_adjust(left=.115, right=.99, top=.91, bottom=.14)
    save(fig, 'matrix')


def topology(table):
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.15))
    topology_panels(table, DESIGNS, axes, show_legend=False)
    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               columnspacing=1.8, handlelength=2.1, fontsize=10, labelspacing=.3)
    fig.subplots_adjust(left=.077, right=.99, bottom=.35, top=.98, wspace=.39)
    save(fig, 'topology')


def feedback(folder):
    paths = pd.read_csv(folder / 'trajectory_summary.csv')
    rates = pd.read_csv(folder / 'feedback_summary.csv').set_index(['attack', 'stratum'])
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.35, 3.12))
    for i, method in enumerate(METHODS):
        data = paths.loc[paths.attack == method].sort_values('actual_fraction')
        color, marker, _ = TOPOLOGY_STYLES[method]
        left.vlines(100*data.actual_fraction, data.ci95_low, data.ci95_high, color=color, lw=.7)
        left.plot(100*data.actual_fraction, data['mean'], color=color, marker=marker,
                  ms=3.6, lw=1.3, mfc='white', mew=.7, label=display_name(method))
        for key, offset, hollow in [('O_plus', -.12, False), ('O_minus', .12, True)]:
            row = rates.loc[(method, key)]
            ci(right, 100*row.mean_p, 100*row.ci95_low, 100*row.ci95_high,
               i+offset, color, open_marker=hollow)
    heading(left, '(a)  Degree-layer connectivity')
    heading(right, '(b)  Objective improvement')
    left.set(xlabel='Evaluation budget used (%)', ylabel=r'Degree-layer connectivity $O$',
             xticks=[0, 25, 50, 75, 100], ylim=(.31, .83))
    right.set(xlabel='Candidates improving robustness (%)', yticks=range(len(METHODS)),
              ylim=(len(METHODS)-.45, -.55), xlim=(0, 53))
    right.set_yticklabels([display_name(m) for m in METHODS])
    tidy(left)
    tidy(right, 'x')
    left.legend(loc='upper left', ncol=2, fontsize=7.3, frameon=False,
                handlelength=1.4, columnspacing=.7, handletextpad=.4)
    handles = [Line2D([], [], color=INK, ls='none', marker='o', ms=4,
                      mfc=mfc, label=label) for mfc, label in
               [(INK, r'$\Delta O>0$'), ('white', r'$\Delta O<0$')]]
    right.legend(handles=handles, loc='upper right', frameon=False,
                 fontsize=8, ncol=2, handlelength=.9, columnspacing=.6)
    fig.subplots_adjust(left=.09, right=.99, bottom=.17, top=.90, wspace=.34)
    save(fig, 'feedback')


def structural_variation(table):
    """Small multiples share bounds down each column and print at paper size."""
    fig, axes = plt.subplots(len(DESIGNS), 4, figsize=(7.35, 6.45),
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
        support = data[data.metric == 'knn']
        axes[row, 3].plot(support.degree_k, support['count'], color='#606b75', lw=1)
        axes[row, 0].text(-.27, .5, display_name(design), transform=axes[row, 0].transAxes,
                          ha='right', va='center', fontsize=8)
        for ax in axes[row]:
            tidy(ax)
            ax.tick_params(labelsize=7, width=.5, pad=2)
    for col, name in enumerate(list(LABELS) + ['Networks']):
        axes[0, col].set_title(name, fontsize=9)
    for col in (0, 1, 3):
        axes[-1,col].set_xticks([0, 40, 80])
    axes[-1,2].set_xticks([2, 4, 6, 8, 10])
    axes[-1,2].set_xlim(1.5, 10.25)
    axes[-1,3].set_yticks([0, 50])
    fig.text(.54, .018, r'Degree $k$', ha='center')
    fig.subplots_adjust(left=.15, right=.993, top=.956, bottom=.072, wspace=.31, hspace=.20)
    save(fig, 'topology_uncertainty')


def onion(folder):
    observations = pd.read_csv(folder / 'tables/per_graph_scores.csv')
    ordered, random = 'backbone_rank_a3', 'backbone_random_rank_a3'
    categories = [('HDA', 'HDA', '#0072B2', 'o'), ('DomiRank', 'DomiRank', '#6A3D9A', 'h'),
                  ('onion', 'Wu–Holme', '#885a45', 's'), (random, 'Random ring', '#808c91', '^'),
                  (ordered, 'Degree-ordered ring', '#167c68', 'D')]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.35, 3.82), sharey=True,
                                      gridspec_kw={'width_ratios':[1.35, 1]})
    for i, key in enumerate(SETTINGS[:6]):
        scores = observations.loc[observations.family == key].pivot(
            index='instance_id', columns='design', values='R7').sort_index()
        if len(scores) != 50 or scores.isnull().any().any():
            raise ValueError('Incomplete paired construction scores in ' + key)
        for off, (design, _, color, marker) in zip(np.linspace(-.31,.31,5), categories):
            estimate = describe(scores[design].values)
            ci(left, estimate['mean_R'], estimate['ci95_low_R'], estimate['ci95_high_R'],
               i+off, color, marker)
        for design, off, color, marker in [(random, -.12, '#58656c', '^'), ('onion', .12, '#885a45', 's')]:
            difference = describe((scores[ordered] - scores[design]).values)
            ci(right, difference['mean_R'], difference['ci95_low_R'], difference['ci95_high_R'],
               i+off, color, marker)
    heading(left, '(a)  Optimized and constructed designs')
    heading(right, '(b)  Paired differences')
    left.set(yticks=range(6), ylim=(5.65, -.65), xlabel=r'Mean robustness across attacks')
    left.set_yticklabels(NETWORKS[:6])
    right.axvline(0, color=INK, linewidth=.8)
    right.set_xlabel('Degree-ordered ring minus reference')
    right.set_xticks([0, .01, .02, .03])
    for ax in (left, right):
        tidy(ax, 'x')
        ax.tick_params(labelsize=9.5)
        ax.xaxis.label.set_size(10)
        for y in np.arange(.5, 5, 1):
            ax.axhline(y, color=GRID, lw=.45)
    legend = [Line2D([], [], color=color, label=label, marker=marker, ls='none', ms=4)
              for _, label, color, marker in categories]
    left.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5,-.20), frameon=False,
                ncol=2, fontsize=9.5, columnspacing=.85, handlelength=1, labelspacing=.3)
    right.legend(handles=[legend[3], legend[2]], loc='upper center', bbox_to_anchor=(.5,-.20),
                 ncol=2, frameon=False, fontsize=9.5, columnspacing=.7, handlelength=1)
    fig.subplots_adjust(left=.095, right=.985, top=.91, bottom=.335, wspace=.14)
    save(fig, 'onion')


def write_rows(filename, rows):
    # The surrounding tabular closes the last row before its bottom rule.
    separator = ' ' + r'\\' + '\n'
    (ASSETS / filename).write_text(separator.join(' & '.join(row) for row in rows) + '%\n',
                                   encoding='utf-8')


def tables(formal):
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
        summary[key] = scores.agg(['mean','std'])
        originals[key] = originals_raw[METHODS].mean()
        times = pd.read_csv(folder / 'attack_runtime.csv')
        stages = times[times.stage.isin(['sa_initial', 'sa_candidate'])]
        if stages.groupby('stage').calls.nunique().max() != 1 or stages.failed_calls.sum():
            raise ValueError('Unequal or incomplete evaluation budget in '+key)
        timing[key] = stages.groupby('attack').cpu_seconds_sum.sum()/3600
    rows = []
    for design in DESIGNS:
        line = [display_name(design)]
        for key in SETTINGS:
            values = summary[key].loc['mean']
            val = ('%.3f' if key in SETTINGS[:6] else '%.4f') % values[design]
            if design == values[METHODS].idxmax():
                val = r'\textbf{' + val + '}'
            line.append(val)
        rows.append(line)
        rows.append([''] + [r'{\scriptsize (%.3f)}' % summary[key].loc['std', design]
                            for key in SETTINGS[:6]] + ['', ''])
    write_rows('design_rows.tex', rows)
    write_rows('timing_rows.tex', [[name] + ['%.2f' % timing[key][m] for m in METHODS]
                                   for key, name in zip(SETTINGS, NETWORKS)])
    write_rows('original_rows.tex', [[name] + [('%.4f' % originals[key][m]) for m in METHODS]
                                     for key, name in zip(SETTINGS, NETWORKS)])
