"""Compact paper figure for an already selected, measured edge exchange.

Four groups: the swap, matched-removal residuals, full curves, and all seven
native score changes. Stored graph positions and numerical evidence are reused.
No attack evaluation, example selection, or topology optimization is performed.
"""
import json
from pathlib import Path

import numpy as np

from .paper_style import METHODS, STYLE, TOPOLOGY_STYLES, display_name
from .small_case_graphs import graph_from_mask, unpack_mask
from ..attacks.common import lcc_curve_from_order


REMOVED, ADDED = '#D55E00', '#0072B2'
COMPONENT_COLORS = ['#91B8D8', '#E8C689', '#B4C8AA', '#C9BBDD', '#E5EAED']
FIGURE_NAME = 'fig_mechanism_exchange'


def _components(graph, deleted):
    active = [v for v in range(graph.vcount()) if v not in deleted]
    return sorted([tuple(active[i] for i in c)
                   for c in graph.induced_subgraph(active).components()],
                  key=lambda c: (-len(c), c))


def _component_colors(groups, previous=None):
    """Retain colors of unchanged components, assign the other colors by size."""
    previous = previous or {}
    assigned = {g: previous[g] for g in groups if g in previous}
    available = [c for c in COMPONENT_COLORS if c not in assigned.values()]
    for g in groups:
        if g not in assigned:
            assigned[g] = available.pop(0) if available else COMPONENT_COLORS[-1]
    return assigned


def _network(ax, graph, positions, focus, changed, edge_color, edge_style,
             deleted=(), component_colors=None):
    """Only changed edges and their endpoints receive extra visual emphasis."""
    from matplotlib.collections import LineCollection

    deleted = set(deleted)
    changed = {tuple(sorted(e)) for e in changed}
    ordinary = [(u, v) for u, v in graph.get_edgelist()
                if (u, v) not in changed and u not in deleted and v not in deleted]
    ax.add_collection(LineCollection([positions[[u, v]] for u, v in ordinary],
        colors='#97A5AF' if deleted else '#BCC5CC',
        linewidths=.8 if deleted else .45, alpha=.85 if deleted else .65, zorder=1))
    for u, v in sorted(changed):
        if u not in deleted and v not in deleted:
            ax.plot(*positions[[u, v]].T, color=edge_color, ls=edge_style,
                    lw=1.8, zorder=2, solid_capstyle='round')
    colors = {v: color for group, color in (component_colors or {}).items() for v in group}
    for v, (x, y) in enumerate(positions):
        if v in deleted:
            ax.scatter([x], [y], s=19, marker='x', color='#BAC2C8', lw=.75, zorder=3)
            if v in focus:
                ax.annotate(str(v+1), (x, y), xytext=(0, -9), textcoords='offset points',
                            ha='center', va='top', fontsize=7.5, color='#87949D')
            continue
        selected = v in focus
        ax.scatter([x], [y], s=123 if selected else 32,
                   c=[colors.get(v, '#F0F4F7')], edgecolors='#455E70' if selected else '#7E909D',
                   lw=.8 if selected else .5, zorder=4)
        if selected:
            ax.text(x, y, str(v+1), ha='center', va='center',
                    fontsize=8, color='#1D3546', zorder=5)
    ax.set(xlim=(-1.10, 1.10), ylim=(-1.10, 1.10), aspect='equal')
    ax.axis('off')


def make_figure(data, positions):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    root, main, n = data['root'], data['main'], data['n']
    archive_methods = data['methods']
    e = main['exchange']; q = main['q']; positions = np.asarray(positions, dtype=float)
    before, after = [graph_from_mask(int(e[k]), n) for k in ('before_mask', 'after_mask')]
    b, a = [data['index'][e[k]] for k in ('before_mask', 'after_mask')]
    j = archive_methods.index(e['method'])
    assert positions.shape == (n, 2) and np.isfinite(positions).all()
    assert before.degree() == after.degree()
    assert sorted(set(before.get_edgelist())-set(after.get_edgelist())) == [tuple(x) for x in e['removed_edges']]
    assert sorted(set(after.get_edgelist())-set(before.get_edgelist())) == [tuple(x) for x in e['added_edges']]
    deleted = set(main['removed_nodes'])
    assert deleted == set(data['orders'][b, j, :q]) == set(data['orders'][a, j, :q])
    for graph, idx, expected in [(before, b, main['before_R']), (after, a, main['after_R'])]:
        counts = np.rint(np.asarray(lcc_curve_from_order(graph, data['orders'][idx, j].tolist()))*n).astype(int)
        assert np.array_equal(counts, data['counts'][idx, j])
        assert abs(counts[1:].sum()/n**2-expected) < 1e-12
    columns = [archive_methods.index(method) for method in METHODS]
    values = (data['counts'][a, columns, 1:].astype(int).sum(axis=1)
              - data['counts'][b, columns, 1:].astype(int).sum(axis=1))/n**2
    assert np.allclose(values, [main['native_delta_R'][m] for m in METHODS], rtol=0, atol=1e-12)
    groups_b, groups_a = [_components(g, deleted) for g in (before, after)]
    assert len(groups_b[0]) == main['before_LCC'] and len(groups_a[0]) == main['after_LCC']
    colors_b = _component_colors(groups_b); colors_a = _component_colors(groups_a, colors_b)
    focus = {v for key in ('removed_edges', 'added_edges') for edge in e[key] for v in edge}

    # Fixed physical size: labels remain readable at a full paper-column width.
    fig = plt.figure(figsize=(7.2, 5.3))
    top = [fig.add_axes([x, .570, .28, .34]) for x in (.015, .315)]
    bottom = [fig.add_axes([x, .090, .30, .365]) for x in (.005, .305)]
    for ax, graph, edges, color, style in [
            (top[0], before, e['removed_edges'], REMOVED, '--'),
            (top[1], after, e['added_edges'], ADDED, '-')]:
        _network(ax, graph, positions, focus, edges, color, style)
    for ax, graph, edges, color, style, colors in [
            (bottom[0], before, e['removed_edges'], REMOVED, '--', colors_b),
            (bottom[1], after, e['added_edges'], ADDED, '-', colors_a)]:
        _network(ax, graph, positions, focus, edges, color, style, deleted, colors)
    for x, title in zip((.155, .455), ('Before', 'After')):
        fig.text(x, .915, title, ha='center', va='bottom', fontsize=9)
    for x, value in zip((.155, .455), (main['before_LCC'], main['after_LCC'])):
        fig.text(x, .078, 'LCC = %d'%value, ha='center', va='top', fontsize=11,
                 fontweight='bold', color='#263945')
    fig.legend(handles=[
        Line2D([0], [0], color=REMOVED, ls='--', lw=1.8, label='Removed edge'),
        Line2D([0], [0], color=ADDED, lw=1.8, label='Added edge'),
        Line2D([0], [0], color='#A4AFB8', marker='x', ls='none', ms=4, label='Removed node')],
        loc='center', bbox_to_anchor=(.305, .535), ncol=3, frameon=False,
        fontsize=8, handlelength=1.8, columnspacing=1.0, handletextpad=.45)

    curve_ax = fig.add_axes([.735, .615, .25, .30])
    bar_ax = fig.add_axes([.735, .135, .25, .30])
    for x, y, letter, title in [
            (.015, .975, 'a', 'Degree-preserving swap'),
            (.015, .495, 'b', 'After the same %d removals'%q),
            (.645, .975, 'c', display_name(e['method'])+' attack'),
            (.645, .495, 'd', 'Response to seven attacks')]:
        fig.text(x, y, '('+letter+')', fontsize=11, fontweight='bold', va='top')
        fig.text(x+.053, y, title, fontsize=9.5, va='top')

    x = np.arange(n+1)/n
    yb, ya = data['counts'][b, j]/n, data['counts'][a, j]/n
    marks = sorted(set(range(0, n+1, max(1, n//10))) | {q, n})
    for y, color, style, label in [(yb, '#727C84', '--', 'Before'), (ya, ADDED, '-', 'After')]:
        curve_ax.plot(x, y, color=color, ls=style, lw=1.25, marker='o', ms=2.5,
                      mfc='white', mew=.7, markevery=marks,
                      label=label+r': $R=%.4f$'%(main['before_R'] if label=='Before' else main['after_R']))
    curve_ax.vlines(q/n, yb[q], ya[q], color='#5A8848', lw=2.2, zorder=5)
    curve_ax.annotate('+%d nodes'%(main['after_LCC']-main['before_LCC']),
        xy=(q/n, (yb[q]+ya[q])/2), xytext=(min(q/n+.20, .65), .55), fontsize=8,
        arrowprops=dict(arrowstyle='-', lw=.7, color='#71808B'))
    curve_ax.set(xlim=(0, 1), ylim=(0, 1.12), xticks=[0, .5, 1], yticks=[0, .5, 1])
    curve_ax.set_xlabel(r'Removed fraction $q/N$', fontsize=9, labelpad=2)
    curve_ax.set_ylabel(r'LCC / $N$', fontsize=9, labelpad=2)
    curve_ax.legend(frameon=False, fontsize=7.5, loc='upper right', handlelength=1.8,
                    handletextpad=.4, borderaxespad=.3, labelspacing=.3)
    curve_ax.grid(axis='y', ls='--', lw=.5, color='#D9D9D9', alpha=.8)

    yy = np.arange(len(METHODS)); span = max(float(np.ptp(values)), .02)
    padding = .42*span
    bar_ax.barh(yy, values, color=[TOPOLOGY_STYLES[m][0] for m in METHODS], height=.58)
    bar_ax.axvline(0, color='#35454F', lw=.75)
    bar_ax.set_yticks(yy); bar_ax.set_yticklabels([display_name(m) for m in METHODS], fontsize=8)
    bar_ax.set_ylim(len(METHODS)-.4, -.6)
    bar_ax.set_xlim(min(0, float(values.min()))-padding, max(0, float(values.max()))+padding)
    for i, v in enumerate(values):
        bar_ax.text(v+(.025*span if v>=0 else -.025*span), i, '%+.4f'%v,
                    va='center', ha='left' if v>=0 else 'right', fontsize=7.5)
    bar_ax.set_xlabel(r'$\Delta R=R_{\mathrm{after}}-R_{\mathrm{before}}$', fontsize=9, labelpad=2)
    bar_ax.grid(axis='x', ls='--', lw=.5, color='#D9D9D9', alpha=.8)
    for ax in (curve_ax, bar_ax):
        ax.tick_params(labelsize=8, length=3, width=.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_linewidth(.8)
    meta = dict(n=n, pair_id=main['pair_id'], source_method=e['method'],
        node_numbering='1-based; only exchanged-edge endpoints are numbered',
        labeled_nodes=[v+1 for v in sorted(focus)],
        removed_nodes=[v+1 for v in sorted(deleted)], q=q,
        before_R=main['before_R'], after_R=main['after_R'],
        delta_R={m:float(v) for m,v in zip(METHODS, values)},
        residual_components_before=[[v+1 for v in g] for g in groups_b],
        residual_components_after=[[v+1 for v in g] for g in groups_a],
        panel_map=dict(a='before/after exchange', b='same removed-node set',
                       c='full native attack curves', d='all seven native score changes'),
        position_policy='Identical stored coordinates in all four topology views',
        ordinary_edges='Equal style within each topology view; no degree-threshold emphasis',
        residual_colors='Connected components; unchanged components retain their colors',
        figure_inches=[7.2,5.3], data_checks='passed')
    return fig, meta


def render_exchange(data, positions, degrees=None, directory=None):
    import matplotlib.pyplot as plt
    folder = Path(directory) if directory is not None else data['root']/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(dict(STYLE, **{'svg.fonttype':'none'})):
        fig, meta = make_figure(data, positions)
        for suffix in ('png', 'pdf', 'svg'):
            path = folder/(FIGURE_NAME+'.'+suffix)
            temporary = folder/(FIGURE_NAME+'.tmp.'+suffix)
            fig.savefig(str(temporary), dpi=300, bbox_inches=None)
            temporary.replace(path)
        plt.close(fig)
    return meta


def load_data(output):
    output = Path(output).resolve()
    config = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    seed = json.loads((output/'selection.json').read_text(encoding='utf-8'))['selected_seed']
    root = output/'selected'/str(seed)
    main = json.loads((root/'analysis/main_example.json').read_text(encoding='utf-8'))
    with np.load(str(root/'shared/graphs.npz')) as z:
        index = {str(unpack_mask(m)):i for i,m in enumerate(z['masks'])}
    with np.load(str(root/'shared_scores.npz')) as z:
        counts, orders = z['counts'], z['orders']
    data = dict(root=root, main=main, n=config['n'], methods=config['methods'],
                index=index, counts=counts, orders=orders)
    positions = json.loads((root/'layout.json').read_text(encoding='utf-8'))['positions']
    return data, positions
