"""Shared typography, colors and topology panels matching the submitted figures."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 11, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'axes.linewidth': 1.0,
    'axes.spines.top': True, 'axes.spines.right': True,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'axes.unicode_minus': False,
}
ROBUSTNESS_CMAP = LinearSegmentedColormap.from_list(
    'paper_robustness', ['#f3de67', '#a5c582', '#5a96cc', '#485098'])
BLUE, GREEN = '#5a96cc', '#a5c582'
METRICS = ('knn', 'ck', 'onion')
LABELS = (r'$\bar{k}_{nn}(k)$', r'$C_k$', r'$S_k/N_k$')
TOPOLOGY_STYLES = {
    'Original': ('#606060', None, '--'),
    'HDA': ('#0072B2', 'o', '-'),
    'CI': ('#E69F00', 's', '-'),
    'CoreHD': ('#009E73', '^', '-'),
    'EI': ('#D55E00', 'D', '-'),
    'FINDER': ('#CC79A7', 'v', '-'),
    'Zheng': ('#56B4E9', 'P', '-'),
    'DomiRank': ('#6A3D9A', 'h', '-'),
    'onion': ('#8C564B', None, '-.'),
}


def method_labels(attack_options=None):
    """Publication labels only; keep saved method IDs and checkpoint keys intact.

    SRA is our shorthand for the adaptive implementation of the
    symbolic-regression-derived topological score, not an author-assigned name.
    Explicit static experiments must retain a distinct display label.
    """
    update = (attack_options or {}).get('Zheng', {}).get('update', 'adaptive')
    return {'Zheng': 'SRA' if update == 'adaptive' else 'SR-' + update, 'onion': 'Onion'}


def display_name(method, labels=None):
    labels = method_labels() if labels is None else labels
    return labels.get(method, method)


def panel_label(ax, letter, outside=False):
    ax.text(-.10 if outside else .02, .98, '(' + letter + ')',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=13, fontweight='bold', clip_on=False)


def topology_panels(table, designs, axes, degree_limits=None, labels=None,
                    shade_metrics=('onion',), show_legend=True):
    """Keep full-degree statistics; only the shell panel is truncated at k=10.

    Main panels (a,b) show means only. Optional shading is sample SD across
    instances (when at least two have this degree), never SEM or a difference.
    Hollow markers match the submitted figure and do not encode sample counts.
    """
    for ax, metric, label, letter in zip(axes, METRICS, LABELS, 'abc'):
        for design in designs:
            data = table[(table['design'] == design) & (table['metric'] == metric)]
            data = data.sort_values('degree_k')
            if degree_limits is not None:
                data = data[data['degree_k'].between(*degree_limits)]
            if metric == 'onion':
                data = data[data['degree_k'] <= 10]
            if data.empty:
                continue
            x, y, sd = (data[k].values.astype(float) for k in ('degree_k', 'mean', 'std'))
            color, marker, linestyle = TOPOLOGY_STYLES.get(design, ('#888888', 'o', '-'))
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.7,
                    marker=marker, markersize=3.8, markerfacecolor='white',
                    markeredgewidth=.9, label=display_name(design, labels), zorder=3)
            valid = (data['count'].values >= 2) & np.isfinite(sd)
            if metric in shade_metrics and valid.any():
                ax.fill_between(x, y-sd, y+sd, where=valid,
                                color=color, alpha=.16 if not show_legend else .07,
                                linewidth=0, zorder=1)
        ax.set(xlabel=r'Degree $k$', ylabel=label)
        ax.grid(color='#d9d9d9', linestyle='--', linewidth=.7, alpha=.7)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        if metric == 'onion':
            ax.set(xlim=(0, 10), ylim=(-.025, 1.08))
            ax.set_xticks(np.arange(0, 11, 2))
        else:
            upper = degree_limits[1] if degree_limits is not None else table['degree_k'].max()
            ax.set_xlim(0, max(1, upper)*1.03)
            if metric == 'ck':
                ax.set_ylim(bottom=0)
        panel_label(ax, letter)
    if show_legend:
        axes[0].legend(loc='upper right', frameon=True, fancybox=False,
                       edgecolor='black', framealpha=1, fontsize=9, handlelength=2.4)
