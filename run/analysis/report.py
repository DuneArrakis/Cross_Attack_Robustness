"""Read saved results, export tables, then draw figures. Safe to rerun alone."""
from __future__ import annotations

from ..config import project_path
from ..experiments.timing import StageRun
from .summarize import read_observations, robustness_tables, curve_tables, runtime_tables
from .rankings import observation_groups, ranking_tables, ranking_plot, collect_groups
from .statistics import statistics_tables
from .paper_style import method_labels


def run(config, source=None):
    labels = method_labels(config.get('attack_options'))
    root = project_path(config['output_dir'])
    source = project_path(source) if source is not None else root
    observations, designs, attacks = read_observations(source)
    with StageRun(root, 'analyze', config):
        tables, figures = root / 'tables', root / 'figures'
        tables.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        timing = runtime_tables(source, tables)
        from . import plotting
        plotting.runtime(timing, figures, labels=labels)
        groups = []
        for name, subset, instances in observation_groups(observations, config):
            group_tables, group_figures = tables / name, figures / name
            group_tables.mkdir(parents=True, exist_ok=True)
            group_figures.mkdir(parents=True, exist_ok=True)
            ranks = None
            if 'Original' in designs:
                ranks = ranking_tables(subset, designs, attacks, group_tables, instances)
                groups.append((name, ranks))
            matrix = robustness_tables(subset, designs, attacks, group_tables)
            summary, paired = statistics_tables(subset, designs, attacks, group_tables)
            if ranks is not None:
                ranks.update(cross_attack_summary=summary, paired_comparisons=paired)
            curves = curve_tables(source, subset, config['analysis']['points'], group_tables)
            plotting.robustness(matrix, group_figures, labels=labels)
            plotting.curves(curves, designs, attacks, config['analysis']['show_ratio'], group_figures,
                            labels=labels)
            if ranks is not None:
                ranking_plot(ranks, group_figures, labels=labels)
                plotting.robustness_comparison(matrix, ranks, group_figures, labels=labels)
                plotting.attack_rank_changes(ranks, group_figures, labels=labels)
            if config['analysis']['topology']:
                from .topology import topology_tables
                topology = topology_tables(subset, group_tables)
                plotting.topology(topology, designs, group_figures, labels=labels)
        collect_groups(groups, tables)
    print('Analysis saved to {} and {}'.format(tables, figures))
