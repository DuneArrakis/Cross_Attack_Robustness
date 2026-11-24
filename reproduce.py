"""Rebuild the manuscript figures and tables from the archived observations."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

from run.reproducibility import THREAD_ENVIRONMENT, evaluation_seed, optimization_seeds
for key, value in THREAD_ENVIRONMENT.items():
    os.environ.setdefault(key, value)

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

from run.datasets.io import load_graph, save_igraph_edgelist
from run.analysis import manuscript_figures as paper
from run.analysis.appendix_tables import write_appendix_tables
from run.analysis.statistics import describe, holm_adjust, statistics_tables
from run.analysis.topology import topology_tables
from run.attacks.common import lcc_curve_from_order, robustness_from_curve

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'reproduction'
GRID = np.linspace(0, 1, 101)
SUPPLEMENT = ROOT / 'results/appendix_c_statistics.zip'


def numeric_csv(archive, path, width):
    data = archive.read(path).partition(b'\n')[2].strip().replace(b'\n', b',')
    return np.fromstring(data, dtype=float, sep=',').reshape(-1, width)


def open_setting(setting):
    path = SOURCES / 'archives' / (setting + '.zip')
    data = path.read_bytes()
    expected = json.loads((SOURCES / 'sha256.json').read_text())[path.name]
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('Archive checksum mismatch: ' + path.name)
    return ZipFile(io.BytesIO(data))


def check_seeds(archive, config, setting):
    tasks = pd.read_csv(archive.open('sa_summary.csv'))
    if len(tasks) != 350 or tasks.duplicated(['instance_id', 'target']).any():
        raise ValueError('Incomplete optimization records in ' + setting)
    for run in tasks.itertuples():
        expected = optimization_seeds(config, run.instance_id, run.target)
        if expected != (run.optimization_seed, run.attack_seed):
            raise ValueError('Inconsistent optimization seed in ' + setting)


def check_graph(archive, row):
    data = archive.read(row.graph_path)
    if hashlib.sha256(data).hexdigest() != row.graph_sha256:
        raise ValueError('Graph checksum mismatch: ' + row.graph_path)
    edges = [tuple(map(int, s.split()[:2])) for s in data.splitlines()
             if s.strip() and not s.startswith(b'#')]
    graph = ig.Graph(n=row.n_nodes, edges=edges, directed=False)
    if graph.ecount() != row.n_edges or not graph.is_simple():
        raise ValueError('Invalid graph: ' + row.graph_path)
    return graph


def check_empirical_input(setting, records):
    """Tie the public edge list to all seven original-network evaluations."""
    metadata = json.loads((SOURCES / 'settings.json').read_text())[setting]
    path = ROOT / metadata['input_file']
    graph = load_graph(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    originals = records[records.design == 'Original']
    if (not graph.is_connected() or set(originals.graph_sha256) != {digest}
            or set(originals.n_nodes) != {graph.vcount()} or graph.vcount() != metadata['nodes']
            or set(originals.n_edges) != {graph.ecount()} or graph.ecount() != metadata['edges']):
        raise ValueError('Empirical input disagrees with the archived observations: ' + str(path))


def check_robustness_statistics(setting, observations, dest):
    """Rebuild the published means, intervals and paired tests from the scored curves."""
    with ZipFile(SUPPLEMENT) as archive:
        summary_ref = pd.read_csv(archive.open('01_robustness.csv'))
        paired_ref = pd.read_csv(archive.open('02_paired_tests.csv'))
    summary_ref = summary_ref.loc[summary_ref.setting == setting].drop(columns='setting')
    # The archived analysis order fixes the sign of each left-minus-right contrast.
    attacks = [a for a in summary_ref.attack.unique() if a != 'Mean_over_attacks']
    designs = list(summary_ref.design.unique())
    labeled = observations.replace({'design': {'Zheng': 'SRA'}, 'attack': {'Zheng': 'SRA'}})
    summary, paired = statistics_tables(labeled, designs, attacks, dest)
    compare_records(summary, summary_ref, ['design', 'attack'])
    if setting in paper.SETTINGS[:6]:
        paired_ref = paired_ref.loc[paired_ref.setting == setting].drop(columns='setting')
        compare_records(paired, paired_ref, ['family', 'left', 'right'])


def compare_records(actual, expected, keys):
    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    if not actual.index.equals(expected.index) or set(actual.columns) != set(expected.columns):
        raise ValueError('Incomplete Appendix C statistical table')
    numeric = expected.select_dtypes(include=np.number).columns
    np.testing.assert_allclose(actual[numeric].values, expected[numeric].values,
                               rtol=1e-10, atol=1e-12, equal_nan=True)
    labels = expected.columns.difference(numeric)
    if not actual[labels].equals(expected[labels]):
        raise ValueError('Appendix C statistical labels differ')


def collect(setting, out, replay):
    dest = out / setting
    dest.mkdir(parents=True, exist_ok=True)
    synthetic = setting in paper.SETTINGS[:6]
    size = 50 if synthetic else 1
    config = json.loads((SOURCES/'configs'/(setting+'.json')).read_text()) if synthetic else None
    curves, scores, graphs = {}, [], {}
    with open_setting(setting) as archive:
        data = pd.read_csv(archive.open('evaluations.csv'))
        counts = data.groupby(['instance_id','design','attack']).size()
        expected = pd.MultiIndex.from_product([sorted(data.instance_id.unique()),
                                               paper.DESIGNS, paper.METHODS])
        if len(data.instance_id.unique()) != size or not (counts.reindex(expected) == 1).all() or len(data) != len(expected):
            raise ValueError('Incomplete evaluation blocks in ' + setting)
        if synthetic:
            check_seeds(archive, config, setting)
        else:
            check_empirical_input(setting, data)
        for row in data.itertuples():
            raw = numeric_csv(archive, row.curve_path, 3)
            steps = np.arange(row.n_nodes + 1)
            if not np.array_equal(raw[:,0], steps) or not np.allclose(raw[:,1], steps / row.n_nodes, rtol=0, atol=1e-12):
                raise ValueError('Incorrect removal grid: ' + row.curve_path)
            score = robustness_from_curve(raw[:,2], row.n_nodes)
            if abs(score - row.R) > 1e-12:
                raise ValueError('Curve/score mismatch: ' + row.curve_path)
            if synthetic:
                if row.attack_seed != evaluation_seed(config, row.instance_id, row.design, row.attack):
                    raise ValueError('Evaluation seed mismatch: ' + row.curve_path)
                if row.graph_path not in graphs:
                    graphs[row.graph_path] = check_graph(archive, row)
                if replay:
                    order = numeric_csv(archive, row.order_path, 2)
                    if not np.array_equal(order[:,0], steps[1:]) or not np.array_equal(order[:,1], order[:,1].astype(int)):
                        raise ValueError('Invalid order: ' + row.order_path)
                    checked = lcc_curve_from_order(graphs[row.graph_path], order[:,1].astype(int).tolist())
                    np.testing.assert_allclose(checked, raw[:,2], rtol=0, atol=1e-12)
            if setting == 'BA_N500':
                curves.setdefault((row.attack,row.design), []).append(np.interp(GRID, raw[:,1], raw[:,2]))
            scores.append(score)
        # Published times are measurements; reformat the original log totals.
        timings = pd.read_csv(archive.open('attack_runtime.csv'))
        stages = timings[timings.stage.isin(['sa_initial','sa_candidate'])]
        measured = stages.groupby('attack').cpu_seconds_sum.sum().reindex(paper.METHODS)
        logs = pd.read_csv(archive.open('sa_summary.csv'))
        by_search = logs.groupby('target').attack_cpu_seconds_sum.sum().reindex(paper.METHODS)
        np.testing.assert_allclose(measured.values, by_search.values, rtol=1e-10, atol=1e-7)
        (dest/'attack_runtime.csv').write_bytes(archive.read('attack_runtime.csv'))
        if setting == 'BA_N500':
            graph_rows = data[['instance_id','design','graph_path','graph_sha256']].drop_duplicates().copy()
            for idx, graph_path in graph_rows.graph_path.items():
                target = dest/graph_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(graph_path))
                graph_rows.at[idx, 'graph_path'] = str(target)
            structure = topology_tables(graph_rows, dest)
    data['R'] = scores
    check_robustness_statistics(setting, data, dest)
    original = data.loc[data.design == 'Original', ['instance_id','attack','R']]
    original.to_csv(dest/'original_attack_per_instance.csv', index=False)
    means = data[data.design != 'Original'].groupby(['instance_id','design']).R.mean().reset_index()
    means.to_csv(dest/'defense_per_instance.csv', index=False)
    matrix = data.pivot_table(index='design', columns='attack', values='R').loc[paper.DESIGNS,paper.METHODS]
    public = matrix.rename(index={'Zheng':'SRA'}, columns={'Zheng':'SRA'})
    public.to_csv(dest/'R_mean.csv')
    if setting == 'BA_N500':
        reference = pd.read_csv(ROOT/'results/R_mean.csv', index_col='design')
        if list(public.index) != list(reference.index) or list(public.columns) != list(reference.columns):
            raise ValueError('Matrix labels or order differ from the published matrix')
        np.testing.assert_allclose(public.values, reference.values, atol=1e-12, rtol=0)
        # Hold the test set fixed across the two designs, excluding both training objectives.
        shared = public.loc[['HDA', 'DomiRank']].drop(columns=['HDA', 'DomiRank']).mean(axis=1)
        shared.rename('mean_R').to_csv(dest / 'five_common_attacks.csv', header=True)
        np.testing.assert_allclose(shared.values, [0.2648, 0.2665], atol=0.00005, rtol=0)
        rankings = pd.DataFrame([dict(describe(group.sort_values('instance_id').R.values), design=design)
                                 for design, group in means.groupby('design')])
        ref = pd.read_csv(SOURCES/'reference/BA500_intervals.csv').set_index('design')
        estimates = rankings.set_index('design').loc[ref.index,ref.columns]
        np.testing.assert_allclose(estimates.values, ref.values, atol=1e-12, rtol=0)
        rankings.to_csv(dest/'defense_ranking.csv', index=False)
        rows = [dict(attack=attack,design=design,x=x,lcc_fraction=value)
                for (attack,design), values in curves.items()
                for x,value in zip(GRID,np.mean(values,axis=0))]
        mean_curves = pd.DataFrame(rows)
        mean_curves.to_csv(dest/'mean_curves.csv', index=False)
        paper.curves(mean_curves)
        paper.matrix_figure(matrix.reset_index(), rankings)
        paper.topology(structure)
        paper.structural_variation(structure)
    print('{}: {} networks, {} curves checked'.format(setting,size,len(data)), flush=True)
    return dest


def check_feedback(folder):
    """Verify the plotted means and intervals using the archived bootstrap draws."""
    with np.load(str(folder/'bootstrap_indices.npz')) as sample:
        indices = sample['indices']
        instances = sample['instance_ids'].tolist()
    if indices.shape != (10000, 50) or len(set(instances)) != 50:
        raise ValueError('Invalid BA500 feedback bootstrap design')
    for source, summary, group, value, mean in [
        ('trajectory_O.csv','trajectory_summary.csv','requested_fraction','O','mean'),
        ('feedback_by_instance.csv','feedback_summary.csv','stratum','p_improve','mean_p')
    ]:
        observations = pd.read_csv(folder/source)
        reference = pd.read_csv(folder/summary).set_index(['attack', group])
        for key, block in observations.groupby(['attack', group]):
            ordered = block.set_index('instance_id').loc[instances]
            values = ordered[value].values
            if value == 'p_improve':
                np.testing.assert_allclose(values, ordered.n_R_plus/ordered.n_candidates, rtol=0, atol=1e-12)
            endpoints = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
            expected = reference.loc[key,[mean, 'ci95_low','ci95_high']].values.astype(float)
            np.testing.assert_allclose(np.r_[values.mean(),endpoints],expected,rtol=0,atol=1e-12)


def check_common_swaps(folder, out):
    """Use one vote per input graph for the fixed-candidate comparison."""
    methods = ['SRA' if attack == 'Zheng' else attack for attack in paper.METHODS]
    swaps = pd.read_csv(folder / 'common_swaps.csv')
    if (swaps.instance_id.nunique() != 20 or swaps.duplicated(['instance_id', 'proposal_id']).any()
            or swaps[methods].isna().any().any()):
        raise ValueError('Incomplete common-candidate observations')
    changing = swaps.loc[swaps.delta_layer_size != 0].copy()
    changing['direction'] = np.where(changing.delta_layer_size > 0, 'increase', 'decrease')
    long = changing.melt(id_vars=['instance_id', 'direction'], value_vars=methods,
                         var_name='attack', value_name='delta_sum_lcc')
    long['improved'] = (long.delta_sum_lcc > 0).astype('int64')
    keys = ['instance_id', 'attack', 'direction']
    counts = long.groupby(keys).improved.agg(['size', 'sum'])
    counts.columns = ['n_scored', 'n_improved']
    reference = pd.read_csv(folder / 'common_swap_counts.csv').set_index(keys).sort_index()
    pd.testing.assert_frame_equal(counts, reference)
    rates = (counts.n_improved / counts.n_scored).unstack('direction')
    if len(rates) != 20 * len(methods) or rates.isna().any().any():
        raise ValueError('Both directions are required for every graph and attack')
    gap = 100 * (rates.increase - rates.decrease).groupby(level='attack').mean()
    gap.rename('difference_percentage_points').to_csv(out / 'common_swap_feedback.csv', header=True)
    np.testing.assert_allclose(gap.loc[['DomiRank', 'HDA', 'EI']], [43.5, 26.9, 13.3],
                               atol=0.05, rtol=0)

    # The common-candidate experiment has its own twenty-network bootstrap.
    with np.load(str(folder/'common_candidate_bootstrap.npz')) as draws:
        ids = draws['instance_ids'].tolist()
        indices = draws['indices']
    if indices.shape != (10000,20) or set(ids) != set(swaps.instance_id):
        raise ValueError('Incomplete common-candidate bootstrap design')
    with ZipFile(SUPPLEMENT) as archive:
        reference = pd.read_csv(archive.open('08_common_candidates.csv'))
        contrasts = pd.read_csv(archive.open('09_common_candidate_contrasts.csv'))
    by_attack = {}
    def check_estimate(values, expected):
        endpoints = np.percentile(values[indices].mean(axis=1), [2.5,97.5])
        np.testing.assert_allclose(np.r_[values.mean(),endpoints],expected,rtol=0,atol=1e-12)
    for row in reference.itertuples():
        sample = rates.xs(row.attack,level='attack').loc[ids]
        direction = 'increase' if row.stratum == 'O_plus' else 'decrease'
        check_estimate(sample[direction].values,[row.mean_p,row.ci95_low,row.ci95_high])
        difference = (sample.increase-sample.decrease).values
        by_attack[row.attack] = difference
        check_estimate(difference,[row.paired_direction_difference,
                                   row.difference_ci95_low,row.difference_ci95_high])
    for row in contrasts.itertuples():
        check_estimate(by_attack[row.left]-by_attack[row.right],
                       [row.mean_difference,row.ci95_low,row.ci95_high])


def check_construction_statistics(scores):
    """Check t intervals and the original 18-test Holm family for each metric."""
    with ZipFile(SUPPLEMENT) as archive:
        means = pd.read_csv(archive.open('10_construction_scores.csv'))
        contrasts = pd.read_csv(archive.open('11_construction_contrasts.csv'))
    labels = {'Wu-Holme':'onion', 'Modified Onion':'backbone_rank_a3', 'SRA':'Zheng'}
    def check_interval(values, row):
        if len(values) != row['count']:
            raise ValueError('Incomplete construction pairs')
        mean = values.mean()
        sd = values.std(ddof=1)
        half = t.ppf(.975, len(values)-1) * sd / np.sqrt(len(values))
        np.testing.assert_allclose([mean,sd,mean-half,mean+half],
            [row['mean'],row['std'],row.ci95_low,row.ci95_high], rtol=0, atol=1e-12)
    by_design = scores.set_index(['family','design','instance_id']).sort_index()
    for _,row in means.iterrows():
        metric = labels.get(row.metric, row.metric)
        check_interval(by_design.loc[(row.family, labels[row.design]),metric].values, row)
    random = 'backbone_random_rank_a3'
    ordered = 'backbone_rank_a3'
    for metric,group in contrasts.groupby('metric'):
        raw_p = []
        column = labels.get(metric, metric)
        for _,row in group.iterrows():
            for left,right in [(ordered,'onion'), (ordered,random), (random,'onion')]:
                # Pandas aligns construction scores on the original input ID.
                difference = (by_design.loc[(row.family,left),column] -
                              by_design.loc[(row.family,right),column]).values
                rounded = np.round(difference,12)
                w,p = wilcoxon(rounded, zero_method='wilcox', alternative='two-sided')
                raw_p.append(p)
                if right == 'onion' and left == ordered:
                    check_interval(difference,row)
                    np.testing.assert_allclose([w,p], [row.wilcoxon_statistic,row.p_value], rtol=1e-10)
        np.testing.assert_allclose(holm_adjust(raw_p)[::3],group.p_holm.values,rtol=1e-10,atol=0)


def check_rings(out, setting, instance):
    """Reconstruct both Fig. 8 rings for one synthetic input and check the saved hashes."""
    from run.optimization.onion_ring import generate_ring

    seeds = pd.read_csv(SOURCES / 'diagnostics/onion/ring_seeds.csv')
    rows = seeds[(seeds.setting == setting) & (seeds.instance_id == instance)]
    if setting not in paper.SETTINGS[:6] or sorted(rows.order) != ['degree', 'random']:
        raise ValueError('Choose a synthetic setting and instance listed in ring_seeds.csv')
    dest = out / 'rings' / setting / instance
    dest.mkdir(parents=True, exist_ok=True)
    with open_setting(setting) as archive:
        input_path = archive.extract('inputs/' + instance + '.txt', str(dest))
    original = load_graph(input_path)
    for record in rows.itertuples():
        graph = generate_ring(original.degree(), record.order, record.seed)
        path = dest / (record.order + '.txt')
        save_igraph_edgelist(graph, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.graph_sha256:
            raise ValueError('Reconstructed ring differs from the measured graph: ' + str(path))
    print('{} {}: both ring constructions match the archived graph hashes.'.format(setting, instance))


def check_example(out):
    """Repeat one full-budget BA100/HDA optimization and all seven final evaluations."""
    from run.config import load_config
    from run.experiments import optimize, evaluate
    setting = 'BA_N100'
    instance = 'graph_0'
    root = out/'example'
    config = load_config(SOURCES/'configs'/(setting+'.json'))
    config['data']['input_dir'] = str(root/'inputs')
    config['output_dir'] = str(root/'rerun')
    config['optimization_attacks'] = ['HDA']
    config['evaluation']['graph_types'] = ['Original','HDA']
    with open_setting(setting) as archive:
        archive.extract('inputs/'+instance+'.txt', str(root))
        expected = pd.read_csv(archive.open('evaluations.csv'))
    optimize.run(config)
    evaluate.run(config)
    actual = pd.read_csv(root/'rerun/evaluations.csv')
    columns = ['instance_id','design','attack']
    reference = expected[(expected.instance_id == instance) & expected.design.isin(['Original','HDA'])]
    actual = actual.set_index(columns).sort_index()
    reference = reference.set_index(columns).sort_index()
    if not actual.index.equals(reference.index) or not actual.graph_sha256.equals(reference.graph_sha256):
        raise ValueError('Reoptimized graph differs from the published BA100 example')
    np.testing.assert_allclose(actual.R.values, reference.R.values, rtol=0, atol=1e-12)
    print('BA100 graph_0: identical HDA topology and all 14 original/design evaluation scores.')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'work')
    parser.add_argument('--replay-orders', action='store_true',
                        help='also reconstruct every synthetic curve from its graph and node order')
    parser.add_argument('--check-example', action='store_true',
                        help='repeat a full-budget BA100 HDA search and evaluate both graphs under all seven methods')
    parser.add_argument('--extract-inputs', choices=paper.SETTINGS[:6], metavar='SETTING',
                        help='only unpack the exact synthetic input graphs for a new optimization run')
    parser.add_argument('--check-rings', nargs=2, metavar=('SETTING', 'INSTANCE'),
                        help='regenerate the modified Onion ring and archived random control for one synthetic input')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.check_rings:
        check_rings(out, *args.check_rings)
        return
    if args.check_example:
        check_example(out)
        return
    if args.extract_inputs:
        with open_setting(args.extract_inputs) as archive:
            for name in archive.namelist():
                if name.startswith('inputs/'):
                    archive.extract(name, str(out/args.extract_inputs))
        return
    paper.ASSETS = out/'figures'
    paper.ASSETS.mkdir(parents=True, exist_ok=True)
    style = dict(paper.STYLE, **{'font.serif':['STIXGeneral'], 'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,
                                 'ytick.labelsize':8,'axes.linewidth':.7})
    with plt.rc_context(style):
        folders = {setting:collect(setting,out, args.replay_orders) for setting in paper.SETTINGS}
        diagnostics = SOURCES/'diagnostics'
        onion_data = diagnostics/'onion'
        synthetic = pd.read_csv(onion_data/'tables/per_graph_scores.csv')
        empirical = pd.read_csv(onion_data/'empirical_construction_scores.csv')
        paper.tables(folders, synthetic, empirical)
        check_construction_statistics(synthetic)
        for name in ['design_rows.tex','original_rows.tex','timing_rows.tex']:
            if (paper.ASSETS/name).read_bytes() != (SOURCES/'reference'/name).read_bytes():
                raise ValueError('Typeset table differs from the manuscript: ' + name)
        write_appendix_tables(SUPPLEMENT, paper.ASSETS)
        check_common_swaps(diagnostics / 'feedback', folders['BA_N500'])
        check_feedback(diagnostics/'feedback')
        paper.feedback(diagnostics/'feedback')
        data, positions = paper.load_data(diagnostics/'exchange')
        figure, _ = paper.make_figure(data, positions)
        paper.save(figure, 'exchange')
        with np.load(onion_data/'BA500_construction_curves.npz', allow_pickle=False) as bundle:
            paper.onion(pd.read_csv(folders['BA_N500']/'mean_curves.csv'), synthetic, bundle)
    print('Both empirical inputs, the BA500 five-attack means and the common-swap feedback verified.')
    print('Verified all 700 BA500 construction curves used in Fig. 8 against their measured scores.')
    print('Fig. 4 matrix and intervals, Table 2, and both Appendix A tables verified.')
    print('Appendix C robustness summaries and 1,470 paired tests verified across settings.')
    print('Figures and LaTeX rows written to {}'.format(paper.ASSETS))


if __name__ == '__main__':
    main()
