"""Instance-level uncertainty and paired comparisons of the fixed attack set.

No attacks, optimization, parameter selection, or extra evaluation sets here.
Wilcoxon uses the conditional sign distribution, including tied average ranks;
SciPy 1.3.1 only supplies a normal approximation, unsuitable for tiny pilots.
See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
for the symmetry assumption, zero handling and subtraction-roundoff issue.
"""
from functools import lru_cache
from itertools import combinations
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261014
ROUNDING_TOLERANCE = 16 * np.finfo(float).eps
MEAN_ATTACK = 'Mean_over_attacks'


@lru_cache(maxsize=8)
def _bootstrap_weights(n):
    # One row is one resampled set of original instances. Reusing the same
    # weights across columns preserves their pairing and leaves global RNGs alone.
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    indices = rng.randint(n, size=(BOOTSTRAP_RESAMPLES, n))
    offsets = n * np.arange(BOOTSTRAP_RESAMPLES)[:, None]
    weights = np.bincount((indices + offsets).ravel(), minlength=BOOTSTRAP_RESAMPLES*n)
    weights = weights.reshape(BOOTSTRAP_RESAMPLES, n).astype(float) / n
    weights.setflags(write=False)
    return weights


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Statistics require a nonempty finite vector of instances')
    n = len(values)
    low, high = (np.percentile(_bootstrap_weights(n).dot(values), [2.5, 97.5])
                 if n >= 2 else (np.nan, np.nan))
    std = float(values.std(ddof=1)) if n >= 2 else np.nan
    return dict(mean_R=float(values.mean()), std_R=std, sem_R=std/np.sqrt(n),
                ci95_low_R=float(low), ci95_high_R=float(high), n_instances=n)


def _signed_rank(difference):
    """Two-sided Wilcoxon, dropping zeros; exact conditional signs for n<=200."""
    nonzero = difference[np.abs(difference) > ROUNDING_TOLERANCE]
    if not len(nonzero):
        return 0.0, 1.0, 0.0, 'all_zero_differences'
    # Coalesce only floating-point arithmetic noise, without a practical-effect
    # threshold. This also gives stable average ranks to numerically tied |d|.
    magnitude = np.abs(nonzero).copy()
    anchor = None
    for index in np.argsort(magnitude):
        if anchor is None or magnitude[index] - anchor > ROUNDING_TOLERANCE:
            anchor = magnitude[index]
        magnitude[index] = anchor
    ranks2 = np.rint(2 * rankdata(magnitude)).astype(int)
    positive = int(ranks2[nonzero > 0].sum())
    total = int(ranks2.sum())
    statistic = min(positive, total-positive) / 2
    effect = (2*positive-total) / total
    if len(nonzero) > 200:
        result = wilcoxon(np.sign(nonzero)*magnitude, zero_method='wilcox',
                          correction=True, alternative='two-sided')
        return statistic, float(result.pvalue), effect, 'normal_approximation_cc'
    # Each nonzero difference has equiprobable signs under the symmetric null.
    # Doubling the average ranks permits exact subset-sum dynamic programming,
    # even with ties; probabilities avoid integer overflow at n=50.
    probability = np.zeros(total+1)
    probability[0] = 1.0
    for weight in ranks2:
        shifted = probability.copy() / 2
        shifted[weight:] += probability[:-weight] / 2
        probability = shifted
    extreme = np.abs(2*np.arange(total+1)-total) >= abs(2*positive-total)
    return statistic, min(1.0, float(probability[extreme].sum())), effect, 'exact_conditional_signs'


def paired_summary(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError('Paired vectors must have the same shape')
    difference = left - right
    summary = describe(difference)
    n = len(difference)
    statistic, p, effect, method = (_signed_rank(difference) if n >= 2 else
                                   (np.nan, np.nan, np.nan, 'insufficient_instances'))
    return dict(n_pairs=n, mean_difference=summary['mean_R'],
                std_difference=summary['std_R'], ci95_low=summary['ci95_low_R'],
                ci95_high=summary['ci95_high_R'],
                left_higher=int((difference > ROUNDING_TOLERANCE).sum()),
                tied=int((np.abs(difference) <= ROUNDING_TOLERANCE).sum()),
                left_lower=int((difference < -ROUNDING_TOLERANCE).sum()),
                wilcoxon_statistic=statistic, p_raw=p,
                rank_biserial=effect, test_method=method)


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        adjusted[order] = np.minimum(1, np.maximum.accumulate(
            values[order] * np.arange(len(order), 0, -1)))
    return adjusted


def statistics_tables(observations, designs, attacks, directory):
    """Complete instance blocks; all pairs, without selecting a winning method."""
    keys = ['instance_id', 'design', 'attack']
    expected = {(i, d, a) for i in observations.instance_id.unique()
                for d in designs for a in attacks}
    if (observations.empty or observations.duplicated(keys).any()
            or set(observations[keys].itertuples(index=False, name=None)) != expected):
        raise ValueError('Statistics require complete paired instance/design/attack blocks')
    matrix = observations.set_index(keys)['R'].unstack(['design', 'attack']).sort_index()
    if not np.isfinite(matrix.values).all():
        raise ValueError('Non-finite R in paired statistics')
    values = {(d, a): matrix[(d, a)].values for d in designs for a in attacks}
    for design in designs:
        values[(design, MEAN_ATTACK)] = np.mean([values[(design, a)] for a in attacks], axis=0)
    records = [dict(design=d, attack=a, **describe(values[(d, a)]))
               for d in designs for a in list(attacks) + [MEAN_ATTACK]]
    summary = pd.DataFrame(records)
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / 'cross_attack_summary.csv', index=False)
    for field, filename in [('std_R', 'R_std'), ('ci95_low_R', 'R_ci95_low'), ('ci95_high_R', 'R_ci95_high')]:
        summary.pivot(index='design', columns='attack', values=field).reindex(
            index=designs, columns=attacks).to_csv(directory / (filename + '.csv'))
    formatted = summary.copy()
    formatted['display'] = ['{:.3f} ± {:.3f} (n={})'.format(r.mean_R, r.std_R, r.n_instances)
                            if r.n_instances >= 2 else '{:.3f} (n=1)'.format(r.mean_R)
                            for r in summary.itertuples()]
    formatted.pivot(index='design', columns='attack', values='display').reindex(
        index=designs, columns=list(attacks)+[MEAN_ATTACK]).to_csv(directory / 'R_mean_sd.csv')

    comparisons = []
    for attack in list(attacks) + [MEAN_ATTACK]:
        for left, right in combinations(designs, 2):
            comparisons.append(dict(family='designs:' + attack, comparison='designs',
                evaluation_attack=attack, left=left, right=right,
                **paired_summary(values[(left, attack)], values[(right, attack)])))
    if 'Original' in designs:
        for left, right in combinations(attacks, 2):
            comparisons.append(dict(family='original_attacks', comparison='attacks_on_original',
                evaluation_attack='Original', left=left, right=right,
                **paired_summary(values[('Original', left)], values[('Original', right)])))
    columns = ['family', 'comparison', 'evaluation_attack', 'left', 'right', 'n_pairs',
               'mean_difference', 'std_difference', 'ci95_low', 'ci95_high', 'left_higher',
               'tied', 'left_lower', 'wilcoxon_statistic', 'p_raw', 'rank_biserial', 'test_method']
    paired = pd.DataFrame(comparisons, columns=columns)
    paired['p_holm'] = np.nan
    paired['family_size'] = 0
    for _, group in paired.groupby('family'):
        paired.loc[group.index, 'p_holm'] = holm_adjust(group.p_raw)
        paired.loc[group.index, 'family_size'] = len(group)
    paired.to_csv(directory / 'paired_comparisons.csv', index=False)
    metadata = dict(
        independent_unit='original input network instance; attack columns are not replicates',
        evaluation_attacks=list(attacks), bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED, interval='95% percentile bootstrap of instance means',
        paired_interval='bootstrap of within-instance left minus right differences',
        comparisons='all unordered pairs; Original included; no winner selection',
        holm_families='within each network setting: original attacks, and designs for each evaluation column or overall mean',
        signed_rank='two-sided; discard zeros; average tied ranks; exact conditional signs for <=200 nonzero pairs',
        signed_rank_null='independent paired differences with a distribution symmetric about zero; not a test of the mean alone',
        effect_size='mean_difference is in R units; rank_biserial is signed by left minus right',
        intervals='pointwise, unadjusted; Holm correction applies to p-values, not intervals',
        single_instance='point estimates only; SD, intervals and tests unavailable',
        numerical_tie_tolerance=ROUNDING_TOLERANCE,
        references=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html'])
    (directory / 'statistics_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return summary, paired
