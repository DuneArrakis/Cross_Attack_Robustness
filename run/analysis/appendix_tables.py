"""Typeset Appendix C from the accompanying statistical supplement."""
import csv
import io
from zipfile import ZipFile

from .paper_style import METHODS, display_name

METHOD_NAMES = [display_name(method) for method in METHODS]
MEAN = 'Mean_over_attacks'
CONTRASTS = [
    ('HDA', 'CI', MEAN), ('CI', 'HDA', 'CI'), ('HDA', 'EI', 'EI'),
    ('DomiRank', 'EI', 'EI'), ('DomiRank', 'HDA', MEAN), ('CI', 'CoreHD', MEAN)]


def read_table(archive, name):
    with archive.open(name + '.csv') as raw:
        return list(csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8')))


def estimate(row, field, low='ci95_low', high='ci95_high', digits=3, scale=1):
    numbers = [format(scale * float(row[k]), '.%df' % digits) for k in (field, low, high)]
    return r'$%s\,[%s, %s]$' % tuple(numbers)


def pvalue(value):
    p = float(value)
    if p < .001:
        return '$<0.001$'
    return '$%s$' % format(p, '.3f' if p >= .1 else '.2g')


def write_rows(folder, name, rows):
    lines = [' & '.join(row) for row in rows]
    (folder/(name+'.tex')).write_text((' ' + r'\\' + '\n').join(lines) + '%\n')


def write_appendix_tables(supplement, folder):
    with ZipFile(supplement) as archive:
        scores = {(r['design'], r['attack']): r for r in read_table(archive,'01_robustness')
                  if r['setting'] == 'BA_N500'}
        # Two blocks of four columns keep the interval endpoints readable at print size.
        rows = []
        tests = METHOD_NAMES + [MEAN]
        for start in (0, 4):
            group = tests[start:start+4]
            rows.append([r'\midrule Design' if start else 'Design'] +
                        ['Seven-attack mean' if m == MEAN else m for m in group])
            for design in ['Original'] + METHOD_NAMES:
                label = (r'\midrule ' if design == 'Original' else '') + design
                rows.append([label] + [estimate(scores[design, m], 'mean_R',
                                                'ci95_low_R', 'ci95_high_R', 4) for m in group])
        write_rows(folder, 'robustness_intervals', rows)

        pairs = {(r['left'], r['right'], r['evaluation_attack']): r
                 for r in read_table(archive,'02_paired_tests')
                 if r['setting'] == 'BA_N500' and r['comparison'] == 'designs'}
        rows = []
        for left, right, attack in CONTRASTS:
            key = (left, right, attack)
            reverse = key not in pairs
            pair = pairs[right, left, attack] if reverse else pairs[key]
            delta, lo, hi = [float(pair[k]) for k in ('mean_difference','ci95_low','ci95_high')]
            if reverse:
                delta, lo, hi = -delta, -hi, -lo
            rows.append(['%s $-$ %s' % (left, right), 'Seven-attack mean' if attack == MEAN else attack,
                         '$%.5f$' % delta, '$[%.5f, %.5f]$' % (lo,hi), pvalue(pair['p_holm'])])
        write_rows(folder, 'paired_robustness', rows)

        path = {(r['attack'], float(r['requested_fraction'])): r
                for r in read_table(archive, '05_trajectory')}
        rows = [[attack] + [estimate(path[attack, fraction], 'mean') for fraction in (.25,.5,.75,1.)]
                for attack in METHOD_NAMES]
        write_rows(folder, 'trajectory_intervals', rows)

        outcomes = {(r['attack'],r['stratum']): r for r in read_table(archive, '06_swap_outcomes')}
        differences = {r['attack']:r for r in read_table(archive,'07_swap_direction_differences')}
        rows = []
        for attack in METHOD_NAMES:
            row = [estimate(outcomes[attack,k], 'mean_p', digits=1, scale=100)
                   for k in ('O_plus','O_minus','O_zero')]
            rows.append([attack] + row + [estimate(differences[attack], 'mean_difference',
                                                   digits=1, scale=100)])
        write_rows(folder, 'swap_intervals', rows)

        means = {(r['family'], r['design']):r for r in read_table(archive,'10_construction_scores')
                 if r['metric'] == 'R7'}
        differences = {r['family']:r for r in read_table(archive,'11_construction_contrasts')
                       if r['metric'] == 'R7'}
        rows = []
        for setting in ('BA_N100','BA_N300','BA_N500','BA_N1000','ER_N500','WS_N500'):
            row = [setting.replace('_N','')]
            row += [estimate(means[setting,design], 'mean', digits=4)
                    for design in ('Wu-Holme','Modified Onion')]
            row += [estimate(differences[setting], 'mean', digits=4), pvalue(differences[setting]['p_holm'])]
            rows.append(row)
        write_rows(folder, 'construction_intervals', rows)
