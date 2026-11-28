Statistical supplement to Appendix C
Do More Effective Attacks Produce More Robust Networks?

Each CSV has a header. Network settings are identified by their model and node
count or by the empirical network name. R is the average largest-component
fraction over the full removal sequence. A lower R means more effective
dismantling on a fixed topology. A higher R means greater robustness under a
fixed test. Mean_over_attacks uses seven equally weighted attacks.

01_robustness.csv: Fig. 4 and Table 2. Every original or optimized design
under each test and under their seven-attack mean in all eight settings.
std_R is the sample SD, sem_R the standard error, and ci95_low_R/ci95_high_R
the pointwise 95% percentile bootstrap limits for mean R (10,000 network
resamples, seed 20261014). The six synthetic settings each have 50 networks.
USAir97 and EUpower217 each have one topology. Blank inferential fields
indicate insufficient independent networks.
02_paired_tests.csv: Every pair of original/optimized designs within each
test and for the seven-attack mean, plus every pair of attacks on the
original network. The 1,470 rows cover the six synthetic settings. Differences
and rank-biserial effects are oriented as left minus right. The interval is a
percentile bootstrap of the paired mean difference. W is the smaller signed
rank sum (column wilcoxon_statistic). Zero differences are excluded from the
two-sided test. p_raw and p_holm are raw and Holm-adjusted p-values. The
family comprises 28 design pairs within a test or mean and 21 attack pairs
on the original networks, separately in each setting. Intervals are pointwise.
03_attack_means_over_designs.csv: Mean R for each test after equal weighting
of the seven optimized designs within each network, with network-level
uncertainty and rank. This averages down columns, whereas design robustness
averages across tests.
04_degree_structure.csv: Figs. 5 and 9. Mean, sample SD, SEM and number of
networks containing each degree. knn is mean neighbor degree, ck is mean
clustering, and onion is degree-layer connectivity. The figure shading is
one SD. The table retains every observed degree and network setting.
05_trajectory.csv: Fig. 6(a). Degree-layer connectivity O at five budget
checkpoints. requested_fraction is the plotted target and actual_fraction
the mean fraction reached. Intervals resample the 50 BA500 inputs.
06_swap_outcomes.csv: Fig. 6(b). For swaps that increased, decreased or
preserved O, the fractions that raised, preserved or lowered objective R,
with SDs, percentile limits and counts. Fractions are averaged over networks.
07_swap_direction_differences.csv: Within-network difference between the
rates of improvement for increasing and decreasing O on each search path.
Files 05 to 07 use the same 10,000 network bootstrap draws (seed 20261115).
08_common_candidates.csv and 09_common_candidate_contrasts.csv: The separate
20-network experiment in response R3-3. Each objective scored identical
candidate swaps on the original BA500 topology. Both files report paired
network bootstrap intervals. In 08 the directional rate difference is
repeated for the two strata. In 09 mean_difference compares these differences
between the two objectives. All rates are fractions; multiply by 100 for
percent or percentage points.
10_construction_scores.csv: Fig. 8 and Table 2. Mean R, sample SD and
two-sided Student t intervals for Wu-Holme and Modified Onion across the 50
networks in each synthetic setting. R7 denotes the seven-attack mean, R5 the
original five, and R6_without_DomiRank the leave-DomiRank-out mean.
11_construction_contrasts.csv: Modified Onion minus Wu-Holme on the same 50
inputs, with two-sided paired t intervals for mean differences and two-sided
Wilcoxon signed-rank tests. For each metric the original Holm family contained
18 tests (three construction contrasts across six settings, including a
random-ring control). The published adjusted p-values retain that family.
12_wu_holme_paired_tests.csv: Comparisons of Wu-Holme with the original and
seven optimized designs. These use the bootstrap and signed-rank analysis in
02, with 36 design pairs per original nine-design family.
13_empirical_constructions.csv: Attack scores by construction realization on
each empirical degree sequence. The five Modified Onion realizations on
USAir97 share one input. They are averaged within that network in Table 2.
