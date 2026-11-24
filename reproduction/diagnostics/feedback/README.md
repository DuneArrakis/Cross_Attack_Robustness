# Degree-layer feedback on BA500

Figure 6 uses the 50 optimization trajectories for each of the seven objectives. `trajectory_O.csv` records degree-layer connectivity at the sampled points on each trajectory. `feedback_by_instance.csv` counts how often the objective improved among swaps that strengthened or weakened degree-layer connectivity along that search. `bootstrap_indices.npz` contains the network-level draws for the plotted intervals.

The common-candidate comparison uses 20 original BA500 networks. Within each network all seven objectives score exactly the same proposals. The two tables below refer to this separate comparison.

| File | Unit | Columns |
| --- | --- | --- |
| [common_swaps.csv](common_swaps.csv) | One row per proposed swap on one input network | `instance_id` and `proposal_id` identify the swap. `delta_layer_size` is the change in the sum of degree-layer component sizes (500 times the change in the paper's layer statistic O). Each method column records the change in the sum of largest-component sizes over the removal sequence. A positive integer denotes an improvement in robustness. |
| [common_swap_counts.csv](common_swap_counts.csv) | One row per input, objective and direction of layer change | `n_scored` counts common candidates in that direction. `n_improved` counts those that strictly improved the objective. |

For each input and objective, divide the number of improvements by the number scored separately for increasing and decreasing layer connectivity. Subtract the decreasing-group rate from the increasing-group rate, then average the paired differences over the 20 inputs. Swaps with unchanged layer connectivity remain in the observations and contribute to neither rate. The main reproduction command checks the individual counts and writes the seven differences, in percentage points, to `work/BA_N500/common_swap_feedback.csv`. The three comparisons reported to the reviewer are 43.5 for DomiRank, 26.9 for HDA and 13.3 for EI.

The file `common_candidate_bootstrap.npz` contains the 10,000 resamples and the ordered IDs of the 20 original networks used for electronic Tables 08–09 in Appendix C.
