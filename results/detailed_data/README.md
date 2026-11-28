# Experimental records for the cross-attack study

Use the CSVs below to trace a reported result back to its inputs. The six synthetic settings each have 50 independent networks. USAir97 and EUpower217 each have one observed network. All seven test attacks have equal weight in the design mean. The CSVs in this directory use the manuscript label SRA; original archives use `Zheng`.

| Question | Files | How to read them |
| --- | --- | --- |
| How robust was each design on each input? | [Per-input robustness](per_input_robustness.csv) · [raw curves by setting](../../reproduction/archives) | One row per setting, input and design. The seven attack columns contain full-sequence robustness scores. The 2,416 rows contain 16,912 measured scores. Pair designs by setting and input. |
| Does rewiring change the attack order? | [Attack ranks and unrounded mean scores](attack_ranks.csv) | For each setting and design, rank 1 is the most effective test attack (smallest mean R). A positive `rank_change` means the attack dropped relative to its rank on the original network. Ties share the smallest occupied rank. |
| How much computation did each optimization use? | [Individual optimization runs](optimization_runs.csv) · [runtime by stage](runtime_by_stage.csv) | The first table records the objective calls, seeds, iterations and acceptance counts for each of 2,114 runs. For Table 4, sum `cpu_seconds_sum` from the `sa_initial` and `sa_candidate` stages for each setting and attack, then divide by 3,600. |
| How variable are the degree-resolved BA500 measurements? | [Structure per network and degree](BA500_degree_structure.csv) · [nodes per degree class](BA500_degree_counts.csv) | The metric labels `knn`, `ck` and `onion` represent mean neighbor degree, clustering and degree-layer connectivity. Average over networks where the degree occurs. |
| Which candidate swaps contributed to Fig. 6? | [Individual candidate records](BA500_candidates.zip) · [per-input feedback counts and bootstrap draws](../../reproduction/diagnostics/feedback) | The ZIP contains every recorded proposal along the 350 BA500 optimization trajectories. See the array definitions below. The separate common-candidate comparison on 20 inputs is in [common_swaps.csv](../../reproduction/diagnostics/feedback/common_swaps.csv). |
| Where are the paired comparisons, intervals and constructions? | [13 electronic tables with field notes](appendix_c) · [synthetic construction scores](../../reproduction/diagnostics/onion/tables/per_graph_scores.csv) · [empirical construction scores](../../reproduction/diagnostics/onion/empirical_construction_scores.csv) | The electronic tables match the [submission supplement](../appendix_c_statistics.zip). For the BA30 exchange in Fig. 7, see the [before/after records](../../reproduction/diagnostics/exchange). |

## BA500 candidate arrays

Each `graph_i/method.npz` holds one trajectory. Matrices have an iteration axis (2,052 rows) and a proposal axis (10 columns). The suffix `sum_S` is the integer sum of the connected degree-layer component sizes; dividing by 500 yields the paper's layer statistic O. The suffix `sum_LCC` is the sum of largest-component sizes across the full node-removal sequence; dividing by 500² yields R.

| Array prefix | Observation |
| --- | --- |
| `base_` | State at the start of the batch. All ten proposals in that batch are generated from this state. |
| `candidate_` | Scores for each proposed swap. Subtract the batch baseline to obtain the changes used in Fig. 6(b). Cast unsigned arrays to a signed integer type before subtraction. |
| `decision_` | Current state just before the acceptance decision for that proposal. |
| `current_`, `best_` | Current state and best-so-far state at iteration boundaries. Entry 0 is the initial state. |
| `valid_swap`, `accepted` | Boolean flags for a changed graph and an accepted proposal. |

From the repository root, run the following check with Python 3 and NumPy. It reads the archive directly and verifies all three directions of layer change against the per-input counts and the five reported trajectory checkpoints.

```sh
python results/detailed_data/verify_candidates.py
```

The [SHA-256 manifest](SHA256SUMS) covers the data files. From this directory, run `sha256sum -c SHA256SUMS` to verify a downloaded copy.
