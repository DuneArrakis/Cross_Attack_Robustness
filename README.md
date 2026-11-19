# Cross-attack robustness of degree-preserving network designs

The study evaluates the original network and seven optimized designs under seven node-removal protocols. The [BA500 robustness matrix](results/R_mean.csv) contains the 56 cell means in Fig. 4(a), including FINDER. Each cell averages 50 independent input networks. HDA and DomiRank give seven-attack design means of 0.280439 and 0.290205 on BA500.

## Reproduce the paper

From the repository root, use Linux x86-64 Docker. The image uses Python 3.7.12 and the [pinned CPU dependencies](environment/requirements-cpu.txt), compiles the FINDER extensions, and includes the trained checkpoint and data. The root-level requirements file records the historical GPU environment. Mount a directory to keep the generated results.

```bash
docker build --platform linux/amd64 -t cross-attack-paper .
mkdir -p work
docker run --platform linux/amd64 --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD/work:/work/work" cross-attack-paper \
  python reproduce.py --replay-orders
```

The command checks the archive hashes, verifies the two [empirical input graphs](real_data/README.md) against the original evaluation records, and computes robustness from the 16,912 original-network and optimized-design curves. It also verifies the 700 BA500 construction curves used in Fig. 8 against their measured scores. On the synthetic networks it also reconstructs the curves from saved graphs and removal orders. Omit `--replay-orders` for the faster curve-based check. Both commands regenerate the following outputs.

| Manuscript | Output under `work/` | Source |
| --- | --- | --- |
| Fig. 3 | `figures/curves.pdf` | BA500 curves averaged on a common removal grid, displayed through 50% removal |
| Fig. 4 | `figures/matrix.pdf`, `BA_N500/R_mean.csv`, `BA_N500/defense_ranking.csv` | Raw curves, one design mean per input, paired input resampling |
| Fig. 5 and Appendix B | `figures/topology.pdf`, `figures/topology_uncertainty.pdf` | Rewired and original BA500 edge lists |
| Figs. 6–7 | `figures/feedback.pdf`, `figures/exchange.pdf` | Measured trajectories, candidate statistics and the selected BA30 exchange |
| Fig. 8 | `figures/onion.pdf` | Curves for HDA, DomiRank, Wu–Holme and Modified Onion on 50 matched BA500 degree sequences |
| Table 2 | `figures/design_rows.tex` | Seven-attack means for the original graph, seven optimization targets and two direct constructions. Synthetic cells give the across-network mean and sample SD |
| Appendix A | `figures/original_rows.tex`, `figures/timing_rows.tex` | Original-network scores and measured CPU time for initial and candidate evaluations |

The script compares all Fig. 4 cells, design means and bootstrap endpoints with the published numerical records. It also checks the three LaTeX tables against the rows used in the manuscript. The input network is the resampling unit for the 10,000 bootstrap draws (seed 20261014). For a graph with `N` nodes, each archived curve gives `LCC(q)/N` in the column named `R`. The scalar robustness is the sum of that column over `q = 1, …, N` divided by `N`.

Section 3.2 compares HDA and DomiRank on the five attacks shared by both designs after excluding their two optimization objectives. The script writes these means to `work/BA_N500/five_common_attacks.csv` (HDA 0.264841488, DomiRank 0.266510368). The [common-candidate data](reproduction/diagnostics/feedback/README.md) document a separate comparison of how seven objectives scored the same BA500 swaps.

## Repeat the optimization

For a short end-to-end check, the command below repeats the complete HDA annealing schedule on BA100 input `graph_0` and runs all seven evaluations on both the original and rewired graph. It compares the resulting graph checksum and 14 robustness scores with the archive.

```bash
docker run --platform linux/amd64 --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD/work:/work/work" cross-attack-paper \
  python reproduce.py --check-example
```

For a full synthetic setting, extract the paper inputs and run its configuration in the same mounted directory. For example, the following repeats all 50 BA500 networks with the original search budgets.

```bash
docker run --platform linux/amd64 --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD/work:/work/work" cross-attack-paper sh -c \
  'python reproduce.py --extract-inputs BA_N500 && python -m run pipeline --config reproduction/configs/BA_N500.json'
```

Each [synthetic configuration](reproduction/configs) records its graph-generation seed, optimization and evaluation seed bases, annealing schedule and method options. Each archive contains the actual task seeds. The machine-readable [settings table](reproduction/settings.json) lists all eight settings. CPU hours in Appendix A were measured on the original runs and will depend on the host during a rerun.

## Direct constructions in Fig. 8 and Table 2

The [BA500 construction curves](reproduction/diagnostics/onion/BA500_construction_curves.npz) store integer largest-component sizes for every removal count from 0 to 500. The array axes are design, input network, test attack and removal count. The first three axes have label arrays in the same file. The keys `onion` and `backbone_rank_a3` identify Wu–Holme and Modified Onion. Fig. 8 plots the first half of each mean curve. All scalar scores use the complete sequence.

The [synthetic scores](reproduction/diagnostics/onion/tables/per_graph_scores.csv) and [empirical scores](reproduction/diagnostics/onion/empirical_construction_scores.csv) supply the two construction rows of Table 2. USAir97 has five Modified Onion realizations on its one 332-node degree sequence. The other empirical construction entries each use one realization.

The [construction seeds and graph checksums](reproduction/diagnostics/onion/ring_seeds.csv) cover Modified Onion and an archived random-order control on all 300 synthetic inputs. For each input the degree-ordered and random rings use the same starting degree sequence and seed. The [constructor](run/optimization/onion_ring.py) reserves the ring and leaf edges before pairing the remaining stubs with the degree-layer preference used in the paper.

The following command builds both variants for one input, checks their graph files against the original measurements, and writes the edge lists under `work/rings/`. Choose any setting and instance listed in the seed table.

```bash
docker run --platform linux/amd64 --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD/work:/work/work" cross-attack-paper \
  python reproduce.py --check-rings BA_N100 graph_0
```

## Data and method names

- The six [synthetic archives](reproduction/archives) contain 300 original graphs, 2,100 optimized graphs, all 16,800 removal orders and curves, the evaluation records and search summaries. Each ZIP is self-contained for its setting and listed in the [SHA-256 manifest](reproduction/sha256.json).
- USAir97 and EUpower217 each contribute one empirical observation. Their ZIPs supply all 56 measured curves, evaluation records, search summaries and timing totals per setting. The [input files and provenance](real_data/README.md) identify the 217-node, 320-edge EUpower graph used in Table 2.
- The [diagnostic data](reproduction/diagnostics) include BA500 per-network feedback, bootstrap draws, BA30 before/after evidence, and per-network construction scores. The figures retain the paper’s labels. `Zheng` in the raw records and configuration denotes SRA.
- The historical root scripts (`generate.py`, `curve_repeat.py` and related scripts) and their [four-method example](results/legacy_demo) belong to the earlier demonstration. The paper workflow uses `reproduce.py` and `python -m run`. [Implementation credits](reproduction/attribution.md).
