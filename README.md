# Cross-attack robustness of degree-preserving network designs

The revised study evaluates the original network and seven optimized designs under seven node-removal protocols. The [BA500 robustness matrix](results/R_mean.csv) contains the 56 cell means in Fig. 4(a), including FINDER. Each cell averages 50 independent input networks. HDA and DomiRank give seven-attack design means of 0.280439 and 0.290205 on BA500.

## Reproduce the paper

From the repository root, use Linux x86-64 Docker. The image pins Python 3.7.12 and the CPU dependencies, compiles the FINDER extensions, and includes the trained checkpoint and data. Mount a directory to keep the generated results.

```bash
docker build --platform linux/amd64 -t cross-attack-paper .
mkdir -p work
docker run --platform linux/amd64 --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD/work:/work/work" cross-attack-paper \
  python reproduce.py --replay-orders
```

The command checks the archive hashes and computes robustness from all 16,912 full-resolution curves. On the synthetic networks it also reconstructs the curves from saved graphs and removal orders. Omit `--replay-orders` for the faster curve-based check. Both commands regenerate the following outputs.

| Manuscript | Output under `work/` | Source |
| --- | --- | --- |
| Fig. 3 | `figures/curves.pdf` | BA500 curves averaged on a common removal grid |
| Fig. 4 | `figures/matrix.pdf`, `BA_N500/R_mean.csv`, `BA_N500/defense_ranking.csv` | Raw curves, one design mean per input, paired input resampling |
| Fig. 5 and Appendix B | `figures/topology.pdf`, `figures/topology_uncertainty.pdf` | Rewired and original BA500 edge lists |
| Figs. 6–8 | `figures/feedback.pdf`, `figures/exchange.pdf`, `figures/onion.pdf` | Measured trajectories and candidate statistics, the selected BA30 exchange, and paired construction scores |
| Table 2 | `figures/design_rows.tex` | Within-network seven-attack means, followed by the across-network mean and sample SD |
| Appendix A | `figures/original_rows.tex`, `figures/timing_rows.tex` | Original-network scores and measured CPU time for initial and candidate evaluations |

The script compares all Fig. 4 cells, design means and bootstrap endpoints with the published numerical records. It also checks the three LaTeX tables against the rows used in the manuscript. The input network is the resampling unit for the 10,000 bootstrap draws (seed 20261014). For a graph with `N` nodes, each archived curve gives `LCC(q)/N` in the column named `R`. The scalar robustness is the sum of that column over `q = 1, …, N` divided by `N`.

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

## Data and method names

- The six [synthetic archives](reproduction/archives) contain 300 original graphs, 2,100 optimized graphs, all 16,800 removal orders and curves, the evaluation records and search summaries. Each ZIP is self-contained for its setting and listed in the [SHA-256 manifest](reproduction/sha256.json).
- USAir97 and EUpower217 each contribute one empirical observation. Their ZIPs supply all 56 measured curves, evaluation records, search summaries and timing totals per setting. The revised EUpower case has 217 nodes and 320 edges.
- The [diagnostic data](reproduction/diagnostics) include BA500 per-network feedback, bootstrap draws, BA30 before/after evidence, and per-network construction scores. The figures retain the paper’s labels. `Zheng` in the raw records and configuration denotes SRA.
- The historical root scripts (`generate.py`, `curve_repeat.py` and related scripts) and their [four-method example](results/legacy_demo) belong to the earlier demonstration. The paper workflow uses `reproduce.py` and `python -m run`. [Implementation credits](reproduction/attribution.md).
