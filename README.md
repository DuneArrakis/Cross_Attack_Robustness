# Cross Attack Robustness

This repository contains the experimental code for evaluating cross-attack
robustness of complex networks. The workflow generates or loads networks,
optimizes network topology against several attack methods, compares robustness
curves, produces an onion-structure rewiring baseline, and computes topology
diagnostics for the resulting graphs.

## Repository Structure

```text
.
|-- generate.py                 # Generate synthetic ER/BA/WS networks
|-- curve_repeat.py             # Main cross-attack robustness experiment
|-- onion_rewire.py             # Degree-preserving onion-style rewiring baseline
|-- topo_index_statistic.py     # Topology diagnostics and plots
|-- HDA.py                      # High-degree adaptive attack baseline
|-- CI.py                       # Collective Influence attack
|-- CoreHD.py                   # CoreHD attack
|-- EI.py                       # Explosive Immunization attack
|-- finder_wrapper.py           # Wrapper around FINDER_ND
|-- FINDER_ND/                  # FINDER model, Cython extensions, and wrapper code
|-- real_data/                  # Example real-world edge-list networks
|-- requirements.txt            # Python package dependencies
`-- results/                    # Generated outputs, created by the scripts
```

## Environment

The dependencies can be installed with:

```bash
pip install -r requirements.txt
```

The bundled FINDER extensions are precompiled for Python 3.7 on Windows
(`*.cp37-win_amd64.pyd`). For the easiest reproduction, use a Python 3.7
environment matching the exported dependencies. If using another Python version
or operating system, rebuild the Cython extensions in `FINDER_ND/` before
running the FINDER-based experiments.

The FINDER wrapper expects the checkpoint files under:

```text
FINDER_ND/models/nrange_30_50_iter_78000.ckpt.*
```

## Input Graph Format

All main scripts expect undirected edge-list files. Each line should contain at
least two node ids:

```text
u v
```

or the repository's default format:

```text
u v {}
```

Files used by the main workflow should be named `graph_*.txt` and placed in the
`generate/` directory. Node ids should be integers. Isolated nodes cannot be
recovered from an edge-only file unless they are represented elsewhere.

## Workflow

### 1. Prepare Networks

Generate synthetic networks:

```bash
python generate.py
```

Edit the configuration at the top of `generate.py` to choose the number of
graphs, graph size, random seed, and network model:

- `GRAPH_TYPE = "ER"` for Erdos-Renyi networks
- `GRAPH_TYPE = "BA"` for Barabasi-Albert networks
- `GRAPH_TYPE = "WS"` for Watts-Strogatz networks

Alternatively, use real-world networks by copying files from `real_data/` into
`generate/` and naming them as `graph_*.txt`, for example:

```bash
mkdir -p generate
cp real_data/graph_USAir97_0.txt generate/graph_USAir97_0.txt
```

Important: `generate.py` clears existing `generate/*.txt` files before writing
new synthetic graphs. Do not run it after copying real-world data unless you
intend to overwrite the current inputs.

### 2. Run Cross-Attack Robustness Experiments

Run:

```bash
python curve_repeat.py
```

This script reads `generate/graph_*.txt`, optimizes each input graph against
each attack method using simulated annealing, then evaluates every optimized
graph under every attack method.

Key settings are at the top of `curve_repeat.py`:

- `GRAPH_DIR`: input graph directory, default `generate`
- `OUT_DIR`: output directory, default `results`
- `SA_CONFIG`: simulated annealing parameters
- `POOL_RATIO`: multiprocessing pool size as a fraction of CPU count
- `SAVE_FINAL_GRAPHS`: whether to save final optimized topologies

Main outputs:

```text
results/R_mean.csv
results/sa_progress.csv
results/curves_each/
results/curves_avg/
results/final_graphs/<graph_name>/Original.txt
results/final_graphs/<graph_name>/Optimized_HDA.txt
results/final_graphs/<graph_name>/Optimized_CI.txt
results/final_graphs/<graph_name>/Optimized_CoreHD.txt
results/final_graphs/<graph_name>/Optimized_EI.txt
results/final_graphs/<graph_name>/Optimized_FINDER.txt
```

The current version saves the robustness matrix `R_mean.csv` and robustness
curves.

### 3. Run Onion Rewiring Baseline

Run:

```bash
python onion_rewire.py
```

This script reads the same `generate/graph_*.txt` files and creates a
degree-preserving onion-style rewired graph for each input.

Main output:

```text
results/final_graphs/<graph_name>/Optimized_onion.txt
```

The main configuration is at the top of `onion_rewire.py`, especially `A`,
`MIN_PAIR_ATTEMPTS`, `PAIR_ATTEMPTS_FACTOR`, `MAX_RESHUFFLE_TRIES`, and
`RESTARTS`.

### 4. Analyze Network Topology

Run:

```bash
python topo_index_statistic.py
```

This script computes topology diagnostics for selected files under
`results/final_graphs/`, including:

- average nearest-neighbor degree by degree class
- clustering coefficient by degree class
- onion-structure statistic `S_k / N_k`

Main outputs:

```text
results/onion_metrics_outputs/per_graph_metrics.csv
results/onion_metrics_outputs/per_method_degree_metrics_long.csv
results/onion_metrics_outputs/topology_diagnostics_triptych.png
```

If you want to include the onion baseline in the topology plots, make sure the
`Optimized_onion.txt` pattern is enabled in `TARGET_FILE_PATHS` near the top of
`topo_index_statistic.py`.

## Attack Methods and References

This repository includes or wraps the following attack methods:

- HDA: high-degree adaptive attack baseline implemented in `HDA.py`
- CI: Collective Influence attack, adapted with reference to
  [zhfkt/ComplexCi](https://github.com/zhfkt/ComplexCi)
- CoreHD: CoreHD attack, adapted with reference to
  [hcmidt/corehd](https://github.com/hcmidt/corehd)
- FINDER: deep reinforcement learning based attack, wrapped from `FINDER_ND/`
  with reference to [FFrankyy/FINDER](https://github.com/FFrankyy/FINDER)
- EI: Explosive Immunization attack, adapted with reference to
  [pclus/explosive-immunization](https://github.com/pclus/explosive-immunization)

Please also check the licenses and citation requirements of the referenced
upstream repositories before redistribution.

## Reproducibility Notes

- Randomness appears in synthetic graph generation, simulated annealing, EI, and
  onion rewiring. Set the seed values in the corresponding scripts when exact
  reproducibility is required.
- `curve_repeat.py` can be computationally expensive because each graph is
  optimized separately for each attack method.
- FINDER may create temporary files under `data/` and `finder_output/`; the
  wrapper removes per-run temporary files by default.
- Generated folders such as `results/`, `data/`, `finder_output/`, and
  `__pycache__/` are runtime artifacts.

## Minimal Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic input networks
python generate.py

# 3. Run robustness optimization and evaluation
python curve_repeat.py

# 4. Generate onion rewiring baseline
python onion_rewire.py

# 5. Compute topology diagnostics
python topo_index_statistic.py
```
