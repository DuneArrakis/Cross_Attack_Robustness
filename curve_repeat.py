# repeat_avg.py - Batch experiments and averaged robustness curve plotting
# ==================================================
# Overview
#   1. Read generate/graph_*.txt into igraph graphs.
#   2. Optimize each original graph for every attack method using simulated annealing.
#   3. Attack Original + every optimized graph with all attack methods and record curves.
#   4. Average curves pointwise across original graphs and plot one averaged curve.
#   5. Save R statistics and curve CSV files.
# --------------------------------------------------

from __future__ import annotations

# Standard library
import os, glob, math, random, gc, multiprocessing as mp

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from finder_wrapper import FinderWrapperND

_finder_singleton = None  # Per-process singleton

def finder_R(G: ig.Graph) -> float:
    """
    Lazily load FINDER:
    - Create the FinderWrapperND instance on the first call.
    - Reuse the same instance afterwards.
    """
    global _finder_singleton
    if _finder_singleton is None:
        _finder_singleton = FinderWrapperND()
        # print("FINDER loaded")
    return _finder_singleton.robustness_R(G)

def finder_curve(G: ig.Graph) -> list[float]:
    """
    Reuse the same singleton when the full FINDER robustness sequence is needed.
    """
    global _finder_singleton
    if _finder_singleton is None:
        _finder_singleton = FinderWrapperND()
        # print("FINDER loaded")
    return _finder_singleton.robustness_sequence(G)

# Attack algorithm modules
#   If some modules are unavailable, wrap them in try/except or comment out the
#   corresponding dictionary entries below.
import CI, HDA, EI, CoreHD
###############################################################################
# 1. Configuration ############################################################
###############################################################################

GRAPH_DIR   = "generate"      # Directory containing original graphs
#GRAPH_DIR   = "generate_real"
OUT_DIR     = "results"       # Output directory
#OUT_DIR     = "results_real"
PTS_RS      = 101             # Number of resampled curve points over [0, 1]
POOL_RATIO  = 0.3             # Use CPU_count * POOL_RATIO worker processes
SA_CONFIG   = dict(           # Simulated annealing parameters for N=500
    T_max      = 4.5 * 0.02,
    T_min      = 1e-10,
    N_per_iter = 10,          # Number of neighbors tested per iteration
    alpha      = 0.99,         # Cooling coefficient
    max_iter   = 2070          # Iteration limit
)

###############################################################################
# 1.5 Save final graphs and attack orders #####################################
###############################################################################

SAVE_FINAL_GRAPHS   = True   # Save final topologies for Original/Optimized_*
SAVE_ATTACK_ORDERS  = False  # Save deletion orders for each final topology
GRAPH_FMT_WITH_BRACE = True  # True: each line is "u v {}"; False: "u v"

FINAL_GRAPH_DIRNAME = "final_graphs"     # Subdirectory under OUT_DIR
ATTACK_ORDER_DIRNAME = "attack_orders"   # Subdirectory under OUT_DIR
_finder_order_singleton = None

###############################################################################
# SA trajectory dataset saving
###############################################################################
SA_TRAJ_SAVE = dict(
    enabled=False,              # Save the best graph from SA iterations
    save_every=1,               # Save check interval in outer iterations
    only_if_improved=True,      # Save only when best_R improves in this iteration
    with_brace=GRAPH_FMT_WITH_BRACE,
)
# Set only_if_improved to False to save every iteration, even without improvement.

###############################################################################
# Save top-percent candidate graphs at the same SA temperature
###############################################################################
SA_TOPK_SAVE = dict(
    enabled=False,             # Enable saving top candidates at each temperature
    top_ratio=0.50,            # Save ratio: 0.50 keeps the top 50%
    max_keep=None,             # Optional maximum number to keep; None means no cap
    only_if_best_improved=False,# Save only when best_R improves in this iteration
    with_brace=GRAPH_FMT_WITH_BRACE,
)

def save_igraph_edgelist(G: ig.Graph, path: str, with_brace: bool = True) -> None:
    """Save an igraph graph as a txt edge list."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    edges = G.get_edgelist()
    with open(path, "w", encoding="utf-8") as f:
        if with_brace:
            for u, v in edges:
                f.write(f"{u} {v} {{}}\n")
        else:
            for u, v in edges:
                f.write(f"{u} {v}\n")

def save_sa_snapshot(best_G: ig.Graph,
                     snapshot_dir: str,
                     it: int,
                     best_R: float,
                     cur_R: float,
                     accepted: int,
                     improved_flag: bool) -> None:
    """
    Save the best graph from simulated annealing iterations as training data.
    File names use only the iteration number to stay stable and sortable.
    A lightweight meta.csv records best_R/cur_R and related metadata.
    """
    os.makedirs(snapshot_dir, exist_ok=True)

    # 1) Save graph
    gpath = os.path.join(snapshot_dir, f"best_it_{it:04d}.txt")
    save_igraph_edgelist(best_G, gpath, with_brace=SA_TRAJ_SAVE["with_brace"])

    # 2) Append meta information
    meta_csv = os.path.join(snapshot_dir, "meta.csv")
    write_header = not os.path.exists(meta_csv)
    with open(meta_csv, "a", encoding="utf-8") as f:
        if write_header:
            f.write("it,best_R,cur_R,accepted,improved,graph_path\n")
        f.write(f"{it},{best_R:.10f},{cur_R:.10f},{accepted},{int(improved_flag)},{gpath}\n")

def save_sa_topk_candidates(cands: list[tuple[ig.Graph, float]],
                            snapshot_dir: str,
                            it: int,
                            top_ratio: float = 0.5,
                            max_keep: int | None = None,
                            with_brace: bool = True) -> None:
    """
    Save the top-percent candidate topologies from the same outer iteration.
    cands: [(G, score), ...]
    Output:
      snapshot_dir/Top_it_0001/rank_01.txt ...
      snapshot_dir/meta_topk.csv  records it/rank/score/path
    """
    if not cands:
        return

    # Filter invalid scores and empty graphs
    clean = []
    for G, s in cands:
        if G is None:
            continue
        try:
            sf = float(s)
        except Exception:
            continue
        if not np.isfinite(sf):
            continue
        clean.append((G, sf))
    if not clean:
        return

    # Sort by score in descending order
    clean.sort(key=lambda x: x[1], reverse=True)

    # Compute the number to save: ceil(n * ratio), at least 1
    ratio = float(top_ratio)
    ratio = max(0.0, min(1.0, ratio))
    k = int(math.ceil(len(clean) * ratio))
    k = max(1, k)
    if max_keep is not None:
        k = min(k, int(max_keep))

    # Use a separate directory per iteration to avoid long names or collisions
    it_dir = os.path.join(snapshot_dir, f"Top_it_{it:04d}")
    os.makedirs(it_dir, exist_ok=True)

    meta_csv = os.path.join(snapshot_dir, "meta_topk.csv")
    write_header = not os.path.exists(meta_csv)

    with open(meta_csv, "a", encoding="utf-8") as f:
        if write_header:
            f.write("it,rank,score,graph_path\n")

        for r, (G, score) in enumerate(clean[:k], start=1):
            gpath = os.path.join(it_dir, f"rank_{r:02d}.txt")
            save_igraph_edgelist(G, gpath, with_brace=with_brace)
            f.write(f"{it},{r},{score:.10f},{gpath}\n")

def save_order_csv(order: list[int], path: str) -> None:
    """Save a deletion order with two columns: step,node."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("step,node\n")
        for i, v in enumerate(order):
            f.write(f"{i},{v}\n")

def save_curve_csv(curve: list[float], out_path: str, x_mode: str = "k_over_N") -> None:
    """
    Save a robustness curve to CSV.
    x_mode:
      - "k": step index k=0..len(curve)-1
      - "k_over_N": q = k/(len(curve)-1), mapped to [0,1]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(curve)
    if n == 0:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("x,R\n")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("x,R\n")
        for k, r in enumerate(curve):
            if x_mode == "k":
                x = k
            else:
                x = (k / (n - 1)) if n > 1 else 0.0
            f.write(f"{x},{r}\n")

def get_attack_order(atk_name: str, G: ig.Graph) -> list[int] | None:
    """
    Get the attack order, i.e. deletion order, through a unified interface.
    Return None when the method cannot provide an order or the order is unused.
    """
    global _finder_order_singleton

    if atk_name == "HDA":
        NodeList, _ = HDA.simulate_attack_once_curve_optimized(G)
        return NodeList

    if atk_name == "CI":
        # Use ci_deletion_order when CI.py provides it.
        if hasattr(CI, "ci_deletion_order"):
            return CI.ci_deletion_order(G)
        return None

    if atk_name == "CoreHD":
        # Use the marker variant when CoreHD.py can return full_order.
        if hasattr(CoreHD, "corehd_robustness_curve_igraph_with_marker"):
            _curve, _qend, full_order = CoreHD.corehd_robustness_curve_igraph_with_marker(G)
            return list(full_order)
        return None

    if atk_name == "EI":
        # Use the internal EI implementation that returns an order and curve.
        if hasattr(EI, "_explosive_immunization_with_curve"):
            deletion_order, _curve = EI._explosive_immunization_with_curve(G)
            return list(deletion_order)
        return None

    if atk_name == "FINDER":
        try:
            if _finder_order_singleton is None:
                _finder_order_singleton = FinderWrapperND()
            return _finder_order_singleton.attack_order(G)
        except Exception as e:
            print(f"!! Failed to get FINDER attack order: {e}")
            return None

    return None

###############################################################################
# 2. Attack method interfaces #################################################
###############################################################################

# attack_methods: functions that return a single robustness score
attack_methods = {
    "HDA":        HDA.simulate_attack_once_optimized,
    "CI":         CI.ci_robustness_total,
    "CoreHD":     CoreHD.corehd_R_igraph,
    "EI":         EI.ei_robustness_total,
    "FINDER":     finder_R,
}

# attack_curve_methods: functions that return the full robustness curve
attack_curve_methods = {
    "HDA":    HDA.hda_curve_only,
    "CI":     CI.ci_robustness_curve,
    "CoreHD": CoreHD.corehd_robustness_curve_igraph,
    "EI":    EI.ei_robustness_curve,
    "FINDER": finder_curve,
}

method_names = list(attack_methods)   # Attack method names
num_methods  = len(method_names)

###############################################################################
# 3. Utility Functions ########################################################
###############################################################################

def load_graph(path: str) -> ig.Graph:
    """Read an undirected graph from an edge-list file."""
    edges, max_id = [], -1
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            u, v, *_ = line.split()
            u, v = int(u), int(v)
            edges.append((u, v))
            max_id = max(max_id, u, v)
    return ig.Graph(n=max_id + 1, edges=edges, directed=False)


def random_edge_reconnect(G: ig.Graph) -> None:
    """Perform one valid random edge swap to generate a neighboring solution."""
    edges = G.get_edgelist()
    if len(edges) < 2:
        return
    for _ in range(100):                # Try at most 100 times
        e1, e2 = random.sample(edges, 2)
        a, b = e1
        c, d = e2
        if len({a, b, c, d}) < 4:
            continue
        if not (G.are_connected(a, c) or G.are_connected(b, d)):
            G.delete_edges([e1, e2])
            G.add_edges([(a, c), (b, d)])
            return                      # Exit after a successful swap


def resample_curve(curve: list[float] | np.ndarray,
                   pts: int = PTS_RS) -> np.ndarray:
    """Resample a variable-length curve to evenly spaced points over [0,1]."""
    if len(curve) == 0:
        return np.zeros(pts)
    x_old = np.linspace(0, 1, num=len(curve))
    x_new = np.linspace(0, 1, num=pts)
    return np.interp(x_new, x_old, curve)

def print_progress(prefix: str, i: int, total: int, bar_len: int = 28) -> None:
    """Print a same-line progress bar using only the standard library."""
    i = min(i, total)
    filled = int(bar_len * i / total) if total > 0 else bar_len
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = (100.0 * i / total) if total > 0 else 100.0
    print(f"\r{prefix} [{bar}] {i}/{total} ({pct:5.1f}%)", end="", flush=True)
    if i >= total:
        print()  # Newline

def append_sa_log(csv_path: str, row: dict) -> None:
    """Append one row to the SA progress log, writing a header if needed."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)

###############################################################################
# 4. Parallel Simulated Annealing #############################################
###############################################################################

def _sa_worker(args):
    """
    Worker for the process pool: swap one random edge and return the new graph
    together with its target robustness.
    args = (Graph, target_method_name)
    """
    G, target = args
    try:
        random_edge_reconnect(G)              # Random perturbation
        score = attack_methods[target](G)     # Objective score
        return G, score
    except Exception as e:
        print(f"[worker] {e}")
        return None, -np.inf

def optimize_sa(*,
                G_init: ig.Graph,
                target: str,
                pool: mp.pool.Pool,
                T_max: float,
                T_min: float,
                N_per_iter: int,
                alpha: float,
                max_iter: int,
                progress_prefix: str | None = None,
                log_R: bool = True,
                snapshot_dir: str | None = None) -> ig.Graph:
    """
    Single-objective simulated annealing. Return the graph that maximizes
    robustness under the target attack method.
    If snapshot_dir is set and SA_TRAJ_SAVE.enabled=True, best_G is saved after
    each outer iteration.
    """
    best_G = G_init.copy()
    best_R = attack_methods[target](best_G)

    G_cur = best_G.copy()
    cur_R = best_R
    T = T_max

    desc = progress_prefix or f"Optimizing defense={target}"

    for it in range(1, max_iter + 1):
        # Termination condition
        if T_min - T > 1e-20:
            break

        tasks = [(G_cur.copy(), target) for _ in range(N_per_iter)]
        results = pool.map(_sa_worker, tasks)

        accepted = 0
        improved_flag = False  # Whether best_R improved in this iteration

        for cand_G, R in results:
            if cand_G is None:
                continue

            delta = R - cur_R
            if delta > 0 or random.random() < math.exp(delta / max(T, 1e-300)):
                G_cur = cand_G
                cur_R = R
                accepted += 1

                if R - best_R > 1e-10:
                    best_G, best_R = cand_G.copy(), R
                    improved_flag = True

        # Save the best graph for this temperature iteration
        if SA_TRAJ_SAVE.get("enabled", False) and snapshot_dir is not None:
            every = int(SA_TRAJ_SAVE.get("save_every", 1))
            only_imp = bool(SA_TRAJ_SAVE.get("only_if_improved", True))
            if (it % max(every, 1) == 0) and ((not only_imp) or improved_flag):
                try:
                    save_sa_snapshot(
                        best_G=best_G,
                        snapshot_dir=snapshot_dir,
                        it=it,
                        best_R=float(best_R),
                        cur_R=float(cur_R),
                        accepted=int(accepted),
                        improved_flag=bool(improved_flag)
                    )
                except Exception as e:
                    print(f"    !! SA snapshot save failed ({target} it={it}): {e}")
        # Save top-percent candidate graphs for this temperature iteration
        if SA_TOPK_SAVE.get("enabled", False) and snapshot_dir is not None:
            if (not SA_TOPK_SAVE.get("only_if_best_improved", False)) or improved_flag:
                try:
                    # results is [(cand_G, R), ...] returned by pool.map
                    save_sa_topk_candidates(
                        cands=[(cand_G, R) for cand_G, R in results if cand_G is not None],
                        snapshot_dir=snapshot_dir,
                        it=it,
                        top_ratio=SA_TOPK_SAVE.get("top_ratio", 0.5),
                        max_keep=SA_TOPK_SAVE.get("max_keep", None),
                        with_brace=SA_TOPK_SAVE.get("with_brace", True),
                    )
                except Exception as e:
                    print(f"    !! SA top-k save failed ({target} it={it}): {e}")

        T *= alpha

        # Progress bar
        print_progress(desc, it, max_iter)

        if log_R and (it == 1 or it % 10 == 0 or it == max_iter):
            print(f"    [SA] it={it:4d}  T={T: .3e}  cur_R={cur_R:.6f}  best_R={best_R:.6f}  "
                  f"accept={accepted}/{N_per_iter}  improved={int(improved_flag)}")

    # best_R = attack_methods[target](best_G)
    # print(f"\n{best_R}")
    return best_G



def get_optimized_graphs(G0: ig.Graph,
                         pool: mp.pool.Pool,
                         *,
                         gname: str,
                         out_dir: str,
                         final_graphs_dir: str) -> dict[str, ig.Graph]:
    """
    Return a dictionary:
        { 'Original': G0,
          'Optimized_xxx': final optimized graph for xxx, ... }
    Also save each defense target's SA best-graph trajectory to:
        final_graphs/<gname>/Trajectory_<defense>/best_it_0001.txt ...
    """
    graphs = {"Original": G0}
    print(">> Starting simulated annealing optimization...")

    sa_log_csv = os.path.join(out_dir, "sa_progress.csv")

    for m in method_names:
        print(f"  - Optimizing for defense = {m}")

        # Baseline on the original graph
        try:
            R_orig_self = float(attack_methods[m](G0))
        except Exception as e:
            print(f"    !! Failed to compute baseline R (Original under {m}): {e}")
            R_orig_self = float("nan")

        # Trajectory save directory for this defense target
        traj_dir = os.path.join(final_graphs_dir, gname, f"Trajectory_{m}")

        try:
            G_opt = optimize_sa(
                G_init=G0,
                target=m,
                pool=pool,
                progress_prefix=f"Optimizing defense={m}",
                snapshot_dir=traj_dir,
                **SA_CONFIG
            )
            graphs[f"Optimized_{m}"] = G_opt

            try:
                R_opt_self = float(attack_methods[m](G_opt))
            except Exception as e:
                print(f"    !! Failed to compute R (Optimized_{m} under {m}): {e}")
                R_opt_self = float("nan")

            delta = R_opt_self - R_orig_self if (np.isfinite(R_opt_self) and np.isfinite(R_orig_self)) else float("nan")
            rel = (delta / R_orig_self) if (
                np.isfinite(delta) and np.isfinite(R_orig_self) and abs(R_orig_self) > 1e-12
            ) else float("nan")

            print(f"    [SA-DONE] {gname} | defense={m} | R_orig({m})={R_orig_self:.6f}  "
                  f"R_opt({m})={R_opt_self:.6f}  delta={delta:.6f}  rel={rel:.3%}")

            append_sa_log(sa_log_csv, {
                "graph": gname,
                "defense_target": m,
                "R_original_under_target": R_orig_self,
                "R_optimized_under_target": R_opt_self,
                "delta": delta,
                "relative_improve": rel,
                "T_max": SA_CONFIG.get("T_max"),
                "T_min": SA_CONFIG.get("T_min"),
                "N_per_iter": SA_CONFIG.get("N_per_iter"),
                "alpha": SA_CONFIG.get("alpha"),
                "max_iter": SA_CONFIG.get("max_iter"),
                "traj_dir": traj_dir,
                "traj_only_if_improved": SA_TRAJ_SAVE.get("only_if_improved", True),
                "traj_save_every": SA_TRAJ_SAVE.get("save_every", 1),
            })

        except Exception as e:
            print(f"    !! Simulated annealing failed ({m}): {e}")
            graphs[f"Optimized_{m}"] = G0

            append_sa_log(sa_log_csv, {
                "graph": gname,
                "defense_target": m,
                "R_original_under_target": R_orig_self,
                "R_optimized_under_target": float("nan"),
                "delta": float("nan"),
                "relative_improve": float("nan"),
                "T_max": SA_CONFIG.get("T_max"),
                "T_min": SA_CONFIG.get("T_min"),
                "N_per_iter": SA_CONFIG.get("N_per_iter"),
                "alpha": SA_CONFIG.get("alpha"),
                "max_iter": SA_CONFIG.get("max_iter"),
                "traj_dir": traj_dir,
                "error": str(e),
            })

        gc.collect()

    return graphs



###############################################################################
# 5. Averaged Curve Plotting ##################################################
###############################################################################

def plot_avg_curves(curve_acc: dict,
                    save_dir: str,
                    show_ratio: float = 0.5):  # Controls only the plotted range
    """
    For each attack method, average the curves for Original + each defense
    target and plot them. Full-length averaged curve data are also saved to CSV.

    curve_acc structure:
        curve_acc[attack][defense] = [curve1, curve2, ...]
    """
    os.makedirs(save_dir, exist_ok=True)

    colors  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
               '#9467bd','#8c564b','#e377c2','#7f7f7f']
    markers = ['o','s','^','D','v','>','<','P']
    linestyles = ['-','--','-.',':']

    for atk_idx, atk in enumerate(method_names):
        # Collect averaged curves for all defense strategies under this attack
        mean_curves = {}   # defense -> ndarray(full length)

        plt.figure(figsize=(12, 8))

        for j, (defense, curves) in enumerate(curve_acc[atk].items()):
            if not curves:                        # No data for this defense
                continue

            # curves: List[np.ndarray], each resampled to length PTS_RS
            mean_curve = np.mean(curves, axis=0)  # Pointwise mean over [0,1]
            mean_curves[defense] = mean_curve

            # Plot only the first show_ratio portion
            end = max(1, int(len(mean_curve) * show_ratio))
            y   = mean_curve[:end]
            x   = np.linspace(0, show_ratio, num=end)

            color     = colors[j % len(colors)]
            marker    = markers[j % len(markers)]
            linestyle = linestyles[(j // len(colors)) % len(linestyles)]

            plt.plot(
                x, y,
                label=defense,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markevery=max(1, end // 40),
                linewidth=2
            )

        # Save averaged curve data for this attack method to CSV
        if mean_curves:
            # All mean_curve arrays should have length PTS_RS
            any_curve = next(iter(mean_curves.values()))
            x_full    = np.linspace(0, 1, num=len(any_curve))

            # Build a DataFrame: x first, then one averaged curve per defense
            data = {"x": x_full}
            # Keep column order: Original first, then Optimized_*
            for dkey in ["Original"] + [f"Optimized_{m}" for m in method_names]:
                if dkey in mean_curves:
                    data[dkey] = mean_curves[dkey]

            df = pd.DataFrame(data)
            csv_path = os.path.join(save_dir, f"avg_curve_data_{atk}.csv")
            df.to_csv(csv_path, index=False)
            print(f"   Saved averaged curve data for {atk}: {csv_path}")

        # Save the plot, showing only the first show_ratio portion
        plt.xlabel(f"Normalized attack steps (0-{show_ratio:.2f})")
        plt.ylabel("Robustness (GCC / N)")
        plt.title(f"Averaged robustness curves - Attack: {atk}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fname = os.path.join(save_dir, f"avg_curve_{atk}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"   Saved averaged curve for {atk}: {fname}")



###############################################################################
# 6. Single-Graph Workflow ####################################################
###############################################################################

def run_single_graph(path: str,
                     pool: mp.pool.Pool,
                     curve_acc: dict,
                     curves_out: str,
                     final_graphs_dir: str,
                     orders_dir: str):
    """
    Process one original graph:
      - Build optimized defense graphs.
      - Record curves and accumulate them into curve_acc.
      - Return the R statistics matrix.
    """
    # ---- Read original graph ----
    G0    = load_graph(path)
    gname = os.path.splitext(os.path.basename(path))[0]
    print(f"-> {gname}:  |V|={G0.vcount()}  |E|={G0.ecount()}")

    # ---- Get Original + optimized defense graphs ----
    graphs_dict = get_optimized_graphs(G0, pool, gname=gname, out_dir=OUT_DIR, final_graphs_dir=final_graphs_dir)


    # ---- Save final topologies: Original + Optimized_* -------------------
    if SAVE_FINAL_GRAPHS:
        gdir = os.path.join(final_graphs_dir, gname)
        for dkey, Gd in graphs_dict.items():
            # File names: Original / Optimized_HDA ...
            out_path = os.path.join(gdir, f"{dkey}.txt")
            try:
                save_igraph_edgelist(Gd, out_path, with_brace=GRAPH_FMT_WITH_BRACE)
            except Exception as e:
                print(f"    !! Failed to save topology ({gname} | {dkey}): {e}")

    # ---- Save attack orders for each final topology and attack method -----
    if SAVE_ATTACK_ORDERS:
        odir = os.path.join(orders_dir, gname)
        for dkey, Gd in graphs_dict.items():
            for atk in method_names:
                try:
                    order = get_attack_order(atk, Gd)
                    if order is None:
                        # Skip methods that do not provide an order.
                        continue
                    out_path = os.path.join(odir, dkey, f"order_{atk}.csv")
                    save_order_csv(order, out_path)
                except Exception as e:
                    print(f"    !! Failed to save attack order ({gname} | {dkey} | {atk}): {e}")

    # ---- Accumulate curves and save them to CSV ---------------------------
    cdir = os.path.join(curves_out, "csv", gname)  # Curve CSV root grouped by graph name
    for atk in method_names:
        for dkey, Gd in graphs_dict.items():
            try:
                curve_raw = attack_curve_methods[atk](Gd)  # Raw length, usually N or N+1
                curve_rs = resample_curve(curve_raw)  # Resample to PTS_RS

                # 1) Add to the averaging accumulator
                curve_acc[atk][dkey].append(curve_rs)

                # 2) Save both raw and resampled CSV files
                out_dir_one = os.path.join(cdir, dkey)
                os.makedirs(out_dir_one, exist_ok=True)

                raw_path = os.path.join(out_dir_one, f"curve_{atk}_raw.csv")
                rs_path = os.path.join(out_dir_one, f"curve_{atk}_rs.csv")

                save_curve_csv(list(curve_raw), raw_path, x_mode="k_over_N")
                save_curve_csv(list(curve_rs), rs_path, x_mode="k_over_N")

            except Exception as e:
                print(f"    !! Failed to compute or save curve ({gname} | {dkey} | {atk}): {e}")

    # ---- Optional: plot per-original-graph curves -------------------------
    # Comment this block out if it creates too many images.
    for atk in method_names:
        plt.figure(figsize=(10,6))
        for j, (dkey, Gd) in enumerate(graphs_dict.items()):
            curve = attack_curve_methods[atk](Gd)
            plt.plot(range(len(curve)), curve,
                     label=dkey, linewidth=1)
        plt.title(f"{gname} - {atk}")
        plt.legend(); plt.tight_layout()
        os.makedirs(curves_out, exist_ok=True)
        plt.savefig(os.path.join(curves_out, f"{gname}_{atk}.png"), dpi=120)
        plt.close()

    # ---- R statistics matrix ---------------------------------------------
    R0   = np.array([attack_methods[m](G0) for m in method_names])
    Rmat = np.zeros((num_methods + 1, num_methods))
    Rmat[0] = R0
    for i, defend in enumerate(method_names):
        G_opt       = graphs_dict[f"Optimized_{defend}"]
        Rmat[i+1]   = [attack_methods[m](G_opt) for m in method_names]

    # ---- Return accumulated statistics ----
    return Rmat


###############################################################################
# 7. Main Program #############################################################
###############################################################################

def main():
    # ---- 1) Find all original graph files ---------------------------------
    files = sorted(glob.glob(os.path.join(GRAPH_DIR, "graph_*.txt")))
    if not files:
        raise FileNotFoundError(f"{GRAPH_DIR} is empty. Please prepare edge-list files first.")
    print(f"Found {len(files)} graph(s)")

    # ---- 2) Prepare output directories ------------------------------------
    out_dir     = OUT_DIR;   os.makedirs(out_dir,        exist_ok=True)
    curves_each = os.path.join(out_dir, "curves_each");  os.makedirs(curves_each, exist_ok=True)
    curves_avg  = os.path.join(out_dir, "curves_avg");   os.makedirs(curves_avg,  exist_ok=True)

    final_graphs_dir = os.path.join(out_dir, FINAL_GRAPH_DIRNAME)
    orders_dir = os.path.join(out_dir, ATTACK_ORDER_DIRNAME)
    os.makedirs(final_graphs_dir, exist_ok=True)
    os.makedirs(orders_dir, exist_ok=True)

    # ---- 3) Process pool --------------------------------------------------
    ctx  = mp.get_context("spawn")
    pool = ctx.Pool(processes=max(1, int(mp.cpu_count() * POOL_RATIO)))
    print(f"Process pool started with {pool._processes} worker(s)")

    # ---- 4) Initialize accumulators ---------------------------------------
    curve_acc = {
        atk: {key: [] for key in ["Original"] + [f"Optimized_{d}" for d in method_names]}
        for atk in method_names
    }
    R_acc = np.zeros((num_methods + 1, num_methods))

    # ---- 5) Iterate over all original graphs ------------------------------
    for path in files:
        try:
            R = run_single_graph(path, pool, curve_acc, curves_each,
                                 final_graphs_dir, orders_dir)

            R_acc   += R
            gc.collect()
        except Exception as e:
            print(f"!! Failed to process {path}: {e}")

    # ---- 6) Close process pool --------------------------------------------
    pool.close(); pool.join()
    print("Process pool closed")

    # ---- 7) Save averaged curve plots -------------------------------------
    print("Plotting and saving averaged robustness curves")
    plot_avg_curves(curve_acc, curves_avg)

    # ---- 8) Save statistics CSV -------------------------------------------
    runs = len(files)
    idx_def = ["Original"] + method_names
    pd.DataFrame(R_acc / runs,   index=idx_def, columns=method_names).to_csv(f"{out_dir}/R_mean.csv")
    print(f"Saved all statistics and curves to {out_dir}")


###############################################################################
# 8. Entry Point ##############################################################
###############################################################################

if __name__ == "__main__":
    # Windows requires the spawn start method.
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Ignore this when the interpreter has already set the start method.
            pass
    main()
