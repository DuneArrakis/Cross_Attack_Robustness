#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute onion-structure diagnostics (Herrmann et al. 2011 / Schneider et al. 2011 style)
for selected graph files (controlled by a list at the top).

Edge list format supported:
    u v {}
or  u v {json...}
or  u v

Outputs:
- per_graph_metrics.csv
- per_method_degree_metrics_long.csv
- figures (PNG only):
    topology_diagnostics_triptych.png

Metrics:
1) k_nn_bar(k): average nearest-neighbor degree for nodes of degree k
2) C_k: average clustering coefficient for nodes of degree k
3) onion structure: S_k / N_k

Notes:
- Degree-binned averages follow "bucket rule":
  For each method and each degree k, average only over graphs that *have* degree k.
- Linear axes for all plots.
- For onion plot, automatically trim x-range after curves saturate near 1.
"""

from __future__ import annotations

import os
import re
import glob
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import igraph as ig
except ImportError as e:
    raise SystemExit("python-igraph is required. Install with: pip install python-igraph") from e


# =========================
# Config (EDIT HERE)
# =========================
TARGET_FILE_PATHS = [
    r"results/final_graphs/graph_*/Original.txt",
    r"results/final_graphs/graph_*/Optimized_HDA.txt",
    r"results/final_graphs/graph_*/Optimized_CI.txt",
    r"results/final_graphs/graph_*/Optimized_CoreHD.txt",
    r"results/final_graphs/graph_*/Optimized_EI.txt",
    r"results/final_graphs/graph_*/Optimized_FINDER.txt",
    # r"results/final_graphs/graph_*/Optimized_onion.txt",
]

# OUT_DIR = "results/onion_metrics_outputs_onion_only"
OUT_DIR = "results/onion_metrics_outputs"

# Onion plot trimming
ONION_PLATEAU = 0.98
ONION_MARGIN_K = 3
ONION_MIN_KMAX = 5

# =========================
# Publication-style plotting
# =========================
FONT_SIZE = 10
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 9.5
TITLE_SIZE = 12
LEGEND_SIZE = 9.5
FIG_DPI = 150
SAVE_DPI = 300

TRIPTYCH_FIGSIZE = (18, 4.8)
# LEGEND_NCOL = 4
# SHOW_SEM_BAND = True
SHOW_SEM_BAND = False
SEM_ALPHA = 0.12

METHOD_ORDER = ["Original", "HDA", "CI", "CoreHD", "EI", "FINDER", "onion"]

STYLE_MAP = {
    "Original": {
        "color": "#4D4D4D",
        "linestyle": "--",
        "marker": None,
        "linewidth": 2.0,
        "alpha": 0.95,
        "zorder": 3,
    },
    "HDA": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
        "markersize": 4.8,
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
        "alpha": 1.0,
        "zorder": 6,
    },
    "CI": {
        "color": "#E69F00",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 4.8,
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
        "alpha": 1.0,
        "zorder": 6,
    },
    "CoreHD": {
        "color": "#009E73",
        "linestyle": "-",
        "marker": "^",
        "linewidth": 2.2,
        "markersize": 5.0,
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
        "alpha": 1.0,
        "zorder": 7,
    },
    "EI": {
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "D",
        "linewidth": 2.2,
        "markersize": 4.8,
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
        "alpha": 1.0,
        "zorder": 6,
    },
    "FINDER": {
        "color": "#CC79A7",
        "linestyle": "-",
        "marker": "v",
        "linewidth": 2.2,
        "markersize": 5.0,
        "markerfacecolor": "white",
        "markeredgewidth": 0.9,
        "alpha": 1.0,
        "zorder": 7,
    },
    "onion": {
        "color": "#56B4E9",
        "linestyle": "-.",
        "marker": None,
        "linewidth": 2.0,
        "alpha": 0.95,
        "zorder": 4,
    },
}


def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,

        "font.size": FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,

        "figure.dpi": FIG_DPI,
        "savefig.dpi": SAVE_DPI,

        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.8,

        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


set_publication_style()


# =========================
# Utilities
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_method_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    if base.lower().startswith("original"):
        return "Original"

    m = re.match(r"Optimized_(.+?)\.txt$", base, flags=re.IGNORECASE)
    if m:
        method = m.group(1).strip()
        for known in ["HDA", "CI", "CoreHD", "EI", "FINDER", "onion"]:
            if method.lower() == known.lower():
                return known
        return method

    return os.path.splitext(base)[0]


def infer_graph_id_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if re.fullmatch(r"graph_\d+", p) or p.startswith("graph_"):
            return p
    return os.path.basename(os.path.dirname(path)) or "graph_unknown"


def _is_comment_or_empty(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("#") or s.startswith("%") or s.startswith("//")


def read_edge_list_txt(path: str) -> ig.Graph:
    numeric_rows: List[Tuple[int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if _is_comment_or_empty(line):
                continue
            parts = re.split(r"[\s,]+", line.strip())
            if len(parts) < 2:
                continue
            try:
                u = int(float(parts[0]))
                v = int(float(parts[1]))
            except ValueError:
                continue
            numeric_rows.append((u, v))

    if not numeric_rows:
        raise ValueError(f"No edges parsed from: {path}")

    header_is_nm = False
    if len(numeric_rows) >= 2:
        N0, M0 = numeric_rows[0]
        edge_rows = numeric_rows[1:]
        if M0 > 0 and abs(len(edge_rows) - M0) <= max(2, int(0.02 * M0)):
            header_is_nm = True
        if not header_is_nm and edge_rows:
            max_id = max(max(u, v) for (u, v) in edge_rows)
            if N0 >= max_id and M0 <= len(edge_rows) + 5:
                header_is_nm = True

    if header_is_nm:
        N, _M = numeric_rows[0]
        edges_raw = numeric_rows[1:]
    else:
        edges_raw = numeric_rows
        max_id = max(max(u, v) for (u, v) in edges_raw)
        N = max_id + 1

    min_id = min(min(u, v) for (u, v) in edges_raw)
    max_id = max(max(u, v) for (u, v) in edges_raw)
    if min_id == 1 and (max_id == N or max_id == N - 1):
        edges = [(u - 1, v - 1) for (u, v) in edges_raw]
        if not header_is_nm:
            N = max(max(u, v) for (u, v) in edges) + 1
    else:
        edges = edges_raw

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.simplify(multiple=True, loops=True)
    return g


def compute_degree_binned_knn(g: ig.Graph) -> Dict[int, float]:
    deg = g.degree()
    n = g.vcount()
    neighs = g.get_adjlist()
    knn_i = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if deg[i] <= 0:
            continue
        nb = neighs[i]
        if not nb:
            continue
        knn_i[i] = float(np.mean([deg[j] for j in nb]))

    byk: Dict[int, List[float]] = {}
    for i, k in enumerate(deg):
        if k <= 0:
            continue
        v = knn_i[i]
        if np.isfinite(v):
            byk.setdefault(k, []).append(v)

    return {k: float(np.mean(vals)) for k, vals in byk.items()}


def compute_degree_binned_clustering(g: ig.Graph) -> Dict[int, float]:
    deg = g.degree()
    c_local = g.transitivity_local_undirected(vertices=None, mode="zero")
    byk: Dict[int, List[float]] = {}
    for i, k in enumerate(deg):
        if k <= 0:
            continue
        byk.setdefault(k, []).append(float(c_local[i]))
    return {k: float(np.mean(v)) for k, v in byk.items()}


def compute_onion_sk_over_nk(g: ig.Graph) -> Dict[int, float]:
    deg = np.array(g.degree(), dtype=int)
    ks = sorted(set(int(x) for x in deg if x > 0))
    if not ks:
        return {}

    unique, counts = np.unique(deg[deg > 0], return_counts=True)
    Nk = {int(k): int(c) for k, c in zip(unique, counts)}

    out: Dict[int, float] = {}
    for k in ks:
        active = np.where(deg <= k)[0].tolist()
        if not active:
            continue
        sub = g.subgraph(active)
        comps = sub.components()
        membership = comps.membership

        comp_counts: Dict[int, int] = {}
        for sub_idx, comp_id in enumerate(membership):
            orig_idx = active[sub_idx]
            if deg[orig_idx] == k:
                comp_counts[comp_id] = comp_counts.get(comp_id, 0) + 1

        Sk = max(comp_counts.values()) if comp_counts else 0
        denom = Nk.get(k, 0)
        if denom > 0:
            out[k] = Sk / denom

    return out


def sem(x: List[float]) -> float:
    if len(x) <= 1:
        return 0.0
    return float(np.std(x, ddof=1) / math.sqrt(len(x)))


@dataclass
class GraphMetrics:
    graph_id: str
    method: str
    path: str
    n: int
    m: int
    knn_by_k: Dict[int, float]
    ck_by_k: Dict[int, float]
    onion_by_k: Dict[int, float]


# =========================
# File selection
# =========================
def collect_selected_files(target_patterns: List[str]) -> List[Tuple[str, str]]:
    all_paths: List[str] = []
    for pat in target_patterns:
        all_paths.extend(glob.glob(pat))

    seen = set()
    uniq_paths = []
    for p in sorted(all_paths):
        rp = os.path.normpath(p)
        if rp not in seen and os.path.isfile(rp):
            uniq_paths.append(rp)
            seen.add(rp)

    items: List[Tuple[str, str]] = []
    for fp in uniq_paths:
        graph_id = infer_graph_id_from_path(fp)
        items.append((graph_id, fp))

    return items


# =========================
# Compute / aggregate
# =========================
def compute_all_metrics(files: List[Tuple[str, str]]) -> List[GraphMetrics]:
    allm: List[GraphMetrics] = []
    for graph_id, fp in files:
        method = parse_method_from_filename(fp)
        try:
            g = read_edge_list_txt(fp)
        except Exception as e:
            print(f"[WARN] failed to read {fp}: {e}")
            continue

        try:
            knn = compute_degree_binned_knn(g)
        except Exception as e:
            print(f"[WARN] knn failed for {fp}: {e}")
            knn = {}

        try:
            ck = compute_degree_binned_clustering(g)
        except Exception as e:
            print(f"[WARN] Ck failed for {fp}: {e}")
            ck = {}

        try:
            onion = compute_onion_sk_over_nk(g)
        except Exception as e:
            print(f"[WARN] onion S_k/N_k failed for {fp}: {e}")
            onion = {}

        allm.append(GraphMetrics(
            graph_id=graph_id,
            method=method,
            path=fp,
            n=g.vcount(),
            m=g.ecount(),
            knn_by_k=knn,
            ck_by_k=ck,
            onion_by_k=onion,
        ))
    return allm


def aggregate_degree_metrics(metrics: List[GraphMetrics]) -> pd.DataFrame:
    agg: Dict[str, Dict[int, Dict[str, List[float]]]] = {}

    for gm in metrics:
        md = gm.method
        agg.setdefault(md, {})
        for k, v in gm.knn_by_k.items():
            agg[md].setdefault(k, {}).setdefault("knn", []).append(v)
        for k, v in gm.ck_by_k.items():
            agg[md].setdefault(k, {}).setdefault("ck", []).append(v)
        for k, v in gm.onion_by_k.items():
            agg[md].setdefault(k, {}).setdefault("onion", []).append(v)

    rows = []
    for md, deg_dict in agg.items():
        for k, metric_lists in deg_dict.items():
            knn_vals = metric_lists.get("knn", [])
            ck_vals = metric_lists.get("ck", [])
            onion_vals = metric_lists.get("onion", [])
            rows.append({
                "method": md,
                "degree_k": int(k),

                "knn_mean": float(np.mean(knn_vals)) if knn_vals else np.nan,
                "knn_sem": sem(knn_vals) if knn_vals else np.nan,
                "knn_n_graphs": int(len(knn_vals)),

                "ck_mean": float(np.mean(ck_vals)) if ck_vals else np.nan,
                "ck_sem": sem(ck_vals) if ck_vals else np.nan,
                "ck_n_graphs": int(len(ck_vals)),

                "onion_mean": float(np.mean(onion_vals)) if onion_vals else np.nan,
                "onion_sem": sem(onion_vals) if onion_vals else np.nan,
                "onion_n_graphs": int(len(onion_vals)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["method", "degree_k"]).reset_index(drop=True)


# =========================
# Plotting
# =========================
def _method_sort_key(md: str) -> Tuple[int, str]:
    return (METHOD_ORDER.index(md) if md in METHOD_ORDER else 999, md)


def find_informative_kmax(
    df_deg: pd.DataFrame,
    value_col: str,
    plateau: float = 0.995,
    margin: int = 3,
    min_kmax: int = 5
) -> int:
    if df_deg.empty:
        return min_kmax
    s = df_deg.groupby("degree_k")[value_col].min()
    if s.empty:
        return min_kmax

    ks_max = int(s.index.max())
    informative = s[s < plateau]
    kmax_inf = ks_max if informative.empty else int(informative.index.max())

    kmax_plot = max(min_kmax, kmax_inf + margin)
    kmax_plot = min(kmax_plot, ks_max)
    return kmax_plot


def apply_axis_style(ax: plt.Axes) -> None:
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.8, alpha=0.25, zorder=0)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=False,
        right=False,
        length=4.5,
        width=1.0
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_degree_curve_on_ax(
    ax: plt.Axes,
    df_deg: pd.DataFrame,
    value_col: str,
    sem_col: str,
    ylabel: str,
    panel_label: str,
    panel_title: str,
    xlim: Optional[Tuple[float, float]] = None,
) -> Dict[str, object]:
    handles = {}

    if df_deg.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return handles

    for md in sorted(df_deg["method"].unique(), key=_method_sort_key):
        sub = df_deg[df_deg["method"] == md].sort_values("degree_k")
        x = sub["degree_k"].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        e = sub[sem_col].to_numpy(dtype=float)

        mask = np.isfinite(y)
        x, y, e = x[mask], y[mask], e[mask]
        if len(x) == 0:
            continue

        st = STYLE_MAP.get(md, {})
        (line,) = ax.plot(
            x,
            y,
            color=st.get("color", None),
            linestyle=st.get("linestyle", "-"),
            linewidth=st.get("linewidth", 1.8),
            marker=st.get("marker", None),
            markersize=st.get("markersize", 4.8),
            markerfacecolor=st.get("markerfacecolor", "white"),
            markeredgewidth=st.get("markeredgewidth", 0.8),
            alpha=st.get("alpha", 1.0),
            zorder=st.get("zorder", 5),
            label=md,
        )
        handles.setdefault(md, line)

        if SHOW_SEM_BAND:
            e = np.nan_to_num(e, nan=0.0)
            if np.any(e > 0):
                ax.fill_between(
                    x, y - e, y + e,
                    color=st.get("color", line.get_color()),
                    alpha=SEM_ALPHA,
                    zorder=max(1, st.get("zorder", 5) - 1),
                )

    ax.set_xlabel("Degree $k$", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    if xlim is not None:
        ax.set_xlim(xlim)

    apply_axis_style(ax)
    ax.margins(y=0.08)

    ax.text(
        0.02, 0.97, panel_label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11.5, fontweight="bold"
    )
    #ax.set_title(panel_title, pad=8)
    # Do not show subplot titles; top spacing is controlled by y-margin/subplot settings below.
    return handles


def plot_triptych(df_deg: pd.DataFrame, outpath_png: str) -> None:
    if df_deg.empty:
        print(f"[WARN] Empty degree dataframe, skip plot: {outpath_png}")
        return

    kmax_onion = find_informative_kmax(
        df_deg=df_deg,
        value_col="onion_mean",
        plateau=ONION_PLATEAU,
        margin=ONION_MARGIN_K,
        min_kmax=ONION_MIN_KMAX
    )

    fig, axes = plt.subplots(1, 3, figsize=TRIPTYCH_FIGSIZE, constrained_layout=False)
    fig.subplots_adjust(wspace=0.28)
    ax1, ax2, ax3 = axes

    handles_all: Dict[str, object] = {}

    h1 = plot_degree_curve_on_ax(
        ax=ax1,
        df_deg=df_deg,
        value_col="knn_mean",
        sem_col="knn_sem",
        ylabel=r"$\bar{k}_{nn}(k)$",
        panel_label="(a)",
        #panel_title=r"Nearest-neighbor degree"
        panel_title=""
    )
    handles_all.update(h1)

    h2 = plot_degree_curve_on_ax(
        ax=ax2,
        df_deg=df_deg,
        value_col="ck_mean",
        sem_col="ck_sem",
        ylabel=r"$C_k$",
        panel_label="(b)",
        #panel_title=r"Degree-dependent clustering"
        panel_title=""
    )
    for k, v in h2.items():
        handles_all.setdefault(k, v)

    h3 = plot_degree_curve_on_ax(
        ax=ax3,
        df_deg=df_deg,
        value_col="onion_mean",
        sem_col="onion_sem",
        ylabel=r"$S_k/N_k$",
        panel_label="(c)",
        #panel_title=r"Onion structure",
        panel_title="",
        xlim=(0.0, float(kmax_onion) + 1.0)
    )
    for k, v in h3.items():
        handles_all.setdefault(k, v)

    ordered_methods = [m for m in METHOD_ORDER if m in handles_all]
    ordered_handles = [handles_all[m] for m in ordered_methods]

    ax1.legend(
        ordered_handles,
        ordered_methods,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        framealpha=0.95,
        ncol=1,
        handlelength=2.0,
        handletextpad=0.6,
        labelspacing=1.2,
        borderpad=0.35,
        borderaxespad=0.4,
    )

    fig.savefig(outpath_png, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# =========================
# Main
# =========================
def main() -> None:
    ensure_dir(OUT_DIR)

    files = collect_selected_files(TARGET_FILE_PATHS)
    if not files:
        raise SystemExit(
            "No files matched TARGET_FILE_PATHS. Please check your patterns.\n"
            f"TARGET_FILE_PATHS={TARGET_FILE_PATHS}"
        )

    print(f"[INFO] Selected {len(files)} files for analysis.")
    metrics = compute_all_metrics(files)
    print(f"[INFO] Computed metrics for {len(metrics)} graphs/files.")

    per_graph_rows = []
    for m in metrics:
        per_graph_rows.append({
            "graph_id": m.graph_id,
            "method": m.method,
            "path": m.path,
            "N": m.n,
            "M": m.m,
        })
    df_per_graph = pd.DataFrame(per_graph_rows)
    df_per_graph.to_csv(os.path.join(OUT_DIR, "per_graph_metrics.csv"), index=False)

    df_deg = aggregate_degree_metrics(metrics)
    df_deg.to_csv(os.path.join(OUT_DIR, "per_method_degree_metrics_long.csv"), index=False)

    plot_triptych(
        df_deg=df_deg,
        outpath_png=os.path.join(OUT_DIR, "topology_diagnostics_triptych.png")
    )

    print(f"[DONE] Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
