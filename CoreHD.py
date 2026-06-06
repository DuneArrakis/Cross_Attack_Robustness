import networkx as nx
import matplotlib.pyplot as plt
from chouffe import destroy
import igraph as ig
import random
import heapq

# ---------- igraph to networkx ----------
def igraph_to_networkx(G_ig: ig.Graph) -> nx.Graph:
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(G_ig.vcount()))
    G_nx.add_edges_from((e.source, e.target) for e in G_ig.es)
    return G_nx

# ---------- Compute the largest connected component ratio s(Q) ----------
def _frac_lcc(G: nx.Graph, N0: int) -> float:
    """Return s(Q) = |LCC| / N0, using the initial node count as denominator."""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / N0

# ---------- Get the node deletion order for the CoreHD attack ----------
def _corehd_sequence_k2(G: nx.Graph):
    ret = destroy(2, G, scm='HD')
    if isinstance(ret, tuple):
        D, _ = ret
    else:
        D = ret
    return list(D)

def _max_degree_full_removal_order(G: nx.Graph):
    """
    Generate a full deletion order by removing the current maximum-degree node
    at every step.
    Tie-break rule: smaller node id first, ensuring deterministic results.
    A lazily updated max heap keeps the iterative deletion efficient.
    """
    G_work = G.copy()
    if G_work.number_of_nodes() == 0:
        return []

    deg = dict(G_work.degree())
    # max-heap: (-deg, node); positive node ids naturally prefer smaller ids.
    heap = [(-d, v) for v, d in deg.items()]
    heapq.heapify(heap)

    order = []
    while G_work.number_of_nodes() > 0:
        # Pop until the entry still exists and matches the current degree.
        while True:
            neg_d, v = heapq.heappop(heap)
            if not G_work.has_node(v):
                continue
            cur_d = G_work.degree(v)
            if -neg_d != cur_d:
                # Stale entry; discard it and keep popping.
                continue
            # Found a valid maximum-degree node.
            break

        order.append(v)

        # Delete v and push updated neighbor degrees. Old entries expire lazily.
        nbrs = list(G_work.neighbors(v))
        G_work.remove_node(v)
        for u in nbrs:
            if G_work.has_node(u):
                heapq.heappush(heap, (-G_work.degree(u), u))

    return order


# ---------- Compute the CoreHD robustness curve from an igraph input ----------
def corehd_robustness_curve_igraph_with_marker(G_ig: ig.Graph):
    # 1) Convert to networkx.
    G = igraph_to_networkx(G_ig)

    # 2) Work copy.
    G_work = G.copy()
    N0 = G_work.number_of_nodes()
    if N0 == 0:
        return [], 0, []

    # 3) CoreHD deletion sequence that dismantles the 2-core.
    D = _corehd_sequence_k2(G.copy())
    q_core_end = len(D)  # End step of 2-core dismantling; forest stage starts here.

    # 4) After deleting D, remove the remaining forest by maximum degree.
    G_forest = G_work.copy()
    for v in D:
        if G_forest.has_node(v):
            G_forest.remove_node(v)

    T = _max_degree_full_removal_order(G_forest)
    full_order = D + T

    # 5) Delete step by step and record s(Q), starting from k=0.
    curve = [_frac_lcc(G_work, N0)]
    for v in full_order:
        if G_work.has_node(v):
            G_work.remove_node(v)
        curve.append(_frac_lcc(G_work, N0))

    return curve, q_core_end, full_order

def corehd_robustness_curve_igraph(G_ig: ig.Graph):
    # 1) Convert to networkx.
    G = igraph_to_networkx(G_ig)

    # 2) Work copy.
    G_work = G.copy()
    N0 = G_work.number_of_nodes()
    if N0 == 0:
        return [], 0, []

    # 3) CoreHD deletion sequence that dismantles the 2-core.
    D = _corehd_sequence_k2(G.copy())
    # q_core_end = len(D)  # End step of 2-core dismantling.

    # 4) After deleting D, remove the remaining forest by maximum degree.
    G_forest = G_work.copy()
    for v in D:
        if G_forest.has_node(v):
            G_forest.remove_node(v)

    T = _max_degree_full_removal_order(G_forest)
    full_order = D + T

    # 5) Delete step by step and record s(Q).
    curve = []
    for v in full_order:
        if G_work.has_node(v):
            G_work.remove_node(v)
        curve.append(_frac_lcc(G_work, N0))

    return curve

# ---------- R metric ----------
def corehd_R_igraph(G_ig: ig.Graph) -> float:
    curve, q_core_end, _ = corehd_robustness_curve_igraph_with_marker(G_ig)
    return (sum(curve) / len(curve)) if curve else 0.0

# ---------- Generate an igraph ER graph ----------
def generate_er_graph_ig(N, p, seed=None):
    # Optional seed for reproducibility.
    return ig.Graph.Erdos_Renyi(n=N, p=p, directed=False, loops=False)

# ---------- Load a graph from txt; each line looks like "1 2 {}" ----------
def load_graph_from_txt_to_igraph(
    path: str,
    relabel_to_0_based: bool = True,
    assume_undirected: bool = True
) -> ig.Graph:
    """
    Read a txt graph file. Each line has at least two columns: u v.
    Extra fields such as "{}" are ignored.
    Example line: 1 2 {}
    """
    edges_raw = []
    nodes_set = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            edges_raw.append((u, v))
            nodes_set.add(u)
            nodes_set.add(v)

    # Isolated nodes cannot be recovered from this edge-only format unless a
    # separate node list is provided. Build the graph from nodes seen in edges.

    if relabel_to_0_based:
        # Relabel nodes to 0..n-1 for stable igraph-to-networkx conversion.
        nodes_sorted = sorted(nodes_set)
        mapping = {old: new for new, old in enumerate(nodes_sorted)}
        edges = [(mapping[u], mapping[v]) for (u, v) in edges_raw]
        n = len(nodes_sorted)
        G_ig = ig.Graph(n=n, edges=edges, directed=not assume_undirected)
    else:
        # Keep original ids. igraph needs vertices 0..max_id, so fill gaps.
        max_id = max(nodes_set) if nodes_set else -1
        n = max_id + 1
        G_ig = ig.Graph(n=n, edges=edges_raw, directed=not assume_undirected)

    if assume_undirected:
        G_ig.to_undirected(combine_edges=None)

    # Remove self-loops and multi-edges for a cleaner simple graph.
    G_ig.simplify(multiple=True, loops=True)

    return G_ig

# ---------- Main program ----------
def run_experiment(N=3000, p=0.001, plot_curve=True, graph_txt_path=None, print_curve=True, save_curve_path=None):
    if graph_txt_path is None:
        G_ig = generate_er_graph_ig(N, p)
        print(f"Generated ER graph: N={N}, p={p:.6g}")
    else:
        G_ig = load_graph_from_txt_to_igraph(
            graph_txt_path,
            relabel_to_0_based=True,
            assume_undirected=True
        )
        print(f"Loaded graph from txt: {graph_txt_path}")
        print(f"Graph stats: N={G_ig.vcount()}, M={G_ig.ecount()}")

    curve, q_core_end, _ = corehd_robustness_curve_igraph_with_marker(G_ig)
    R_val = (sum(curve) / len(curve)) if curve else 0.0

    N0 = G_ig.vcount()
    q_end_frac = q_core_end / N0 if N0 > 0 else 0.0

    print(f"CoreHD robustness R = {R_val:.6f}")
    print(f"2-core dismantling ends at step = {q_core_end} / {N0}, q = {q_end_frac:.6f}")

    # 1) Print the curve sequence.
    if print_curve:
        # Save to a file instead if the sequence is too long.
        print("R sequence, s(Q) after each deletion step:")
        print(curve)

    # 2) Optionally save to file.
    if save_curve_path is not None:
        with open(save_curve_path, "w", encoding="utf-8") as f:
            f.write("# step\ts(Q)\n")
            for i, val in enumerate(curve, start=0):
                f.write(f"{i}\t{val}\n")
            f.write(f"# 2core_end_step={q_core_end}\n")
            f.write(f"# 2core_end_q={q_end_frac}\n")
        print(f"Saved curve to: {save_curve_path}")

    # 3) Plot and mark the 2-core end point.
    if plot_curve and curve:
        rho = [i / N0 for i in range(len(curve))]
        plt.figure(figsize=(4, 4))
        plt.plot(rho, curve, label='s(Q)')
        # Vertical line marking the end of 2-core dismantling.
        plt.axvline(x=q_end_frac, linestyle='--', label=f'2-core end (q={q_end_frac:.3f})')

        plt.xlabel('q')
        plt.ylabel('s(Q)')
        plt.title('CoreHD + Forest(max-degree) full removal')
        plt.xlim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Run experiment.
if __name__ == "__main__":
    # 1) Original mode: generate ER.
    # run_experiment(N=3000, p=0.001, plot_curve=True)

    # 2) New mode: load from txt.
    run_experiment(plot_curve=True, graph_txt_path="generate/graph_0.txt")
