# generate.py
import igraph as ig
import os
import glob
import random
import math

# =========================================================
# 0) Parameters
# =========================================================
num_runs = 50                # Number of graphs to generate
n_nodes = 500                # Number of nodes
out_dir = "generate"
SEED = None                  # Random seed; set to None when reproducibility is not needed
GRAPH_TYPE = "BA"            # "ER", "BA", or "WS"
MAX_TRIES_PER_GRAPH = 1000   # Maximum retries per graph

# ---------- ER Parameters ----------
er_p = (math.log(n_nodes) / n_nodes) * 1.1

# ---------- BA Parameters ----------
N0 = 3                       # Number of nodes in the initial complete BA seed graph
m_attach = 3                 # Number of edges added by each new node

# ---------- WS Parameters ----------
ws_dim = 1                   # Small-world dimension; 1 gives the classic ring lattice
ws_nei = 3                   # Neighbors on each side, average degree about 2*ws_nei
ws_rewire_p = 0.1            # Rewiring probability

# =========================================================
# 1) Create or clear the output directory
# =========================================================
os.makedirs(out_dir, exist_ok=True)

old_files = glob.glob(os.path.join(out_dir, "*.txt"))
for f in old_files:
    os.remove(f)
print(f"Cleared {len(old_files)} old file(s) under {out_dir}/")

# =========================================================
# 2) Random seed
# =========================================================
if SEED is not None:
    random.seed(SEED)
    try:
        ig.set_random_number_generator(random)
    except Exception:
        pass

# =========================================================
# 3) Validation functions
# =========================================================
def is_valid_undirected_simple_connected_graph(G: ig.Graph, expected_n: int) -> bool:
    """
    Check whether the graph satisfies:
    1) undirected,
    2) correct node count,
    3) no self-loops or multi-edges,
    4) connected.
    """
    if G is None:
        return False
    if G.is_directed():
        return False
    if G.vcount() != expected_n:
        return False
    if not G.is_simple():
        return False
    if not G.is_connected():
        return False
    return True


def validate_edge_list(edges, expected_n: int) -> bool:
    """
    Strictly validate an edge list:
    1) every edge satisfies 0 <= u < v < expected_n,
    2) no duplicate edges,
    3) the constructed graph is undirected, simple, and connected.
    """
    seen = set()

    for u, v in edges:
        if not (0 <= u < v < expected_n):
            return False
        if (u, v) in seen:
            return False
        seen.add((u, v))

    G_check = ig.Graph(n=expected_n, edges=edges, directed=False)
    return is_valid_undirected_simple_connected_graph(G_check, expected_n)

# =========================================================
# 4) Graph generation functions
# =========================================================
def generate_er_graph(n: int, p: float) -> ig.Graph:
    """
    Generate an ER graph.
    """
    G = ig.Graph.Erdos_Renyi(
        n=n,
        p=p,
        directed=False,
        loops=False
    )
    return G


def generate_ba_graph(n: int, n0: int, m: int) -> ig.Graph:
    """
    Generate a BA graph.
    Use an initial complete graph as the seed to ensure connectivity.
    """
    if n0 < 1:
        raise ValueError("N0 must be >= 1")
    if m < 1:
        raise ValueError("m_attach must be >= 1")
    if n <= n0:
        raise ValueError("n_nodes must be greater than N0")
    if m > n0:
        raise ValueError("m_attach cannot be greater than N0")

    G0 = ig.Graph.Full(n0)
    G = ig.Graph.Barabasi(
        n=n,
        m=m,
        start_from=G0,
        directed=False
    )
    return G


def generate_ws_graph(n: int, dim: int, nei: int, p: float) -> ig.Graph:
    """
    Generate a WS small-world network.
    Parameters:
    - dim=1: one-dimensional ring lattice,
    - size=n: number of nodes,
    - nei: neighbors on each side, so average degree is about 2*nei,
    - p: rewiring probability.
    """
    if dim < 1:
        raise ValueError("ws_dim must be >= 1")
    if nei < 1:
        raise ValueError("ws_nei must be >= 1")
    if not (0 <= p <= 1):
        raise ValueError("ws_rewire_p must be within [0, 1]")

    G = ig.Graph.Watts_Strogatz(
        dim=dim,
        size=n,
        nei=nei,
        p=p,
        loops=False,
        multiple=False
    )
    return G


def generate_one_base_graph() -> ig.Graph:
    """
    Generate one base graph according to GRAPH_TYPE.
    """
    gt = GRAPH_TYPE.upper()

    if gt == "ER":
        return generate_er_graph(n_nodes, er_p)
    elif gt == "BA":
        return generate_ba_graph(n_nodes, N0, m_attach)
    elif gt == "WS":
        return generate_ws_graph(n_nodes, ws_dim, ws_nei, ws_rewire_p)
    else:
        raise ValueError(f"Unsupported GRAPH_TYPE: {GRAPH_TYPE}")

# =========================================================
# 5) Relabeling function
# =========================================================
def relabel_graph_by_permutation(G: ig.Graph):
    """
    Randomly relabel the graph and return:
    1) the relabeled edge list with u < v,
    2) the permutation used.
    """
    perm = list(range(G.vcount()))
    random.shuffle(perm)

    edges_new = set()
    for u, v in G.get_edgelist():
        a, b = perm[u], perm[v]

        # This should not occur, but keep the defensive check.
        if a == b:
            continue

        if a > b:
            a, b = b, a

        edges_new.add((a, b))

    edges_new = sorted(edges_new)
    return edges_new, perm

# =========================================================
# 6) Main workflow
# =========================================================
for run in range(num_runs):
    success = False

    for attempt in range(1, MAX_TRIES_PER_GRAPH + 1):
        # 1) Generate a base graph.
        G = generate_one_base_graph()

        # 2) Validate the base graph.
        if not is_valid_undirected_simple_connected_graph(G, n_nodes):
            continue

        # 3) Randomly relabel nodes.
        edges_new, perm = relabel_graph_by_permutation(G)

        # 4) Validate again after relabeling.
        if not validate_edge_list(edges_new, n_nodes):
            continue

        # 5) Write the graph file.
        file_path = os.path.join(out_dir, f"graph_{run}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for u, v in edges_new:
                f.write(f"{u} {v} {{}}\n")

        print(
            f"Saved: {file_path}  "
            f"(type={GRAPH_TYPE}, attempt={attempt}, |V|={n_nodes}, |E|={len(edges_new)})"
        )
        success = True
        break

    if not success:
        raise RuntimeError(
            f"graph_{run} could not be generated after {MAX_TRIES_PER_GRAPH} attempts.\n"
            f"Current parameters: GRAPH_TYPE={GRAPH_TYPE}, n_nodes={n_nodes}\n"
            f"For ER, consider increasing er_p. For BA, check N0 and m_attach. "
            f"For WS, check ws_nei and ws_rewire_p."
        )

print("All networks have been generated and validated: undirected, connected, no self-loops, no multi-edges")
