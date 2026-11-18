"""CoreHD decycling followed by Appendix A greedy tree breaking.

Source: Zdeborova, Zhang & Zhou (2016), doi:10.1038/srep37954.
Core ties retain the project's deterministic minimum original ID convention.
The forest rule is extended to N deletions for full R; no reinsertion is used.
forest_strategy='legacy_hd' reproduces the previous maximum-degree tail.
"""
import networkx as nx
from ._corehd import destroy
import igraph as ig
import heapq
from .common import robustness_from_curve

# ---------- igraph → networkx ----------
def igraph_to_networkx(G_ig: ig.Graph) -> nx.Graph:
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(G_ig.vcount()))
    G_nx.add_edges_from((e.source, e.target) for e in G_ig.es)
    return G_nx

# ---------- 计算最大连通分量占比 s(Q) ----------
def _frac_lcc(G: nx.Graph, N0: int) -> float:
    """返回 s(Q) = |LCC| / N0（分母固定为初始节点总数）"""
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / N0

# ---------- 获取 CoreHD 攻击的节点删除顺序 ----------
def _corehd_sequence_k2(G: nx.Graph):
    ret = destroy(2, G, scm='HD')
    if isinstance(ret, tuple):
        D, _ = ret
    else:
        D = ret
    return list(D)

def _max_degree_full_removal_order(G: nx.Graph):
    """
    对图 G 生成“每一步删除当前最大度节点”的完整删除序列。
    tie-break: 节点编号更小者优先（保证确定性、可复现）。
    使用懒更新最大堆，适合上万点以内的迭代删除。
    """
    G_work = G.copy()
    if G_work.number_of_nodes() == 0:
        return []

    deg = dict(G_work.degree())
    # max-heap: (-deg, node)；node 用正数，天然实现 tie 时 node 小者优先
    heap = [(-d, v) for v, d in deg.items()]
    heapq.heapify(heap)

    order = []
    while G_work.number_of_nodes() > 0:
        # 弹出直到拿到一个“仍存在且度值匹配当前”的条目
        while True:
            neg_d, v = heapq.heappop(heap)
            if not G_work.has_node(v):
                continue
            cur_d = G_work.degree(v)
            if -neg_d != cur_d:
                # 旧条目，丢弃，继续弹
                continue
            # 找到有效最大度节点
            break

        order.append(v)

        # 删除 v，并更新邻居度（推入新条目即可，旧条目懒淘汰）
        nbrs = list(G_work.neighbors(v))
        G_work.remove_node(v)
        for u in nbrs:
            if G_work.has_node(u):
                heapq.heappush(heap, (-G_work.degree(u), u))

    return order


# ---------- 计算 CoreHD 鲁棒性曲线（输入 igraph） ----------
def _greedy_tree_breaking_order(G: nx.Graph):
    """Split a largest tree at a centroid, repeating down to isolated nodes.

    Subtree sizes give the largest branch after removal in O(tree size).
    Minimizing that size in a largest tree minimizes the forest's next LCC.
    Unspecified ties: largest trees by minimum original ID, then minimum
    largest branch within that tree, then minimum original vertex ID.
    This also defines plateau steps when several largest trees coexist.
    """
    if not G:
        return []
    if not nx.is_forest(G):
        raise ValueError('CoreHD tree breaking requires a forest')
    adjacency = {v: set(G[v]) for v in G}
    heap = [(-len(c), min(c), c) for c in nx.connected_components(G)]
    heapq.heapify(heap)
    order = []
    while heap:
        neg_size, root, nodes = heapq.heappop(heap)
        size = -neg_size
        parent, traversal = {root: None}, [root]
        for v in traversal:
            for u in adjacency[v]:
                if u != parent[v]:
                    parent[u] = v
                    traversal.append(u)
        subtree = dict.fromkeys(nodes, 1)
        largest_branch = dict.fromkeys(nodes, 0)
        for v in reversed(traversal):
            largest_branch[v] = max(largest_branch[v], size - subtree[v])
            p = parent[v]
            if p is not None:
                subtree[p] += subtree[v]
                largest_branch[p] = max(largest_branch[p], subtree[v])
        chosen = min(nodes, key=lambda v: (largest_branch[v], v))
        order.append(chosen)
        neighbors = adjacency.pop(chosen)
        for u in neighbors:
            adjacency[u].remove(chosen)
            # Each neighbor belongs to a distinct subtree after the cut.
            component, stack = {u}, [u]
            while stack:
                v = stack.pop()
                for w in adjacency[v]:
                    if w not in component:
                        component.add(w)
                        stack.append(w)
            heapq.heappush(heap, (-len(component), min(component), component))
    return order


def corehd_robustness_curve_igraph_with_marker(G_ig: ig.Graph, forest_strategy='tree_breaking'):
    if forest_strategy not in ('tree_breaking', 'legacy_hd'):
        raise ValueError('Unknown CoreHD forest_strategy: ' + str(forest_strategy))
    # 1) 转换成 networkx
    G = igraph_to_networkx(G_ig)

    # 2) 工作副本
    G_work = G.copy()
    N0 = G_work.number_of_nodes()
    if N0 == 0:
        return [0.0], 0, []

    # 3) CoreHD 阶段删除序列（瓦解 2-core）
    D = _corehd_sequence_k2(G.copy())
    q_core_end = len(D)  # 2-core 瓦解结束点（步数），森林阶段从这里开始

    # 4) Appendix A: greedily minimize the next largest component in the forest.
    G_forest = G_work.copy()
    for v in D:
        if G_forest.has_node(v):
            G_forest.remove_node(v)

    tail = (_greedy_tree_breaking_order if forest_strategy == 'tree_breaking'
            else _max_degree_full_removal_order)
    T = tail(G_forest)
    full_order = D + T

    # 5) 逐步删除并记录 s(Q) —— 从 k=0 开始
    curve = [_frac_lcc(G_work, N0)]
    for v in full_order:
        if G_work.has_node(v):
            G_work.remove_node(v)
        curve.append(_frac_lcc(G_work, N0))

    return curve, q_core_end, full_order

def corehd_robustness_curve_igraph(G_ig: ig.Graph, **kwargs):
    """Full q=0..N curve, using the same order as the scalar R interface."""
    curve, _, _ = corehd_robustness_curve_igraph_with_marker(G_ig, **kwargs)
    return curve


# ---------- R 指标 ----------
def corehd_R_igraph(G_ig: ig.Graph, **kwargs) -> float:
    curve, q_core_end, _ = corehd_robustness_curve_igraph_with_marker(G_ig, **kwargs)
    return robustness_from_curve(curve, G_ig.vcount())

# ---------- 生成 igraph 的 ER 图 ----------

# ---------- 从 txt 读入图：每行形如 "1 2 {}" ----------

# ---------- 主程序 ----------

# 运行实验
