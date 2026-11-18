"""Original ER/BA/WS construction functions, now controlled by external config."""
from __future__ import annotations
import csv
import json
import math
import random
from time import perf_counter
import igraph as ig
from ..config import project_path
from .io import save_igraph_edgelist
from ..experiments.timing import StageRun

def generate_er_graph(n: int, p: float) -> ig.Graph:
    """
    生成 ER 图
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
    生成 BA 图
    使用一个初始完全图作为种子，保证起始连通。
    """
    if n0 < 1:
        raise ValueError("N0 必须 >= 1")
    if m < 1:
        raise ValueError("m_attach 必须 >= 1")
    if n <= n0:
        raise ValueError("n_nodes 必须大于 N0")
    if m > n0:
        raise ValueError("m_attach 不能大于 N0，否则初始阶段无法附着")

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
    生成 WS 小世界网络
    参数解释：
    - dim=1: 一维环形格点
    - size=n: 节点数
    - nei: 每侧连接的近邻数，因此平均度约为 2*nei
    - p: 重连概率
    """
    if dim < 1:
        raise ValueError("ws_dim 必须 >= 1")
    if nei < 1:
        raise ValueError("ws_nei 必须 >= 1")
    if not (0 <= p <= 1):
        raise ValueError("ws_rewire_p 必须在 [0, 1] 内")

    G = ig.Graph.Watts_Strogatz(
        dim=dim,
        size=n,
        nei=nei,
        p=p,
        loops=False,
        multiple=False
    )
    return G

def relabel_graph_by_permutation(G: ig.Graph):
    """
    对图做随机重标号，并返回：
    1) 重标号后的边列表（保证 u < v）
    2) 所用置换 perm
    """
    perm = list(range(G.vcount()))
    random.shuffle(perm)

    edges_new = set()
    for u, v in G.get_edgelist():
        a, b = perm[u], perm[v]

        # 理论上不会出现，但保险起见仍然检查
        if a == b:
            continue

        if a > b:
            a, b = b, a

        edges_new.add((a, b))

    edges_new = sorted(edges_new)
    return edges_new, perm


def run(config):
    options = config['generation']
    output = project_path(options['output_dir'])
    targets = [output / ('graph_{}.txt'.format(i)) for i in range(options['count'])]
    if any(p.exists() for p in targets):
        raise FileExistsError('Generation output already exists; choose another generation.output_dir')
    if options['seed'] is not None:
        random.seed(options['seed'])
        ig.set_random_number_generator(random)
    records = []
    with StageRun(output, 'generate', config):
        for index, path in enumerate(targets):
            started = perf_counter()
            for attempt in range(1, options['max_tries'] + 1):
                kind, n = options['type'].upper(), options['n']
                if kind == 'BA':
                    graph = generate_ba_graph(n, options['ba_n0'], options['ba_m'])
                elif kind == 'ER':
                    p = options['er_p']
                    graph = generate_er_graph(n, p if p is not None else 1.1 * math.log(n) / n)
                elif kind == 'WS':
                    graph = generate_ws_graph(n, options['ws_dim'], options['ws_nei'], options['ws_p'])
                else:
                    raise ValueError('Unsupported graph type: ' + kind)
                if (graph.vcount() != n or graph.is_directed() or not graph.is_simple()
                        or (options['require_connected'] and not graph.is_connected())):
                    continue
                if options['relabel']:
                    edges, permutation = relabel_graph_by_permutation(graph)
                    graph = ig.Graph(n=n, edges=edges)
                else:
                    permutation = list(range(n))
                save_igraph_edgelist(graph, path)
                path.with_suffix('.nodes.json').write_text(json.dumps(permutation) + '\n')
                records.append(dict(instance_id=path.stem, type=kind, n_nodes=n,
                                    n_edges=graph.ecount(), batch_seed=options['seed'],
                                    instance_index=index, attempts=attempt,
                                    wall_seconds=perf_counter()-started, path=str(path.resolve())))
                print('Generated {}: N={}, M={}'.format(path, n, graph.ecount()))
                break
            else:
                raise RuntimeError('No valid graph after {} attempts'.format(options['max_tries']))
        if records:
            with (output / 'manifest.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
