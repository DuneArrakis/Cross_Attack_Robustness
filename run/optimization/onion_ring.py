"""Degree-ordered and random-ring constructions used in Fig. 8."""
import random

import igraph as ig

from .onion import layer_indices_from_degrees


def generate_ring(degrees, order, seed):
    """Return a simple graph with the supplied vertex degrees and ring order.

    Both variants reserve two edges per non-leaf and use the same layer
    preference (a=3). The seed controls leaf placement, ring ties, and pairing.
    """
    if order not in ('degree', 'random'):
        raise ValueError("Ring order must be 'degree' or 'random'")
    if not degrees or min(degrees) < 1 or max(degrees) >= len(degrees) or sum(degrees) % 2:
        raise ValueError('Expected a positive degree sequence for a simple graph')
    core = [i for i, degree in enumerate(degrees) if degree > 1]
    leaves = [i for i, degree in enumerate(degrees) if degree == 1]
    if len(core) < 3 or sum(degrees[i] - 2 for i in core) < len(leaves):
        raise ValueError('The degree sequence cannot support a ring and its leaves')

    rng = random.Random(seed)
    layers = layer_indices_from_degrees(degrees)
    for _ in range(20):
        neighbors = [set() for _ in degrees]
        edges = []
        remaining = list(degrees)

        def add_edge(u, v):
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append((min(u, v), max(u, v)))

        leaf_order = leaves.copy()
        rng.shuffle(leaf_order)
        for leaf in leaf_order:
            hosts = [i for i in core if remaining[i] > 2]
            host = rng.choices(hosts, weights=[1 / degrees[i]**2 for i in hosts])[0]
            add_edge(leaf, host)
            remaining[leaf] = 0
            remaining[host] -= 1

        ring = core.copy()
        rng.shuffle(ring)
        if order == 'degree':
            # A stable sort preserves the random order within each degree class.
            ring.sort(key=lambda i: degrees[i])
        for u, v in zip(ring, ring[1:] + ring[:1]):
            add_edge(u, v)
            remaining[u] -= 1
            remaining[v] -= 1

        fixed = len(edges)
        stubs = [i for i, count in enumerate(remaining) for _ in range(count)]
        rng.shuffle(stubs)
        for _ in range(max(200000, 4000 * sum(degrees))):
            if len(stubs) < 2:
                break
            i = rng.randrange(len(stubs))
            j = rng.randrange(len(stubs) - 1)
            if j >= i:
                j += 1
            v, u = stubs[i], stubs[j]
            if u == v or u in neighbors[v]:
                continue
            if rng.random() >= 1 / (1 + 3 * abs(layers[v] - layers[u])):
                continue
            add_edge(u, v)
            for index in sorted((i, j), reverse=True):
                stubs[index] = stubs[-1]
                stubs.pop()

        # Absorb the last two stubs by splitting an unprotected edge.
        # The leaf edges and the ring stay in place throughout the reshuffle.
        def absorb_pair(v, u):
            if len(edges) == fixed:
                return False
            for _ in range(100000):
                index = rng.randrange(fixed, len(edges))
                x, y = edges[index]
                neighbors[x].remove(y)
                neighbors[y].remove(x)
                first, second = (x, y) if rng.random() < .5 else (y, x)
                edge1 = (min(v, first), max(v, first))
                edge2 = (min(u, second), max(u, second))
                if (v != first and u != second and first not in neighbors[v]
                        and second not in neighbors[u] and edge1 != edge2):
                    neighbors[v].add(first)
                    neighbors[first].add(v)
                    neighbors[u].add(second)
                    neighbors[second].add(u)
                    edges[index] = edge1
                    edges.append(edge2)
                    return True
                neighbors[x].add(y)
                neighbors[y].add(x)
            return False

        while stubs:
            if not absorb_pair(stubs.pop(), stubs.pop()):
                break
        else:
            graph = ig.Graph(n=len(degrees), edges=sorted(edges), directed=False)
            if graph.degree() != list(degrees) or not graph.is_simple() or not graph.is_connected():
                raise ValueError('Invalid ring realization')
            return graph
        rng.seed(rng.randrange(1 << 30))
    raise RuntimeError('Could not complete a ring realization in 20 starts')
