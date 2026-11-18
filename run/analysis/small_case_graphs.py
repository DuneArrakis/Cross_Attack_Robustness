"""Graph identity/storage and legal swaps for mechanism examples up to N=64.

Separate from the archived BA15 two-word format. BA20 needs 190 adjacency bits,
so mask storage must retain three uint64 words rather than truncate at 128 bits.
"""
from functools import lru_cache
from itertools import combinations
import random

import igraph as ig
import numpy as np

from ..datasets.generate import generate_ba_graph


@lru_cache(None)
def pairs(n):
    if not 1<=n<=64:raise ValueError('Small-case masks support 1<=N<=64')
    return tuple(combinations(range(n),2))


@lru_cache(None)
def edge_bits(n):
    return {edge:1<<i for i,edge in enumerate(pairs(n))}


def graph_mask(graph):
    if graph.is_directed() or not graph.is_simple():raise ValueError('Simple undirected graphs only')
    bits=edge_bits(graph.vcount())
    return sum(bits[tuple(sorted(e))] for e in graph.get_edgelist())


def graph_from_mask(mask,n):
    mask=int(mask)
    if mask<0 or mask.bit_length()>n*(n-1)//2:raise ValueError('Mask outside graph size')
    return ig.Graph(n=n,edges=[e for i,e in enumerate(pairs(n)) if mask&(1<<i)])


def pack_masks(masks,n):
    width=max(1,(n*(n-1)//2+63)//64)
    values=[]
    for mask in masks:
        mask=int(mask)
        if mask<0 or mask.bit_length()>n*(n-1)//2:raise ValueError('Mask overflow')
        values.append([(mask>>(64*w))&((1<<64)-1) for w in range(width)])
    return np.asarray(values,dtype=np.uint64).reshape((-1,width))


def unpack_mask(row):
    return sum(int(v)<<(64*w) for w,v in enumerate(row))


def canonical_mask(graph):
    p=graph.canonical_permutation();bits=edge_bits(graph.vcount())
    return sum(bits[tuple(sorted((p[u],p[v])))] for u,v in graph.get_edgelist())


def make_graph(n,seed):
    ig.set_random_number_generator(random.Random(int(seed)))
    g=generate_ba_graph(n,3,3)
    return graph_from_mask(graph_mask(g),n)


def switch_edges(before,after):
    old,new=set(before.get_edgelist()),set(after.get_edgelist())
    removed,added=sorted(old-new),sorted(new-old)
    if removed or added:
        if len(removed)!=2 or len(added)!=2 or before.degree()!=after.degree():
            raise ValueError('Not a degree-preserving two-edge switch')
    return removed,added


def neighbors(graph):
    old=set(graph.get_edgelist());states={}
    for (a,b),(c,d) in combinations(sorted(old),2):
        if len({a,b,c,d})<4:continue
        for e,f in [((a,c),(b,d)),((a,d),(b,c))]:
            e,f=tuple(sorted(e)),tuple(sorted(f))
            if e in old or f in old:continue
            candidate=ig.Graph(n=graph.vcount(),edges=sorted(old-{(a,b),(c,d)}|{e,f}))
            mask=graph_mask(candidate)
            states.setdefault(mask,dict(removed_edges=sorted([(a,b),(c,d)]),added_edges=sorted([e,f])))
    return states
