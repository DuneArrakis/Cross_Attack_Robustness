# ei_interfaces_igraph_parametric.py
# ------------------------------------------------------------
# Explosive Immunization (EI) 算法实现（参数集中管理 + 安全外部调用）
# 对外接口（只需 G）：
#   ei_robustness_curve(G) -> List[float]
#   ei_robustness_total(G) -> float
# 可选全局参数设置：
#   set_ei_params(m=None, k=None, iters=None, seed=None)
# ------------------------------------------------------------

import math
import random
from typing import List, Tuple, Optional
import numpy as np
import igraph as ig
from .common import robustness_from_curve

__all__ = [
    "ei_robustness_curve",
    "ei_robustness_total",
    "set_ei_params",
]

# ==========================================================
# ===============  全局配置（对外可修改）  ===================
# ==========================================================

class EIConfig:
    # EI 算法参数默认值（符合你之前的要求：m=10, K=6）
    M_CAND: int = 1000       # 每轮候选节点数 m
    K_STRONG: int = 6      # 强枢纽阈值 K
    KEFF_ITERS: int = 3    # k_eff 迭代次数
    RANDOM_SEED: Optional[int] = 42  # 随机种子（None 表示不固定）

def set_ei_params(m: Optional[int] = None,
                  k: Optional[int] = None,
                  iters: Optional[int] = None,
                  seed: Optional[int] = None) -> None:
    """
    可选：全局设置 EI 参数（保持接口只需 G）。
    不传的参数不改变当前配置。
    """
    if m is not None:
        # EIConfig.M_CAND = int(m)
        EIConfig.M_CAND = 1000  # 固定为 1000，不允许外部改动
    if k is not None:
        EIConfig.K_STRONG = int(k)
    if iters is not None:
        EIConfig.KEFF_ITERS = int(iters)
    if seed is not None:
        EIConfig.RANDOM_SEED = seed

# ==========================================================
# ===============  核心算法部分  ============================
# ==========================================================

class UF:
    """并查集结构，用于动态追踪连通簇"""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size   = [0]*n
        self.active = [False]*n
        self.max_size = 0

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def activate(self, x: int) -> None:
        if self.active[x]:
            return
        self.active[x] = True
        self.parent[x] = x
        self.size[x]   = 1
        if self.size[x] > self.max_size:
            self.max_size = self.size[x]

    def union_if_active(self, a: int, b: int) -> None:
        if not (self.active[a] and self.active[b]):
            return
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra]  += self.size[rb]
        if self.size[ra] > self.max_size:
            self.max_size = self.size[ra]


def _iterate_keff(G: ig.Graph) -> List[int]:
    """计算 k_eff(i) = k(i) - L_i - M_i"""
    K = EIConfig.K_STRONG
    deg   = G.degree()
    n     = G.vcount()
    keff  = deg[:]
    neigh = [G.neighbors(v) for v in range(n)]
    for _ in range(EIConfig.KEFF_ITERS):
        strong   = [1 if keff[v] >= K else 0 for v in range(n)]
        new_keff = [0]*n
        for i in range(n):
            Li = sum(1 for j in neigh[i] if deg[j] == 1)
            Mi = sum(1 for j in neigh[i] if strong[j] == 1)
            val = deg[i] - Li - Mi
            new_keff[i] = val if val > 0 else 0
        keff = new_keff
    return keff


def _neighbor_sus_clusters(i: int, UF_obj: UF, neigh: List[List[int]]) -> Tuple[List[int], int]:
    """返回 i 邻域中已恢复簇的大小列表（降序）和数量"""
    roots = {}
    for j in neigh[i]:
        if UF_obj.active[j]:
            r = UF_obj.find(j)
            roots[r] = UF_obj.size[r]
    sizes = sorted(roots.values(), reverse=True)
    return sizes, len(sizes)


def _explosive_immunization_with_curve(G: ig.Graph, seed=None) -> Tuple[List[int], List[float]]:
    """反向恢复过程：同时返回删除序列与鲁棒性曲线 R=GCC/N（每删1个点记录一次）"""
    n     = G.vcount()
    if n == 0:
        return [], [0.0]
    m     = EIConfig.M_CAND
    neigh = [G.neighbors(v) for v in range(n)]
    keff  = _iterate_keff(G)

    #每次评估用局部 RNG（同一张图多次调用结果一致）
    rng = random.Random(EIConfig.RANDOM_SEED if seed is None else seed)

    UF_obj = UF(n)
    deleted = set(range(n))
    restored_order: List[int] = []
    # lcc_by_restored[r] = 恢复 r 个节点后的 GCC 大小（r=0..n）
    lcc_by_restored = [0] * (n + 1)

    def _S() -> float:
        return UF_obj.max_size / n

    while len(restored_order) < n:
        if len(deleted) <= m:
            candidates = sorted(deleted)
        else:
            candidates = rng.sample(sorted(deleted), m)        # --- 关键判分：sigma(1) or sigma(2) ---
        # 论文：当巨型簇开始主导后（S > 1/sqrt(N)）切到 σ(2)
        use_sigma1 = (_S() <= 1 / math.sqrt(n))
        best, best_key = None, None

        # σ(2) 阶段（方案 B，论文口径）：
        #   - 候选若不邻接 giant，则等价于 σ(2)=∞（无效）
        #   - 若本轮采样候选全无效：继续重采样，直到出现至少一个有效候选
        #   - 例外：当 len(deleted) <= m 时，候选已是全体 deleted；
        #           若仍全无效，说明“邻接 giant 的 vaccinated 节点集合为空”，σ(2) 无法执行，
        #           只能退回 σ(1) 完成一步以保证过程可终止。
        if not use_sigma1:
            while True:
                any_valid = False
                for i in candidates:
                    sizes, _ = _neighbor_sus_clusters(i, UF_obj, neigh)
                    has_giant = (UF_obj.max_size > 0 and len(sizes) > 0 and sizes[0] == UF_obj.max_size)
                    if has_giant:
                        any_valid = True
                        break

                if any_valid:
                    break

                # 若候选集已经覆盖全部 deleted，则全局无有效候选，σ(2) 不可执行
                if len(deleted) <= m:
                    use_sigma1 = True
                    break

                # 否则继续重采样（不设 MAX_TRIES，不回退 σ(1)）
                candidates = rng.sample(sorted(deleted), m)

        for i in candidates:
            sizes, card = _neighbor_sus_clusters(i, UF_obj, neigh)
            if use_sigma1:
                nonlocal_term = sum((math.sqrt(s) - 1.0) for s in sizes if s > 1)
                key = (keff[i] + nonlocal_term, )
            else:
                has_giant = (len(sizes) > 0 and sizes[0] == UF_obj.max_size and UF_obj.max_size > 0)
                if not has_giant:
                    continue  # σ(2) 下无效候选直接跳过
                second = sizes[1] if len(sizes) >= 2 else 0
                key = (card, second)

            if best_key is None or key < best_key:
                best, best_key = i, key

        if best is None:
            best = candidates[0]

        deleted.remove(best)
        UF_obj.activate(best)
        for j in neigh[best]:
            UF_obj.union_if_active(best, j)
        restored_order.append(best)

        # 记录恢复到当前规模后的 GCC
        lcc_by_restored[len(restored_order)] = UF_obj.max_size

    deletion_order = list(reversed(restored_order))

    # 删除 k 个节点后，剩余 r=n-k 个节点；对应反向恢复 r 个节点时的 GCC
    R_curve = [(lcc_by_restored[n - k] / n) for k in range(0, n + 1)]
    return deletion_order, R_curve


def _explosive_immunization(G: ig.Graph) -> List[int]:
    """反向恢复过程，输出删除序列（读取 EIConfig）"""
    deletion_order, _ = _explosive_immunization_with_curve(G)
    return deletion_order


def _robustness_curve_from_order(G: ig.Graph, deletion_order: List[int]) -> List[float]:
    """按删除顺序依次删除节点，返回 R=GCC/N 曲线"""
    n = G.vcount()
    H = G.copy()
    R: List[float] = []
    id_map = list(range(n))
    deleted_flag = [False]*n

    for v in deletion_order:
        idx = id_map[v]
        H.delete_vertices([idx])
        deleted_flag[v] = True
        for u in range(n):
            if not deleted_flag[u] and id_map[u] > idx:
                id_map[u] -= 1
        if H.vcount() == 0:
            R.append(0.0)
        else:
            comps = H.components()
            gcc = max(comps.sizes()) if len(comps) > 0 else 0
            R.append(gcc / n)
    return R


# ==========================================================
# ===============  对外接口（仅需 G 参数）  =================
# ==========================================================

def ei_robustness_curve(G: ig.Graph) -> List[float]:
    """接口1：传入 igraph 图 G，返回鲁棒性变化曲线 R（列表）"""
    _, R_curve = _explosive_immunization_with_curve(G)
    return R_curve



def ei_robustness_total(G: ig.Graph) -> float:
    """接口2：传入 igraph 图 G，返回鲁棒性积分 sum(R)/N"""
    _, R = _explosive_immunization_with_curve(G)
    N = G.vcount()
    return robustness_from_curve(R, N)


# ==========================================================
# ===============  主程序（绘图示例，仅示范）  ===============
# ==========================================================
