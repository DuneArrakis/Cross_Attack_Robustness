#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import igraph as ig
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import uuid
from typing import List, Optional
import glob
from .common import lcc_curve_from_order, robustness_from_curve

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vendor'))
RUNTIME_ROOT = os.path.abspath(os.path.join(ROOT, '..', '..', 'work', 'runtime', 'finder'))
sys.path.append(os.path.join(ROOT, "FINDER_ND"))
from FINDER import FINDER

# ---- TF: 强制 TF1 风格，并提供 reset_default_graph ----
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # 强烈建议关闭 eager（避免某些环境下 TF2 默认行为干扰）
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass
except Exception:
    tf = None


MODEL_FILE_CKPT = 'nrange_30_50_iter_78000.ckpt'
STEP_RATIO_DEFAULT = 0.01
STRATEGY_ID_DEFAULT = 0
WORK_TAG_DEFAULT = 'wrapper_runtime'


@contextmanager
def _tensorflow_thread_limit():
    """Override the vendored 100-thread session only when explicitly requested.

    FINDER is a compiled extension, so environment TF thread hints alone cannot
    override its ConfigProto. Restore the factory as soon as initialization ends.
    This changes execution resources, not the model or attack policy.
    """
    value = os.environ.get('RESS_FINDER_THREADS')
    if value is None or tf is None:
        yield
        return
    threads = int(value)
    if threads < 1:
        raise ValueError('RESS_FINDER_THREADS must be positive')
    original = tf.ConfigProto

    def limited_config(*args, **kwargs):
        config = original(*args, **kwargs)
        config.inter_op_parallelism_threads = threads
        config.intra_op_parallelism_threads = threads
        return config

    tf.ConfigProto = limited_config
    try:
        yield
    finally:
        tf.ConfigProto = original


class FinderWrapperND:
    """
    FINDER ND 封装（进程内单例）：
      - 输入 igraph.Graph
      - robustness_R: 返回完整 q=1..N 曲线对应的论文 R
      - robustness_sequence: 返回从完整删除序列重算的 q=0..N 曲线
      - attack_order: 从 solution 文件解析“删点序列/集合”，并补齐为长度 N 的完整序列
    """

    _INSTANCE = None
    _INIT_DONE = False

    def __new__(cls, *args, **kwargs):
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    def __init__(self,
                 model_ckpt=MODEL_FILE_CKPT,
                 step_ratio=STEP_RATIO_DEFAULT,
                 strategy_id=STRATEGY_ID_DEFAULT,
                 work_tag=WORK_TAG_DEFAULT,
                 keep_artifacts: bool = False):

        # ---- 单例：避免重复 LoadModel 导致 restore 漂移 ----
        if self.__class__._INIT_DONE:
            return

        self.root = ROOT
        self.model_path = os.path.join(self.root, "FINDER_ND", "models", model_ckpt)

        self.rel_data_dir = os.path.join(RUNTIME_ROOT, "data", work_tag).replace("\\", "/")
        self.rel_result_root = os.path.join(RUNTIME_ROOT, "output", work_tag).replace("\\", "/")

        abs_data_dir = os.path.join(self.root, self.rel_data_dir)
        abs_result_root = os.path.join(self.root, self.rel_result_root)
        os.makedirs(abs_data_dir, exist_ok=True)
        os.makedirs(abs_result_root, exist_ok=True)

        self.step_ratio = float(step_ratio)
        self.strategy_id = int(strategy_id)
        self.keep_artifacts = bool(keep_artifacts)

        # ---- 关键：初始化前 reset_default_graph，保证变量名与 ckpt 对齐 ----
        if tf is not None:
            try:
                tf.compat.v1.reset_default_graph()
            except Exception:
                pass

        with _tensorflow_thread_limit(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.dqn = FINDER()
            self.dqn.LoadModel(self.model_path)

        self.__class__._INIT_DONE = True

    def _auto_name(self) -> str:
        # Parallel workers share the runtime directory. Keep the full UUID to
        # avoid collisions across the many calls in a complete SA batch.
        return f"auto_{os.getpid()}_{uuid.uuid4().hex}"

    def _cleanup_run_files(self, name: str, step_str: str):
        # data 文件及其同名前缀产物
        data_prefix = os.path.join(self.root, self.rel_data_dir, name)
        for p in glob.glob(data_prefix + "*"):
            try:
                os.remove(p)
            except Exception:
                pass

        # solution 目录及其同名前缀产物
        sol_dir_rel = f"{self.rel_result_root}/StepRatio_{step_str}"
        sol_dir_abs = os.path.join(self.root, sol_dir_rel)
        sol_prefix = os.path.join(sol_dir_abs, name)
        for p in glob.glob(sol_prefix + "*"):
            try:
                os.remove(p)
            except Exception:
                pass

    def _save_igraph(self, G: ig.Graph, name: str) -> str:
        rel_path = f"{self.rel_data_dir}/{name}.txt"
        abs_path = os.path.join(self.root, rel_path)
        with open(abs_path, "w", encoding="utf-8") as f:
            for u, v in G.get_edgelist():
                f.write(f"{u} {v} {{}}\n")
        return rel_path

    def _read_solution_order(self, solution_abs_path: str, n: int) -> List[int]:
        # This vendored FINDER writes one vertex ID per line, never a binary mask.
        with open(solution_abs_path, encoding='utf-8') as stream:
            values = [int(line.strip()) for line in stream if line.strip()]
        if (not values or len(set(values)) != len(values)
                or any(v < 0 or v >= n for v in values)):
            raise ValueError('FINDER returned an empty, duplicate or out-of-range solution')
        return values

    def _complete_to_full_order(self, partial: List[int], n: int) -> List[int]:
        if n == 0:
            return []
        if not partial or len(set(partial)) != len(partial) or any(v < 0 or v >= n for v in partial):
            raise ValueError('Invalid FINDER attack prefix')
        seen = set(partial)
        return list(partial) + [v for v in range(n) if v not in seen]

    def _run_finder(self, G: ig.Graph, name: str):
        if self.strategy_id != 0:
            raise ValueError('The frozen FINDER protocol uses strategy_id=0 (no reinsertion)')
        n = G.vcount()
        # The external edge-list reader cannot represent isolates. Compact the
        # active vertices and map predictions back; append original isolates last.
        active = [v for v, degree in enumerate(G.degree()) if degree > 0]
        isolates = [v for v, degree in enumerate(G.degree()) if degree == 0]
        if not active:
            order = list(range(n))
            curve = lcc_curve_from_order(G, order)
            return robustness_from_curve(curve), curve, order
        compact = G.induced_subgraph(active)
        data_rel = self._save_igraph(compact, name)
        step_str = f"{self.step_ratio:.4f}"
        solution_dir = os.path.join(self.rel_result_root, 'StepRatio_' + step_str)
        solution_abs = os.path.join(solution_dir, name + '.txt')
        try:
            # The vendor uses a check-then-mkdir here; precreate the shared
            # directory safely before concurrent processes enter that code.
            os.makedirs(solution_dir, exist_ok=True)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                solution, _ = self.dqn.EvaluateRealData(
                    self.model_path, data_rel, self.rel_result_root, self.step_ratio)
            prefix = self._read_solution_order(solution_abs, len(active))
            if prefix != list(solution):
                raise ValueError('FINDER returned and saved different attack sequences')
            remaining = [v for v in range(len(active)) if v not in set(prefix)]
            if compact.induced_subgraph(remaining).ecount():
                raise ValueError('FINDER stopped before removing all edges')
            order = [active[v] for v in self._complete_to_full_order(prefix, len(active))] + isolates
            curve = lcc_curve_from_order(G, order)
            return robustness_from_curve(curve), curve, order
        finally:
            if not self.keep_artifacts:
                self._cleanup_run_files(name, step_str)

    def robustness_R(self, G: ig.Graph, name: Optional[str] = None) -> float:
        if name is None:
            name = self._auto_name()
        score, _, _ = self._run_finder(G, name)
        return float(score)

    def robustness_sequence(self, G: ig.Graph, name: Optional[str] = None) -> List[float]:
        if name is None:
            name = self._auto_name()
        _, seq, _ = self._run_finder(G, name)
        return list(seq)

    def attack_order(self, G: ig.Graph, name: Optional[str] = None) -> List[int]:
        if name is None:
            name = self._auto_name()
        _, _, order = self._run_finder(G, name)
        return order



# ============== 测试（可选）=============
