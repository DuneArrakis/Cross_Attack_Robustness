#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import igraph as ig
import warnings
from contextlib import redirect_stdout, redirect_stderr
import io
import uuid
from typing import List, Optional
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "FINDER_ND"))
from FINDER import FINDER

# ---- TF: force TF1-style behavior and provide reset_default_graph ----
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Disable eager execution to avoid TF2 default behavior interfering here.
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


class FinderWrapperND:
    """
    FINDER ND wrapper, implemented as an in-process singleton:
      - Input: igraph.Graph.
      - robustness_R: return robustness integral by averaging MaxCCList plus 0.
      - robustness_sequence: pad MaxCCList to N+1 with a trailing 0.
      - attack_order: parse the deletion sequence/set from the solution file
        and complete it to a full sequence of length N.
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

        # ---- Singleton: avoid repeated LoadModel calls and restore drift ----
        if self.__class__._INIT_DONE:
            return

        self.root = ROOT
        self.model_path = os.path.join(self.root, "FINDER_ND", "models", model_ckpt)

        self.rel_data_dir = os.path.join("data", work_tag).replace("\\", "/")
        self.rel_result_root = os.path.join("finder_output", "FINDER_ND", work_tag).replace("\\", "/")

        abs_data_dir = os.path.join(self.root, self.rel_data_dir)
        abs_result_root = os.path.join(self.root, self.rel_result_root)
        os.makedirs(abs_data_dir, exist_ok=True)
        os.makedirs(abs_result_root, exist_ok=True)

        cwd_result_root = os.path.join(os.getcwd(), "finder_output", "FINDER_ND", work_tag)
        os.makedirs(cwd_result_root, exist_ok=True)

        self.step_ratio = float(step_ratio)
        self.strategy_id = int(strategy_id)
        self.keep_artifacts = bool(keep_artifacts)

        # ---- Reset before initialization so variable names align with ckpt ----
        if tf is not None:
            try:
                tf.compat.v1.reset_default_graph()
            except Exception:
                pass

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.dqn = FINDER()
            self.dqn.LoadModel(self.model_path)

        self.__class__._INIT_DONE = True

    def _auto_name(self) -> str:
        return f"auto_{uuid.uuid4().hex[:6]}"

    def _cleanup_run_files(self, name: str, step_str: str):
        # Data file and any same-prefix artifacts.
        data_prefix = os.path.join(self.root, self.rel_data_dir, name)
        for p in glob.glob(data_prefix + "*"):
            try:
                os.remove(p)
            except Exception:
                pass

        # Solution directory and any same-prefix artifacts.
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

    def _read_solution_order(self, solution_abs_path: str, n: int) -> Optional[List[int]]:
        """
        Parse a solution file:
          - One node id per line is treated as a sequence/set with stable
            de-duplication.
          - A 0/1 mask with length >= n is treated as a set, returning indices
            where mask == 1.
        """
        if not os.path.exists(solution_abs_path):
            return None

        vals: List[int] = []
        try:
            with open(solution_abs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    vals.append(int(line.split()[0]))
        except Exception:
            return None

        if not vals:
            return None

        # Mask case.
        if all(v in (0, 1) for v in vals) and len(vals) >= n:
            mask = vals[:n]
            return [i for i, b in enumerate(mask) if b == 1]

        # Sequence/set case: stable de-duplication plus valid range filtering.
        seen = set()
        out: List[int] = []
        for v in vals:
            if 0 <= v < n and v not in seen:
                seen.add(v)
                out.append(v)
        return out if out else None

    def _complete_to_full_order(self, partial: Optional[List[int]], n: int) -> List[int]:
        """
        FINDER solutions are often prefixes/sets with length < n.
        Complete them to a full deletion sequence of length n for full-process
        robustness analysis:
          full = partial after stable de-duplication + remaining nodes sorted by id
        """
        if n <= 0:
            return []
        if not partial:
            return list(range(n))

        seen = set()
        prefix: List[int] = []
        for v in partial:
            if 0 <= v < n and v not in seen:
                seen.add(v)
                prefix.append(v)

        tail = [v for v in range(n) if v not in seen]
        return prefix + tail

    import glob

    def _run_finder(self, G: ig.Graph, name: str):
        n = G.vcount()

        # 1) Compute all paths first; they are needed in finally.
        data_rel = self._save_igraph(G, name)
        data_abs_prefix = os.path.join(self.root, data_rel)  # e.g. .../data/work_tag/name.txt

        step_str = f"{self.step_ratio:.4f}"
        sol_dir_rel = f"{self.rel_result_root}/StepRatio_{step_str}"
        solution_rel = f"{sol_dir_rel}/{name}.txt"
        solution_abs = os.path.join(self.root, solution_rel)
        sol_abs_prefix = os.path.join(self.root, sol_dir_rel, name)  # e.g. .../StepRatio_xxxx/name

        def _safe_remove(pattern: str):
            for p in glob.glob(pattern):
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                except Exception:
                    pass

        try:
            # 2) Keep the original FINDER call flow unchanged.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                _solution_list, _ = self.dqn.EvaluateRealData(
                    self.model_path,
                    data_rel,
                    self.rel_result_root,
                    self.step_ratio
                )

                score_raw, MaxCCList = self.dqn.EvaluateSol(
                    data_rel,
                    solution_rel,
                    self.strategy_id,
                    reInsertStep=0.001
                )

            # ---- 1) Normalize robustness sequence to length N+1, trailing 0 ----
            seq = list(MaxCCList) if MaxCCList is not None else []
            if len(seq) == n:
                seq.append(0.0)

            # ---- 2) Normalize score: mean over curve points ----
            score = float(sum(seq) / len(seq)) if seq else 0.0

            # ---- 3) Parse attack order from solution and complete to length N ----
            partial_order = self._read_solution_order(solution_abs, n)
            attack_order = self._complete_to_full_order(partial_order, n)

            return score, seq, attack_order

        finally:
            # 3) Clean temporary input/output files after each run by default.
            #    Set self.keep_artifacts=True in __init__ when debugging.
            if not getattr(self, "keep_artifacts", False):
                # Data: remove name.txt and possible same-prefix artifacts.
                _safe_remove(data_abs_prefix.replace(".txt", "") + "*")
                _safe_remove(data_abs_prefix + "*")  # Fallback for name.txt.bak, etc.

                # Solution: remove name.txt and possible same-prefix artifacts.
                _safe_remove(sol_abs_prefix + "*")

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



# ============== Optional Test ==============
if __name__ == "__main__":
    G = ig.Graph.Barabasi(n=100, m=2)
    fw = FinderWrapperND()

    R = fw.robustness_R(G)
    seq = fw.robustness_sequence(G)

    print(R)
    print(len(seq))
    print(seq)
