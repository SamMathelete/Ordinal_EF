from collections import deque
from dataclasses import dataclass, field
from typing import Iterable
import numpy as np
from score import local_score, LocalScoreResult

def _is_acyclic(parents_of: list[set[int]]) -> bool:
    d = len(parents_of)
    children_of: list[set[int]] = [set() for _ in range(d)]
    for i, ps in enumerate(parents_of):
        for j in ps:
            children_of[j].add(i)
    indeg = [len(parents_of[i]) for i in range(d)]
    q = deque(i for i in range(d) if indeg[i] == 0)
    visited = 0
    while q:
        u = q.popleft()
        visited += 1
        for c in children_of[u]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    return visited == d

class _ScoreCache:
    def __init__(self, X, node_info, fit_kwargs):
        self.X = X
        self.node_info = node_info
        self.fit_kwargs = fit_kwargs
        self._cache: dict[tuple[int, frozenset], LocalScoreResult] = {}
        self.n_fits = 0

    def get(self, node: int, parents: Iterable[int]) -> LocalScoreResult:
        key = (node, frozenset(parents))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        res = local_score(node, sorted(parents), self.X, self.node_info, **self.fit_kwargs)
        self._cache[key] = res
        self.n_fits += 1
        return res

@dataclass
class GreedyDAGResult:
    W: np.ndarray
    parents_of: list[set[int]]
    total_score: float
    insert_steps: int
    delete_steps: int
    reverse_steps: int
    n_fits: int
    history: list[dict] = field(default_factory=list)

def estimate_dag_greedy(
    X: np.ndarray,
    node_info: dict,
    bp_mask: np.ndarray,
    *,
    score_tol: float = 1e-6,
    max_outer_passes: int = 10,
    max_inner_steps: int | None = None,
    fit_max_iter: int = 200,
    fit_ftol: float = 1e-8,
    verbose: bool = False,
) -> GreedyDAGResult:
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, d); got {X.shape}")
    N, d = X.shape
    bp_mask = np.asarray(bp_mask, dtype=np.int8)
    if bp_mask.shape != (d, d):
        raise ValueError(f"bp_mask shape {bp_mask.shape} != (d, d) = ({d}, {d})")
    if max_inner_steps is None:
        max_inner_steps = d * d
    cache = _ScoreCache(X, node_info, dict(max_iter=fit_max_iter, ftol=fit_ftol))
    parents_of: list[set[int]] = [set() for _ in range(d)]
    node_score: list[float] = [cache.get(i, frozenset()).score for i in range(d)]
    total_score = sum(node_score)
    history: list[dict] = []
    cand_pairs = [(int(child), int(parent))
                  for child in range(d) for parent in range(d)
                  if bp_mask[child, parent] == 1 and child != parent]
    if verbose:
        print(f"[greedy DAG] d={d}, |candidates|={len(cand_pairs)}, "
              f"initial BIC={total_score:.4f}")
    insert_steps = 0
    delete_steps = 0
    reverse_steps = 0

    def _try_insert() -> bool:
        nonlocal total_score
        best_delta = 0.0
        best_op = None
        for (child, parent) in cand_pairs:
            if parent in parents_of[child]:
                continue
            new_parents = parents_of[child] | {parent}
            tmp = [ps for ps in parents_of]
            tmp[child] = new_parents
            if not _is_acyclic(tmp):
                continue
            new_score_child = cache.get(child, new_parents).score
            delta = new_score_child - node_score[child]
            if delta < best_delta - 1e-15:
                best_delta = delta
                best_op = (child, parent, new_score_child)
        if best_op is None or best_delta > -score_tol:
            return False
        child, parent, new_score_child = best_op
        parents_of[child] = parents_of[child] | {parent}
        total_score += (new_score_child - node_score[child])
        node_score[child] = new_score_child
        history.append({"op": "insert", "edge": (parent, child),
                        "delta": best_delta, "total_score": total_score})
        if verbose:
            print(f"  +  {parent}->{child}  delta={best_delta:+.5f}  total={total_score:.4f}")
        return True

    def _try_delete() -> bool:
        nonlocal total_score
        best_delta = 0.0
        best_op = None
        for child in range(d):
            if not parents_of[child]:
                continue
            for parent in list(parents_of[child]):
                new_parents = parents_of[child] - {parent}
                new_score_child = cache.get(child, new_parents).score
                delta = new_score_child - node_score[child]
                if delta < best_delta - 1e-15:
                    best_delta = delta
                    best_op = (child, parent, new_score_child)
        if best_op is None or best_delta > -score_tol:
            return False
        child, parent, new_score_child = best_op
        parents_of[child] = parents_of[child] - {parent}
        total_score += (new_score_child - node_score[child])
        node_score[child] = new_score_child
        history.append({"op": "delete", "edge": (parent, child),
                        "delta": best_delta, "total_score": total_score})
        if verbose:
            print(f"  -  {parent}->{child}  delta={best_delta:+.5f}  total={total_score:.4f}")
        return True

    def _try_reverse() -> bool:
        nonlocal total_score
        best_delta = 0.0
        best_op = None
        for child in range(d):
            for parent in list(parents_of[child]):
                if bp_mask[parent, child] == 0:
                    continue
                new_parents_child = parents_of[child] - {parent}
                new_parents_parent = parents_of[parent] | {child}
                if child in parents_of[parent]:
                    continue
                tmp = [ps for ps in parents_of]
                tmp[child] = new_parents_child
                tmp[parent] = new_parents_parent
                if not _is_acyclic(tmp):
                    continue
                new_score_child = cache.get(child, new_parents_child).score
                new_score_parent = cache.get(parent, new_parents_parent).score
                delta = (new_score_child - node_score[child]) + \
                        (new_score_parent - node_score[parent])
                if delta < best_delta - 1e-15:
                    best_delta = delta
                    best_op = (child, parent, new_score_child, new_score_parent,
                               new_parents_child, new_parents_parent)
        if best_op is None or best_delta > -score_tol:
            return False
        (child, parent, new_score_child, new_score_parent,
         new_parents_child, new_parents_parent) = best_op
        parents_of[child] = new_parents_child
        parents_of[parent] = new_parents_parent
        total_score += (new_score_child - node_score[child])
        total_score += (new_score_parent - node_score[parent])
        node_score[child] = new_score_child
        node_score[parent] = new_score_parent
        history.append({"op": "reverse", "edge": (parent, child),
                        "delta": best_delta, "total_score": total_score})
        if verbose:
            print(f"  ~  {parent}->{child} flipped to {child}->{parent}  "
                  f"delta={best_delta:+.5f}  total={total_score:.4f}")
        return True
    for outer in range(max_outer_passes):
        any_change = False
        for _ in range(max_inner_steps):
            if not _try_insert():
                break
            insert_steps += 1
            any_change = True
        for _ in range(max_inner_steps):
            if not _try_delete():
                break
            delete_steps += 1
            any_change = True
        for _ in range(max_inner_steps):
            if not _try_reverse():
                break
            reverse_steps += 1
            any_change = True
        if not any_change:
            if verbose:
                print(f"[greedy DAG] converged after outer pass {outer + 1}")
            break
        if verbose:
            print(f"[greedy DAG] outer pass {outer + 1}: insert={insert_steps}, "
                  f"delete={delete_steps}, reverse={reverse_steps}, BIC={total_score:.4f}")
    W = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        if not parents_of[i]:
            continue
        ps = sorted(parents_of[i])
        res = cache.get(i, parents_of[i])
        for k, j in enumerate(ps):
            W[i, j] = res.weights[k]
    return GreedyDAGResult(
        W=W,
        parents_of=parents_of,
        total_score=total_score,
        insert_steps=insert_steps,
        delete_steps=delete_steps,
        reverse_steps=reverse_steps,
        n_fits=cache.n_fits,
        history=history,
    )