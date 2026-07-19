from dataclasses import dataclass
import numpy as np

from score import local_score
from utils import enumerate_orientations


@dataclass(frozen=True)
class ESResult:
    best_name: str
    best_W: np.ndarray              
    best_score: float                
    per_candidate_score: dict        


def _parents_from_adj(A: np.ndarray) -> list[list[int]]:
    d = A.shape[0]
    return [sorted(int(j) for j in np.nonzero(A[i])[0]) for i in range(d)]


def score_candidate(
    X: np.ndarray,
    node_info: dict,
    candidate_W: np.ndarray,
    *,
    max_iter: int = 200,
    ftol: float = 1e-8,
) -> float:
    d = candidate_W.shape[0]
    parents_per_node = _parents_from_adj(candidate_W)
    total = 0.0
    for i in range(d):
        res = local_score(i, parents_per_node[i], X, node_info,
                          max_iter=max_iter, ftol=ftol)
        total += res.score
    return total


def exhaustive_search(
    X: np.ndarray,
    node_info: dict,
    skeleton: np.ndarray,
    *,
    max_iter: int = 200,
    ftol: float = 1e-8,
) -> ESResult:
    candidates = enumerate_orientations(skeleton)
    if not candidates:
        raise ValueError("skeleton admits no acyclic orientation")
    per_candidate: dict[str, float] = {}
    for name, W in candidates.items():
        per_candidate[name] = score_candidate(X, node_info, W,
                                              max_iter=max_iter, ftol=ftol)
    best_name = min(sorted(per_candidate), key=per_candidate.__getitem__)
    best_W = (np.asarray(candidates[best_name]) != 0).astype(np.float64)
    return ESResult(
        best_name=best_name,
        best_W=best_W,
        best_score=per_candidate[best_name],
        per_candidate_score=per_candidate,
    )