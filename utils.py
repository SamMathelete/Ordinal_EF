from collections import deque
import hashlib
import itertools
import numpy as np

def topo_order(
        W: np.ndarray
) -> list[int] | None:
    A = (W != 0).astype(np.int8)
    d = A.shape[0]
    indeg = A.sum(axis=1).tolist()       
    q = deque(i for i in range(d) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for i in np.nonzero(A[:, u])[0]:
            indeg[i] -= 1
            if indeg[i] == 0:
                q.append(int(i))
    return order if len(order) == d else None

def shd(
        W_est: np.ndarray, 
        W_true: np.ndarray
) -> int:
    A_est = (W_est != 0).astype(np.int8)
    A_true = (W_true != 0).astype(np.int8)
    FP = int(((1 - A_true) * A_est).sum())
    FN = int((A_true * (1 - A_est)).sum())
    A_est_T = A_est.T
    R = int((A_true * (1 - A_est) * A_est_T).sum())
    return FP + FN - R

def nshd(
        W_est: np.ndarray, 
        W_true: np.ndarray
) -> float:
    d = W_true.shape[0]
    return 2.0 * shd(W_est, W_true) / (d * (d - 1))

def fnr(
        W_est: np.ndarray, 
        W_true: np.ndarray
) -> float:
    A_est = (W_est != 0).astype(np.int8)
    A_true = (W_true != 0).astype(np.int8)
    P = int(A_true.sum())
    if P == 0:
        return 0.0
    FN = int((A_true * (1 - A_est)).sum())
    return FN / P

def fpr(
        W_est: np.ndarray, 
        W_true: np.ndarray
) -> float:
    A_est = (W_est != 0).astype(np.int8)
    A_true = (W_true != 0).astype(np.int8)
    Nneg = int((1 - A_true).sum())
    if Nneg == 0:
        return 0.0
    FP = int(((1 - A_true) * A_est).sum())
    return FP / Nneg

def bipartite_mask(
        d: int, 
        ord_nodes: list[int], 
        expfam_nodes: list[int]
) -> np.ndarray:
    is_ord = np.zeros(d, dtype=bool)
    is_ord[ord_nodes] = True
    row_ord = is_ord[:, None]
    col_ord = is_ord[None, :]
    mask = (row_ord ^ col_ord).astype(np.int8)
    np.fill_diagonal(mask, 0)
    return mask

def skeleton_edges(
        A_skeleton: np.ndarray
) -> list[tuple[int, int]]:
    d = A_skeleton.shape[0]
    return [(i, j) for i in range(d) for j in range(i + 1, d)
            if A_skeleton[i, j] or A_skeleton[j, i]]

def random_orientation(
        A_skeleton: np.ndarray,
        rng: np.random.Generator
) -> list[set[int]]:
    d = A_skeleton.shape[0]
    order = rng.permutation(d).tolist()
    rank = {node: r for r, node in enumerate(order)}
    parents_of: list[set[int]] = [set() for _ in range(d)]
    for (i, j) in skeleton_edges(A_skeleton):
        if rank[i] < rank[j]:
            parents_of[j].add(i)
        else:
            parents_of[i].add(j)
    return parents_of

_ORIENT_MEMO: dict[bytes, dict[str, np.ndarray]] = {}

def enumerate_orientations(
        A_skeleton: np.ndarray
) -> dict[str, np.ndarray]:
    A_skeleton = np.asarray(A_skeleton)
    key = A_skeleton.astype(np.int8).tobytes() + bytes(str(A_skeleton.shape), "ascii")
    hit = _ORIENT_MEMO.get(key)
    if hit is not None:
        return hit
    d = A_skeleton.shape[0]
    edges = skeleton_edges(A_skeleton)
    cands: dict[str, np.ndarray] = {}
    for bits in itertools.product((0, 1), repeat=len(edges)):
        W = np.zeros((d, d), dtype=np.float64)
        for (u, v), b in zip(edges, bits):
            if b == 0:
                W[v, u] = 1.0
            else:
                W[u, v] = 1.0
        if topo_order(W) is None:
            continue
        cands["O_" + "".join("F" if b == 0 else "R" for b in bits)] = W
    _ORIENT_MEMO[key] = cands
    return cands

def orientation_error(
        W_est: np.ndarray,
        W_true: np.ndarray
) -> float:
    A_est = (W_est != 0).astype(np.int8)
    A_true = (W_true != 0).astype(np.int8)
    P = int(A_true.sum())
    if P == 0:
        return 0.0
    return int((A_true * A_est.T).sum()) / P

def assign_node_types(
        d: int, 
        rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    n_ord = int(max(1, min(d - 1, rng.integers(d // 2 - 1, d // 2 + 2))))
    perm = rng.permutation(d)
    ord_nodes = sorted(perm[:n_ord].tolist())
    expfam_nodes = sorted(perm[n_ord:].tolist())
    return ord_nodes, expfam_nodes

def generate_bipartite_dag(
        d: int,
        ord_nodes: list[int],
        expfam_nodes: list[int],
        edge_prob: float,
        rng: np.random.Generator,
) -> np.ndarray:
    order = rng.permutation(d).tolist()
    rank = {node: r for r, node in enumerate(order)}
    ord_set = set(ord_nodes)
    A = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if (i in ord_set) == (j in ord_set):
                continue
            if rank[j] >= rank[i]:
                continue
            if rng.random() < edge_prob:
                A[i, j] = 1.0
    return A

def generate_ordinal_cutpoints(
        S: int, 
        rng: np.random.Generator
) -> np.ndarray:
    raw = np.sort(rng.uniform(-1.0, 1.0, size=S - 1))
    gamma = np.empty(S + 1, dtype=np.float64)
    gamma[0] = -1e20
    gamma[1:S] = raw
    gamma[S] = 1e20
    return gamma

def _stable_hash(keys) -> int:
    h = hashlib.md5(repr(keys).encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF

def make_trial_seed(seed_base: int, *keys) -> int:
    return (seed_base + _stable_hash(keys)) & 0x7FFFFFFF

def make_trial_rng(seed_base: int, *keys) -> np.random.Generator:
    return np.random.default_rng(make_trial_seed(seed_base, *keys))
 
def run_trials(fn, args_list: list, n_workers: int) -> list:
    if n_workers <= 1 or len(args_list) <= 1:
        return [fn(*a) for a in args_list]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = [None] * len(args_list)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(fn, *a): i for i, a in enumerate(args_list)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    return out