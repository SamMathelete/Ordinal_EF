import os
import pickle
import time
import numpy as np
from config import (
    SEED, N_WORKERS, STATE_DIR, ALL_EXPFAM_DISTS, DIST_HYPERPARAMS, DIST_MODES,
)
from utils import (
    assign_node_types, generate_bipartite_dag, random_orientation,
    generate_ordinal_cutpoints, make_trial_seed, make_trial_rng, run_trials,
    orientation_error,
)
from data_gen import generate_data
from greedy_dag_search import orient_dag_greedy

d = 20
B = 1000
N_VALUES = sorted(set(np.round(
    np.linspace(5, 500, 20)
).astype(int).tolist()))
EDGE_PROBS = {"ER-2": 0.1, "ER-4": 0.9}

GREEDY_KW = dict(
    score_tol=1e-6,
    fit_max_iter=200,
    fit_ftol=1e-8,
)

def _assign_expfam_dists(expfam_nodes, mode, rng):
    if mode.startswith("all_"):
        return {i: mode[4:] for i in expfam_nodes}
    if mode == "full_mix":
        return {i: rng.choice(ALL_EXPFAM_DISTS) for i in expfam_nodes}
    raise ValueError(f"Unknown mode: {mode}")

def _build_node_info(d, ord_nodes, expfam_dists, rng):
    ord_set = set(ord_nodes)
    info = {}
    for i in range(d):
        if i in ord_set:
            info[i] = {"dist": "ordinal", "S": int(rng.integers(3, 5))}
        else:
            dist = expfam_dists[i]
            info[i] = {"dist": dist, **DIST_HYPERPARAMS.get(dist, {})}
    return info

def _build_weighted_dag(edge_prob: float):
    rng = np.random.default_rng(SEED)
    ord_nodes, expfam_nodes = assign_node_types(d, rng)
    A_bin = generate_bipartite_dag(d, ord_nodes, expfam_nodes, edge_prob, rng)
    magnitudes = rng.uniform(0.5, 1.0, size=(d, d))
    signs = np.where(rng.random((d, d)) < 0.5, -1.0, 1.0)
    W_true = A_bin * magnitudes * signs
    skeleton = ((A_bin + A_bin.T) != 0).astype(np.float64)
    return W_true, A_bin, skeleton, ord_nodes, expfam_nodes, int(A_bin.sum())

def _build_mode_info(ord_nodes, expfam_nodes, mode, density):
    rng = make_trial_rng(SEED, density, mode)
    expfam_dists = _assign_expfam_dists(expfam_nodes, mode, rng)
    node_info = _build_node_info(d, ord_nodes, expfam_dists, rng)
    gammas = {i: generate_ordinal_cutpoints(info["S"], rng)
              for i, info in node_info.items() if info["dist"] == "ordinal"}
    return node_info, gammas

def _trial(W_true, A_bin, skeleton, node_info, gammas, N, trial_seed, kw):
    X = generate_data(W=W_true, N=N, node_info=node_info,
                      ordinal_gammas=gammas, seed=trial_seed)
    rng = np.random.default_rng(trial_seed)
    result = orient_dag_greedy(X, node_info, skeleton, rng=rng, **kw)
    return {"orient_err": orientation_error(result.W, A_bin)}

def main():
    print(f"CPU workers: {N_WORKERS}")
    print(f"d = {d},  B = {B},  N values = {N_VALUES}")
    print(f"Modes: {list(DIST_MODES)}")
    dags = {}
    for density, edge_prob in EDGE_PROBS.items():
        W_true, A_bin, skeleton, ord_nodes, expfam_nodes, n_edges = _build_weighted_dag(edge_prob)
        dags[density] = dict(W_true=W_true, A_bin=A_bin, skeleton=skeleton,
                             ord_nodes=ord_nodes, expfam_nodes=expfam_nodes,
                             n_edges=n_edges)
        print(f"  {density}: edges = {n_edges}")
    def _empty():
        return {density: {mode: {n: [] for n in N_VALUES} for mode in DIST_MODES}
                for density in EDGE_PROBS}
    results = {"orient_err": _empty()}
    total = len(EDGE_PROBS) * len(DIST_MODES) * len(N_VALUES) * B
    done = 0
    t0 = time.time()
    for density, info in dags.items():
        for mode in DIST_MODES:
            node_info, gammas = _build_mode_info(
                info["ord_nodes"], info["expfam_nodes"], mode, density
            )
            for N in N_VALUES:
                seed_base = make_trial_seed(SEED, density, mode, N, "data")
                args = [(info["W_true"], info["A_bin"], info["skeleton"],
                         node_info, gammas, N, seed_base + b, GREEDY_KW)
                        for b in range(B)]
                trial_out = run_trials(_trial, args, N_WORKERS)
                for k in results:
                    results[k][density][mode][N] = [r[k] for r in trial_out]
                done += B
                elapsed = time.time() - t0
                print(f"  [{done:>5d}/{total}]  {density:>5s}  "
                      f"{DIST_MODES[mode]['label']:>20s}  N={N:>4d}  "
                      f"rho={np.mean(results['orient_err'][density][mode][N]):.4f}  "
                      f"[{elapsed:.0f}s]")
    os.makedirs(STATE_DIR, exist_ok=True)
    pkl_path = os.path.join(STATE_DIR, "greedy_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({**results, "dags": dags,
                     "config": {"d": d, "B": B, "N_VALUES": N_VALUES}}, f)
    print(f"\nDone in {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f} min)")
    print(f"Saved: {pkl_path}")
    print(f"To generate plots: python plot_all.py")

if __name__ == "__main__":
    main()