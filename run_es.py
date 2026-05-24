import os
import pickle
import time

import numpy as np

from config import (
    SEED, N_WORKERS, STATE_DIR, ALL_EXPFAM_DISTS, DIST_HYPERPARAMS,
)
from utils import (
    generate_ordinal_cutpoints, make_trial_seed, run_trials, nshd,
)
from data_gen import generate_data
from es import exhaustive_search

B = 10000
N_VALUES = sorted(set(np.round(
    np.linspace(5, 500, 20)
).astype(int).tolist()))
SIGMA2_VALUES = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_FIXED = 1000     
S_ORD = 4          

GRAPHS = {
    "G1": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),  
    "G2": np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=float),  
    "G3": np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=float),  
}
LABELS = {
    "G1": r"$\mathcal{G}_1$: $X_1 \to X_2 \to X_3$",
    "G2": r"$\mathcal{G}_2$: $X_1 \to X_2 \leftarrow X_3$",
    "G3": r"$\mathcal{G}_3$: $X_1 \leftarrow X_2 \to X_3$",
}

def _node_info(y_dist: str) -> dict:
    return {
        0: {"dist": "ordinal", "S": S_ORD},
        1: {"dist": y_dist, **DIST_HYPERPARAMS.get(y_dist, {})},
        2: {"dist": "ordinal", "S": S_ORD},
    }


def _ordinal_gammas(node_info: dict, rng_seed: int) -> dict:
    rng = np.random.default_rng(rng_seed)
    return {i: generate_ordinal_cutpoints(info["S"], rng)
            for i, info in node_info.items() if info["dist"] == "ordinal"}

def _trial_binary(W_struct, node_info, gammas, candidates, N, trial_seed):
    X = generate_data(W=W_struct, N=N, node_info=node_info,
                      ordinal_gammas=gammas, seed=trial_seed)
    result = exhaustive_search(X, node_info, candidates)
    return float(nshd(result.best_W, W_struct))


def _trial_edge(W_struct, node_info, gammas, candidates, N, sigma2, trial_seed):
    rng = np.random.default_rng(trial_seed)
    abs_bound = float(np.sqrt(3.0 * sigma2))
    d = W_struct.shape[0]
    W_random = rng.uniform(-abs_bound, abs_bound, size=(d, d))
    W = W_struct * W_random
    X = generate_data(W=W, N=N, node_info=node_info,
                      ordinal_gammas=gammas, seed=trial_seed)
    result = exhaustive_search(X, node_info, candidates)
    return float(nshd(result.best_W, W_struct))

def _run_sweep(x_values: list, x_key: str, experiment_key: str):
    print(f"\n=== {experiment_key} ===")
    candidates = GRAPHS

    results = {g: {d_: {x: [] for x in x_values} for d_ in ALL_EXPFAM_DISTS}
               for g in GRAPHS}
    t0 = time.time()

    for truth_name, W_struct in GRAPHS.items():
        print(f"  Truth: {LABELS[truth_name]}")
        for y_dist in ALL_EXPFAM_DISTS:
            node_info = _node_info(y_dist)
            gammas = _ordinal_gammas(
                node_info,
                make_trial_seed(SEED, "gammas", experiment_key, truth_name, y_dist),
            )

            for x in x_values:
                seed_base = make_trial_seed(SEED, experiment_key, truth_name, y_dist, x, "data")
                if x_key == "N":
                    args = [(W_struct, node_info, gammas, candidates, x, seed_base + b)
                            for b in range(B)]
                    fn = _trial_binary
                else:  
                    args = [(W_struct, node_info, gammas, candidates, N_FIXED, x, seed_base + b)
                            for b in range(B)]
                    fn = _trial_edge
                nshd_vec = run_trials(fn, args, N_WORKERS)
                results[truth_name][y_dist][x] = nshd_vec

                elapsed = time.time() - t0
                x_label = f"N={x:>5d}" if x_key == "N" else f"sigma^2={x:.2f}"
                print(f"    {truth_name} {y_dist:>12s}  {x_label}  "
                      f"nSHD={np.mean(nshd_vec):.3f}  [{elapsed:.0f}s]")

    os.makedirs(STATE_DIR, exist_ok=True)
    pkl_path = os.path.join(STATE_DIR, f"es_{experiment_key}.pkl")
    bundle = {
        "results":        results,
        "graphs":         {g: G.tolist() for g, G in GRAPHS.items()},
        "labels":         LABELS,
        "x_values":       x_values,
        "x_key":          x_key,
        "experiment_key": experiment_key,
        "config":         {"B": B, "SEED": SEED},
    }
    if x_key == "sigma2":
        bundle["config"]["N"] = N_FIXED
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  saved: {pkl_path}")

def main():
    print(f"CPU workers: {N_WORKERS},  B = {B}")
    _run_sweep(N_VALUES,      "N",      "exp1_3node_samples")
    _run_sweep(SIGMA2_VALUES, "sigma2", "exp2_3node_edge")
    print("\nAll exhaustive-search experiments done.")
    print(f"To generate plots: python plot_all.py")


if __name__ == "__main__":
    main()