import os
import pickle
import time

import numpy as np

from config import (
    SEED, N_WORKERS, STATE_DIR, ALL_EXPFAM_DISTS, DIST_HYPERPARAMS,
)
from utils import (
    generate_ordinal_cutpoints, make_trial_seed, run_trials, orientation_error,
)
from data_gen import generate_data
from es import exhaustive_search

B = 10000
N_VALUES = sorted(set(np.round(
    np.linspace(5, 500, 20)
).astype(int).tolist()))
S_ORD = 4          

W_EDGE = 0.6

SKELETON = np.array([[0., 1., 0.],
                     [1., 0., 1.],
                     [0., 1., 0.]])

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

def _trial_binary(W_struct, node_info, gammas, skeleton, N, trial_seed):
    X = generate_data(W=W_struct * W_EDGE, N=N, node_info=node_info,
                      ordinal_gammas=gammas, seed=trial_seed)
    result = exhaustive_search(X, node_info, skeleton)
    return float(orientation_error(result.best_W, W_struct))


def _run_sweep(x_values: list, experiment_key: str):
    print(f"\n=== {experiment_key} ===")
    skeleton = SKELETON

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
                args = [(W_struct, node_info, gammas, skeleton, x, seed_base + b)
                        for b in range(B)]
                rho_vec = run_trials(_trial_binary, args, N_WORKERS)
                results[truth_name][y_dist][x] = rho_vec

                elapsed = time.time() - t0
                print(f"    {truth_name} {y_dist:>12s}  N={x:>5d}  "
                      f"rho={np.mean(rho_vec):.4f}  [{elapsed:.0f}s]")

    os.makedirs(STATE_DIR, exist_ok=True)
    pkl_path = os.path.join(STATE_DIR, f"es_{experiment_key}.pkl")
    bundle = {
        "results":        results,
        "graphs":         {g: G.tolist() for g, G in GRAPHS.items()},
        "labels":         LABELS,
        "x_values":       x_values,
        "x_key":          "N",
        "experiment_key": experiment_key,
        "config":         {"B": B, "SEED": SEED, "W_EDGE": W_EDGE, "S_ORD": S_ORD},
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  saved: {pkl_path}")

def main():
    print(f"CPU workers: {N_WORKERS},  B = {B}")
    _run_sweep(N_VALUES, "exp1_3node_samples")
    print("\nAll exhaustive-search experiments done.")
    print(f"To generate plots: python plot_all.py")


if __name__ == "__main__":
    main()