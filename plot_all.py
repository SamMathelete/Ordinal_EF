import glob
import os
import pickle

import numpy as np

from config import (
    STATE_DIR, DIST_STYLE, DIST_MODES, apply_plot_style,
)
from plotting import plot_curves

_X_LABELS = {
    "N":      r"number of samples ($N$)",
    "sigma2": r"edge strength  ($\sigma^2$)",
}


def _es_series(results_truth: dict, dist_label_prefix: str) -> list[dict]:
    series = []
    for y_dist, by_x in results_truth.items():
        xs = sorted(by_x)
        means = np.array([np.mean(by_x[x]) for x in xs])
        style = DIST_STYLE.get(y_dist, {"color": "#333", "marker": "o", "label": y_dist})
        series.append({
            "xs":     xs,
            "means":  means,
            "color":  style["color"],
            "marker": style["marker"],
            "label":  f"{dist_label_prefix} ~ {style['label']}",
        })
    return series


def _plot_es_file(pkl_path: str):
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    results = bundle["results"]
    labels = bundle["labels"]
    x_key = bundle["x_key"]
    key = bundle["experiment_key"]

    dist_label_prefix = "$X_2$"
    xlabel = _X_LABELS.get(x_key, x_key)

    print(f"\n[{key}]")
    for truth_name, label_truth in labels.items():
        series = _es_series(results[truth_name], dist_label_prefix)
        plot_curves(
            series,
            title=f"Truth: {label_truth}",
            xlabel=xlabel, ylabel=r"average nSHD",
            out_basename=f"es_{key}_{truth_name}_linear",
            log_x=False, log_y=False,
        )
        plot_curves(
            series,
            title=f"Truth: {label_truth}  ",
            xlabel=xlabel, ylabel=r"average nSHD",
            out_basename=f"es_{key}_{truth_name}",
            log_x=False, log_y=True,
        )

_GREEDY_METRICS = [
    ("nshd", "nSHD", r"average nSHD"),
    ("fnr",  "FNR",  r"False Negative Rate"),
    ("fpr",  "FPR",  r"False Positive Rate"),
]

_DENSITY_LABELS = {
    "ER-2": "Sparse",
    "ER-4": "Dense",
}


def _greedy_series(results_density: dict) -> list[dict]:
    series = []
    for mode, meta in DIST_MODES.items():
        if mode not in results_density:
            continue
        by_N = results_density[mode]
        Ns = sorted(by_N)
        series.append({
            "xs":     Ns,
            "means":  np.array([np.mean(by_N[n]) for n in Ns]),
            "color":  meta["color"],
            "marker": meta["marker"],
            "label":  meta["label"],
        })
    return series


def _plot_greedy_file(pkl_path: str):
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    dags = bundle["dags"]
    d = bundle.get("config", {}).get("d", "?")

    print(f"\n[greedy_results]")
    for density in bundle["nshd"]:
        n_edges = dags[density]["n_edges"]
        density_label = _DENSITY_LABELS.get(density, density)
        for key, name, ylabel in _GREEDY_METRICS:
            plot_curves(
                _greedy_series(bundle[key][density]),
                title=f"{density_label} Bipartite DAG  "
                      f"($d$={d}, edges = {n_edges})",
                xlabel=r"number of samples ($N$)", ylabel=ylabel,
                out_basename=f"greedy_{density}_{key}",
                log_x=False, legend_ncol=3,
            )

def main():
    apply_plot_style()
    es_paths = sorted(glob.glob(os.path.join(STATE_DIR, "es_*.pkl")))
    greedy_paths = sorted(glob.glob(os.path.join(STATE_DIR, "greedy_results.pkl")))
    if not es_paths and not greedy_paths:
        print(f"  (no state pickles found in {STATE_DIR}/)")
        return
    for p in es_paths:
        _plot_es_file(p)
    for p in greedy_paths:
        _plot_greedy_file(p)
    print("\nAll plots regenerated.")


if __name__ == "__main__":
    main()