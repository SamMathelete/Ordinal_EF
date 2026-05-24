import os
from typing import Sequence
import numpy as np

from config import PLOTS_DIR, PLOT_LOG_FLOOR


def plot_curves(
    series: Sequence[dict],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_basename: str,
    log_x: bool = False,
    log_y: bool = True,
    legend_ncol: int = 3,
    legend_loc: str = "upper right",
    figsize: tuple = (6.5, 5.5),
):
    import matplotlib.pyplot as plt
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    for s in series:
        means = np.asarray(s["means"], dtype=float)
        y = np.clip(means, PLOT_LOG_FLOOR, None) if log_y else means
        ax.plot(
            s["xs"], y, "-",
            color=s["color"],
            linewidth=2.8,
            marker=s["marker"],
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=s["label"],
            zorder=3,
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.legend(
        ncol=legend_ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.6,
        frameon=True,
    )
    ax.grid(True, which="major", color="#BDC3C7", linewidth=0.6, zorder=0)
    ax.grid(True, which="minor", color="#D5DBDB", linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")

    plt.tight_layout()
    out_base = os.path.join(PLOTS_DIR, out_basename)
    fig.savefig(out_base + ".png")
    fig.savefig(out_base + ".pdf")
    plt.close(fig)
    print(f"  saved: {out_base}.png/pdf")