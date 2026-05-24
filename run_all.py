import argparse
import os
import time

from config import STATE_DIR, PLOTS_DIR


def _section(t0: float, title: str, fn):
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}", flush=True)
    t = time.time()
    fn()
    print(f"\n[{title} done in {time.time() - t:.0f}s, total {time.time() - t0:.0f}s]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-es",     action="store_true")
    parser.add_argument("--skip-greedy", action="store_true")
    parser.add_argument("--skip-plots",  action="store_true")
    parser.add_argument("--plots-only",  action="store_true")
    args = parser.parse_args()

    if args.plots_only:
        args.skip_es = True
        args.skip_greedy = True
        args.skip_plots = False

    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    t0 = time.time()

    if not args.skip_es:
        import run_es
        _section(t0, "PART A: Exhaustive search (2-node, 3-node)", run_es.main)

    if not args.skip_greedy:
        import run_greedy
        _section(t0, "PART B: Greedy DAG search (d-node)", run_greedy.main)

    if not args.skip_plots:
        import plot_all
        _section(t0, "PART C: Regenerate plots", plot_all.main)

    print(f"\nTotal wall time: {time.time() - t0:.0f}s "
          f"({(time.time() - t0)/60:.1f} min).")


if __name__ == "__main__":
    main()