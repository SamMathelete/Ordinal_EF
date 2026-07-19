# Ordinal–Exponential Family Causal DAGs

Code accompanying *On the Identifiability of Mixed Ordinal and Exponential Family Causal DAGs*.

The experiments demonstrate, on finite samples, the population-level identifiability
results proved in the paper. They are not a contribution to structure learning, and
no procedure here discovers edges. **The skeleton is supplied throughout and only edge
orientations are estimated**, which is the setting the identifiability theorems address.

## Requirements

Python 3.10 or later, with

```
numpy
scipy
matplotlib
```

## Running

```bash
python run_es.py       # three-node experiment
python run_greedy.py   # d = 20 experiment
python plot_all.py     # figures, from whatever pickles are present
```

The two experiments are independent and can be run in either order, or singly;
`plot_all.py` draws whatever it finds in `state/`. Results are cached as
pickles there and figures written to `plots/` as matched `.png` and `.pdf`.
Both directories are gitignored. Since `plot_all.py` reads only the pickles,
figures can be restyled without recomputing anything.

Set `N_WORKERS` in `config.py` to your core count before launching. On 8 workers
the three-node experiment takes roughly 6 hours; the `d = 20` experiment is
considerably cheaper.

## What each experiment does

**Three-node (`run_es.py`).** Ground truths are the chain, the inverse fork and
the fork over `X1, X3` ordinal and `X2` exponential family. For each, the
estimator enumerates all four acyclic orientations of the skeleton
`X1 – X2 – X3` and returns the lowest-BIC candidate. Because enumeration is
exhaustive, any recovery error is finite-sample noise rather than search failure.
The chain and the fork are Markov equivalent, so recovering the correct one is
the empirical counterpart of identifiability *within* an equivalence class.

**`d = 20` (`run_greedy.py`).** One bipartite skeleton is drawn per density
regime, sparse (10 edges) and dense (90 edges), then held fixed. Each trial
starts from a uniformly drawn acyclic orientation of that skeleton and greedily
reverses the single edge that most reduces BIC until no reversal improves the
score. Trials are data replicates from one fixed model, not fresh graphs.

## Metric

`orientation_error` (written ρ in the paper) is the fraction of skeleton edges
assigned the wrong direction. Since every candidate carries the true skeleton,
there are no absent or spurious edges and this equals the structural Hamming
distance normalised by the edge count.

## Layout

| File | Role |
| --- | --- |
| `config.py` | Seeds, worker count, distribution families and hyperparameters, plot style |
| `utils.py` | Graph helpers, orientation enumeration, the ρ metric, parallel trial runner |
| `data_gen.py` | Ancestral sampling from an ordinal–exponential LPM |
| `score.py` | Per-node conditional MLE and the local BIC score |
| `es.py` | Exhaustive search over acyclic orientations (Algorithm 1) |
| `greedy_dag_search.py` | Greedy reversal-only orientation search (Algorithm 2) |
| `run_es.py` | Three-node experiment driver |
| `run_greedy.py` | `d = 20` experiment driver |
| `plotting.py`, `plot_all.py` | Figure generation |

## Configuration

Distributions are the seven regular one-parameter exponential families of
Table 1: Poisson, Exponential, Gaussian (σ² = 1), Gamma with fixed shape
(α = 2), Binomial (n = 5), Pascal (r = 3), and Gamma with fixed rate (β = 1).

Key settings, all in `config.py` unless noted:

| Setting | Value | Where |
| --- | --- | --- |
| `SEED` | 7 | `config.py` |
| `N_WORKERS` | 8 | `config.py` |
| `PLOT_LOG_FLOOR` | 2e-5 | `config.py` |
| `B` (three-node trials) | 10000 | `run_es.py` |
| `B` (`d = 20` trials) | 1000 | `run_greedy.py` |
| `N_VALUES` | 20 points, 5 to 500 | both drivers |
| `W_EDGE` | 0.6 | `run_es.py` |
| `S_ORD` | 4 | `run_es.py` |

`W_EDGE = 0.6` was chosen empirically. At 1.0 the exponential-family node
reaches values large enough to pin the downstream ordinal node in its top
category for several families, which makes that edge nearly uninformative; at
0.6 all seven families recover the correct orientation in every trial at
N = 500. Any nonzero weight is consistent with the theory, which requires only
`w ≠ 0`.

`PLOT_LOG_FLOOR` sits below `0.5 / B = 5e-5`, the smallest nonzero mean ρ
attainable in the three-node experiment, so a single misoriented trial stays
distinguishable from exact zero on the log axis.

## Reproducibility

Every random draw is seeded through `make_trial_seed` / `make_trial_rng`, which
hash a tuple of experiment identifiers together with the global `SEED`. Runs are
reproducible and independent of worker count and scheduling order. Ties in the
exhaustive search are broken by sorted candidate name.