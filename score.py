from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, expit, digamma

_EPS = 1e-12

def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)

def _linear_predictor(X: np.ndarray, parents: Sequence[int], w: np.ndarray) -> np.ndarray:
    if len(parents) == 0:
        return np.zeros(X.shape[0], dtype=np.float64)
    return X[:, list(parents)] @ w

def _poisson_nll_grad(w, X, parents, x_target, **_):
    N = x_target.shape[0]
    kappa = _linear_predictor(X, parents, w)
    lam = np.exp(kappa)
    nll = float(np.mean(-x_target * kappa + lam))
    if len(parents) == 0:
        return nll, np.zeros(0)
    grad = X[:, list(parents)].T @ (lam - x_target) / N
    return nll, grad

def _exponential_nll_grad(w, X, parents, x_target, **_):
    N = x_target.shape[0]
    kappa = _linear_predictor(X, parents, w)
    lam = np.exp(kappa)
    nll = float(np.mean(-kappa + x_target * lam))
    if len(parents) == 0:
        return nll, np.zeros(0)
    grad = X[:, list(parents)].T @ (x_target * lam - 1.0) / N
    return nll, grad

def _gaussian_nll_grad(w, X, parents, x_target, sigma2=1.0, **_):
    N = x_target.shape[0]
    mu = _linear_predictor(X, parents, w)
    resid = x_target - mu
    const = 0.5 * math.log(2.0 * math.pi * sigma2)
    nll = float(np.mean(0.5 * resid * resid / sigma2 + const))
    if len(parents) == 0:
        return nll, np.zeros(0)
    grad = -X[:, list(parents)].T @ resid / (sigma2 * N)
    return nll, grad

def _gamma_shape_nll_grad(w, X, parents, x_target, alpha=2.0, **_):
    N = x_target.shape[0]
    kappa = _linear_predictor(X, parents, w)
    beta = np.exp(kappa)
    log_x = np.log(x_target + _EPS)
    nll = float(np.mean(
        -alpha * kappa - (alpha - 1.0) * log_x + beta * x_target + gammaln(alpha)
    ))
    if len(parents) == 0:
        return nll, np.zeros(0)
    grad = X[:, list(parents)].T @ (beta * x_target - alpha) / N
    return nll, grad

def _binomial_nll_grad(w, X, parents, x_target, n_trials=10, **_):
    N = x_target.shape[0]
    kappa = _linear_predictor(X, parents, w)
    log_binom = (gammaln(n_trials + 1.0)
                 - gammaln(x_target + 1.0)
                 - gammaln(n_trials - x_target + 1.0))
    nll = float(np.mean(-x_target * kappa + n_trials * _softplus(kappa) - log_binom))
    if len(parents) == 0:
        return nll, np.zeros(0)
    p = expit(kappa)
    grad = X[:, list(parents)].T @ (n_trials * p - x_target) / N
    return nll, grad

def _pascal_nll_grad(w, X, parents, x_target, r=3, **_):
    N = x_target.shape[0]
    kappa = _linear_predictor(X, parents, w)
    log_binom = (gammaln(x_target + r) - gammaln(x_target + 1.0) - gammaln(float(r)))
    nll = float(np.mean(
        r * _softplus(-kappa) + x_target * _softplus(kappa) - log_binom
    ))
    if len(parents) == 0:
        return nll, np.zeros(0)
    p = expit(kappa)
    grad = X[:, list(parents)].T @ ((r + x_target) * p - r) / N
    return nll, grad

def _gamma_rate_nll_grad(w, X, parents, x_target, beta=1.0, **_):
    N = x_target.shape[0]
    kappa = _linear_predictor(X, parents, w)
    alpha = np.exp(kappa) + _EPS
    log_x = np.log(x_target + _EPS)
    log_beta = math.log(beta)
    nll = float(np.mean(
        -alpha * log_beta - (alpha - 1.0) * log_x + beta * x_target + gammaln(alpha)
    ))
    if len(parents) == 0:
        return nll, np.zeros(0)
    coeff = alpha * (-log_beta - log_x + digamma(alpha))
    grad = X[:, list(parents)].T @ coeff / N
    return nll, grad

_EXPFAM_NLLS = {
    "poisson":     _poisson_nll_grad,
    "exponential": _exponential_nll_grad,
    "gaussian":    _gaussian_nll_grad,
    "gamma_shape": _gamma_shape_nll_grad,
    "binomial":    _binomial_nll_grad,
    "pascal":      _pascal_nll_grad,
    "gamma_rate":  _gamma_rate_nll_grad,
}

_DISTS_NEEDING_BOUNDS = {"poisson", "exponential", "gamma_shape", "gamma_rate"}

_KAPPA_MAX = 50.0
_BOUND_QUANTILE = 0.99   


def _compute_w_bounds(X, parents, dist):
    if dist not in _DISTS_NEEDING_BOUNDS or len(parents) == 0:
        return None
    Xp = X[:, list(parents)]
    row_L1 = np.abs(Xp).sum(axis=1)
    row_L1_q = float(np.quantile(row_L1, _BOUND_QUANTILE))
    if row_L1_q < 1e-12:
        return None
    b = _KAPPA_MAX / row_L1_q
    return [(-b, b)] * len(parents)


def _fit_expfam_node(dist, X, parents, x_target, hyperparams, *, max_iter, ftol):
    nll_grad_fn = _EXPFAM_NLLS[dist]
    if len(parents) == 0:
        nll, _ = nll_grad_fn(np.zeros(0), X, parents, x_target, **hyperparams)
        return nll, np.zeros(0)
    w0 = np.zeros(len(parents), dtype=np.float64)
    bounds = _compute_w_bounds(X, parents, dist)
    res = minimize(
        lambda w: nll_grad_fn(w, X, parents, x_target, **hyperparams),
        w0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max_iter, "ftol": ftol, "gtol": 1e-7},
    )
    return float(res.fun), res.x

def _gamma_from_alpha(alpha: np.ndarray) -> np.ndarray:
    inc = np.empty_like(alpha)
    inc[0] = alpha[0]
    if alpha.shape[0] > 1:
        inc[1:] = _softplus(alpha[1:])
    return np.cumsum(inc)


def _ordinal_nll_grad_full(theta, X, parents, x_target, S, P_w):
    w = theta[:P_w]
    alpha = theta[P_w:]
    N = x_target.shape[0]

    gamma_interior = _gamma_from_alpha(alpha)
    kappa = _linear_predictor(X, parents, w)
    s_int = x_target.astype(np.int64)

    upper_is_inf = (s_int == S)
    lower_is_neg_inf = (s_int == 1)

    g_upper = np.where(upper_is_inf, 0.0, gamma_interior[np.clip(s_int - 1, 0, S - 2)])
    g_lower = np.where(lower_is_neg_inf, 0.0, gamma_interior[np.clip(s_int - 2, 0, S - 2)])
    a_upper = np.where(upper_is_inf, 1.0, expit(g_upper - kappa))
    a_lower = np.where(lower_is_neg_inf, 0.0, expit(g_lower - kappa))

    P = np.clip(a_upper - a_lower, _EPS, None)
    nll = float(-np.mean(np.log(P)))

    du = np.where(upper_is_inf, 0.0, a_upper * (1.0 - a_upper))
    dl = np.where(lower_is_neg_inf, 0.0, a_lower * (1.0 - a_lower))
    inv_P = 1.0 / P

    if P_w > 0:
        coeff_w = (du - dl) * inv_P
        grad_w = X[:, list(parents)].T @ coeff_w / N
    else:
        grad_w = np.zeros(0)

    grad_gamma = np.zeros(S - 1)
    upper_valid = ~upper_is_inf
    if upper_valid.any():
        np.add.at(grad_gamma, (s_int - 1)[upper_valid],
                  -du[upper_valid] * inv_P[upper_valid] / N)
    lower_valid = ~lower_is_neg_inf
    if lower_valid.any():
        np.add.at(grad_gamma, (s_int - 2)[lower_valid],
                  dl[lower_valid] * inv_P[lower_valid] / N)

    suffix_sum = np.cumsum(grad_gamma[::-1])[::-1]
    grad_alpha = np.empty(S - 1)
    grad_alpha[0] = suffix_sum[0]
    if S - 1 > 1:
        grad_alpha[1:] = expit(alpha[1:]) * suffix_sum[1:]

    return nll, np.concatenate([grad_w, grad_alpha])


def _init_alpha_empirical(x_target: np.ndarray, S: int) -> np.ndarray:
    counts = np.bincount(x_target.astype(np.int64), minlength=S + 1)[1:]
    freq = np.clip(counts.astype(np.float64) / max(counts.sum(), 1), 1e-3, 1 - 1e-3)
    freq = freq / freq.sum()
    cum = np.cumsum(freq)
    gamma_init = np.log(cum[:-1] / (1.0 - cum[:-1]))
    alpha = np.empty(S - 1)
    alpha[0] = gamma_init[0]
    if S - 1 > 1:
        diffs = np.clip(np.diff(gamma_init), 1e-6, None)
        alpha[1:] = np.log(np.expm1(diffs))
    return alpha


def _fit_ordinal_node(X, parents, x_target, S, *, max_iter, ftol):
    P_w = len(parents)
    alpha0 = _init_alpha_empirical(x_target, S)
    theta0 = np.concatenate([np.zeros(P_w), alpha0])
    res = minimize(
        lambda theta: _ordinal_nll_grad_full(theta, X, parents, x_target, S, P_w),
        theta0, jac=True, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": ftol, "gtol": 1e-7},
    )
    return float(res.fun), res.x[:P_w], res.x[P_w:]

@dataclass(frozen=True)
class LocalScoreResult:
    score: float          
    nll: float
    n_params: int
    weights: np.ndarray
    extras: dict

def local_score(
    node_idx: int,
    parents: Sequence[int],
    X: np.ndarray,
    node_info: dict,
    *,
    max_iter: int = 200,
    ftol: float = 1e-8,
) -> LocalScoreResult:
    info = node_info[node_idx]
    dist = info["dist"]
    x_target = X[:, node_idx]
    N = X.shape[0]
    parents = tuple(int(p) for p in parents)
    penalty = 0.5 * math.log(N)

    if dist == "ordinal":
        S = int(info["S"])
        nll, w, alpha = _fit_ordinal_node(X, parents, x_target, S,
                                          max_iter=max_iter, ftol=ftol)
        n_params = len(parents) + (S - 1)
        return LocalScoreResult(
            score=N * nll + penalty * n_params,
            nll=nll, n_params=n_params, weights=w,
            extras={"alpha": alpha, "S": S},
        )

    if dist not in _EXPFAM_NLLS:
        raise ValueError(f"Unknown distribution '{dist}' at node {node_idx}")

    hyperparams = {k: v for k, v in info.items() if k != "dist"}
    nll, w = _fit_expfam_node(dist, X, parents, x_target, hyperparams,
                               max_iter=max_iter, ftol=ftol)
    n_params = len(parents)
    return LocalScoreResult(
        score=N * nll + penalty * n_params,
        nll=nll, n_params=n_params, weights=w, extras={},
    )