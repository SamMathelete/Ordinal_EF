import numpy as np
from scipy.special import expit
from utils import topo_order

def generate_data(
        W: np.ndarray,
        N: int,
        node_info: dict,
        ordinal_gammas: dict,
        seed: int
) -> np.ndarray:
    d = W.shape[0]
    order = topo_order(W)
    if order is None:
        raise ValueError("W is not acyclic")
    rng = np.random.default_rng(seed = seed)
    X = np.zeros((N, d), dtype = np.float64)
    for i in order:
        info = node_info[i]
        dist = info["dist"]
        parents = np.nonzero(W[i])[0]
        if parents.size > 0:
            kappa = X[:, parents] @ W[i, parents]
        else:
            kappa = np.zeros(N)
        if dist == "poisson":
            X[:, i] = rng.poisson(np.exp(kappa))
        elif dist == "exponential":
            X[:, i] = rng.exponential(scale = 1.0 / np.exp(kappa))
        elif dist == "gaussian":
            sigma2 = info.get("sigma2", 1.0)
            X[:, i] = rng.normal(loc = kappa, scale = np.sqrt(sigma2))
        elif dist == "gamma_shape":
            alpha = info.get("alpha", 2.0)
            beta = np.exp(kappa)
            X[:, i] = rng.gamma(shape = alpha, scale = 1.0 / beta)
        elif dist == "binomial":
            n_trials = info.get("n_trials", 10)
            X[:, i] = rng.binomial(n = n_trials, p = expit(kappa))
        elif dist == "pascal":
            r = info.get("r", 3)
            p_success = np.clip(expit(kappa), 1e-8, 1.0 - 1e-8)
            X[:, i] = rng.negative_binomial(n = r, p = p_success)
        elif dist == "gamma_rate":
            beta = info.get("beta", 1.0)
            alpha_param = np.clip(np.exp(kappa), 1e-4, None)
            X[:, i] = rng.gamma(shape = alpha_param, scale = 1.0/ beta)
        elif dist == "ordinal":
            S = int(info["S"])
            gamma_interior = ordinal_gammas[i][1:S]
            cdf = expit(gamma_interior[:, None] - kappa[None, :])
            u = rng.uniform(size = N)
            X[:, i] = (1 + (u[None, :] > cdf).sum(axis = 0)).astype(np.float64)
        else:
            raise ValueError(f"Unknown distribution '{dist}' at node {i}")
    return X
