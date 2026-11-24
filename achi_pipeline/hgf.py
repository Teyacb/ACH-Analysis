"""Outline 5.3: Hierarchical Gaussian Filter with a prior-weighting (nu) response model.

Perceptual model: 3-level binary HGF (Mathys et al., 2011, 2014), same update
equations as TAPAS ``tapas_hgf_binary``. Free: omega2. Fixed: kappa = 1, omega3 = -6,
mu2_0 = 0, sa2_0 = 1, mu3_0 = 1, sa3_0 = 1.

Response model (after the conditioned-hallucinations idea of Powers et al., 2017):
    logit P(yes_k) = zeta * [ nu * logit(muhat1_k) + (1 - nu) * sensory_k ]
    sensory_k      = logit(clip(intensity_k, c, 1 - c))      (silent -> logit(c))
nu in [0, 1] is the weight on the prior expectation relative to sensory evidence.

This is an APPROXIMATION of the published model and has not been validated against
TAPAS. Run :func:`parameter_recovery` before trusting nu, or supply nu_estimates.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from . import config as C

OM3 = -6.0
PRIOR = {"om2": (-3.0, 4.0), "logit_nu": (0.0, 2.0), "log_zeta": (0.0, 2.0)}  # (mean, sd)


def hgf_binary(u: np.ndarray, om2: float, resets: np.ndarray | None = None) -> np.ndarray | None:
    """Return muhat1 (prior P(u=1)) for each trial, or None if the parameters are invalid."""
    with np.errstate(all="ignore"):
        return _hgf_binary(u, om2, resets)


def _hgf_binary(u, om2, resets):
    n = len(u)
    muhat1 = np.empty(n)
    mu2, pi2, mu3, pi3 = 0.0, 1.0, 1.0, 1.0
    for k in range(n):
        if resets is not None and resets[k]:
            mu2, pi2, mu3, pi3 = 0.0, 1.0, 1.0, 1.0
        # predictions
        m1 = expit(mu2)
        if not 1e-12 < m1 < 1 - 1e-12 or not np.isfinite(mu3) or mu3 + om2 > 50:
            return None
        muhat1[k] = m1
        pihat1 = 1.0 / (m1 * (1 - m1))
        v2 = np.exp(mu3 + om2)
        pihat2 = 1.0 / (1.0 / pi2 + v2)
        pihat3 = 1.0 / (1.0 / pi3 + np.exp(OM3))
        # level 2 update
        da1 = u[k] - m1
        pi2_new = pihat2 + 1.0 / pihat1
        mu2_new = mu2 + da1 / pi2_new
        da2 = (1.0 / pi2_new + (mu2_new - mu2) ** 2) * pihat2 - 1.0
        # level 3 update
        w2 = v2 * pihat2
        pi3_new = pihat3 + 0.5 * w2 * (w2 + (2 * w2 - 1) * da2)
        if not np.isfinite(pi3_new) or pi3_new <= 0:
            return None
        mu3 = mu3 + 0.5 / pi3_new * w2 * da2
        mu2, pi2, pi3 = mu2_new, pi2_new, pi3_new
    return muhat1


def _unpack(x):
    return x[0], expit(x[1]), np.exp(x[2])


def neg_log_posterior(x, u, intensity, y, use, resets):
    om2, nu, zeta = _unpack(x)
    mh = hgf_binary(u, om2, resets)
    if mh is None:
        return 1e10
    c = C.HGF_SENSORY_CLIP
    prior_lo = logit(np.clip(mh, 1e-6, 1 - 1e-6))
    sens_lo = logit(np.clip(intensity, c, 1 - c))
    p = expit(zeta * (nu * prior_lo + (1 - nu) * sens_lo))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = np.sum((y * np.log(p) + (1 - y) * np.log(1 - p))[use])
    lp = sum(-0.5 * ((v - PRIOR[k][0]) / PRIOR[k][1]) ** 2 for k, v in zip(PRIOR, x))
    return -(ll + lp)


def fit_participant(trials: pd.DataFrame, run_order: list | None = None) -> dict:
    """Fit one participant. ``trials`` from io_utils.load_trials (all runs, session order)."""
    t = trials.copy()
    if run_order is not None:
        t["_ord"] = t["run_id"].map({r: i for i, r in enumerate(run_order)})
        within = ["block_order", "trial"] if "block_order" in t else ["trial"]   # blocks restart at trial 0
        t = t.sort_values(["_ord", *within], kind="mergesort")
    t = t[t["response_yes"].notna()]
    u = t[C.HGF_INPUT].to_numpy(dtype=float)
    intensity = t["intensity"].to_numpy(dtype=float)
    y = t["response_yes"].to_numpy(dtype=float)
    use = t["silent"].to_numpy() if C.HGF_LIKELIHOOD_TRIALS == "silent" else np.ones(len(t), bool)
    resets = (t["run_id"] != t["run_id"].shift()).to_numpy() if C.HGF_RESET_EACH_RUN else None
    if use.sum() < 10:
        return {"nu": np.nan, "om2": np.nan, "zeta": np.nan, "converged": False, "n_trials": int(use.sum())}

    rng = np.random.default_rng(C.HGF_SEED)
    best = None
    for i in range(C.HGF_N_STARTS):
        x0 = np.array([PRIOR[k][0] for k in PRIOR]) if i == 0 else \
            np.array([rng.normal(m, s / 2) for m, s in PRIOR.values()])
        res = minimize(neg_log_posterior, x0, args=(u, intensity, y, use, resets), method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    om2, nu, zeta = _unpack(best.x)
    return {"nu": float(nu), "om2": float(om2), "zeta": float(zeta), "neg_log_post": float(best.fun),
            "converged": bool(best.success), "n_trials": int(use.sum())}


def simulate(n_trials=200, om2=-3.0, nu=0.5, zeta=2.0, p_stim=0.5, seed=0) -> pd.DataFrame:
    """Simulate one participant (used for parameter recovery and tests)."""
    rng = np.random.default_rng(seed)
    # volatile contingency: probability of a word changes in blocks
    probs = np.repeat(rng.choice([0.2, 0.5, 0.8], size=n_trials // 25 + 1), 25)[:n_trials]
    u = (rng.random(n_trials) < probs).astype(int)
    intensity = u * rng.choice([0.25, 0.5, 0.75], size=n_trials)
    mh = hgf_binary(u.astype(float), om2)
    c = C.HGF_SENSORY_CLIP
    p = expit(zeta * (nu * logit(np.clip(mh, 1e-6, 1 - 1e-6)) + (1 - nu) * logit(np.clip(intensity, c, 1 - c))))
    y = (rng.random(n_trials) < p).astype(int)
    return pd.DataFrame({"run_id": "run-1", "trial": np.arange(n_trials), "stimulus_present": u,
                         "intensity": intensity, "valence": rng.choice(["negative", "positive"], n_trials),
                         "response_yes": y, "silent": u == 0})


def parameter_recovery(nus=(0.2, 0.4, 0.6, 0.8), n_sims=5, n_trials=200) -> pd.DataFrame:
    """Simulate with known nu, refit, and report the true-vs-recovered correlation."""
    rows = []
    for nu in nus:
        for s in range(n_sims):
            d = simulate(n_trials=n_trials, nu=nu, seed=1000 * s + int(nu * 100))
            rows.append({"true_nu": nu, **fit_participant(d)})
    df = pd.DataFrame(rows)
    print("true vs recovered nu: r =", round(df[["true_nu", "nu"]].corr().iloc[0, 1], 3))
    return df
