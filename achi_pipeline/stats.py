"""Outline 6 (statistics) and 7 (descriptive summary).

Every test runs on the participant table (one row per participant), except the order
comparison, which uses the run table.

    H1        paired t-tests, stress vs non-stress MIST (RMSSD, SCL)
    check     paired t-tests, stress MIST vs baseline (manipulation check)
    SCR       paired t-test, stress vs non-stress MIST (secondary, uncorrected)
    H2        ANCOVA  delta_rmssd ~ group + baseline RMSSD (primary); unadjusted and
              baseline t-tests; binomial (logit) yes / silent ~ delta_rmssd + group for
              all words (primary), negative and positive words (secondary);
              linear nu ~ delta_rmssd + group (secondary)
    H3        linear nu and binomial FP (all words) ~ recovery time * group (exploratory)
    order     run 1 vs later runs on delta_rmssd (mixed model) and FP rate (binomial,
              participant-clustered SEs), adjusting for run type (secondary)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from . import config as C

RESULT_COLUMNS = ["test", "family", "model", "formula", "term", "scale", "n_rows", "n_participants",
                  "estimate", "se", "ci_low", "ci_high", "statistic", "p", "p_holm", "significant", "note"]


def _base(label, family, formula, term, scale, d):
    return {"test": label, "family": family, "formula": formula, "term": term, "scale": scale,
            "n_rows": len(d), "n_participants": d["participant_id"].nunique() if len(d) else 0}


def _vars(formula):
    return [c.strip() for c in formula.replace("~", "+").replace("*", "+").replace(":", "+").split("+")]


def paired(table, a, b, label, family):
    """Paired t-test of a vs b within participants. Estimate = mean(a - b)."""
    d = table[["participant_id", a, b]].dropna()
    out = _base(label, family, f"{a} - {b}", "mean difference", "units of the measure", d)
    if len(d) < 3:
        return {**out, "model": "not run", "note": "too few participants with both values"}
    res = sps.ttest_rel(d[a], d[b])
    ci = res.confidence_interval()
    diff = d[a] - d[b]
    return {**out, "model": "paired t-test", "estimate": diff.mean(), "se": diff.std(ddof=1) / np.sqrt(len(d)),
            "ci_low": ci.low, "ci_high": ci.high, "statistic": res.statistic, "p": res.pvalue}


def group_ttest(table, var, label, family):
    """Independent-samples t-test, patients minus controls."""
    d = table[["participant_id", "group", var]].dropna()
    pat, con = d.loc[d["group"] == "patient", var], d.loc[d["group"] == "control", var]
    out = _base(label, family, f"{var} by group", "patient - control", "units of the measure", d)
    if len(pat) < 2 or len(con) < 2:
        return {**out, "model": "not run", "note": "fewer than 2 participants in a group"}
    res = sps.ttest_ind(pat, con, equal_var=not C.WELCH_TTEST)
    ci = res.confidence_interval()
    se = (pat.mean() - con.mean()) / res.statistic if res.statistic else np.nan
    return {**out, "model": "Welch t-test" if C.WELCH_TTEST else "independent-samples t-test",
            "estimate": pat.mean() - con.mean(), "se": se, "ci_low": ci.low, "ci_high": ci.high,
            "statistic": res.statistic, "p": res.pvalue}


def _from_fit(out, res, term, model, note=""):
    if res.df_resid < 2:
        return {**out, "model": "not run", "note": f"too few observations ({out['n_rows']}) for "
                                                   f"{len(res.params)} model parameters"}
    if term not in res.params.index:
        return {**out, "model": model, "note": f"term {term} not in model: {list(res.params.index)}"}
    ci = res.conf_int().loc[term]
    return {**out, "model": model, "estimate": res.params[term], "se": res.bse[term], "ci_low": ci.iloc[0],
            "ci_high": ci.iloc[1], "statistic": res.tvalues[term], "p": res.pvalues[term], "note": note}


def linear(formula, data, term, label, family):
    """OLS regression (ANCOVA when a covariate is included)."""
    d = data.dropna(subset=[c for c in _vars(formula) if c in data.columns]).copy()
    out = _base(label, family, formula, term, "slope / adjusted difference", d)
    if len(d) < 4 or (d.get("patient") is not None and "patient" in formula and d["patient"].nunique() < 2):
        return {**out, "model": "not run", "note": "too few observations or only one group"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.ols(formula, d).fit()
    return _from_fit(out, res, term, "linear regression (OLS)")


def binomial(rhs, data, yes, n, term, label, family, cluster=False):
    """Logistic regression of yes-count out of n silent trials: logit(p) = rhs.

    Overdispersed fits (Pearson chi2 / df > OVERDISPERSION_LIMIT) get quasi-binomial
    SEs. With ``cluster`` (several rows per participant) SEs are participant-clustered.
    """
    formula = f"{yes} / {n} ~ {rhs}"
    d = data.dropna(subset=[c for c in _vars(rhs) if c in data.columns] + [yes, n]).copy()
    d = d[d[n] > 0]
    out = _base(label, family, formula, term, "log-odds", d)
    if len(d) < 4 or ("patient" in rhs and d["patient"].nunique() < 2):
        return {**out, "model": "not run", "note": "too few observations or only one group"}
    d["_p"] = d[yes] / d[n]
    model = smf.glm(f"_p ~ {rhs}", d, family=sm.families.Binomial(), var_weights=d[n])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit()
        disp = res.pearson_chi2 / res.df_resid if res.df_resid > 0 else np.nan
        note = f"dispersion {disp:.2f}"
        name = "binomial GLM (logit)"
        if cluster:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["participant_id"])[0]})
            name += ", participant-clustered SE"
        elif np.isfinite(disp) and disp > C.OVERDISPERSION_LIMIT:
            res = model.fit(scale="X2")
            name = "quasi-binomial GLM (logit)"
            note += "; overdispersed, SEs scaled"
    return _from_fit(out, res, term, name, note)


def mixed(formula, data, term, label, family):
    """Random intercept for participant; OLS with clustered SEs if it fails."""
    d = data.dropna(subset=[c for c in _vars(formula) if c in data.columns] + ["participant_id"]).copy()
    out = _base(label, family, formula, term, "units of the measure", d)
    if len(d) < 4 or d["participant_id"].nunique() < 3:
        return {**out, "model": "not run", "note": "too few observations"}
    try:
        if d.groupby("participant_id").size().max() < 2:
            raise RuntimeError("one row per participant")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = smf.mixedlm(formula, d, groups=d["participant_id"]).fit(reml=True)
            conv = [str(x.message) for x in w if "converge" in str(x.message).lower()
                    or "singular" in str(x.message).lower()]
        if conv:
            raise RuntimeError("; ".join(conv)[:200])
        return _from_fit(out, res, term, "linear mixed model (random intercept: participant)")
    except Exception as err:
        res = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["participant_id"])[0]})
        return _from_fit(out, res, term, "OLS, participant-clustered SE (mixed model fallback)", str(err)[:200])


def _patient(df):
    df = df.copy()
    df["patient"] = (df["group"] == "patient").astype(float)
    df.loc[df["group"].isna(), "patient"] = np.nan
    return df


def run_all(table: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    t = _patient(table)
    # order comparison: MIST runs of known type only; "first run" = the first MIST run
    r = runs[runs["run_type"].isin(["stress", "non-stress"])].copy()
    r["first_run"] = (r["run_number"] == r.groupby("participant_id")["run_number"].transform("min")).astype(int)
    r["stress_run"] = (r["run_type"] == "stress").astype(int)
    rec = "rmssd_recovery_time_s"

    res = [
        paired(t, "rmssd_stress", "rmssd_nonstress", "H1: RMSSD stress vs non-stress MIST (expect <0)", "primary"),
        paired(t, "scl_stress", "scl_nonstress", "H1: SCL stress vs non-stress MIST (expect >0)", "primary"),
        paired(t, "rmssd_stress", "rmssd_baseline", "Manipulation check: RMSSD stress MIST vs baseline (expect <0)",
               "manipulation check"),
        paired(t, "scl_stress", "scl_baseline", "Manipulation check: SCL stress MIST vs baseline (expect >0)",
               "manipulation check"),
        paired(t, "scr_freq_stress", "scr_freq_nonstress", "SCR frequency stress vs non-stress MIST", "secondary"),
        linear("delta_rmssd ~ patient + rmssd_baseline", t, "patient",
               "H2: dRMSSD patients vs controls, adjusted for baseline RMSSD (ANCOVA, expect <0)", "primary"),
        group_ttest(t, "delta_rmssd", "H2: dRMSSD patients vs controls, unadjusted", "secondary"),
        group_ttest(t, "rmssd_baseline", "H2: baseline RMSSD patients vs controls", "secondary"),
        binomial("delta_rmssd + patient", t, "n_yes_all", "n_silent_all", "delta_rmssd",
                 "H2: FP rate (all words) ~ dRMSSD + group (expect <0)", "primary"),
        binomial("delta_rmssd + patient", t, "n_yes_neg", "n_silent_neg", "delta_rmssd",
                 "H2: FP rate (negative words) ~ dRMSSD + group (expect <0)", "secondary"),
        binomial("delta_rmssd + patient", t, "n_yes_pos", "n_silent_pos", "delta_rmssd",
                 "H2: FP rate (positive words) ~ dRMSSD + group (expect <0)", "secondary"),
        linear("nu ~ delta_rmssd + patient", t, "delta_rmssd", "H2: nu ~ dRMSSD + group (expect <0)", "secondary"),
        linear(f"nu ~ {rec} * patient", t, rec, "H3: nu ~ recovery time (expect >0)", "exploratory"),
        linear(f"nu ~ {rec} * patient", t, f"{rec}:patient", "H3: nu, recovery x group (expect >0)", "exploratory"),
        binomial(f"{rec} * patient", t, "n_yes_all", "n_silent_all", rec,
                 "H3: FP rate (all words) ~ recovery time (expect >0)", "exploratory"),
        binomial(f"{rec} * patient", t, "n_yes_all", "n_silent_all", f"{rec}:patient",
                 "H3: FP rate (all words), recovery x group (expect >0)", "exploratory"),
    ]
    cov = " + stress_run" if r["run_type"].nunique() > 1 else ""
    res.append(mixed(f"delta_rmssd ~ first_run{cov}", r, "first_run", "Order: dRMSSD first MIST run vs later MIST runs", "secondary"))
    res.append(binomial(f"first_run{cov}", r, "n_yes_all", "n_silent_all", "first_run",
                        "Order: FP rate (all words) first MIST run vs later MIST runs", "secondary", cluster=True))

    out = pd.DataFrame(res).reindex(columns=RESULT_COLUMNS)
    if C.HOLM_CORRECTION:
        prim = (out["family"] == "primary") & out["p"].notna()
        if prim.any():
            out.loc[prim, "p_holm"] = multipletests(out.loc[prim, "p"], method="holm")[1]
        p_used = out["p_holm"].where(out["family"] == "primary", out["p"])
    else:
        p_used = out["p"]
    out["significant"] = p_used < C.ALPHA
    return out


def descriptives(table: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and n participants by group, for every measure in the participant table."""
    vars_ = [f"{m}_{s}" for m in ("rmssd", "scl", "scr_freq", "phasic_area") for s in ("baseline", "stress", "nonstress")] + \
            ["rmssd_post", "delta_rmssd_stress", "delta_scl_stress", "delta_rmssd", "delta_scl",
             "rmssd_recovery_time_s", "n_stress_runs_not_recovered", "fp_rate_neg", "fp_rate_pos", "fp_rate_neu", "fp_rate_all", "nu"]
    rows = []
    for g, d in table.groupby("group", dropna=False):
        for v in vars_:
            x = d[["participant_id", v]].dropna()
            rows.append({"group": g, "variable": v, "mean": x[v].mean(), "sd": x[v].std(ddof=1),
                         "n_participants": x["participant_id"].nunique()})
    return pd.DataFrame(rows)


def descriptives_scr(runs: pd.DataFrame) -> pd.DataFrame:
    """SCR frequency and phasic area per minute summarised by group and run type (outline: secondary, descriptive)."""
    rows = []
    for (g, rt), d in runs.groupby(["group", "run_type"], dropna=False):
        for v in ("scr_freq_baseline", "scr_freq_task", "phasic_area_baseline", "phasic_area_task"):
            x = d[["participant_id", v]].dropna()
            rows.append({"group": g, "run_type": rt, "variable": v, "mean": x[v].mean(), "sd": x[v].std(ddof=1),
                         "n_runs": len(x), "n_participants": x["participant_id"].nunique()})
    return pd.DataFrame(rows)


def supplementary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, d in table.groupby("group", dropna=False):
        ap = d["antipsychotic"].astype(str).str.lower().isin({"yes", "y", "1", "true"})
        rows.append({"group": g, "n": len(d), "age_mean": d["age"].mean(), "age_sd": d["age"].std(ddof=1),
                     "antipsychotic_yes_n": int(ap.sum()), "antipsychotic_yes_pct": 100 * ap.mean() if len(d) else np.nan})
    return pd.DataFrame(rows)
