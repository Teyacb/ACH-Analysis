"""Outline 5 (measures, change scores, recovery, false-positive rates).

Measures are first calculated per run (run table), then combined into one row per
participant (participant table): stress MIST = the task segment of stress runs,
non-stress MIST = the task segment of non-stress runs, runs of the same type averaged.
The session has ONE baseline (RestBaseline); every run's change scores and recovery
time use it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .eda import find_scrs

SEGMENTS = ("baseline", "task", "post")
EDA_SEGMENTS = ("baseline", "task")


def valid_rr(peaks: np.ndarray, fs: float, bad_spans: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """RR intervals whose both beats lie in [a, b] and that don't cross a bad span."""
    p = np.sort(np.asarray(peaks))
    p = p[(p / fs >= a) & (p / fs <= b)]
    if len(p) < 2:
        return pd.DataFrame(columns=["t0", "t1", "rr_ms", "ok"])
    t0, t1 = p[:-1] / fs, p[1:] / fs
    ok = np.ones(len(t0), bool)
    for _, s in bad_spans.iterrows():
        ok &= ~((t0 < s["end_s"]) & (t1 > s["start_s"]))
    return pd.DataFrame({"t0": t0, "t1": t1, "rr_ms": (t1 - t0) * 1000, "ok": ok})


def rmssd(rr: pd.DataFrame) -> float:
    """Successive differences only between adjacent, valid intervals."""
    if len(rr) < 3:
        return np.nan
    x = rr["rr_ms"].to_numpy()
    ok = rr["ok"].to_numpy()
    adjacent = np.isclose(rr["t1"].to_numpy()[:-1], rr["t0"].to_numpy()[1:])
    use = ok[:-1] & ok[1:] & adjacent
    if use.sum() < 2:
        return np.nan
    d = np.diff(x)[use]
    return float(np.sqrt(np.mean(d ** 2)))


def rolling_rmssd(rr: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    w, step = C.RECOVERY_WINDOW_S, C.RECOVERY_STEP_S
    rows = []
    s = a
    while s + w <= b + 1e-9:
        sub = rr[(rr["t0"] >= s) & (rr["t1"] <= s + w)]
        rows.append({"win_start": s, "win_end": s + w, "rmssd": rmssd(sub)})
        s += step
    return pd.DataFrame(rows, columns=["win_start", "win_end", "rmssd"])


def recovery_time(rr_base, base_ab, rr_post, post_ab) -> tuple[float, bool, float]:
    """Seconds from post-task start until rolling RMSSD is back at baseline RMSSD.

    Baseline RMSSD = mean of the baseline's rolling windows (minus RECOVERY_SD_BELOW SDs,
    0 by default). Must hold for RECOVERY_SUSTAIN_WINDOWS consecutive windows. Returns
    (time_s or NaN, reached, threshold).
    """
    base = rolling_rmssd(rr_base, *base_ab)["rmssd"].dropna()
    if len(base) < 2:
        return np.nan, False, np.nan
    thr = base.mean() - C.RECOVERY_SD_BELOW * base.std(ddof=1)
    post = rolling_rmssd(rr_post, *post_ab)
    above = (post["rmssd"] >= thr).to_numpy()
    k = C.RECOVERY_SUSTAIN_WINDOWS
    for i in range(len(above) - k + 1):
        if above[i:i + k].all():
            return float(post["win_end"].iloc[i] - post_ab[0]), True, float(thr)
    return np.nan, False, float(thr)


def eda_bad_mask(eda: pd.DataFrame, ignore_spans: pd.DataFrame | None = None) -> np.ndarray:
    """Artefact samples, minus spans the reviewer marked not_artifact."""
    t = eda["time_s"].to_numpy()
    bad = eda["artifact"].to_numpy().astype(bool).copy()
    if ignore_spans is not None:
        for _, s in ignore_spans.iterrows():
            bad[(t >= s["time_s"]) & (t <= s["end_s"])] = False
    return bad


def mean_scl(eda: pd.DataFrame, a: float, b: float, bad: np.ndarray) -> float:
    t = eda["time_s"].to_numpy()
    use = (t >= a) & (t <= b) & ~bad
    return float(eda["scl"].to_numpy()[use].mean()) if use.any() else np.nan


def scr_frequency(scrs: pd.DataFrame, eda: pd.DataFrame, eda_fs: float, a: float, b: float,
                  bad: np.ndarray) -> tuple[int, float, float]:
    """(n SCRs, artefact-free minutes, SCRs per artefact-free minute) in [a, b].

    An SCR counts if its peak is in the segment and neither its onset nor peak is flagged.
    """
    t = eda["time_s"].to_numpy()
    clean_min = float(((t >= a) & (t <= b) & ~bad).sum()) / eda_fs / 60
    if len(scrs):
        inside = (scrs["peak_s"] >= a) & (scrs["peak_s"] <= b)
        i_on = np.clip(np.round(scrs["onset_s"] * eda_fs).astype(int), 0, len(bad) - 1)
        i_pk = np.clip(np.round(scrs["peak_s"] * eda_fs).astype(int), 0, len(bad) - 1)
        n = int((inside & ~bad[i_on] & ~bad[i_pk]).sum())
    else:
        n = 0
    return n, clean_min, (n / clean_min if clean_min > 0 else np.nan)


def phasic_area_per_min(eda: pd.DataFrame, eda_fs: float, a: float, b: float, bad: np.ndarray) -> float:
    """Area of the phasic component above zero (µS·s) per artefact-free minute in [a, b].

    Does not depend on SCR detection, so overlapping SCRs that merge into one count still
    add their full area. Only the part above zero is summed: the 0.05 Hz high-pass split
    makes the phasic signal swing below zero, and a plain mean would cancel out.
    """
    t = eda["time_s"].to_numpy()
    use = (t >= a) & (t <= b) & ~bad
    clean_min = float(use.sum()) / eda_fs / 60
    if clean_min <= 0:
        return np.nan
    area = float(np.clip(eda["scr"].to_numpy()[use], 0, None).sum()) / eda_fs
    return area / clean_min


def run_rows(pid, runs, segments, qc, ecg=None, eda=None) -> pd.DataFrame:
    """One row per run: RMSSD per segment, SCL, SCR frequency and phasic area per minute (baseline, task),
    task - baseline change scores, recovery time and QC columns.

    The baseline segment is the session's single baseline, repeated on every run row.
    ecg: {"peaks", "bad_spans", "fs"} or None.  eda: {"eda", "fs", "ignore"} or None.
    """
    if eda is not None:
        bad = eda_bad_mask(eda["eda"], eda["ignore"])
        scrs = find_scrs(eda["eda"]["eda_clean"].to_numpy(), eda["fs"])
    segments = segments.assign(run_id=segments["run_id"].astype(str))
    qc = qc.assign(run_id=qc["run_id"].astype(str))
    base_seg = segments[segments["segment"] == "baseline"].head(1)
    base_q = qc[qc["segment"] == "baseline"].head(1)
    rows = []
    for _, run in runs.iterrows():
        rid = str(run["run_id"])
        r = {"participant_id": pid, **{k: run[k] for k in run.index}}
        segs = pd.concat([base_seg, segments[(segments["run_id"] == rid) & (segments["segment"] != "baseline")]])
        segs = segs.set_index("segment")
        q = pd.concat([base_q, qc[(qc["run_id"] == rid) & (qc["segment"] != "baseline")]]).set_index("segment")
        rr_by = {}
        for seg in SEGMENTS:
            have = seg in segs.index
            a, b = (float(segs.loc[seg, "start_s"]), float(segs.loc[seg, "end_s"])) if have else (np.nan, np.nan)
            ecg_keep = have and seg in q.index and bool(q.loc[seg, "ecg_keep"])
            r[f"duration_s_{seg}"] = b - a
            r[f"pct_ecg_bad_{seg}"] = q.loc[seg, "pct_ecg_bad"] if seg in q.index else np.nan
            r[f"ecg_keep_{seg}"] = ecg_keep
            r[f"rmssd_{seg}"] = np.nan
            if have and ecg is not None:
                rr = valid_rr(ecg["peaks"], ecg["fs"], ecg["bad_spans"], a, b)
                rr_by[seg] = (rr, (a, b))
                if ecg_keep:
                    r[f"rmssd_{seg}"] = rmssd(rr)
            if seg not in EDA_SEGMENTS:
                continue
            eda_keep = have and seg in q.index and bool(q.loc[seg, "eda_keep"])
            r[f"pct_eda_removed_{seg}"] = q.loc[seg, "pct_eda_removed"] if seg in q.index else np.nan
            r[f"eda_keep_{seg}"] = eda_keep
            r[f"scl_{seg}"] = r[f"scr_freq_{seg}"] = r[f"n_scr_{seg}"] = r[f"eda_clean_min_{seg}"] = np.nan
            r[f"phasic_area_{seg}"] = np.nan
            if eda_keep and eda is not None:
                r[f"scl_{seg}"] = mean_scl(eda["eda"], a, b, bad)
                n, minutes, freq = scr_frequency(scrs, eda["eda"], eda["fs"], a, b, bad)
                r[f"n_scr_{seg}"], r[f"eda_clean_min_{seg}"], r[f"scr_freq_{seg}"] = n, minutes, freq
                r[f"phasic_area_{seg}"] = phasic_area_per_min(eda["eda"], eda["fs"], a, b, bad)
        r["delta_rmssd"] = r["rmssd_task"] - r["rmssd_baseline"]
        r["delta_scl"] = r["scl_task"] - r["scl_baseline"]
        if r["ecg_keep_baseline"] and r["ecg_keep_post"] and "post" in rr_by and "baseline" in rr_by:
            t, reached, thr = recovery_time(*rr_by["baseline"], *rr_by["post"])
        else:
            t, reached, thr = np.nan, False, np.nan
        r["rmssd_recovery_time_s"] = t
        r["recovery_reached"] = reached
        r["recovery_threshold_ms"] = thr
        rows.append(r)
    return pd.DataFrame(rows)


def _pct_excluded(runs: pd.DataFrame, signal: str, segments) -> float:
    """% of the analysed time of a signal that was not used: excluded segments count in
    full, kept segments count their unusable share."""
    total = lost = 0.0
    for seg in segments:
        # the single session baseline is repeated on every run row: count it once
        rows = runs.head(1) if seg == "baseline" else runs
        dur = rows[f"duration_s_{seg}"].to_numpy(dtype=float)
        pct_col = "pct_ecg_bad" if signal == "ecg" else "pct_eda_removed"
        pct = rows[f"{pct_col}_{seg}"].to_numpy(dtype=float)
        keep = rows[f"{signal}_keep_{seg}"].to_numpy(dtype=bool)
        ok = np.isfinite(dur)
        total += dur[ok].sum()
        lost += np.where(keep, np.nan_to_num(pct, nan=100.0) / 100 * dur, dur)[ok].sum()
    return 100 * lost / total if total else np.nan


def participant_row(pid, runs: pd.DataFrame) -> dict:
    """Combine a participant's runs into one row (physiology only)."""
    stress = runs[runs["run_type"] == "stress"]
    non = runs[runs["run_type"] == "non-stress"]
    m = lambda d, c: float(pd.to_numeric(d[c], errors="coerce").mean()) if len(d) else np.nan  # noqa: E731
    r = {"participant_id": pid, "n_stress_runs": len(stress), "n_nonstress_runs": len(non)}
    if "has_mist" in runs:
        r["n_unknown_runs"] = int((runs["has_mist"].astype(bool) & ~runs["run_type"].isin(["stress", "non-stress"])).sum())
    for sig in ("rmssd", "scl", "scr_freq", "phasic_area"):
        r[f"{sig}_baseline"] = m(runs.head(1), f"{sig}_baseline")     # one session baseline
        r[f"{sig}_stress"] = m(stress, f"{sig}_task")
        r[f"{sig}_nonstress"] = m(non, f"{sig}_task")
    r["rmssd_post"] = m(stress, "rmssd_post")
    r["delta_rmssd_stress"] = r["rmssd_stress"] - r["rmssd_nonstress"]
    r["delta_scl_stress"] = r["scl_stress"] - r["scl_nonstress"]
    r["delta_rmssd"] = r["rmssd_stress"] - r["rmssd_baseline"]
    r["delta_scl"] = r["scl_stress"] - r["scl_baseline"]
    r["rmssd_recovery_time_s"] = m(stress, "rmssd_recovery_time_s")
    # runs where recovery could be computed (threshold set) but was not reached in the post window
    if "recovery_threshold_ms" in runs:
        r["n_stress_runs_not_recovered"] = int((stress["recovery_threshold_ms"].notna()
                                                & ~stress["recovery_reached"].astype(bool)).sum())
    r["pct_excluded_ecg"] = _pct_excluded(runs, "ecg", SEGMENTS)
    r["pct_removed_eda"] = _pct_excluded(runs, "eda", EDA_SEGMENTS)
    return r


def fp_rates(trials: pd.DataFrame) -> dict:
    """False-positive rate = 'yes' on silent trials / silent trials, for negative,
    positive and neutral words and for "all words" (= the valences in
    config.FP_ALL_VALENCES). Trials without a response are left out."""
    s = trials[trials["silent"].astype(bool) & trials["response_yes"].notna()]
    out = {}
    groups = (("neg", s[s["valence"] == "negative"]), ("pos", s[s["valence"] == "positive"]),
              ("neu", s[s["valence"] == "neutral"]), ("all", s[s["valence"].isin(C.FP_ALL_VALENCES)]))
    for key, sv in groups:
        n_yes = int(sv["response_yes"].sum())
        out[f"fp_rate_{key}"] = n_yes / len(sv) if len(sv) else np.nan
        out[f"n_yes_{key}"] = n_yes
        out[f"n_silent_{key}"] = int(len(sv))
    return out


def fp_rates_by_run(pid, trials: pd.DataFrame) -> pd.DataFrame:
    rows = [{"participant_id": pid, "run_id": run_id, **fp_rates(g)} for run_id, g in trials.groupby("run_id")]
    return pd.DataFrame(rows)
