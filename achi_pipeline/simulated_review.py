"""A simulated human reviewer for the SYNTHETIC data only.

Real data must be reviewed by a person (outline step 4). For synthetic participants we
know where every true heartbeat and every injected artefact is
(synthetic_data/_ground_truth/), so we can:

* ``expected_decisions(pid)``  work out what a careful reviewer looking at the plots
  would decide for every flag (the "answer key"), with a human-style note;
* ``apply(pid, error_rate)``   fill the review sheets with those decisions, making a
  realistic share of mistakes (missed corrections) at ``error_rate``;
* ``score(pid)``               compare whatever is in the review sheets (yours, if you
  reviewed in the notebook) against the answer key.

What the simulated reviewer "sees":
  ECG  - a flat line             -> accept (signal loss confirmed; already excluded)
       - unreadable noise burst  -> exclude_span covering the burst
       - peak on a spike/T-wave  -> remove_peak
       - peak shifted off the R  -> move_peak to the visible R wave
       - visible R with no peak  -> add_peak
       - interpolated beat where no QRS is visible (missed beat) -> accept
       - otherwise               -> accept (detection is right; RR change is real)
  EDA  - square jump / contact loss          -> accept (it is an artefact)
       - smooth fast rise + slow recovery    -> not_artifact (genuine large SCR)
       - unclear                             -> accept (conservative)
  Segments: exclude the ECG with a reason when > 20% of the segment is unusable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .review import review_dir

MATCH_S = 0.04      # a peak within 40 ms of a true R wave is "on" it
SEARCH_S = 0.15     # tolerance used by apply_review to find the peak a flag refers to


def _truth(pid):
    gt = C.DATA_DIR / "_ground_truth"
    if not gt.exists():
        raise FileNotFoundError("No ground truth found. This only works on synthetic data "
                                "(config.use_synthetic(True); make_synthetic_data.py).")
    beats = pd.read_csv(gt / f"{pid}_true_beats.csv")["time_s"].to_numpy()
    arts = pd.read_csv(gt / f"{pid}_injected_artifacts.csv")
    return beats, arts


def _near(arr, t, tol):
    if len(arr) == 0:
        return None
    j = int(np.argmin(np.abs(arr - t)))
    return arr[j] if abs(arr[j] - t) <= tol else None


def expected_decisions(pid) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (ecg_answers, eda_answers, segment_answers)."""
    from .io_utils import participant_dir
    beats, arts = _truth(pid)
    d = review_dir(pid)
    arr = np.load(participant_dir(pid) / "ecg_arrays.npz")
    fs = float(arr["fs"])
    peaks = arr["peaks_auto"] / fs
    ecg = pd.read_csv(d / "flags_ecg.csv", keep_default_na=False)
    eda = pd.read_csv(d / "flags_eda.csv", keep_default_na=False)

    bursts = arts[(arts["signal"] == "ecg") & (arts["type"] == "movement_burst")]
    missed = arts.loc[arts["type"] == "missed_beat", "start_s"].to_numpy()
    spikes = arts.loc[arts["type"] == "false_peak", "start_s"].to_numpy()

    handled_peaks, excluded, rows = set(), {}, []   # excluded: burst index -> (start, end) actually excluded
    in_excluded = lambda x: any(a <= x <= b for a, b in excluded.values())  # noqa: E731
    for _, f in ecg.iterrows():
        t = float(f["time_s"])
        dec, new_t, note = "accept", "", ""
        burst = bursts[(bursts["start_s"] - 1.5 <= t) & (bursts["end_s"] + 1.5 >= t)]
        b = burst.index[0] if len(burst) else None
        if f["type"] == "flatline":
            note = "flat line, no QRS: signal loss confirmed"
        elif b is not None and b not in excluded:
            excluded[b] = (burst.loc[b, "start_s"] - 0.5, burst.loc[b, "end_s"] + 0.5)
            dec = "exclude_span"
            new_t = f"{excluded[b][0]:.3f}-{excluded[b][1]:.3f}"
            note = "movement artefact, QRS complexes not identifiable"
        elif in_excluded(t):
            note = "inside movement artefact already excluded"
        elif f["type"] == "auto_removed":
            true_r = _near(beats, t, MATCH_S)
            if true_r is not None and _near(peaks, true_r, MATCH_S) is None and round(true_r, 3) not in handled_peaks \
                    and not in_excluded(true_r):
                dec, new_t, note = "add_peak", f"{true_r:.3f}", "algorithm removed a real R wave; put it back"
                handled_peaks.add(round(true_r, 3))
            else:
                note = "removed peak was not a real R wave"
        else:
            p = _near(peaks, t, SEARCH_S)
            if p is not None and round(p, 3) in handled_peaks:
                note = "already corrected under an earlier flag"
            elif p is not None and _near(beats, p, MATCH_S) is None:
                # nearest visible R wave within half a beat that has no peak yet
                cand = [b for b in beats[np.abs(beats - p) <= 0.5] if _near(peaks, b, MATCH_S) is None
                        and _near(missed, b, 0.01) is None and round(b, 3) not in handled_peaks
                        and not in_excluded(b)]
                true_r = min(cand, key=lambda b: abs(b - p)) if cand else None
                if _near(missed, p, 0.35) is not None:
                    note = "no QRS visible here (missed beat); interpolated beat kept"
                elif _near(spikes, p, 0.05) is not None or true_r is None:
                    dec, note = "remove_peak", "peak sits on a sharp spike, not an R wave"
                else:
                    dec, new_t, note = "move_peak", f"{true_r:.3f}", "peak is off the R wave; moved to the R wave"
                handled_peaks.add(round(p, 3))
            else:
                # is there a visible R wave with no peak just before this point?
                prev = peaks[peaks < t - 0.05]
                lo = prev[-1] if len(prev) else t - 2
                cand = beats[(beats > lo + MATCH_S) & (beats < t - MATCH_S)]
                cand = [b for b in cand if _near(peaks, b, MATCH_S) is None and _near(missed, b, 0.01) is None
                        and round(b, 3) not in handled_peaks and not in_excluded(b)]
                if cand:
                    dec, new_t, note = "add_peak", f"{cand[0]:.3f}", "visible R wave without a detected peak"
                    handled_peaks.add(round(cand[0], 3))
                else:
                    note = "detection correct; RR change is physiological"
        rows.append({"flag_id": f["flag_id"], "type": f["type"], "time_s": t, "decision": dec,
                     "new_time_s": new_t, "notes": note})
    ecg_ans = pd.DataFrame(rows)

    eda_rows = []
    e_arts = arts[arts["signal"] == "eda"]
    for _, f in eda.iterrows():
        a, b = float(f["time_s"]), float(f["end_s"])
        hit = e_arts[(e_arts["start_s"] <= b) & (e_arts["end_s"] >= a)]
        if len(hit) and (hit["type"] == "genuine_large_scr").all():
            dec, note = "not_artifact", "smooth fast rise with slow recovery: genuine SCR"
        elif len(hit):
            dec, note = "accept", f"artefact ({hit['type'].iloc[0].replace('_', ' ')})"
        else:
            dec, note = "accept", "unclear; treated as artefact (conservative)"
        eda_rows.append({"flag_id": f["flag_id"], "time_s": a, "decision": dec, "notes": note})
    eda_ans = pd.DataFrame(eda_rows)

    seg = pd.read_csv(d / "segment_decisions.csv", keep_default_na=False)
    seg_rows = []
    for _, s in seg.iterrows():
        a, b = float(s["start_s"]), float(s["end_s"])
        lost = sum(max(0, min(b, r.end_s) - max(a, r.start_s)) for r in
                   arts[(arts["signal"] == "ecg") & arts["type"].isin(["movement_burst", "signal_loss"])].itertuples())
        pct = 100 * lost / (b - a)
        if pct > C.ECG_EXCLUDE_PCT:
            seg_rows.append({"run_id": s["run_id"], "segment": s["segment"], "ecg_decision": "exclude",
                             "eda_decision": "keep", "reason": f"ECG unusable for ~{lost:.0f} s ({pct:.0f}%)"})
        else:
            seg_rows.append({"run_id": s["run_id"], "segment": s["segment"], "ecg_decision": "keep",
                             "eda_decision": "keep", "reason": ""})
    return ecg_ans, eda_ans, pd.DataFrame(seg_rows)


def apply(pid, error_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Fill the review sheets like a reviewer would, with occasional mistakes.

    A mistake = a needed correction is overlooked and the flag is accepted as is.
    Returns a log of every decision (and whether it was a simulated mistake).
    """
    rng = np.random.default_rng(seed + int(pid[-3:]))
    ecg_ans, eda_ans, seg_ans = expected_decisions(pid)
    d = review_dir(pid)

    ecg = pd.read_csv(d / "flags_ecg.csv", keep_default_na=False)
    log = []
    for i, a in ecg_ans.iterrows():
        dec, new_t, note, mistake = a["decision"], a["new_time_s"], a["notes"], False
        if dec != "accept" and rng.random() < error_rate:
            dec, new_t, note, mistake = "accept", "", "looks fine", True
        ecg.loc[i, ["decision", "new_time_s", "notes"]] = [dec, new_t, note]
        log.append({"participant_id": pid, "signal": "ecg", "flag_id": a["flag_id"], "type": a["type"],
                    "time_s": a["time_s"], "correct_decision": a["decision"], "decision": dec,
                    "simulated_mistake": mistake, "notes": note})
    ecg.to_csv(d / "flags_ecg.csv", index=False)

    eda = pd.read_csv(d / "flags_eda.csv", keep_default_na=False)
    for i, a in eda_ans.iterrows():
        dec, note, mistake = a["decision"], a["notes"], False
        if dec != "accept" and rng.random() < error_rate:
            dec, note, mistake = "accept", "looks like an artefact", True
        eda.loc[i, ["decision", "notes"]] = [dec, note]
        log.append({"participant_id": pid, "signal": "eda", "flag_id": a["flag_id"], "type": "eda_artifact",
                    "time_s": a["time_s"], "correct_decision": a["decision"], "decision": dec,
                    "simulated_mistake": mistake, "notes": note})
    eda.to_csv(d / "flags_eda.csv", index=False)

    seg = pd.read_csv(d / "segment_decisions.csv", keep_default_na=False)
    seg = seg.drop(columns=["ecg_decision", "eda_decision", "reason"]).merge(
        seg_ans, on=["run_id", "segment"], how="left")
    seg.to_csv(d / "segment_decisions.csv", index=False)
    return pd.DataFrame(log)


def score(pid) -> pd.DataFrame:
    """Compare the decisions currently in the review sheets with the answer key."""
    ecg_ans, eda_ans, _ = expected_decisions(pid)
    d = review_dir(pid)
    rows = []
    for sig, ans in (("ecg", ecg_ans), ("eda", eda_ans)):
        given = pd.read_csv(d / f"flags_{sig}.csv", keep_default_na=False)
        for (_, a), (_, g) in zip(ans.iterrows(), given.iterrows()):
            ok = str(g["decision"]) == a["decision"]
            if ok and a["decision"] in ("add_peak", "move_peak"):
                try:
                    ok = abs(float(g["new_time_s"]) - float(a["new_time_s"])) <= 0.05
                except ValueError:
                    ok = False
            rows.append({"signal": sig, "flag_id": a["flag_id"], "time_s": a["time_s"],
                         "your_decision": g["decision"] or "(blank)", "expected": a["decision"],
                         "expected_new_time_s": a.get("new_time_s", ""), "correct": ok, "why": a["notes"]})
    out = pd.DataFrame(rows)
    if len(out):
        print(f"{pid}: {out['correct'].sum()}/{len(out)} decisions match the answer key "
              f"({100 * out['correct'].mean():.0f}%)")
    return out


def true_rmssd(pid, segments: pd.DataFrame) -> pd.DataFrame:
    """RMSSD from the true heartbeats, per run and segment (for checking results)."""
    beats, _ = _truth(pid)
    rows = []
    for _, s in segments.iterrows():
        b = beats[(beats >= s["start_s"]) & (beats <= s["end_s"])]
        rr = np.diff(b) * 1000
        rows.append({"participant_id": pid, "run_id": s["run_id"], "segment": s["segment"],
                     "true_rmssd": float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 2 else np.nan})
    return pd.DataFrame(rows)


def true_scr_frequency(pid, segments: pd.DataFrame) -> pd.DataFrame:
    """SCRs per minute from the true SCRs (>= SCR_MIN_AMPLITUDE_US), per run and segment.

    Uses the whole segment as the denominator, so segments with removed artefacts read
    slightly lower than the pipeline's per-artefact-free-minute value.
    """
    scrs = pd.read_csv(C.DATA_DIR / "_ground_truth" / f"{pid}_true_scrs.csv")
    scrs = scrs[scrs["amplitude_us"] >= C.SCR_MIN_AMPLITUDE_US]
    rows = []
    for _, s in segments.iterrows():
        n = int(((scrs["peak_s"] >= s["start_s"]) & (scrs["peak_s"] <= s["end_s"])).sum())
        rows.append({"participant_id": pid, "run_id": s["run_id"], "segment": s["segment"],
                     "true_scr_freq": n / ((s["end_s"] - s["start_s"]) / 60)})
    return pd.DataFrame(rows)
