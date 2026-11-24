"""Outline 3.2: clean ECG, detect signal loss, detect and correct beats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import neurokit2 as nk

from . import config as C


def clean_ecg(ecg_raw: np.ndarray, fs: float) -> np.ndarray:
    # method="neurokit": 0.5 Hz high-pass, 5th-order Butterworth, plus power-line filter
    return nk.ecg_clean(ecg_raw, sampling_rate=fs, method="neurokit", powerline=C.POWERLINE_HZ)


def detect_flatline(ecg_raw: np.ndarray, fs: float) -> pd.DataFrame:
    """Signal-loss spans: consecutive windows whose peak-to-peak range < threshold."""
    w = int(round(C.FLATLINE_WINDOW_S * fs))
    n = len(ecg_raw) // w
    if n == 0:
        return pd.DataFrame(columns=["start_s", "end_s"])
    rng = np.ptp(ecg_raw[: n * w].reshape(n, w), axis=1)
    flat = rng < C.FLATLINE_RANGE_MV
    return _runs_to_spans(flat, C.FLATLINE_WINDOW_S)


def _runs_to_spans(mask: np.ndarray, step_s: float) -> pd.DataFrame:
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return pd.DataFrame({"start_s": starts * step_s, "end_s": ends * step_s})


def good_chunks(n_samples: int, fs: float, bad_spans: pd.DataFrame) -> list[tuple[int, int]]:
    """Sample ranges between bad spans."""
    edges, cur = [], 0
    for _, s in bad_spans.sort_values("start_s").iterrows():
        a, b = int(s["start_s"] * fs), int(s["end_s"] * fs)
        if a > cur:
            edges.append((cur, a))
        cur = max(cur, b)
    if cur < n_samples:
        edges.append((cur, n_samples))
    return edges


def detect_and_correct_peaks(ecg_clean: np.ndarray, fs: float, flat: pd.DataFrame):
    """R-peak detection + Lipponen & Tarvainen (Kubios) correction, per good chunk.

    Returns (peaks_initial, peaks_corrected, corrections) where ``corrections`` lists
    every beat the algorithm changed: action in {inserted, removed, moved}.
    """
    initial, corrected = [], []
    min_chunk = int(10 * fs)
    for a, b in good_chunks(len(ecg_clean), fs, flat):
        if b - a < min_chunk:
            continue
        try:
            _, info = nk.ecg_peaks(ecg_clean[a:b], sampling_rate=fs)
            pk = np.asarray(info["ECG_R_Peaks"], dtype=int)
        except Exception:
            continue
        if len(pk) < 5:
            continue
        initial.append(pk + a)
        try:
            _, pk_c = nk.signal_fixpeaks(pk, sampling_rate=fs, method="Kubios", iterative=True,
                                         alpha=C.KUBIOS_ALPHA, window_width=C.KUBIOS_WINDOW_BEATS)
            pk_c = np.unique(np.asarray(pk_c, dtype=int))
        except Exception:
            pk_c = pk
        pk_c = pk_c[(pk_c >= 0) & (pk_c < b - a)]
        corrected.append(pk_c + a)
    peaks_initial = np.concatenate(initial) if initial else np.array([], int)
    peaks_corrected = np.concatenate(corrected) if corrected else np.array([], int)
    corrections = diff_peaks(peaks_initial, peaks_corrected, fs)
    return peaks_initial, peaks_corrected, corrections


def diff_peaks(before: np.ndarray, after: np.ndarray, fs: float, source: str = "auto") -> pd.DataFrame:
    """Describe how ``after`` differs from ``before`` as inserted / removed / moved beats."""
    removed = np.setdiff1d(before, after)
    added = np.setdiff1d(after, before)
    rows = []
    used = np.zeros(len(added), bool)
    med_rr = np.median(np.diff(before)) if len(before) > 2 else fs
    for r in removed:
        if len(added):
            j = np.argmin(np.abs(added - r) + used * 1e12)
            if not used[j] and abs(added[j] - r) < 0.5 * med_rr:
                used[j] = True
                rows.append({"action": "moved", "old_sample": r, "sample": added[j]})
                continue
        rows.append({"action": "removed", "old_sample": r, "sample": r})
    for j in np.flatnonzero(~used):
        rows.append({"action": "inserted", "old_sample": np.nan, "sample": added[j]})
    df = pd.DataFrame(rows, columns=["action", "old_sample", "sample"])
    df["time_s"] = df["sample"] / fs
    df["source"] = source
    return df.sort_values("sample").reset_index(drop=True)


def rr_table(peaks: np.ndarray, fs: float, bad_spans: pd.DataFrame | None = None) -> pd.DataFrame:
    """RR intervals with plausibility flags. Intervals spanning signal loss are invalid."""
    peaks = np.sort(np.asarray(peaks, dtype=int))
    rr = np.diff(peaks) / fs * 1000
    df = pd.DataFrame({"start_sample": peaks[:-1], "end_sample": peaks[1:],
                       "time_s": peaks[1:] / fs, "rr_ms": rr})
    df["spans_bad"] = False
    if bad_spans is not None and len(bad_spans):
        s0, s1 = df["start_sample"].to_numpy() / fs, df["end_sample"].to_numpy() / fs
        for _, b in bad_spans.iterrows():
            df.loc[(s0 < b["end_s"]) & (s1 > b["start_s"]), "spans_bad"] = True
    df["range_flag"] = (df["rr_ms"] < C.RR_MIN_MS) | (df["rr_ms"] > C.RR_MAX_MS)
    df["jump_flag"] = df["rr_ms"].diff().abs() > C.RR_JUMP_MS
    df["artifact_flag"] = (df["range_flag"] | df["jump_flag"]) & ~df["spans_bad"]
    return df


def process_ecg(ecg_raw: np.ndarray, fs: float) -> dict:
    ecg_c = clean_ecg(ecg_raw, fs)
    flat = detect_flatline(ecg_raw, fs)
    p0, p1, corr = detect_and_correct_peaks(ecg_c, fs, flat)
    rr = rr_table(p1, fs, flat)
    return {"ecg_clean": ecg_c.astype(np.float32), "flatline": flat, "peaks_initial": p0,
            "peaks_auto": p1, "auto_corrections": corr, "rr": rr}
