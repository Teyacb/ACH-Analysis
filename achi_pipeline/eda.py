"""Outline 3: clean EDA, flag artifacts (Kleckner et al., 2018), split SCL / SCR, count SCRs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import find_peaks

from . import config as C


def process_eda(eda_raw: np.ndarray, fs: float) -> dict:
    """Return a DataFrame at EDA_TARGET_FS with columns time_s, eda_raw, eda_clean,
    scl, scr, artifact (bool), plus the artifact spans."""
    eda_raw = np.asarray(eda_raw, dtype=float)
    # 1 Hz low-pass on the whole recording (4th-order Butterworth, zero phase)
    eda_lp = nk.signal_filter(eda_raw, sampling_rate=fs, highcut=C.EDA_LOWPASS_HZ,
                              method="butterworth", order=4)
    # downsample: content is < 1 Hz, so 50 Hz keeps everything
    step = max(1, int(round(fs / C.EDA_TARGET_FS)))
    fs_d = fs / step
    raw_d = eda_raw[::step]
    clean = eda_lp[::step]
    t = np.arange(len(clean)) / fs_d

    # Kleckner rules
    out_of_range = clean < C.EDA_MIN_US
    if C.EDA_MAX_US is not None:
        out_of_range |= clean > C.EDA_MAX_US
    slope = np.abs(np.gradient(clean) * fs_d)
    too_fast = slope > C.EDA_MAX_SLOPE_US_PER_S
    flagged = out_of_range | too_fast
    pad = int(round(C.EDA_ARTIFACT_PAD_S * fs_d))
    if pad and flagged.any():
        kernel = np.ones(2 * pad + 1)
        flagged = np.convolve(flagged.astype(float), kernel, mode="same") > 0

    # tonic / phasic split (on the clean signal; artifacts masked later when averaging)
    try:
        comp = nk.eda_phasic(clean, sampling_rate=fs_d, method=C.EDA_DECOMPOSITION)
        scl, scr = comp["EDA_Tonic"].to_numpy(), comp["EDA_Phasic"].to_numpy()
    except Exception as err:  # keep going; SCL falls back to the clean signal
        print(f"  EDA decomposition failed ({err}); using low-passed EDA as SCL")
        scl, scr = clean, np.zeros_like(clean)

    df = pd.DataFrame({"time_s": t, "eda_raw": raw_d.astype(np.float32), "eda_clean": clean.astype(np.float32),
                       "scl": scl.astype(np.float32), "scr": scr.astype(np.float32), "artifact": flagged,
                       "out_of_range": out_of_range, "too_fast": too_fast})
    spans = _mask_to_spans(flagged, fs_d)
    return {"eda": df, "fs": fs_d, "artifact_spans": spans}


def _mask_to_spans(mask: np.ndarray, fs: float) -> pd.DataFrame:
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return pd.DataFrame({"start_s": s / fs, "end_s": e / fs})


def find_scrs(x: np.ndarray, fs: float) -> pd.DataFrame:
    """Every SCR in the 1 Hz low-passed EDA, as onset-to-peak rises (Boucsein et al., 2012).

    Each local peak is one SCR, so overlapping responses count separately. The onset is
    the lowest point since the previous accepted peak, looking back at most SCR_MAX_RISE_S.
    Only rises of at least SCR_MIN_AMPLITUDE_US are kept. A rejected peak (a small notch on
    a rise) does not move the onset search, so it cannot split one SCR into two too-small ones.
    """
    x = np.asarray(x, dtype=float)
    peaks, _ = find_peaks(x)
    rows, prev = [], 0
    back = int(round(C.SCR_MAX_RISE_S * fs))
    for p in peaks:
        a = max(prev, p - back)
        if p > a:
            onset = a + int(np.argmin(x[a:p]))
            amp = x[p] - x[onset]
            if amp >= C.SCR_MIN_AMPLITUDE_US:
                rows.append({"onset_s": onset / fs, "peak_s": p / fs, "amplitude_us": amp})
                prev = p
    return pd.DataFrame(rows, columns=["onset_s", "peak_s", "amplitude_us"])
