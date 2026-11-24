"""SCR overlap check: SCR frequency vs phasic area per minute (does not touch the pipeline outputs).

1. Controlled test: two SCRs of known size, onset gap 0.5-8 s, through the real
   process_eda path. Records the SCR count, the phasic area and the dip between the SCRs.
2. Notch case: two sub-threshold rises stacked on each other (count should be 1).
3. Plots of detected SCRs and phasic area on synthetic participants.
4. SCR frequency vs phasic area per minute per segment for the same participants.

Run from ACHI_analysis/:  python3 checks/scr_overlap/scr_overlap_check.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PROJECT = Path(__file__).resolve().parents[2]      # ACHI_analysis/
sys.path.insert(0, str(PROJECT))
from achi_pipeline import eda, measures  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
FS = 250
PRE_S = 60          # flat lead-in so the 0.05 Hz split has settled
WIN_S = 40          # fixed window for the area, so it is comparable across gaps


def scr_shape(t, onset, amp, tau_rise=0.75, tau_decay=4.0):
    """Bateman-type SCR scaled so its peak is `amp` µS."""
    s = np.zeros_like(t)
    k = t >= onset
    tt = t[k] - onset
    s[k] = np.exp(-tt / tau_decay) - np.exp(-tt / tau_rise)
    return amp * s / s.max() if s.max() > 0 else s


def run_pair(a1, a2, gap, noise_sd=0.0, seed=0):
    t = np.arange(0, PRE_S + WIN_S, 1 / FS)
    on1 = PRE_S + 5
    x = 5.0 + scr_shape(t, on1, a1) + scr_shape(t, on1 + gap, a2)
    if noise_sd:
        x = x + np.random.default_rng(seed).normal(0, noise_sd, len(t))
    res = eda.process_eda(x, FS)
    df, fs = res["eda"], res["fs"]
    clean = df["eda_clean"].to_numpy()
    s = eda.find_scrs(clean, fs)
    s = s[(s["peak_s"] >= on1 - 2) & (s["peak_s"] <= on1 + gap + 15)]
    no_bad = np.zeros(len(df), bool)
    out = {"n_scr": len(s),
           "phasic_area_per_min": measures.phasic_area_per_min(df, fs, PRE_S, PRE_S + WIN_S, no_bad)}
    # dip between the first two maxima of the low-passed signal (0 = no dip, merged)
    win = ((df["time_s"] >= on1 - 2) & (df["time_s"] <= on1 + gap + 15)).to_numpy()
    seg = clean[win]
    pk, _ = find_peaks(seg)
    out["dip_us"] = float(seg[pk[0]] - seg[pk[0]:pk[1]].min()) if len(pk) >= 2 else 0.0
    return out, df, fs


# ---------------------------------------------------------------- 1. overlap grid
rows = []
gaps = np.arange(0.5, 8.01, 0.25)
pairs = [(0.3, 0.3), (0.3, 0.1), (0.1, 0.3), (0.1, 0.1), (0.06, 0.06)]
single = {a: run_pair(a, 0.0, 1.0)[0]["phasic_area_per_min"] for a in {a for p in pairs for a in p}}
for a1, a2 in pairs:
    expected = single[a1] + single[a2]          # area if both SCRs were fully kept
    for gap in gaps:
        r, _, _ = run_pair(a1, a2, gap)
        rows.append({"amp1": a1, "amp2": a2, "gap_s": gap, **r,
                     "area_vs_two_singles": r["phasic_area_per_min"] / expected})
grid = pd.DataFrame(rows)
grid.to_csv(OUT / "overlap_grid.csv", index=False)

# with noise: 20 repeats at a few gaps
noisy = []
for gap in (1.0, 2.0, 3.0, 5.0):
    for seed in range(20):
        r, _, _ = run_pair(0.3, 0.3, gap, noise_sd=0.02, seed=seed)
        noisy.append({"gap_s": gap, "seed": seed, "n_scr": r["n_scr"], "phasic_area_per_min": r["phasic_area_per_min"]})
noisy = pd.DataFrame(noisy)
noisy.to_csv(OUT / "overlap_noisy.csv", index=False)

# ---------------------------------------------------------------- 2. notch case
notch = []
for a, gap in [(0.04, 2.0), (0.04, 3.0), (0.045, 2.5), (0.03, 2.0)]:
    r, df, fs = run_pair(a, a, gap)
    seg = df[(df["time_s"] >= PRE_S + 3) & (df["time_s"] <= PRE_S + 5 + gap + 15)]["eda_clean"]
    notch.append({"amp_each": a, "gap_s": gap, "total_rise_us": round(seg.max() - seg.iloc[0], 3),
                  "dip_us": round(r["dip_us"], 4), "n_found": r["n_scr"]})
notch = pd.DataFrame(notch)
notch.to_csv(OUT / "notch_case.csv", index=False)

# ---------------------------------------------------------------- overlap figure
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, gap in zip(axes[0], (1.0, 2.0, 4.0)):
    r, df, fs = run_pair(0.3, 0.3, gap)
    d = df[(df["time_s"] >= PRE_S) & (df["time_s"] <= PRE_S + 25)]
    s = eda.find_scrs(df["eda_clean"].to_numpy(), fs)
    s = s[(s["peak_s"] >= PRE_S) & (s["peak_s"] <= PRE_S + 25)]
    tt = d["time_s"] - PRE_S
    ax.plot(tt, d["eda_clean"], color="k", lw=1.2, label="low-passed EDA (count)")
    ax.plot(s["onset_s"] - PRE_S, np.interp(s["onset_s"], df["time_s"], df["eda_clean"]), "o", color="tab:blue", label="onset")
    ax.plot(s["peak_s"] - PRE_S, np.interp(s["peak_s"], df["time_s"], df["eda_clean"]), "v", color="tab:red", label="peak")
    ax.set_ylabel("µS")
    ax2 = ax.twinx()
    ax2.fill_between(tt, 0, np.clip(d["scr"], 0, None), color="tab:orange", alpha=.25, label="phasic area")
    ax2.set_ylabel("phasic µS", color="tab:orange")
    ax.set_title(f"two 0.3 µS SCRs {gap:g} s apart: {len(s)} counted, area {r['phasic_area_per_min']:.2f}")
    ax.set_xlabel("s")
axes[0, 0].legend(fontsize=8, loc="upper right")
for ax, (a1, a2) in zip(axes[1], [(0.3, 0.3), (0.3, 0.1), (0.1, 0.1)]):
    g = grid[(grid.amp1 == a1) & (grid.amp2 == a2)]
    ax.step(g.gap_s, g.n_scr, where="mid", color="tab:blue", label="SCRs counted")
    ax.set_ylim(-0.2, 2.6); ax.set_ylabel("SCRs counted", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(g.gap_s, g.area_vs_two_singles, color="tab:orange", label="area ÷ two separate SCRs")
    ax2.set_ylim(0, 1.3); ax2.set_ylabel("area ÷ (area of the two SCRs alone)", color="tab:orange")
    ax.set_title(f"SCR 1 = {a1} µS, SCR 2 = {a2} µS"); ax.set_xlabel("onset gap (s)")
fig.tight_layout(); fig.savefig(OUT / "overlap_test.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 3 & 4. synthetic participants
DER = PROJECT / "synthetic_derived" / "participants"
comp = []
for pid in ("ACHI_2026_901", "ACHI_2026_902"):
    z = np.load(DER / pid / "eda_arrays.npz", allow_pickle=True)
    fs = float(z["fs"])
    df = pd.DataFrame({k: z[k] for k in ("time_s", "eda_clean", "scl", "scr", "artifact")})
    bad = df["artifact"].to_numpy(bool)       # automatic flags only (no manual review spans here)
    scrs = eda.find_scrs(df["eda_clean"].to_numpy(), fs)
    segs = pd.read_csv(DER / pid / "segments.csv")
    segs = segs[segs.segment.isin(["baseline", "task"])]
    for _, sg in segs.iterrows():
        a, b = sg.start_s, sg.end_s
        n, mins, freq = measures.scr_frequency(scrs, df, fs, a, b, bad)
        comp.append({"participant_id": pid, "run_id": sg.run_id, "segment": sg.segment, "n_scr": n,
                     "clean_min": round(mins, 2), "scr_freq_per_min": round(freq, 2),
                     "phasic_area_per_min": round(measures.phasic_area_per_min(df, fs, a, b, bad), 2)})
    # one 60 s window from baseline and from the first task
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex="col")
    for col, (_, sg) in enumerate(segs.drop_duplicates("segment").iterrows()):
        a = sg.start_s + 30; b = a + 60
        m = ((df.time_s >= a) & (df.time_s <= b)).to_numpy()
        d = df[m]
        s = scrs[(scrs.peak_s >= a) & (scrs.peak_s <= b)]
        ax = axes[0, col]
        ax.plot(d.time_s, d.eda_clean, color="k", lw=1)
        ax.plot(d.time_s, d.scl, color="tab:green", lw=1, alpha=.7, label="SCL (tonic)")
        ax.plot(s.onset_s, np.interp(s.onset_s, df.time_s, df.eda_clean), "o", color="tab:blue", ms=4, label="onset")
        ax.plot(s.peak_s, np.interp(s.peak_s, df.time_s, df.eda_clean), "v", color="tab:red", ms=5, label="peak")
        ax.set_title(f"{pid} {sg.segment} ({sg.run_id}): low-passed EDA, {len(s)} SCRs counted")
        ax.set_ylabel("µS"); ax.legend(fontsize=8, loc="upper right")
        ax = axes[1, col]
        ax.plot(d.time_s, d.scr, color="k", lw=1)
        ax.fill_between(d.time_s, 0, np.clip(d.scr, 0, None), color="tab:orange", alpha=.35, label="area above zero")
        ax.axhline(0, color="grey", lw=.5)
        flagged = d.artifact.to_numpy(bool)
        if flagged.any():
            ax.fill_between(d.time_s, d.scr.min(), d.scr.max(), where=flagged, color="grey", alpha=.2, label="artefact (left out)")
        area = measures.phasic_area_per_min(df, fs, a, b, bad)
        ax.set_title(f"phasic component: area {area:.2f} µS·s per artefact-free min")
        ax.set_ylabel("µS"); ax.set_xlabel("time (s)"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / f"scr_detection_{pid}.png", dpi=130); plt.close(fig)
comp = pd.DataFrame(comp)
comp.to_csv(OUT / "count_vs_area.csv", index=False)

pd.set_option("display.width", 200)
print("== overlap grid: SCRs counted by gap ==")
print(grid.pivot_table(index="gap_s", columns=["amp1", "amp2"], values="n_scr").T.to_string())
print("\n== overlap grid: area ÷ area of the two SCRs alone, by gap ==")
print(grid.pivot_table(index="gap_s", columns=["amp1", "amp2"], values="area_vs_two_singles").T.round(2).to_string())
print("\n== noisy (0.02 µS noise), 0.3+0.3 ==")
print(noisy.groupby("gap_s")[["n_scr", "phasic_area_per_min"]].agg(["mean", "min", "max"]).round(2))
print("\n== notch ==\n", notch.to_string())
print("\n== count vs area ==\n", comp.to_string())
