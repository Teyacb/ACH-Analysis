"""SCR frequency vs phasic area per minute on synthetic participants: which separates
stress from non-stress MIST more clearly? (does not touch the pipeline outputs)

Uses the saved EDA arrays and segments in synthetic_derived/, automatic artefact flags only.
Synthetic data: stress differs from non-stress only in SCR rate (6 vs 3 per min, same sizes).

Run from ACHI_analysis/:  python3 checks/scr_overlap/compare_measures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]      # ACHI_analysis/
sys.path.insert(0, str(PROJECT))
from achi_pipeline import eda, measures  # noqa: E402

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
DER = PROJECT / "synthetic_derived" / "participants"
GT = PROJECT / "synthetic_data" / "_ground_truth"
SHOW = ["ACHI_2026_901", "ACHI_2026_902", "ACHI_2026_906"]
ALL = sorted(p.name for p in DER.iterdir() if (p / "eda_arrays.npz").exists())
groups = pd.read_csv(PROJECT / "synthetic_data" / "participants.csv").set_index("participant_id")["group"]


def load(pid):
    z = np.load(DER / pid / "eda_arrays.npz", allow_pickle=True)
    df = pd.DataFrame({k: z[k] for k in ("time_s", "eda_clean", "scl", "scr", "artifact")})
    return df, float(z["fs"])


rows = []
for pid in ALL:
    df, fs = load(pid)
    bad = df["artifact"].to_numpy(bool)
    scrs = eda.find_scrs(df["eda_clean"].to_numpy(), fs)
    true = pd.read_csv(GT / f"{pid}_true_scrs.csv")
    types = pd.read_csv(DER / pid / "runs.csv").set_index("run_id")["run_type"]
    segs = pd.read_csv(DER / pid / "segments.csv")
    for _, sg in segs[segs.segment.isin(["baseline", "task"])].iterrows():
        a, b = sg.start_s, sg.end_s
        n, mins, freq = measures.scr_frequency(scrs, df, fs, a, b, bad)
        n_true = int(((true.peak_s >= a) & (true.peak_s <= b)).sum())
        rows.append({"participant_id": pid, "group": groups.get(pid), "run_id": sg.run_id,
                     "condition": "baseline" if sg.segment == "baseline" else types[sg.run_id],
                     "n_scr": n, "n_true_scr": n_true, "scr_freq": freq,
                     "phasic_area": measures.phasic_area_per_min(df, fs, a, b, bad)})
seg = pd.DataFrame(rows)
seg.to_csv(OUT / "compare_segments.csv", index=False)

# per participant: baseline, mean of stress runs, mean of non-stress runs
per = seg.groupby(["participant_id", "group", "condition"])[["scr_freq", "phasic_area", "n_scr", "n_true_scr"]].mean()
per = per.unstack("condition")
summary = []
for (pid, grp), r in per.iterrows():
    out = {"participant_id": pid, "group": grp}
    for m in ("scr_freq", "phasic_area"):
        b, s, n = r[(m, "baseline")], r[(m, "stress")], r[(m, "non-stress")]
        out |= {f"{m}_baseline": b, f"{m}_stress": s, f"{m}_nonstress": n,
                f"{m}_stress_vs_non_pct": 100 * (s - n) / n}
    out["pct_true_scrs_counted_stress"] = 100 * r[("n_scr", "stress")] / r[("n_true_scr", "stress")]
    out["pct_true_scrs_counted_nonstress"] = 100 * r[("n_scr", "non-stress")] / r[("n_true_scr", "non-stress")]
    summary.append(out)
summary = pd.DataFrame(summary)
summary.to_csv(OUT / "compare_participants.csv", index=False)

# all participants: paired effect size (dz) of stress vs non-stress, per measure
dz = []
for m in ("scr_freq", "phasic_area"):
    d = summary[f"{m}_stress"] - summary[f"{m}_nonstress"]
    dz.append({"measure": m, "n": len(d), "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
               "dz": d.mean() / d.std(ddof=1), "n_participants_stress_higher": int((d > 0).sum())})
dz = pd.DataFrame(dz)
dz.to_csv(OUT / "compare_effect_sizes.csv", index=False)

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, len(SHOW) + 1, figsize=(18, 8),
                         gridspec_kw={"width_ratios": [1] * len(SHOW) + [1.2]})
cond_order, colors = ["baseline", "non-stress", "stress"], {"baseline": "grey", "non-stress": "tab:blue", "stress": "tab:red"}
for row, (m, lab) in enumerate((("scr_freq", "SCR frequency (per min)"),
                                ("phasic_area", "phasic area (µS·s per min)"))):
    for col, pid in enumerate(SHOW):
        ax = axes[row, col]
        d = seg[seg.participant_id == pid]
        for i, c in enumerate(cond_order):
            v = d[d.condition == c][m]
            ax.bar(i, v.mean(), color=colors[c], alpha=.35)
            ax.plot(np.full(len(v), i), v, "o", color=colors[c])
        pct = summary.set_index("participant_id").loc[pid, f"{m}_stress_vs_non_pct"]
        ax.set_xticks(range(3), cond_order); ax.set_ylabel(lab)
        ax.set_title(f"{pid} ({groups[pid]}): stress {pct:+.0f}% vs non-stress", fontsize=10)
    ax = axes[row, -1]
    for _, r in summary.iterrows():
        ax.plot([0, 1], [r[f"{m}_nonstress"], r[f"{m}_stress"]], "-o",
                color="tab:purple" if r.group == "patient" else "tab:green", alpha=.7)
    e = dz.set_index("measure").loc[m]
    ax.set_xticks([0, 1], ["non-stress", "stress"]); ax.set_xlim(-.3, 1.3); ax.set_ylabel(lab)
    ax.set_title(f"all {int(e.n)} participants: dz = {e.dz:.2f}, higher in stress {int(e.n_participants_stress_higher)}/{int(e.n)}",
                 fontsize=10)
axes[0, -1].plot([], [], "-o", color="tab:purple", label="patient")
axes[0, -1].plot([], [], "-o", color="tab:green", label="control")
axes[0, -1].legend(fontsize=8)
fig.suptitle("SCR frequency vs phasic area per minute, synthetic data (dots = runs)")
fig.tight_layout(); fig.savefig(OUT / "compare_measures.png", dpi=130); plt.close(fig)

pd.set_option("display.width", 220)
cols = ["participant_id", "group"] + [f"{m}_{c}" for m in ("scr_freq", "phasic_area")
                                       for c in ("baseline", "nonstress", "stress", "stress_vs_non_pct")]
print("== shown participants ==")
print(summary[summary.participant_id.isin(SHOW)][cols].round(2).to_string(index=False))
print("\n== true SCRs counted (%) ==")
print(summary[["participant_id", "pct_true_scrs_counted_nonstress", "pct_true_scrs_counted_stress"]].round(0).to_string(index=False))
print("\n== all participants, stress vs non-stress ==")
print(dz.round(2).to_string(index=False))
