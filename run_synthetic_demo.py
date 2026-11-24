"""Run the pipeline on the synthetic data the way it will run on real data.

    python run_synthetic_demo.py                     # simulated reviewer, 5% mistakes
    python run_synthetic_demo.py --error-rate 0      # perfect reviewer
    python run_synthetic_demo.py --review manual     # stop after stage 1: YOU review

--review simulated : achi_pipeline/simulated_review.py plays the reviewer. It makes the
                     decisions a careful person would make from the plots (correct bad
                     peaks, exclude movement artefacts, keep genuine skin-conductance
                     responses), with occasional overlooked corrections (--error-rate).
--review manual    : runs stage 1 only. Review in the notebook (USE_SYNTHETIC = True),
                     check yourself with simulated_review.score(pid), then run stages 3-4
                     from the notebook.

The script also reports what the review changed: RMSSD accuracy against the true heartbeats
before review (automatic correction only) and after review, and SCR frequency against the
true skin-conductance responses.
"""
import argparse
import shutil

import numpy as np
import pandas as pd

from achi_pipeline import config, pipeline, simulated_review

ap = argparse.ArgumentParser()
ap.add_argument("--review", choices=["simulated", "manual"], default="simulated")
ap.add_argument("--error-rate", type=float, default=0.05, help="share of needed corrections the reviewer overlooks")
args = ap.parse_args()
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

config.use_synthetic()
if not (config.DATA_DIR / "_ground_truth").exists():
    raise SystemExit("No synthetic data (or old version without ground truth). Run: "
                     "python make_synthetic_data.py")
if config.DERIVED_DIR.exists():
    shutil.rmtree(config.DERIVED_DIR)

# ---------------------------------------------------------------- stage 1
print("=" * 70, "\nSTAGE 1: preprocessing\n" + "=" * 70)
pipeline.run_all_preprocessing()
pids = list(pipeline.discover_participants())

work = []
for pid in pids:
    d = pipeline.participant_dir(pid) / "review"
    e = pd.read_csv(d / "flags_ecg.csv", keep_default_na=False)
    a = pd.read_csv(d / "flags_eda.csv", keep_default_na=False)
    work.append({"participant_id": pid, **e["type"].value_counts().to_dict(), "ecg_flags": len(e), "eda_flags": len(a)})
work = pd.DataFrame(work).fillna(0).set_index("participant_id").astype(int)
n_flags = int(work[["ecg_flags", "eda_flags"]].sum().sum())
print("\nFlags waiting for review (per participant):")
print(work.to_string())
print(f"\nTotal: {n_flags} flags. At ~15-30 s per flag that is roughly "
      f"{n_flags * 15 / 60:.0f}-{n_flags * 30 / 60:.0f} minutes of reviewing.")

if args.review == "manual":
    print("\nStopped after stage 1. Next:\n"
          "  1. open phsyiological_data_analysis.ipynb with USE_SYNTHETIC = True\n"
          "  2. skip the stage 1 cell (already done) and review in the stage 2 cells\n"
          "  3. check yourself:  from achi_pipeline import simulated_review; simulated_review.score(pid)\n"
          "  4. run the stage 3 and 4 cells")
    raise SystemExit

# ---------------------------------------------------------------- before review (for comparison only)
config.ALLOW_UNREVIEWED = True
pipeline.run_all_compute()
before = pipeline.analyse()["runs"]
config.ALLOW_UNREVIEWED = False

# ---------------------------------------------------------------- stage 2
print("\n" + "=" * 70, f"\nSTAGE 2: simulated visual review (mistake rate {args.error_rate:.0%})\n" + "=" * 70)
log = pd.concat([simulated_review.apply(pid, error_rate=args.error_rate) for pid in pids], ignore_index=True)
log.to_csv(config.DERIVED_DIR / "simulated_review_log.csv", index=False)
print("\nDecisions made:")
print(pd.crosstab(log["signal"], log["decision"]).to_string())
changes = log[log["decision"] != "accept"]
print(f"\nExamples of corrections ({len(changes)} in total):")
print(changes[["participant_id", "signal", "type", "time_s", "decision", "notes"]].head(12).round(2).to_string(index=False))
mistakes = log[log["simulated_mistake"]]
print(f"\nSimulated reviewer mistakes (overlooked corrections): {len(mistakes)}")
if len(mistakes):
    print(mistakes[["participant_id", "signal", "time_s", "correct_decision"]].round(2).to_string(index=False))

# ---------------------------------------------------------------- what the automatic flags missed
missed_rows = []
for pid in pids:
    arts = pd.read_csv(config.DATA_DIR / "_ground_truth" / f"{pid}_injected_artifacts.csv")
    d = pipeline.participant_dir(pid) / "review"
    fe = pd.read_csv(d / "flags_ecg.csv", keep_default_na=False)
    fa = pd.read_csv(d / "flags_eda.csv", keep_default_na=False)
    segs = pd.read_csv(pipeline.participant_dir(pid) / "segments.csv")
    for _, a in arts.iterrows():
        if a["type"] in ("missed_beat", "genuine_large_scr"):
            continue
        in_seg = ((segs["start_s"] <= a["end_s"]) & (segs["end_s"] >= a["start_s"])).any()
        if not in_seg:
            continue
        if a["signal"] == "ecg":
            caught = ((fe["time_s"].astype(float) >= a["start_s"] - 1.5) & (fe["time_s"].astype(float) <= a["end_s"] + 1.5)).any()
        else:
            caught = ((fa["time_s"].astype(float) <= a["end_s"]) & (fa["end_s"].astype(float) >= a["start_s"])).any()
        if not caught:
            missed_rows.append({"participant_id": pid, "signal": a["signal"], "artifact": a["type"],
                                "start_s": round(a["start_s"], 1), "duration_s": round(a["end_s"] - a["start_s"], 1)})
print("\nInjected artefacts the automatic flags did NOT catch (a reviewer never sees these unless")
print("they scan qc_overview.png):")
print(pd.DataFrame(missed_rows).to_string(index=False) if missed_rows else "  none")

# ---------------------------------------------------------------- stages 3-4
print("\n" + "=" * 70, "\nSTAGES 3-4: measures and statistics\n" + "=" * 70)
pipeline.run_all_compute()
res = pipeline.analyse()
after = res["runs"]

qc = pd.concat([pd.read_csv(p).assign(participant_id=p.parent.name)
                for p in sorted((config.DERIVED_DIR / "participants").glob("*/segment_qc_final.csv"))])
used_eda = qc["segment"].isin(["baseline", "task"])
excl = qc[~qc["ecg_keep"].astype(bool) | (used_eda & ~qc["eda_keep"].astype(bool))]
print("\nSegments excluded:")
if len(excl):
    print(excl[["participant_id", "run_id", "segment", "pct_ecg_bad", "pct_eda_removed", "ecg_keep", "eda_keep",
                "ecg_exclusion_reason", "eda_exclusion_reason"]].fillna("").to_string(index=False))
else:
    print("  none")

# ---------------------------------------------------------------- accuracy against truth
truth = pd.concat([simulated_review.true_rmssd(pid, pd.read_csv(pipeline.participant_dir(pid) / "segments.csv"))
                   for pid in pids])


def errors(table, label):
    rows = []
    for seg in ("baseline", "task", "post"):
        t = table[["participant_id", "run_id", f"rmssd_{seg}"]].rename(columns={f"rmssd_{seg}": "est"})
        m = t.merge(truth[truth["segment"] == seg], on=["participant_id", "run_id"]).dropna(subset=["est"])
        err = (m["est"] - m["true_rmssd"]).abs()
        rows.append({"when": label, "segment": seg, "n_segments": len(m), "mean_abs_error_ms": err.mean(),
                     "worst_error_ms": err.max()})
    return rows


acc = pd.DataFrame(errors(before, "before review (automatic only)") + errors(after, "after review"))
print("\nRMSSD accuracy vs the true heartbeats:")
print(acc.round(2).to_string(index=False))

scr_truth = pd.concat([simulated_review.true_scr_frequency(pid, pd.read_csv(pipeline.participant_dir(pid) / "segments.csv"))
                       for pid in pids])
rows = []
for seg in ("baseline", "task"):
    m = after[["participant_id", "run_id", f"scr_freq_{seg}"]].merge(
        scr_truth[scr_truth["segment"] == seg], on=["participant_id", "run_id"]).dropna()
    rows.append({"segment": seg, "n_segments": len(m), "mean_estimated": m[f"scr_freq_{seg}"].mean(),
                 "mean_true": m["true_scr_freq"].mean(), "r": m[f"scr_freq_{seg}"].corr(m["true_scr_freq"])})
print("\nSCR frequency (per min) vs the true SCRs:")
print(pd.DataFrame(rows).round(2).to_string(index=False))

stress = after[after["run_type"] == "stress"]
print(f"\nRecovery time reached in {stress['rmssd_recovery_time_s'].notna().sum()}/{len(stress)} stress runs "
      f"(RECOVERY_SD_BELOW = {config.RECOVERY_SD_BELOW}).")

cols = ["test", "model", "n_participants", "estimate", "ci_low", "ci_high", "p"]
print("\nStatistics:")
print(res["statistics"][cols].round(4).to_string(index=False))

gt = pd.read_csv(config.DATA_DIR / "ground_truth.csv")
nu = res["table"][["participant_id", "nu"]].merge(gt, on="participant_id")
print(f"\nnu: true vs estimated r = {nu['true_nu'].corr(nu['nu']):.2f} (HGF_LIKELIHOOD_TRIALS = "
      f"'{config.HGF_LIKELIHOOD_TRIALS}')")
print("\nBuilt into the data: stress MIST lowers RMSSD and raises SCL and SCR frequency more than the non-stress")
print("MIST (H1); larger RMSSD drop and slower recovery in patients (H2, H3); higher nu (so more false positives)")
print("in patients. Physiology is NOT directly linked to false positives: any association arises through group.")
print("Nothing is built in for order effects.")
print(f"\nReview log: {config.DERIVED_DIR / 'simulated_review_log.csv'}")
