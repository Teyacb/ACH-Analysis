"""Stage drivers.

    stage 1  preprocess(pid)      load, clean ECG/EDA, cut segments, write review files
    stage 2  review.ReviewApp     visual check (notebook)
    stage 3  compute(pid)         apply review, RMSSD/SCL/SCR, change scores, FP rates, nu
    stage 4  analyse()            participant table, run table, statistics, descriptives

``run_all_preprocessing()`` and ``run_all_compute()`` loop over every participant
and log failures instead of stopping.
"""
from __future__ import annotations

import json
import traceback

import numpy as np
import pandas as pd

from . import config as C
from . import ecg as ecg_mod
from . import eda as eda_mod
from . import hgf, measures, review, stats, triggers
from .io_utils import (check_inputs, discover_participants, load_participants_table, load_recording,
                       load_session_json, load_trials, log_problem, missing_trial_blocks, participant_dir,
                       session_events_table, session_runs)

# Outline step 2: the participant table (one row per participant), defined before processing.
TABLE_COLUMNS = [
    "participant_id", "group", "age", "antipsychotic",
    "rmssd_baseline", "rmssd_stress", "rmssd_nonstress", "rmssd_post",
    "scl_baseline", "scl_stress", "scl_nonstress",
    "scr_freq_baseline", "scr_freq_stress", "scr_freq_nonstress",
    "phasic_area_baseline", "phasic_area_stress", "phasic_area_nonstress",
    "delta_rmssd_stress", "delta_scl_stress", "delta_rmssd", "delta_scl",
    "rmssd_recovery_time_s", "n_stress_runs_not_recovered", "nu", "nu_source",
    "fp_rate_neg", "fp_rate_pos", "fp_rate_neu", "fp_rate_all",
    "n_yes_neg", "n_silent_neg", "n_yes_pos", "n_silent_pos", "n_yes_neu", "n_silent_neu", "n_yes_all", "n_silent_all",
    "pct_excluded_ecg", "pct_removed_eda", "n_stress_runs", "n_nonstress_runs", "n_unknown_runs", "reviewed",
]
# The run table (one row per run), used for the order comparison.
RUN_COLUMNS = [
    "participant_id", "run_id", "run_type", "manipulation_valence", "run_number", "mist_number", "has_mist", "group",
    "mist_correct", "mist_incorrect", "mist_timed_out", "timeout_check",
    "rmssd_baseline", "rmssd_task", "rmssd_post", "scl_baseline", "scl_task",
    "scr_freq_baseline", "scr_freq_task", "phasic_area_baseline", "phasic_area_task", "n_scr_baseline", "n_scr_task",
    "eda_clean_min_baseline", "eda_clean_min_task",
    "delta_rmssd", "delta_scl", "rmssd_recovery_time_s", "recovery_reached", "recovery_threshold_ms",
    "fp_rate_neg", "fp_rate_pos", "fp_rate_neu", "fp_rate_all",
    "n_yes_neg", "n_silent_neg", "n_yes_pos", "n_silent_pos", "n_yes_neu", "n_silent_neu", "n_yes_all", "n_silent_all",
    "pct_ecg_bad_baseline", "pct_ecg_bad_task", "pct_ecg_bad_post",
    "ecg_keep_baseline", "ecg_keep_task", "ecg_keep_post",
    "pct_eda_removed_baseline", "pct_eda_removed_task", "eda_keep_baseline", "eda_keep_task",
    "duration_s_baseline", "duration_s_task", "duration_s_post", "reviewed",
]
ECG_LOG_COLUMNS = ["participant_id", "run_id", "segment", "start_s", "end_s", "n_beats", "n_beats_corrected_auto",
                   "n_beats_corrected_manual", "pct_beats_corrected", "pct_time_signal_loss", "pct_ecg_bad",
                   "ecg_keep", "ecg_exclusion_reason"]


def empty_participant_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_COLUMNS)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def inspect_participant(pid: str) -> None:
    """Print the files found, the session log per run, run types, trials and triggers."""
    files = discover_participants().get(pid)
    if files is None:
        print(f"{pid}: no files found under {C.DATA_DIR}")
        return
    for k in ("acq", "json"):
        print(f"{k:5s}:", [p.name for p in files[k]])
    if files["json"]:
        s = load_session_json(files["json"][0])
        s.attrs["participant_id"] = pid
        runs = session_runs(s, pid)
        print("\nSession events per run (duration s):")
        for run_id in runs["run_id"]:
            g = s[s["runId"] == run_id]
            print(f"  {run_id}: " + ", ".join(f"{n} ({d:.0f})" for n, d in
                                              zip(g["eventName"], g["end_epoch_s"] - g["start_epoch_s"])))
        print("\nRuns (run_type from config.RUN_TYPE_RULE / run_types.csv):")
        print(runs.to_string(index=False))
        miss = missing_trial_blocks(s, runs)
        print("\nMissing Trial-TT blocks:", miss or "none")
        t = load_trials(s, list(runs["run_id"]))
        print(f"\nTrials: {len(t)}; per run / valence (silent trials, 'yes' on silent, no response):")
        print(t.groupby(["run_id", "valence"], sort=False).agg(
            n=("trial", "size"), silent=("silent", "sum"),
            yes_on_silent=("response_yes", lambda x: int(x[t.loc[x.index, "silent"]].eq(1).sum())),
            no_response=("response_yes", lambda x: int(x.isna().sum()))).to_string())
    if files["acq"]:
        rec, fs = load_recording(files["acq"][0])
        print(f"\nRecording: {len(rec) / fs / 60:.1f} min at {fs:g} Hz, columns {list(rec.columns)}")
        ev = triggers.decode_events(rec, fs)
        print(ev["event_label"].value_counts().to_string())
        bl = triggers.trigger_blocks(ev)
        if files["json"]:
            bl, al = triggers.align_session_to_blocks(s, bl)
            print("\nAlignment:", al)
            print(triggers.build_segments(pid, s, bl, al, runs, len(rec) / fs).to_string(index=False))
        print(bl.to_string(index=False))


def _save_df(df, pid, name):
    df.to_csv(participant_dir(pid) / name, index=False)


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------
def preprocess(pid: str, files: dict | None = None) -> bool:
    """Missing ECG skips the ECG steps, missing EDA the EDA steps (both logged).
    Without the session log the participant is skipped."""
    files = files or discover_participants().get(pid)
    if files is None:
        log_problem(pid, "-", "participant folder not found")
        return False
    if not check_inputs(pid, files):
        return False
    out = participant_dir(pid)
    out.mkdir(parents=True, exist_ok=True)
    json_name = files["json"][0].name
    session = load_session_json(files["json"][0])
    session.attrs["participant_id"] = pid
    runs = session_runs(session, pid)
    _save_df(runs, pid, "runs.csv")
    _save_df(session_events_table(session), pid, "session_events.csv")
    for _, run in runs.iterrows():
        if run["run_type"] not in ("stress", "non-stress", "none"):
            log_problem(pid, json_name, f"{run['run_id']}: run type '{run['run_type']}' (manipulationValence "
                                        f"'{run['manipulation_valence']}'); not used as stress or non-stress")
        elif run["timeout_check"] == "disagrees":
            log_problem(pid, json_name, f"{run['run_id']}: run type '{run['run_type']}' but {run['mist_timed_out']:g} "
                                        "timed-out MIST questions (only the stress MIST has a time limit); check")
    for block in missing_trial_blocks(session, runs):
        log_problem(pid, json_name, f"{block} not in the session log; those trials are missing")
    trials = load_trials(session, list(runs["run_id"]))
    _save_df(trials, pid, "trials.csv")
    if trials.empty:
        log_problem(pid, json_name, "no Trial-TT trials; false-positive rates and nu will be missing")
    elif (trials["valence"] == "unknown").any():
        log_problem(pid, json_name, "some Trial-TT blocks have no /neutral|negative|positive/ in the audio file "
                                    "path; their valence is 'unknown'")

    signals = {"ecg": False, "eda": False}
    segments = pd.DataFrame(columns=["participant_id", "run_id", "segment", "event_name", "start_s", "end_s",
                                     "source", "duration_s"])
    rec = None
    if files["acq"]:
        print(f"[{pid}] loading {files['acq'][0].name}")
        rec, fs = load_recording(files["acq"][0])
        if not any(c in rec for c in ("m0", "m1", "m2")) and not C.CLOCK_OFFSETS_FILE.exists():
            log_problem(pid, files["acq"][0].name, "digital trigger channels not found and no clock_offsets.csv; "
                                                   "ECG and EDA steps skipped")
            rec = None
    if rec is not None:
        for ch in ("ECG", "EDA"):
            signals[ch.lower()] = ch in rec
            if ch not in rec:
                log_problem(pid, files["acq"][0].name, f"{ch} channel not found; {ch} steps skipped")
        events = triggers.decode_events(rec, fs)
        blocks = triggers.trigger_blocks(events)
        blocks, align = triggers.align_session_to_blocks(session, blocks)
        segments = triggers.build_segments(pid, session, blocks, align, runs, len(rec) / fs)
        _save_df(events, pid, "events_decoded.csv")
        _save_df(blocks, pid, "trigger_blocks_aligned.csv")
        (out / "alignment.json").write_text(json.dumps(align, indent=2, default=float))
        print(f"[{pid}] alignment: {align.get('n_matched')}/{align.get('n_blocks')} trigger blocks matched "
              f"(offset {align.get('offset_s')}, {align.get('method')})")
        if not np.isfinite(align.get("offset_s", np.nan)):
            log_problem(pid, files["acq"][0].name, "trigger blocks could not be matched to the session log and "
                                                   "no clock_offsets.csv entry; no segments, ECG/EDA measures missing")
        if "baseline" not in set(segments["segment"]):
            log_problem(pid, json_name, f"{C.BASELINE_EVENT} segment not found; baseline skipped")
        for _, run in runs[runs["has_mist"].astype(bool)].iterrows():
            have = set(segments.loc[segments["run_id"].astype(str) == str(run["run_id"]), "segment"])
            for seg in ("task", "post"):
                if seg not in have and (seg == "task" or C.POST_TASK_S):
                    log_problem(pid, json_name, f"{run['run_id']}: {seg} segment not found; skipped")
    _save_df(segments, pid, "segments.csv")
    (out / "stage1.json").write_text(json.dumps({"signals": signals}))

    ecg_res = eda_res = None
    if signals["ecg"]:
        print(f"[{pid}] ECG")
        ecg_raw = rec["ECG"].to_numpy(dtype=float)
        ecg_res = ecg_mod.process_ecg(ecg_raw, fs)
        ecg_res["fs"] = fs
        np.savez_compressed(out / "ecg_arrays.npz", ecg_raw=ecg_raw.astype(np.float32), ecg_clean=ecg_res["ecg_clean"],
                            peaks_initial=ecg_res["peaks_initial"], peaks_auto=ecg_res["peaks_auto"], fs=fs)
        _save_df(ecg_res["auto_corrections"], pid, "ecg_auto_corrections.csv")
        _save_df(ecg_res["flatline"], pid, "ecg_flatline_spans.csv")
        _save_df(ecg_res["rr"], pid, "rr_intervals_auto.csv")
    if signals["eda"]:
        print(f"[{pid}] EDA")
        eda_res = eda_mod.process_eda(rec["EDA"].to_numpy(dtype=float), fs)
        e = eda_res["eda"]
        np.savez_compressed(out / "eda_arrays.npz", time_s=e["time_s"].to_numpy(), eda_raw=e["eda_raw"].to_numpy(),
                            eda_clean=e["eda_clean"].to_numpy(), scl=e["scl"].to_numpy(), scr=e["scr"].to_numpy(),
                            artifact=e["artifact"].to_numpy(), fs=eda_res["fs"])
        _save_df(eda_res["artifact_spans"], pid, "eda_artifact_spans.csv")

    review.build_flag_tables(pid, ecg_res, eda_res, segments)
    if ecg_res is not None or eda_res is not None:
        qc_plot(pid, ecg_res, eda_res, segments)
    st = review.review_status(pid)
    print(f"[{pid}] done. Flags to review: " + (", ".join(f"{k[6:].upper()} {v[1]}" for k, v in st.items()) or "none"))
    return True


def qc_plot(pid, ecg_res, eda_res, segments):
    # Draw on a standalone Figure (no pyplot), so saving this file never changes the
    # notebook's plotting backend; switching to "Agg" here hid the Stage 2 review plots.
    from matplotlib.figure import Figure
    panels = [p for p, res in (("ecg", ecg_res), ("eda", eda_res)) if res is not None]
    fig = Figure(figsize=(16, 3 * len(panels)))
    ax = fig.subplots(len(panels), 1, sharex=True, squeeze=False)
    ax = dict(zip(panels, ax[:, 0]))
    if ecg_res is not None:
        rr = ecg_res["rr"]
        ax["ecg"].plot(rr["time_s"], rr["rr_ms"].where(~rr["spans_bad"]).clip(upper=2500), lw=0.6)
        ax["ecg"].scatter(rr.loc[rr["artifact_flag"], "time_s"], rr.loc[rr["artifact_flag"], "rr_ms"].clip(upper=2500),
                          color="red", s=10, label="RR flag")
        ax["ecg"].set_ylabel("RR (ms)")
        for _, f in ecg_res["flatline"].iterrows():
            ax["ecg"].axvspan(f["start_s"], f["end_s"], color="black", alpha=0.3)
    if eda_res is not None:
        e = eda_res["eda"]
        ax["eda"].plot(e["time_s"], e["eda_clean"], lw=0.6, label="EDA (1 Hz LP)")
        ax["eda"].plot(e["time_s"], e["scl"], lw=0.8, label="SCL")
        ax["eda"].set_ylabel("µS")
    colors = {"baseline": "tab:green", "task": "tab:red", "post": "tab:blue"}
    top = ax[panels[0]]
    for _, s in segments.iterrows():
        for a in ax.values():
            a.axvspan(s["start_s"], s["end_s"], color=colors.get(s["segment"], "grey"), alpha=0.12)
        top.text(s["start_s"], top.get_ylim()[1], f"{s['run_id']}\n{s['segment']}", fontsize=7, va="top")
    for a in ax.values():
        a.legend(loc="upper right")
    ax[panels[-1]].set_xlabel("Time (s)")
    fig.suptitle(f"{pid}: overview (green = baseline, red = MIST, blue = post, black = ECG signal loss)")
    fig.tight_layout()
    fig.savefig(participant_dir(pid) / "qc_overview.png", dpi=110)


# --------------------------------------------------------------------------
# Stage 3
# --------------------------------------------------------------------------
def _load_nu_overrides() -> dict:
    if C.NU_ESTIMATES_FILE.exists():
        d = pd.read_csv(C.NU_ESTIMATES_FILE, dtype={"participant_id": str})
        return dict(zip(d["participant_id"], d["nu"]))
    return {}


def compute(pid: str, files: dict | None = None, demographics: pd.DataFrame | None = None) -> dict | None:
    """Returns {"runs": run-table rows, "participant": one-row participant table}."""
    out = participant_dir(pid)
    if not (out / "stage1.json").exists():
        log_problem(pid, "-", "stage 1 output missing; run preprocess first")
        return None
    reviewed = review.is_reviewed(pid)
    if not reviewed and not C.ALLOW_UNREVIEWED:
        st = review.review_status(pid)
        print(f"[{pid}] not fully reviewed {st}; skipped (set config.ALLOW_UNREVIEWED = True for a provisional run)")
        return None
    files = files or discover_participants().get(pid, {})
    signals = review.signals_recorded(pid)
    segments = pd.read_csv(out / "segments.csv", dtype={"run_id": str})
    runs = pd.read_csv(out / "runs.csv", dtype={"run_id": str})

    ecg = eda = None
    if signals["ecg"]:
        arr = np.load(out / "ecg_arrays.npz")
        ecg = {"peaks_auto": arr["peaks_auto"], "fs": float(arr["fs"]),
               "auto_corrections": pd.read_csv(out / "ecg_auto_corrections.csv"),
               "flatline": pd.read_csv(out / "ecg_flatline_spans.csv")}
    if signals["eda"]:
        e = np.load(out / "eda_arrays.npz")
        eda = {"eda": pd.DataFrame({k: e[k] for k in ("time_s", "eda_raw", "eda_clean", "scl", "scr", "artifact")}),
               "fs": float(e["fs"])}

    peaks, bad_spans, all_corr, qc = review.apply_review(pid, ecg, eda, segments)
    ecg_in = None
    if ecg is not None:
        pd.DataFrame({"sample": peaks, "time_s": peaks / ecg["fs"]}).to_csv(out / "rpeaks_final.csv", index=False)
        all_corr.to_csv(out / "ecg_all_corrections.csv", index=False)
        ecg_in = {"peaks": peaks, "bad_spans": bad_spans, "fs": ecg["fs"]}
    if eda is not None:
        flags = pd.read_csv(review.review_dir(pid) / "flags_eda.csv", keep_default_na=False)
        eda["ignore"] = flags[flags["decision"].astype(str) == "not_artifact"][["time_s", "end_s"]]

    rows = measures.run_rows(pid, runs, segments, qc, ecg_in, eda)
    person = measures.participant_row(pid, rows)

    # behaviour (from the session JSON): FP rates per run and per participant, one nu per participant
    try:
        trials = pd.read_csv(out / "trials.csv", dtype={"run_id": str})
        if trials.empty:
            raise ValueError("no Trial-TT trials in the session log")
        trials["silent"] = trials["silent"].astype(str).str.lower().eq("true")
        fp = measures.fp_rates_by_run(pid, trials)
        rows = rows.merge(fp.drop(columns="participant_id"), on="run_id", how="left")
        nu_over = _load_nu_overrides()
        if pid in nu_over:
            fit = {"nu": nu_over[pid], "nu_source": "nu_estimates.csv"}
        else:
            fit = {**hgf.fit_participant(trials, run_order=list(runs["run_id"])), "nu_source": "python_hgf"}
        person.update(measures.fp_rates(trials))
        person.update({"nu": fit["nu"], "nu_source": fit["nu_source"]})
        pd.DataFrame([{"participant_id": pid, **fit}]).to_csv(out / "hgf_fit.csv", index=False)
    except Exception as err:
        log_problem(pid, "trials.csv (from session log)", f"behaviour failed: {err}")

    demographics = demographics if demographics is not None else load_participants_table()
    demo = demographics[demographics["participant_id"] == pid]
    for c in ("group", "age", "antipsychotic"):
        person[c] = demo[c].iloc[0] if len(demo) else np.nan
    if not len(demo):
        log_problem(pid, C.PARTICIPANTS_FILE.name, "participant not listed; group/age/antipsychotic missing")
    rows["group"] = person["group"]
    rows["reviewed"] = person["reviewed"] = reviewed

    rows = rows.reindex(columns=RUN_COLUMNS + [c for c in rows.columns if c not in RUN_COLUMNS])
    person = pd.DataFrame([person]).reindex(columns=TABLE_COLUMNS)
    rows.to_csv(out / "run_rows.csv", index=False)
    person.to_csv(out / "participant_row.csv", index=False)
    return {"runs": rows, "participant": person}


# --------------------------------------------------------------------------
# Loops and stage 4
# --------------------------------------------------------------------------
def _loop(func, pids=None):
    found = discover_participants()
    pids = pids or list(found)
    results = {}
    for pid in pids:
        try:
            results[pid] = func(pid, found.get(pid))
        except Exception as err:
            log_problem(pid, "-", f"{func.__name__} crashed: {err!r}")
            traceback.print_exc()
            results[pid] = None
    return results


def run_all_preprocessing(pids=None):
    return _loop(preprocess, pids)


def run_all_compute(pids=None):
    demo = load_participants_table()
    return _loop(lambda pid, f: compute(pid, f, demo), pids)


def _collect(name, columns):
    frames = [pd.read_csv(p, dtype={"run_id": str, "participant_id": str})
              for p in sorted((C.DERIVED_DIR / "participants").glob(f"*/{name}"))]
    frames = [f for f in frames if len(f)]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return table.reindex(columns=columns + [c for c in table.columns if c not in columns])


def _numeric(table, skip):
    cols = [c for c in table.columns if c not in skip]
    table[cols] = table[cols].apply(pd.to_numeric, errors="coerce")
    return table


def analyse() -> dict:
    table = _collect("participant_row.csv", TABLE_COLUMNS)
    runs = _collect("run_rows.csv", RUN_COLUMNS)
    qc = [pd.read_csv(p, dtype={"run_id": str}).assign(participant_id=p.parent.name)
          for p in sorted((C.DERIVED_DIR / "participants").glob("*/segment_qc_final.csv"))]
    ecg_log = pd.concat(qc, ignore_index=True).reindex(columns=ECG_LOG_COLUMNS) if qc else \
        pd.DataFrame(columns=ECG_LOG_COLUMNS)
    res_dir = C.DERIVED_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (C.DERIVED_DIR / "logs").mkdir(parents=True, exist_ok=True)
    ecg_log.to_csv(C.DERIVED_DIR / "logs" / "ecg_corrected_beats.csv", index=False)
    if table.empty:
        print("No participant rows found. Run stage 3 (compute) first; participants are skipped until "
              "every flag is reviewed (or set config.ALLOW_UNREVIEWED = True).")
    elif not table["reviewed"].astype(str).str.lower().eq("true").all():
        print("WARNING: table contains unreviewed participants (reviewed = False). Results are provisional.")
    text = {"participant_id", "group", "antipsychotic", "nu_source", "reviewed", "run_id", "run_type",
            "recovery_reached", "has_mist"} | {c for c in runs.columns if "_keep_" in c}
    table = _numeric(table, text)
    runs = _numeric(runs, text)
    table.to_csv(res_dir / "participant_table.csv", index=False)
    runs.to_csv(res_dir / "run_table.csv", index=False)
    # Trial table (one row per trial, all participants) so trial-level analyses remain possible.
    tr = [pd.read_csv(p, dtype={"run_id": str}).assign(participant_id=p.parent.name)
          for p in sorted((C.DERIVED_DIR / "participants").glob("*/trials.csv"))]
    trials = pd.concat([f for f in tr if len(f)], ignore_index=True) if any(len(f) for f in tr) else \
        pd.DataFrame(columns=["participant_id", "run_id"])
    trials = trials.merge(runs[["participant_id", "run_id", "run_number", "run_type", "group"]],
                          on=["participant_id", "run_id"], how="left")
    trials.insert(0, "participant_id", trials.pop("participant_id"))
    trials.to_csv(res_dir / "trial_table.csv", index=False)
    results = stats.run_all(table, runs)
    desc = stats.descriptives(table)
    desc_scr = stats.descriptives_scr(runs)
    supp = stats.supplementary(table)
    results.to_csv(res_dir / "statistics.csv", index=False)
    desc.to_csv(res_dir / "descriptive_summary.csv", index=False)
    desc_scr.to_csv(res_dir / "descriptive_scr_by_run_type.csv", index=False)
    supp.to_csv(res_dir / "supplementary_demographics.csv", index=False)
    print(f"Saved results to {res_dir}")
    return {"table": table, "runs": runs, "statistics": results, "descriptives": desc,
            "descriptives_scr": desc_scr, "supplementary": supp, "ecg_log": ecg_log}
