"""Outline 4: visual check of flagged values and keep / exclude decisions.

Stage 1 writes, per participant, to derived/participants/<pid>/review/:
    flags_ecg.csv           every auto-corrected beat, residual RR flag and flatline span
                            that falls inside an analysis segment
    flags_eda.csv           every EDA artifact span inside a baseline or MIST segment
    segment_decisions.csv   one row per run x segment, with automatic % bad
A flag file is only written for a signal that was recorded. The reviewer fills the
``decision`` / ``new_time_s`` / ``notes`` columns, either by editing the CSVs or with
:class:`ReviewApp` in the notebook. Stage 3 calls :func:`apply_review` to turn the
decisions into final peaks and segment exclusions.

ECG decisions:  accept | add_peak | remove_peak | move_peak | exclude_span
                (new_time_s: the new peak time for add/move; for exclude_span either the
                 end time, or "start-end" to exclude an exact range, e.g. "512.4-518.0")
EDA decisions:  accept (= is an artifact, removed) | not_artifact
Segment decisions: keep | exclude, with a reason. ECG segments where artefacts or
corrected beats cover > ECG_EXCLUDE_PCT (20%) of the duration are always excluded.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .ecg import diff_peaks
from .io_utils import participant_dir

ECG_DECISIONS = ["", "accept", "add_peak", "remove_peak", "move_peak", "exclude_span"]
EDA_DECISIONS = ["", "accept", "not_artifact"]
EDA_SEGMENTS = ("baseline", "task")     # SCL and SCR frequency are not calculated post-task


def review_dir(pid):
    d = participant_dir(pid) / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d


def signals_recorded(pid) -> dict:
    """{"ecg": bool, "eda": bool} as found by stage 1."""
    p = participant_dir(pid) / "stage1.json"
    if not p.exists():
        return {"ecg": False, "eda": False}
    return json.loads(p.read_text())["signals"]


# --------------------------------------------------------------------------
# Stage 1 output
# --------------------------------------------------------------------------
def build_flag_tables(pid, ecg_res, eda_res, segments) -> None:
    """``ecg_res`` / ``eda_res`` are None when that signal was not recorded."""
    d = review_dir(pid)
    if ecg_res is not None:
        rows = []
        for _, c in ecg_res["auto_corrections"].iterrows():
            rows.append({"type": f"auto_{c['action']}", "time_s": c["time_s"], "end_s": np.nan})
        rr = ecg_res["rr"]
        for _, r in rr[rr["artifact_flag"]].iterrows():
            kind = "rr_range" if r["range_flag"] else "rr_jump"
            rows.append({"type": kind, "time_s": r["time_s"], "end_s": np.nan, "rr_ms": r["rr_ms"]})
        for _, f in ecg_res["flatline"].iterrows():
            rows.append({"type": "flatline", "time_s": f["start_s"], "end_s": f["end_s"]})
        ecg_flags = pd.DataFrame(rows, columns=["type", "time_s", "end_s", "rr_ms"])
        ecg_flags = _keep_in_segments(ecg_flags, segments)
        for col in ["decision", "new_time_s", "notes"]:
            ecg_flags[col] = ""
        ecg_flags.insert(0, "flag_id", np.arange(len(ecg_flags)))
        _write_preserving(ecg_flags, d / "flags_ecg.csv")

    if eda_res is not None:
        eda_flags = eda_res["artifact_spans"].rename(columns={"start_s": "time_s"}).assign(type="eda_artifact")
        eda_flags = _keep_in_segments(eda_flags[["type", "time_s", "end_s"]],
                                      segments[segments["segment"].isin(EDA_SEGMENTS)])
        for col in ["decision", "notes"]:
            eda_flags[col] = ""
        eda_flags.insert(0, "flag_id", np.arange(len(eda_flags)))
        _write_preserving(eda_flags, d / "flags_eda.csv")

    seg = segments[["run_id", "segment", "start_s", "end_s"]].copy()
    ecg = None if ecg_res is None else {"peaks": ecg_res["peaks_auto"], "corrections": ecg_res["auto_corrections"],
                                        "bad_spans": ecg_res["flatline"], "fs": ecg_res["fs"]}
    qc = segment_qc(seg, ecg, eda_res)
    qc["ecg_decision"] = ""
    qc["eda_decision"] = ""
    qc["reason"] = ""
    _write_preserving(qc, d / "segment_decisions.csv", key=["run_id", "segment"],
                      keep_cols=["ecg_decision", "eda_decision", "reason"])


def _keep_in_segments(flags, segments):
    if len(flags) == 0:
        flags = flags.copy()
        flags["run_id"] = []
        flags["segment"] = []
        return flags
    end = flags["end_s"].fillna(flags["time_s"])
    keep, runs, segs = [], [], []
    for t0, t1 in zip(flags["time_s"], end):
        hit = segments[(segments["start_s"] <= t1) & (segments["end_s"] >= t0)]
        keep.append(len(hit) > 0)
        runs.append(";".join(hit["run_id"].astype(str)) if len(hit) else None)
        segs.append(";".join(hit["segment"]) if len(hit) else None)
    flags = flags.assign(run_id=runs, segment=segs)[keep].reset_index(drop=True)
    return flags


def _write_preserving(new, path, key=("type", "time_s"), keep_cols=("decision", "new_time_s", "notes")):
    """Re-running stage 1 must not wipe decisions already entered."""
    key = list(key)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        keep_cols = [c for c in keep_cols if c in old.columns and c in new.columns]
        if keep_cols and all(k in old.columns for k in key):
            for k in key:
                if k == "time_s":
                    old[k] = pd.to_numeric(old[k]).round(3)
                    new[k] = new[k].round(3)
                else:
                    old[k] = old[k].astype(str)
                    new[k] = new[k].astype(str)
            new = new.drop(columns=keep_cols).merge(old[key + keep_cols], on=key, how="left")
            for c in keep_cols:
                new[c] = new[c].fillna("")
    new.to_csv(path, index=False)


# --------------------------------------------------------------------------
# QC numbers per segment
# --------------------------------------------------------------------------
def _covered_s(starts, ends, a, b) -> float:
    """Seconds of [a, b] covered by the union of the intervals."""
    s = np.clip(np.asarray(starts, float), a, b)
    e = np.clip(np.asarray(ends, float), a, b)
    keep = e > s
    s, e = s[keep], e[keep]
    if len(s) == 0:
        return 0.0
    order = np.argsort(s)
    total, cur_s, cur_e = 0.0, s[order[0]], e[order[0]]
    for i in order[1:]:
        if s[i] > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s[i], e[i]
        else:
            cur_e = max(cur_e, e[i])
    return float(total + cur_e - cur_s)


def corrected_beat_spans(peaks, corrections, fs) -> pd.DataFrame:
    """The RR intervals touching each corrected beat: from the beat before to the beat after."""
    tp = np.sort(np.asarray(peaks)) / fs
    rows = []
    for t in corrections["time_s"].to_numpy(dtype=float) if len(corrections) else []:
        before, after = tp[tp < t - 1e-3], tp[tp > t + 1e-3]
        rows.append({"start_s": before[-1] if len(before) else t, "end_s": after[0] if len(after) else t})
    return pd.DataFrame(rows, columns=["start_s", "end_s"])


def segment_qc(seg, ecg=None, eda_res=None, eda_ignore=None) -> pd.DataFrame:
    """% of each segment that is unusable.

    ecg: {"peaks", "corrections", "bad_spans", "fs"} or None. pct_ecg_bad = % of the
    duration covered by corrected-beat intervals, signal loss and excluded spans.
    eda_res: stage-1 EDA result or None. pct_eda_removed = % of samples flagged
    (baseline and task only).
    """
    rows = []
    if ecg is not None:
        t_peaks = np.asarray(ecg["peaks"]) / ecg["fs"]
        corr = ecg["corrections"]
        cspans = corrected_beat_spans(ecg["peaks"], corr, ecg["fs"])
        bad = pd.concat([cspans, ecg["bad_spans"][["start_s", "end_s"]]], ignore_index=True)
    if eda_res is not None:
        eda_df = eda_res["eda"]
        eda_t = eda_df["time_s"].to_numpy()
        eda_bad = eda_df["artifact"].to_numpy().copy()
        if eda_ignore is not None:
            for _, s in eda_ignore.iterrows():
                eda_bad[(eda_t >= s["time_s"]) & (eda_t <= s["end_s"])] = False
    for _, s in seg.iterrows():
        a, b = s["start_s"], s["end_s"]
        dur = max(b - a, 1e-9)
        r = {**s.to_dict()}
        if ecg is not None:
            in_seg = (corr["time_s"] >= a) & (corr["time_s"] <= b) if len(corr) else np.array([], bool)
            src = corr.loc[in_seg, "source"] if len(corr) else pd.Series(dtype=str)
            n_beats = int(((t_peaks >= a) & (t_peaks <= b)).sum())
            n_corr = int(in_seg.sum())
            r.update({"n_beats": n_beats, "n_beats_corrected_auto": int((src == "auto").sum()),
                      "n_beats_corrected_manual": int((src == "manual").sum()),
                      "pct_beats_corrected": round(100 * n_corr / n_beats, 2) if n_beats else 100.0,
                      "pct_time_signal_loss": round(100 * _covered_s(ecg["bad_spans"]["start_s"],
                                                                     ecg["bad_spans"]["end_s"], a, b) / dur, 2),
                      "pct_ecg_bad": round(100 * _covered_s(bad["start_s"], bad["end_s"], a, b) / dur, 2)})
        else:
            r.update({"n_beats": np.nan, "pct_beats_corrected": np.nan, "pct_ecg_bad": np.nan})
        if eda_res is not None and s["segment"] in EDA_SEGMENTS:
            m = (eda_t >= a) & (eda_t <= b)
            r["pct_eda_removed"] = round(100 * float(eda_bad[m].mean()), 2) if m.any() else 100.0
        else:
            r["pct_eda_removed"] = np.nan
        rows.append(r)
    if not rows:            # no segments (e.g. no recording): empty table with the usual columns
        return pd.DataFrame(columns=list(seg.columns) + ["n_beats", "pct_beats_corrected", "pct_ecg_bad",
                                                          "pct_eda_removed"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Stage 3 input
# --------------------------------------------------------------------------
def review_status(pid) -> dict:
    """{flag file: (n reviewed, n flags)} for each recorded signal."""
    d = review_dir(pid)
    out = {}
    for sig, on in signals_recorded(pid).items():
        p = d / f"flags_{sig}.csv"
        if on and p.exists():
            df = pd.read_csv(p, keep_default_na=False)
            out[f"flags_{sig}"] = (int((df["decision"].astype(str) != "").sum()), len(df))
    return out


def is_reviewed(pid) -> bool:
    """True when stage 1 has run and every flag of every recorded signal has a decision."""
    if not (participant_dir(pid) / "stage1.json").exists():
        return False
    return all(done == total for done, total in review_status(pid).values())


def apply_review(pid, ecg, eda_res, segments):
    """Apply decisions. ``ecg`` = {"peaks_auto", "auto_corrections", "flatline", "fs"} or None.

    Returns (final_peaks or None, bad_spans, all_corrections, segment_qc_table).
    """
    d = review_dir(pid)
    peaks, bad_spans, all_corr, ecg_qc = None, pd.DataFrame(columns=["start_s", "end_s"]), None, None
    if ecg is not None:
        fs = ecg["fs"]
        peaks_auto = np.asarray(ecg["peaks_auto"], int)
        ecg_flags = pd.read_csv(d / "flags_ecg.csv", keep_default_na=False)
        peaks = np.sort(peaks_auto)
        bad = [ecg["flatline"][["start_s", "end_s"]]]
        tol = int(0.15 * fs)  # 150 ms to find the peak a decision refers to

        for _, f in ecg_flags.iterrows():
            dec = str(f["decision"]).strip()
            t = float(f["time_s"])
            raw_new = str(f["new_time_s"]).strip()
            if dec in ("", "accept"):
                continue
            if dec == "exclude_span" and "-" in raw_new[1:]:      # "start-end"
                a_s, b_s = raw_new.split("-", 1) if not raw_new.startswith("-") else raw_new[1:].split("-", 1)
                bad.append(pd.DataFrame({"start_s": [float(a_s)], "end_s": [float(b_s)]}))
                continue
            new_t = float(raw_new) if raw_new not in ("", "nan") else np.nan
            if dec == "add_peak":
                peaks = np.unique(np.append(peaks, int(round((new_t if np.isfinite(new_t) else t) * fs))))
            elif dec == "remove_peak":
                peaks = _remove_near(peaks, int(round(t * fs)), tol)
            elif dec == "move_peak":
                if not np.isfinite(new_t):
                    raise ValueError(f"{pid}: flag {f['flag_id']} move_peak needs new_time_s")
                peaks = _remove_near(peaks, int(round(t * fs)), tol)
                peaks = np.unique(np.append(peaks, int(round(new_t * fs))))
            elif dec == "exclude_span":
                end = new_t if np.isfinite(new_t) else (float(f["end_s"]) if str(f["end_s"]) not in ("", "nan") else t + 1)
                bad.append(pd.DataFrame({"start_s": [min(t, end)], "end_s": [max(t, end)]}))
            else:
                raise ValueError(f"{pid}: unknown ECG decision '{dec}' (flag {f['flag_id']})")

        bad_spans = pd.concat(bad, ignore_index=True)
        # beats inside excluded spans are not usable
        peaks = peaks[~_inside(peaks / fs, bad_spans)]
        manual = diff_peaks(peaks_auto[~_inside(peaks_auto / fs, bad_spans)], peaks, fs, source="manual")
        all_corr = pd.concat([ecg["auto_corrections"], manual], ignore_index=True)
        ecg_qc = {"peaks": peaks, "corrections": all_corr, "bad_spans": bad_spans, "fs": fs}

    eda_ignore = None
    if eda_res is not None:
        eda_flags = pd.read_csv(d / "flags_eda.csv", keep_default_na=False)
        eda_ignore = eda_flags[eda_flags["decision"].astype(str) == "not_artifact"][["time_s", "end_s"]]
    qc = segment_qc(segments[["run_id", "segment", "start_s", "end_s"]], ecg_qc, eda_res, eda_ignore)

    # merge reviewer segment decisions; automatic rules always apply
    sd_path = d / "segment_decisions.csv"
    sd = pd.read_csv(sd_path, keep_default_na=False) if sd_path.exists() else pd.DataFrame(columns=["run_id", "segment"])
    sd["run_id"] = sd["run_id"].astype(str)
    qc["run_id"] = qc["run_id"].astype(str)
    qc = qc.merge(sd[[c for c in ["run_id", "segment", "ecg_decision", "eda_decision", "reason"] if c in sd]],
                  on=["run_id", "segment"], how="left")
    reason = qc["reason"].fillna("").astype(str) if "reason" in qc else pd.Series([""] * len(qc))
    rules = {"ecg": ("pct_ecg_bad", C.ECG_EXCLUDE_PCT, ecg is not None),
             "eda": ("pct_eda_removed", C.EDA_EXCLUDE_PCT, eda_res is not None)}
    for sig, (col, limit, recorded) in rules.items():
        rev = qc.get(f"{sig}_decision", pd.Series([""] * len(qc))).fillna("").astype(str).str.strip()
        auto_ex = (qc[col] > limit) if limit is not None else pd.Series(False, index=qc.index)
        used = qc["segment"].isin(EDA_SEGMENTS) if sig == "eda" else pd.Series(True, index=qc.index)
        qc[f"{sig}_keep"] = recorded & used & ~(auto_ex | (rev == "exclude"))
        qc[f"{sig}_exclusion_reason"] = np.where(
            ~pd.Series(recorded, index=qc.index), "not recorded",
            np.where(auto_ex, f">{limit}% of segment unusable",
                     np.where(rev == "exclude", reason.where(reason != "", "reviewer"), "")))
    qc.to_csv(participant_dir(pid) / "segment_qc_final.csv", index=False)
    return peaks, bad_spans, all_corr, qc


def _inside(t, spans):
    m = np.zeros(len(t), bool)
    for _, s in spans.iterrows():
        m |= (t >= s["start_s"]) & (t <= s["end_s"])
    return m


def _remove_near(peaks, target, tol):
    if len(peaks) == 0:
        return peaks
    j = np.argmin(np.abs(peaks - target))
    if abs(peaks[j] - target) <= tol:
        return np.delete(peaks, j)
    print(f"  no peak within tolerance of sample {target}; nothing removed")
    return peaks


# --------------------------------------------------------------------------
# Notebook reviewer (adapted from inspo/01_verify_inputs.ipynb)
# --------------------------------------------------------------------------
class ReviewApp:
    """Interactive reviewer for one participant. Use in Jupyter:

        app = ReviewApp("ACHI_2026_003"); app.show()
    """

    def __init__(self, pid, padding_s=4.0):
        from .io_utils import participant_dir as pdir
        self.pid = pid
        self.dir = pdir(pid)
        self.padding_s = padding_s
        self.signals = signals_recorded(pid)
        if self.signals["ecg"]:
            arr = np.load(self.dir / "ecg_arrays.npz")
            self.fs = float(arr["fs"])
            self.ecg_raw, self.ecg_clean, self.peaks = arr["ecg_raw"], arr["ecg_clean"], arr["peaks_auto"]
        if self.signals["eda"]:
            e = np.load(self.dir / "eda_arrays.npz")
            self.eda_t, self.eda_raw, self.eda_clean, self.scl = e["time_s"], e["eda_raw"], e["eda_clean"], e["scl"]
        self.rdir = review_dir(pid)
        self.tables = {k: pd.read_csv(self.rdir / f"flags_{k}.csv", keep_default_na=False)
                       for k in ("ecg", "eda") if self.signals[k]}
        self.segdec = pd.read_csv(self.rdir / "segment_decisions.csv", keep_default_na=False)

    def plot_flag(self, kind, i):
        import matplotlib.pyplot as plt
        f = self.tables[kind].iloc[i]
        t0 = float(f["time_s"])
        t1 = float(f["end_s"]) if str(f["end_s"]) not in ("", "nan") else t0
        a, b = max(0, t0 - self.padding_s), t1 + self.padding_s
        if kind == "ecg":
            sa, sb = int(a * self.fs), min(len(self.ecg_raw), int(b * self.fs))
            t = np.arange(sa, sb) / self.fs
            pk = self.peaks[(self.peaks >= sa) & (self.peaks < sb)]
            fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
            ax[0].plot(t, self.ecg_raw[sa:sb], lw=0.8, color="steelblue")
            ax[0].set_ylabel("Raw ECG")
            ax[1].plot(t, self.ecg_clean[sa:sb], lw=0.8, color="black")
            ax[1].scatter(pk / self.fs, self.ecg_clean[pk], color="red", s=25, zorder=3, label="peaks (auto-corrected)")
            for x in ax:
                x.axvspan(t0 - 0.02, t1 + 0.02, color="orange", alpha=0.25)
            ax[1].set_ylabel("Clean ECG")
            rr = np.diff(pk) / self.fs * 1000
            for p, r in zip(pk[1:], rr):
                ax[1].annotate(f"{r:.0f}", (p / self.fs, ax[1].get_ylim()[1]), fontsize=7, ha="right", va="top")
            ax[1].legend(loc="lower right")
        else:
            m = (self.eda_t >= a - 20) & (self.eda_t <= b + 20)
            fig, ax = plt.subplots(1, 1, figsize=(15, 4))
            ax.plot(self.eda_t[m], self.eda_raw[m], lw=0.8, alpha=0.6, label="raw")
            ax.plot(self.eda_t[m], self.eda_clean[m], lw=1, color="black", label="1 Hz low-pass")
            ax.plot(self.eda_t[m], self.scl[m], lw=1, color="green", label="SCL")
            ax.axvspan(t0, t1, color="orange", alpha=0.25, label="flagged")
            ax.set_ylabel("µS")
            ax.legend()
        fig.suptitle(f"{self.pid} | {kind.upper()} flag {i + 1}/{len(self.tables[kind])} | {f['type']} | "
                     f"run {f['run_id']} {f['segment']} | t = {t0:.3f} s")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    def show(self, kind="ecg"):
        import ipywidgets as w
        from IPython.display import display, clear_output
        if kind not in self.tables:
            print(f"No {kind.upper()} recorded for {self.pid}.")
            return
        tab = self.tables[kind]
        if len(tab) == 0:
            print(f"No {kind.upper()} flags for {self.pid}.")
            return
        opts = ECG_DECISIONS if kind == "ecg" else EDA_DECISIONS
        slider = w.IntSlider(0, 0, len(tab) - 1, description="Flag", continuous_update=False)
        dec = w.Dropdown(options=opts, description="Decision")
        new_t = w.Text(description="new time s", placeholder="add/move: peak time | exclude_span: end, or start-end")
        notes = w.Text(description="Notes")
        save = w.Button(description="Save & next", button_style="success")
        prev, nxt = w.Button(description="Previous"), w.Button(description="Next")
        out, status = w.Output(), w.Output()

        def refresh(_=None):
            r = tab.iloc[slider.value]
            dec.value = r["decision"] if r["decision"] in opts else ""
            if kind == "ecg":
                new_t.value = str(r["new_time_s"])
            notes.value = str(r["notes"])
            with out:
                clear_output(wait=True)
                self.plot_flag(kind, slider.value)

        def do_save(_=None):
            i = slider.value
            with status:
                clear_output()
                if dec.value == "":
                    print("Choose a decision first.")
                    return
                tab.loc[i, "decision"] = dec.value
                if kind == "ecg":
                    tab.loc[i, "new_time_s"] = new_t.value
                tab.loc[i, "notes"] = notes.value
                tab.to_csv(self.rdir / f"flags_{kind}.csv", index=False)
                done = (tab["decision"].astype(str) != "").sum()
                print(f"Saved flag {i + 1}: {dec.value}   ({done}/{len(tab)} reviewed)")
            if i < slider.max:
                slider.value += 1

        slider.observe(refresh, names="value")
        save.on_click(do_save)
        prev.on_click(lambda _: setattr(slider, "value", max(0, slider.value - 1)))
        nxt.on_click(lambda _: setattr(slider, "value", min(slider.max, slider.value + 1)))
        items = [slider, out, dec] + ([new_t] if kind == "ecg" else []) + [notes, w.HBox([prev, save, nxt]), status]
        display(w.VBox(items))
        refresh()

    def accept_all_remaining(self, kind):
        """Mark every un-reviewed flag as 'accept' (use only after scanning them)."""
        tab = self.tables[kind]
        tab.loc[tab["decision"].astype(str) == "", "decision"] = "accept"
        tab.to_csv(self.rdir / f"flags_{kind}.csv", index=False)

    def segment_table(self):
        return self.segdec

    def set_segment_decision(self, run_id, segment, signal, decision, reason):
        """signal: 'ecg' or 'eda'; decision: 'keep' or 'exclude'."""
        m = (self.segdec["run_id"].astype(str) == str(run_id)) & (self.segdec["segment"] == segment)
        if not m.any():
            raise KeyError(f"No segment {run_id}/{segment}")
        self.segdec.loc[m, f"{signal}_decision"] = decision
        self.segdec.loc[m, "reason"] = reason
        self.segdec.to_csv(self.rdir / "segment_decisions.csv", index=False)
