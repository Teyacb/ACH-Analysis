"""Outline 1 (match files to session log) and 3.4 (cut segments).

Digital lines m0/m1/m2 are bits 1/2/4 of an event code (see config.EVENT_CODE_MAP).
Trigger *blocks* (blockStart..blockEnd) carry no task name, so each block is matched
to a session-JSON event by its start AND end time: the constant offset between the
JSON clock (epoch seconds) and the recording clock (seconds from the start of the
recording) that pairs the most blocks with events is found, then every segment is
placed with it. Without usable triggers, an offset can be given in clock_offsets.csv.

Segments (one row each):
    baseline  the session's single RestBaseline (stored under the run it was logged in)
    task      per MIST run: start of the first to end of the last config.TASK_EVENTS event
    post      per MIST run: POST_TASK_S seconds after config.POST_TASK_AFTER_EVENT
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def decode_events(rec: pd.DataFrame, fs: float) -> pd.DataFrame:
    bits = [rec[c].to_numpy(dtype=np.uint8) if c in rec else np.zeros(len(rec), np.uint8)
            for c in ("m0", "m1", "m2")]
    code = bits[0] + 2 * bits[1] + 4 * bits[2]
    change = np.flatnonzero(code[1:] != code[:-1]) + 1
    ev = pd.DataFrame({"sample": change, "time_s": change / fs, "code": code[change]})
    ev = ev[ev["code"] != 0].reset_index(drop=True)
    ev["event_label"] = ev["code"].map(C.EVENT_CODE_MAP)
    return ev


def trigger_blocks(events: pd.DataFrame) -> pd.DataFrame:
    """Pair each blockStart with the next blockEnd."""
    starts = events.loc[events["event_label"] == "blockStart", "time_s"].to_numpy()
    ends = events.loc[events["event_label"] == "blockEnd", "time_s"].to_numpy()
    rows = []
    for i, s in enumerate(starts):
        nxt_end = ends[ends > s]
        nxt_start = starts[i + 1] if i + 1 < len(starts) else np.inf
        if len(nxt_end) and nxt_end[0] < nxt_start:
            rows.append({"block": len(rows), "start_s": s, "end_s": nxt_end[0]})
        else:
            rows.append({"block": len(rows), "start_s": s, "end_s": np.nan})
    return pd.DataFrame(rows, columns=["block", "start_s", "end_s"])


def align_session_to_blocks(session: pd.DataFrame, blocks: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Assign a session event to every trigger block.

    A block matches an event when both its start and end lie within ALIGN_TOLERANCE_S of
    the event's start and end shifted by one common offset (recording s - epoch s).
    Returns the blocks with event_name / run_id columns, and a dict describing the fit.
    """
    blocks = blocks.dropna(subset=["end_s"]).reset_index(drop=True).copy()
    ev = session[(session["time_source"] == "data") & (session["end_epoch_s"] - session["start_epoch_s"] > 0.5)]
    js, je = ev["start_epoch_s"].to_numpy(), ev["end_epoch_s"].to_numpy()
    tol = C.ALIGN_TOLERANCE_S
    best = {"n_matched": 0, "offset_s": np.nan, "method": "triggers", "n_blocks": len(blocks),
            "n_timed_events": len(ev)}
    blocks["event_name"], blocks["run_id"], blocks["json_start_epoch_s"] = None, None, np.nan
    if len(blocks) and len(ev):
        bs, be = blocks["start_s"].to_numpy(), blocks["end_s"].to_numpy()
        same_len = np.abs((be - bs)[:, None] - (je - js)[None, :]) <= 2 * tol
        cands = np.unique(np.round((bs[:, None] - js[None, :])[same_len], 1))
        best_n, best_off = 0, np.nan
        for off in cands:
            ok = (np.abs(bs[:, None] - js[None, :] - off) <= tol) & (np.abs(be[:, None] - je[None, :] - off) <= tol)
            n = int(ok.any(axis=1).sum())
            if n > best_n:
                best_n, best_off = n, off
        if best_n:
            err = np.abs(bs[:, None] - js[None, :] - best_off) + np.abs(be[:, None] - je[None, :] - best_off)
            err[(np.abs(bs[:, None] - js[None, :] - best_off) > tol) | (np.abs(be[:, None] - je[None, :] - best_off) > tol)] = np.inf
            used_b, used_e, pairs = set(), set(), []
            for flat in np.argsort(err, axis=None):        # greedy: closest pairs first, one each
                i, j = np.unravel_index(flat, err.shape)
                if not np.isfinite(err[i, j]):
                    break
                if i in used_b or j in used_e:
                    continue
                pairs.append((i, j))
                used_b.add(i)
                used_e.add(j)
            bi, ej = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            off = float(np.median(np.r_[bs[bi] - js[ej], be[bi] - je[ej]]))
            blocks.loc[bi, "event_name"] = ev["eventName"].to_numpy()[ej]
            blocks.loc[bi, "run_id"] = ev["runId"].to_numpy()[ej]
            blocks.loc[bi, "json_start_epoch_s"] = js[ej]
            resid = np.r_[bs[bi] - js[ej] - off, be[bi] - je[ej] - off]
            best.update({"n_matched": len(pairs), "offset_s": off,
                         "median_abs_residual_s": float(np.median(np.abs(resid)))})
    manual = _manual_offset(session.attrs.get("participant_id"))
    if manual is not None:
        best.update({"offset_s": manual, "method": "clock_offsets.csv"})
    return blocks, best


def _manual_offset(pid):
    if pid is None or not C.CLOCK_OFFSETS_FILE.exists():
        return None
    d = pd.read_csv(C.CLOCK_OFFSETS_FILE, dtype={"participant_id": str})
    d = d[d["participant_id"] == pid]
    return float(d["offset_s"].iloc[0]) if len(d) else None


def _event_times(ev: pd.DataFrame, blocks: pd.DataFrame, off: float) -> tuple[float, float, str]:
    """Recording-clock start / end of the session events in ``ev`` (first start, last end).
    Uses matched trigger blocks where available, else the JSON time + offset."""
    starts, ends, src = [], [], set()
    for _, e in ev.iterrows():
        b = blocks[(blocks["run_id"] == e["runId"]) & (blocks["event_name"] == e["eventName"])
                   & np.isclose(blocks["json_start_epoch_s"].astype(float), e["start_epoch_s"])]
        if len(b):
            starts.append(float(b["start_s"].iloc[0]))
            ends.append(float(b["end_s"].iloc[0]))
            src.add("trigger")
        else:
            starts.append(e["start_epoch_s"] + off)
            ends.append(e["end_epoch_s"] + off)
            src.add("json")
    return min(starts), max(ends), "+".join(sorted(src))


def build_segments(pid: str, session: pd.DataFrame, blocks: pd.DataFrame, align: dict,
                   runs: pd.DataFrame, rec_duration_s: float) -> pd.DataFrame:
    """One row per (run, segment) with start/end seconds in the recording clock.

    source = "trigger" (matched trigger blocks), "json" (JSON time + fitted offset),
    "post_window" or "override". Segments are clipped to the recording.
    """
    cols = ["participant_id", "run_id", "segment", "event_name", "start_s", "end_s", "source"]
    rows = []
    off = align.get("offset_s", np.nan)
    if np.isfinite(off):
        base = session[session["eventName"] == C.BASELINE_EVENT]
        if len(base):
            s, e, src = _event_times(base.iloc[:1], blocks, off)
            rows.append([pid, base["runId"].iloc[0], "baseline", C.BASELINE_EVENT, s, e, src])
        for _, run in runs[runs["has_mist"].astype(bool)].iterrows():
            sr = session[session["runId"] == run["run_id"]]
            task = sr[sr["eventName"].isin(C.TASK_EVENTS)]
            if not len(task):
                continue
            s, e, src = _event_times(task, blocks, off)
            rows.append([pid, run["run_id"], "task", "+".join(dict.fromkeys(task["eventName"])), s, e, src])
            if C.POST_TASK_S:
                after = sr[sr["eventName"] == C.POST_TASK_AFTER_EVENT]
                p0 = _event_times(after.iloc[:1], blocks, off)[1] if len(after) else e
                rows.append([pid, run["run_id"], "post", f"{C.POST_TASK_S:g} s after "
                             f"{C.POST_TASK_AFTER_EVENT if len(after) else 'task'}", p0, p0 + C.POST_TASK_S,
                             "post_window"])
    seg_df = pd.DataFrame(rows, columns=cols)
    seg_df = apply_overrides(pid, seg_df)
    seg_df = cap_segments(seg_df)
    seg_df["start_s"] = seg_df["start_s"].astype(float).clip(lower=0, upper=rec_duration_s)
    seg_df["end_s"] = seg_df["end_s"].astype(float).clip(lower=0, upper=rec_duration_s)
    seg_df = seg_df[seg_df["end_s"] > seg_df["start_s"]].reset_index(drop=True)
    seg_df["duration_s"] = seg_df["end_s"] - seg_df["start_s"]
    return seg_df


def apply_overrides(pid: str, seg_df: pd.DataFrame) -> pd.DataFrame:
    if not C.SEGMENT_OVERRIDES_FILE.exists():
        return seg_df
    ov = pd.read_csv(C.SEGMENT_OVERRIDES_FILE, dtype={"participant_id": str, "run_id": str})
    ov = ov[ov["participant_id"] == pid]
    for _, o in ov.iterrows():
        mask = (seg_df["run_id"] == o["run_id"]) & (seg_df["segment"] == o["segment"])
        seg_df = seg_df[~mask]
        seg_df = pd.concat([seg_df, pd.DataFrame([{
            "participant_id": pid, "run_id": o["run_id"], "segment": o["segment"],
            "event_name": "override", "start_s": float(o["start_s"]), "end_s": float(o["end_s"]),
            "source": "override"}])], ignore_index=True)
    return seg_df


def cap_segments(seg_df: pd.DataFrame) -> pd.DataFrame:
    if not C.MAX_SEGMENT_S:
        return seg_df
    seg_df = seg_df.copy()
    n = C.MAX_SEGMENT_S
    is_base = seg_df["segment"] == "baseline"
    seg_df.loc[is_base, "start_s"] = np.maximum(seg_df.loc[is_base, "start_s"], seg_df.loc[is_base, "end_s"] - n)
    seg_df.loc[~is_base, "end_s"] = np.minimum(seg_df.loc[~is_base, "end_s"], seg_df.loc[~is_base, "start_s"] + n)
    return seg_df
