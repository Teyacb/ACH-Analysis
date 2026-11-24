"""Outline step 1 (inputs) and 3.1 (load): find files, load recordings, log problems."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Logging of missing files / errors (outline 3.1)
# --------------------------------------------------------------------------
def log_problem(participant_id: str, file_name: str, reason: str) -> None:
    log_file = C.DERIVED_DIR / "logs" / "missing_and_errors.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp}\t{participant_id}\t{file_name}\t{reason}\n")
    print(f"[LOGGED] {participant_id} | {file_name} | {reason}")


def participant_dir(pid: str) -> Path:
    return C.DERIVED_DIR / "participants" / pid


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def discover_participants() -> dict[str, dict]:
    """Return {participant_id: {"acq": [...], "json": [...]}} from the flat data folder.

    Files are grouped by the participant ID in their file name (PARTICIPANT_ID_REGEX),
    so they can sit directly in DATA_DIR or in any sub-folder. Behaviour is inside the
    session JSON, so there is no separate behavioural file.
    """
    out: dict[str, dict] = {}
    if not C.DATA_DIR.exists():
        return out
    pattern = re.compile(C.PARTICIPANT_ID_REGEX)
    kinds = {"acq": [C.ACQ_GLOB, C.PHYSIO_CSV_GLOB], "json": [C.SESSION_JSON_GLOB]}
    for kind, globs in kinds.items():
        for g in globs:
            for path in sorted(C.DATA_DIR.glob(g)):
                if any(part.startswith("_") for part in path.relative_to(C.DATA_DIR).parts[:-1]):
                    continue                     # e.g. synthetic _ground_truth/
                m = pattern.search(path.name)
                if m is None:
                    continue
                entry = out.setdefault(m.group(0), {"folder": path.parent, "acq": [], "json": []})
                entry[kind].append(path)
    return dict(sorted(out.items()))


def check_inputs(pid: str, files: dict) -> bool:
    """Log missing inputs. Returns False if the participant must be skipped.

    No session log: skipped (runs, segments and trials all come from it).
    No recording: ECG and EDA steps skipped; behaviour (FP rates, nu) still computed.
    """
    if not files.get("json"):
        log_problem(pid, "session log JSON", "file not found; participant skipped")
        return False
    if not files.get("acq"):
        log_problem(pid, "BIOPAC .acq", "file not found; ECG and EDA steps skipped")
    for kind, label in (("acq", "recording"), ("json", "session log")):
        if len(files.get(kind, [])) > 1:
            log_problem(pid, ";".join(p.name for p in files[kind]),
                        f"more than one {label}; only the first is used - check this")
    return True


def load_participants_table() -> pd.DataFrame:
    if not C.PARTICIPANTS_FILE.exists():
        log_problem("ALL", C.PARTICIPANTS_FILE.name, "participants.csv not found; group/age/antipsychotic will be missing")
        return pd.DataFrame(columns=["participant_id", "group", "age", "antipsychotic"])
    df = pd.read_csv(C.PARTICIPANTS_FILE, dtype={"participant_id": str})
    df["group"] = df["group"].str.strip().str.lower().replace(
        {"hc": "control", "healthy control": "control", "healthy": "control", "ep": "patient",
         "early psychosis": "patient", "fep": "patient"})
    return df


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------
def _pick_channel(columns, candidates):
    for cand in candidates:
        for col in columns:
            if cand.lower() in str(col).lower():
                return col
    return None


def load_recording(path: Path) -> tuple[pd.DataFrame, float]:
    """Load ECG, EDA and digital trigger lines.

    Returns a DataFrame with columns ECG, EDA (either may be absent), m0..m3, and the
    sampling rate. ``*_physio.csv`` files (columns ECG, EDA, m0..m3 plus a header
    comment ``# sampling_rate=...``) are accepted for testing / exported data.
    """
    path = Path(path)
    if path.suffix.lower() == ".acq":
        import neurokit2 as nk
        raw, fs = nk.read_acqknowledge(str(path))
    else:
        with open(path) as fh:
            first = fh.readline()
        fs = float(first.split("=")[1])
        raw = pd.read_csv(path, comment="#")

    # the ECG candidate "ECG" would also match "ECG - RSP"; prefer explicit ECG-ECG
    ecg_col = _pick_channel([c for c in raw.columns if "rsp" not in str(c).lower()], C.ECG_CHANNEL_CANDIDATES)
    eda_col = _pick_channel(raw.columns, C.EDA_CHANNEL_CANDIDATES)
    digital = [c for c in raw.columns if str(c).startswith(C.DIGITAL_CHANNEL_PREFIX) or str(c) in {"m0", "m1", "m2", "m3"}]

    out = pd.DataFrame(index=raw.index)
    if ecg_col is not None:
        out["ECG"] = raw[ecg_col].astype(np.float32)
    if eda_col is not None:
        out["EDA"] = raw[eda_col].astype(np.float32)
    for i, col in enumerate(digital[:4]):
        out[f"m{i}"] = (raw[col].to_numpy() > C.DIGITAL_HIGH_THRESHOLD_V).astype(np.uint8)
    return out, float(fs)




# --------------------------------------------------------------------------
# Session log JSON
# --------------------------------------------------------------------------
def _epoch_s(value) -> float:
    """data.startTimestamp / endTimestamp -> epoch seconds. Accepts epoch ms (number or
    numeric string) and ISO strings; anything else -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) / 1000
    text = str(value).strip()
    try:
        return float(text) / 1000
    except ValueError:
        pass
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    return np.nan if pd.isna(ts) else ts.timestamp()


def load_session_json(path: Path) -> pd.DataFrame:
    """One row per event, in the order they happened.

    Columns: eventName, runId, createdAt, data (the raw dict), start_epoch_s,
    end_epoch_s (from data.start/endTimestamp; createdAt when missing; time_source says
    which), t_rel_s (start, seconds after the first event).
    """
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    df = pd.DataFrame({
        "eventName": [r.get("eventName") for r in records],
        "runId": [str(r.get("runId")) for r in records],
        "createdAt": pd.to_datetime([r.get("createdAt") for r in records], utc=True, errors="coerce",
                                    format="mixed"),
        "data": [r.get("data") or {} for r in records],
    })
    df["start_epoch_s"] = [_epoch_s(d.get("startTimestamp")) for d in df["data"]]
    df["end_epoch_s"] = [_epoch_s(d.get("endTimestamp")) for d in df["data"]]
    created = df["createdAt"].map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
    has = df["start_epoch_s"].notna() & df["end_epoch_s"].notna()
    df["time_source"] = np.where(has, "data", "createdAt")
    df["start_epoch_s"] = df["start_epoch_s"].fillna(created)
    df["end_epoch_s"] = df["end_epoch_s"].fillna(created)
    df = df.sort_values(["start_epoch_s", "end_epoch_s"], kind="stable").reset_index(drop=True)
    df["t_rel_s"] = df["start_epoch_s"] - df["start_epoch_s"].min()
    return df


def session_events_table(session: pd.DataFrame) -> pd.DataFrame:
    """The session log without the raw data payloads (for saving / printing)."""
    out = session.drop(columns="data").copy()
    out["duration_s"] = out["end_epoch_s"] - out["start_epoch_s"]
    out["start_utc"] = pd.to_datetime(out["start_epoch_s"], unit="s", utc=True)
    return out


def run_order(session: pd.DataFrame) -> list[str]:
    """Run IDs in session order: Start-Session data.allRunIds, then any others seen."""
    listed = []
    start = session[session["eventName"] == "Start-Session"]
    if len(start):
        listed = [str(r) for r in (start["data"].iloc[0].get("allRunIds") or [])]
    seen = list(dict.fromkeys(session["runId"]))
    return listed + [r for r in seen if r not in listed]


def mist_performance(session: pd.DataFrame) -> pd.DataFrame:
    """One row per MIST round (practice excluded): run_id, eventName, correct, incorrect,
    timedOut, start_epoch_s, end_epoch_s."""
    m = session[session["eventName"].astype(str).str.startswith(C.MIST_EVENT_PREFIX)
                & ~session["eventName"].astype(str).str.contains("practice", case=False)]
    rows = [{"run_id": r["runId"], "eventName": r["eventName"],
             **{k: pd.to_numeric(r["data"].get(k), errors="coerce") for k in ("correct", "incorrect", "timedOut")},
             "start_epoch_s": r["start_epoch_s"], "end_epoch_s": r["end_epoch_s"]} for _, r in m.iterrows()]
    return pd.DataFrame(rows, columns=["run_id", "eventName", "correct", "incorrect", "timedOut",
                                       "start_epoch_s", "end_epoch_s"])


def _run_types_file(pid: str) -> dict:
    if not C.RUN_TYPES_FILE.exists():
        return {}
    d = pd.read_csv(C.RUN_TYPES_FILE, dtype=str)
    d = d[d["participant_id"] == pid]
    return dict(zip(d["run_id"], d["run_type"].str.strip().str.lower()))


def _find_key(obj, key):
    """Every value stored under `key` anywhere in a nested dict / list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            else:
                yield from _find_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find_key(v, key)


def manipulation_valence(session: pd.DataFrame, run_id: str) -> str:
    """The run's config.MANIPULATION_FIELD from any of its events' data, lower case.
    "" when absent; distinct values joined with "|" when the events disagree."""
    values = {str(v).strip().lower() for d in session.loc[session["runId"] == run_id, "data"]
              for v in _find_key(d, C.MANIPULATION_FIELD) if v is not None and str(v).strip()}
    return "|".join(sorted(values))


def timeout_check(run_type: str, timed_out: float) -> str:
    """Only the stress MIST has a time limit: stress runs should have timed-out questions,
    non-stress runs none. "agrees" / "disagrees" / "not checked"."""
    if run_type not in ("stress", "non-stress") or pd.isna(timed_out):
        return "not checked"
    return "agrees" if (timed_out > 0) == (run_type == "stress") else "disagrees"


def session_runs(session: pd.DataFrame, pid: str | None = None) -> pd.DataFrame:
    """One row per run in session order.

    run_type: "none" (no MIST, e.g. run-1), else from RUN_TYPES_FILE or config.RUN_TYPE_RULE
    on the run's manipulationValence ("stress" / "non-stress" / "unknown"). run_number
    counts all runs; mist_number only MIST runs. Also the summed MIST performance of the
    run and timeout_check (does the run type match the time-outs?).
    """
    perf = mist_performance(session)
    from_file = _run_types_file(pid) if pid else {}
    rows, k = [], 0
    for order, run_id in enumerate(run_order(session), start=1):
        mist = perf[perf["run_id"] == run_id].reset_index(drop=True)
        names = set(session.loc[session["runId"] == run_id, "eventName"])
        has_task = bool(names & set(C.TASK_EVENTS))
        manipulation = manipulation_valence(session, run_id)
        if not has_task:
            run_type = "none"
        elif run_id in from_file:
            run_type = from_file[run_id]
        else:
            run_type = str(C.RUN_TYPE_RULE(run_id, mist, manipulation))
        k += has_task
        timed_out = mist["timedOut"].sum(min_count=1)
        rows.append({"run_id": run_id, "run_number": order, "mist_number": k if has_task else np.nan,
                     "run_type": run_type, "manipulation_valence": manipulation, "has_mist": has_task,
                     "mist_correct": mist["correct"].sum(min_count=1),
                     "mist_incorrect": mist["incorrect"].sum(min_count=1),
                     "mist_timed_out": timed_out, "timeout_check": timeout_check(run_type, timed_out)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Behaviour (Trial-TT blocks inside the session JSON)
# --------------------------------------------------------------------------
def _valence(audio_file) -> str:
    path = str(audio_file or "").lower()
    for v in C.VALENCES:
        if f"/{v}/" in path:
            return v
    return "unknown"


def _num_list(x, n):
    x = list(x) if isinstance(x, (list, tuple)) else []
    x = (x + [None] * n)[:n]
    return pd.to_numeric(pd.Series(x, dtype=object), errors="coerce").to_numpy(dtype=float)


def load_trials(session: pd.DataFrame, run_ids: list[str] | None = None) -> pd.DataFrame:
    """Every Trial-TT trial in session order, one row per trial.

    Columns: run_id, block, block_order, trial, decibels, stimulus_present (0/1),
    intensity (0-1, for the HGF), valence, response_yes (1/0/NaN = not recorded),
    response_time_ms, rating, silent (bool).
    """
    blocks = session[session["eventName"].astype(str).str.startswith(C.TRIAL_BLOCK_PREFIX)]
    frames = []
    for b, (_, ev) in enumerate(blocks.iterrows()):
        d = ev["data"]
        db = _num_list(d.get("decibels"), len(d.get("decibels") or []))
        n = len(db)
        if n == 0:
            continue
        resp = _num_list(d.get("response"), n)
        frames.append(pd.DataFrame({
            "run_id": ev["runId"], "block": ev["eventName"], "block_order": b, "trial": np.arange(n),
            "decibels": db, "stimulus_present": (np.nan_to_num(db) > 0).astype(int),
            "valence": _valence((d.get("stimulusInfo") or {}).get("audioFile")),
            "response_yes": np.where(np.isin(resp, [0, 1]), resp, np.nan),
            "response_time_ms": _num_list(d.get("responseTime"), n),
            "rating": _num_list(d.get("ratings"), n),
            "block_start_epoch_s": ev["start_epoch_s"],
        }))
    cols = ["run_id", "block", "block_order", "trial", "decibels", "stimulus_present", "intensity", "valence",
            "response_yes", "response_time_ms", "rating", "silent", "block_start_epoch_s"]
    if not frames:
        return pd.DataFrame(columns=cols)
    t = pd.concat(frames, ignore_index=True)
    if run_ids is not None:                       # session order of the runs, then time
        rank = {r: i for i, r in enumerate(run_ids)}
        t["_r"] = t["run_id"].map(rank).fillna(len(rank))
        t = t.sort_values(["_r", "block_start_epoch_s", "block_order", "trial"], kind="stable").drop(columns="_r")
    t["intensity"] = _intensity(t["decibels"].to_numpy())
    t["silent"] = t["stimulus_present"] == 0
    return t.reset_index(drop=True)[cols]


def _intensity(db: np.ndarray) -> np.ndarray:
    """Word intensity on 0-1 for the HGF (config.INTENSITY_FROM_DB); 0 for silent trials."""
    db = np.nan_to_num(np.asarray(db, float))
    out = np.zeros(len(db))
    on = db > 0
    if C.INTENSITY_FROM_DB == "present":
        out[on] = 1.0
        return out
    levels = np.unique(np.round(db[on], 3))
    if len(levels):
        rank = np.searchsorted(levels, np.round(db[on], 3)) + 1
        out[on] = rank / (len(levels) + 1)
    return out


def missing_trial_blocks(session: pd.DataFrame, runs: pd.DataFrame, n_blocks: int = 3) -> list[str]:
    """'run-x:Trial-TT-n' for every expected word-detection block absent from the log."""
    have = set(zip(session["runId"], session["eventName"]))
    return [f"{r}:{C.TRIAL_BLOCK_PREFIX}{k}" for r in runs["run_id"] for k in range(1, n_blocks + 1)
            if (r, f"{C.TRIAL_BLOCK_PREFIX}{k}") not in have]
