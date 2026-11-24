"""Synthetic participants in the same format as the real ACHI session files, for testing.

    from achi_pipeline import synthetic; synthetic.make_dataset(out_dir)

Layout (flat, like the real Data/ folder):
    out_dir/ACHI_2026_9XX_session-1_sessiondata.json   session log; trials are INSIDE it
    out_dir/ACHI_2026_9XX_session-1_physio.csv         ECG, EDA, m0..m3 (stand-in for .acq)
    out_dir/participants.csv, out_dir/ground_truth.csv
    out_dir/_ground_truth/                              true beats, artefacts, SCRs, trials, runs

Session JSON = list of {createdAt (ISO "...Z"), eventName, runId, sessionId, userId, data}.
Runs run-1, run-2, run-3, run-5, run-7.
  run-1 : Start-Session, RestBaseline (the ONE baseline for the whole session), Setup, IMS,
          VAS1, ACH-Trial-Practice x2, ACH-Trial-Practice-Rating, ACH-WordConfirm,
          ACH-TrialQ, Trial-TT-1, VAS2, NWT, VAS3, Trial-TT-2, Trial-TT-3, VAS4
  others: IMS, VAS1, ACH-WordConfirm, ACH-TrialQ, Trial-TT-1, VAS2, MIST-Round-practice,
          MIST-Round-1, MIST-Feedback-1, MIST-Round-2, MIST-Feedback-2, VAS3,
          Trial-TT-2, Trial-TT-3, VAS4  (the last run ends with an extra IMS)
Quirks copied from the real format:
  * data.startTimestamp / endTimestamp are epoch-ms floats for RestBaseline, NWT and MIST
    events, but ISO strings for IMS, VAS, ACH and Trial-TT events
  * createdAt can lag the real end of an event (by up to a minute for RestBaseline)
  * VAS arousal / pleasure are sometimes numbers and sometimes strings
  * a Trial-TT block (any of 1-3, any run) is occasionally missing from a run
  * Trial-TT-n: decibels[45] (0 = silent trial), response[45] (1/0/null), responseTime[45]
    (null, or -1 when no response), ratings[45] / ratingsRaw[45] (null when no response),
    stimulusInfo.audioFile ".../<neutral|negative|positive>/...", timestamps (relative ms),
    timestampsUTC and triggerTS ([label, epoch ms] lists on two slightly different clocks)

Stress vs non-stress: every MIST-Round event carries manipulationValence (negative = stress,
positive = non-stress), independent of the word valence; stress rounds also have
timedOut > 0. The true run types are in _ground_truth/<pid>_true_runs.csv. Where the field
sits in the real JSON is not known; the pipeline looks for it in any event of the run.

All names, words and texts are placeholders. Built-in effects: the stress MIST lowers
RMSSD and raises SCL and SCR frequency more than the non-stress MIST; patients drop more,
recover more slowly and have higher nu.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import hgf

FS = 200
RUN_IDS = ["run-1", "run-2", "run-3", "run-5", "run-7"]
MIST_RUNS = RUN_IDS[1:]
N_TT = 45            # trials per Trial-TT block
N_Q = 40             # items per ACH-TrialQ staircase
TRIAL_MS = 4550      # one Trial-TT trial
POST_S = 300         # recovery window after the last MIST feedback
JSON_OFFSET_S = 1234.0
T0 = dt.datetime(2026, 3, 2, 10, 0, 0, tzinfo=dt.timezone.utc)
SCR_PER_MIN = {"baseline": 2.0, "stress": 6.0, "non-stress": 3.0, "post": 3.0}   # elsewhere 2/min
MIST_TASK = ("MIST-Round-1", "MIST-Feedback-1", "MIST-Round-2", "MIST-Feedback-2")
LOREM = ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
         "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud")


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
def _ecg_from_beats(beat_times, n, fs, rng):
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for comp_shift, amp, sd in [(-0.16, 0.12, 0.025), (-0.03, -0.12, 0.008), (0.0, 1.2, 0.010),
                                (0.03, -0.25, 0.008), (0.25, 0.3, 0.04)]:
        idx = np.round((beat_times + comp_shift) * fs).astype(int)
        imp = np.zeros(n)
        ok = (idx >= 0) & (idx < n)
        np.add.at(imp, idx[ok], amp)
        half = int(4 * sd * fs)
        k = np.exp(-0.5 * (np.arange(-half, half + 1) / (sd * fs)) ** 2)
        sig += np.convolve(imp, k, mode="same")
    sig += 0.15 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * rng.standard_normal(n)
    return sig


def _rr_series(duration, mean_rr, rmssd_target, rng):
    """AR(1) RR series, scaled so successive differences ~ rmssd_target."""
    n = int(duration / (mean_rr / 1000)) + 5
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + rng.standard_normal()
    d = np.diff(x)
    x = x * rmssd_target / np.sqrt(np.mean(d ** 2))
    return mean_rr + x


# --------------------------------------------------------------------------
# Clock helpers (recording seconds -> JSON clock)
# --------------------------------------------------------------------------
def _ms(t_rec: float) -> float:
    return round((T0.timestamp() + JSON_OFFSET_S + t_rec) * 1000, 1)


def _iso(t_rec: float) -> str:
    ms = int(round(_ms(t_rec)))
    d = dt.datetime.fromtimestamp(ms // 1000, tz=dt.timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms % 1000:03d}Z"


def _num_or_str(x: float, rng):
    """VAS values: the real files mix numbers and strings."""
    x = round(float(x), 2)
    return x if (x == 0.5 or rng.random() < 0.3) else f"{x:g}"


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------
RUN1_EVENTS = [("Start-Session", 0), ("RestBaseline", 240), ("Setup", 5), ("IMS", 60), ("VAS1", 12),
               ("ACH-Trial-Practice", 32), ("ACH-Trial-Practice", 31), ("ACH-Trial-Practice-Rating", 46),
               ("ACH-WordConfirm", 4), ("ACH-TrialQ", 245), ("Trial-TT-1", 205), ("VAS2", 12),
               ("NWT", 180), ("VAS3", 10), ("Trial-TT-2", 205), ("Trial-TT-3", 205), ("VAS4", 6)]
MIST_EVENTS = [("IMS", 50), ("VAS1", 8), ("ACH-WordConfirm", 5), ("ACH-TrialQ", 245), ("Trial-TT-1", 205),
               ("VAS2", 8), ("MIST-Round-practice", 15), ("MIST-Round-1", 110), ("MIST-Feedback-1", 18),
               ("MIST-Round-2", 95), ("MIST-Feedback-2", 16), ("VAS3", 5), ("Trial-TT-2", 205),
               ("Trial-TT-3", 205), ("VAS4", 6)]


def _timeline(rng) -> list[dict]:
    """Events in recording seconds. ``logged`` = False: happened, but missing from the JSON."""
    t, out = 30.0, []
    for run_id in RUN_IDS:
        events = list(RUN1_EVENTS if run_id == "run-1" else MIST_EVENTS)
        if run_id == RUN_IDS[-1]:
            events.append(("IMS", 35))
        drop_tt = f"Trial-TT-{rng.integers(1, 4)}" if rng.random() < 0.15 else None
        for name, dur in events:
            dur = dur * rng.uniform(0.9, 1.1) if dur else 0.0
            out.append({"run_id": run_id, "name": name, "start": t, "end": t + dur,
                        "logged": name != drop_tt})
            t += dur + rng.uniform(2, 10)
        t += rng.uniform(150, 300)        # break between runs
    return out


# --------------------------------------------------------------------------
# Event payloads
# --------------------------------------------------------------------------
def _trial_lists(t_start, t_end, n, present, responded, rng, label_ms_offset):
    """timestamps (relative ms), timestampsUTC and triggerTS for n trials."""
    rel, utc, trig = [["start", 0]], [["start", int(_ms(t_start))]], [["start", _ms(t_start) + label_ms_offset]]
    step = (t_end - t_start) * 1000 / n
    for i in range(n):
        t0 = i * step
        stim = t0 + rng.uniform(400, 700)
        resp = stim + rng.uniform(500, 1400)
        rate = resp + rng.uniform(300, 1100)
        rel.append(["stim", int(stim)])
        if responded[i]:
            rel += [["resp", int(resp)], ["rating", int(rate)]]
        base = _ms(t_start)
        utc += [[lab, int(base + v)] for lab, v in rel[-(3 if responded[i] else 1):]]
        trig.append(["trialStart", round(base + t0 + 50 + label_ms_offset, 1)])
        trig += [[lab, round(base + v + label_ms_offset, 1)] for lab, v in rel[-(3 if responded[i] else 1):]]
        trig.append(["trialEnd", round(base + t0 + step - 10 + label_ms_offset, 1)])
    return rel, utc, trig


def _stimulus_info(valence, word):
    ext = "mp3" if valence == "neutral" else "wav"
    return {"audioFile": f"/ACH/audio/fr/{valence}/{word}_voiceA_final.{ext}",
            "visualStimulus": {"color": "yellow", "pattern": "diamond"}, "whiteNoiseDb": 80}


def _trial_tt(ev, trials, valence, word, db_levels, trig_offset, rng):
    """One Trial-TT block from simulated trials (stimulus_present, intensity, response_yes)."""
    n = len(trials)
    db = np.where(trials["stimulus_present"] == 1,
                  trials["intensity"].map(dict(zip([0.25, 0.5, 0.75], db_levels))).fillna(0), 0.0)
    resp = trials["response_yes"].astype(int).to_numpy()
    r = rng.random(n)
    missing = r < 0.02                    # nothing recorded
    timeout = (r >= 0.02) & (r < 0.04)    # no answer in time: response 0, responseTime -1
    response = [None if m else (0 if to else int(x)) for x, m, to in zip(resp, missing, timeout)]
    rt = [None if m else (-1 if to else int(rng.uniform(450, 1500))) for m, to in zip(missing, timeout)]
    ratings = [None if m else int(np.clip(rng.integers(3, 6) if x == 1 else rng.integers(1, 4), 1, 5))
               for x, m in zip(response, missing)]
    raw = [None if v is None else int(v * 258 + rng.integers(-15, 16)) for v in ratings]
    rel, utc, trig = _trial_lists(ev["start"], ev["end"], n, db > 0, [x is not None for x in response],
                                  rng, trig_offset)
    return {"decibels": [float(x) for x in db], "endTimestamp": _iso(ev["end"]), "ratings": ratings,
            "ratingsRaw": raw, "response": response, "responseTime": rt,
            "startTimestamp": _iso(ev["start"]), "stimulusInfo": _stimulus_info(valence, word),
            "timestamps": rel, "timestampsUTC": utc, "triggerTS": trig}


def _practice(ev, valence, word, trig_offset, rng, with_ratings):
    present = rng.random(10) < 0.6
    db = np.where(present, 65, 0)
    response = [int(p) if rng.random() < 0.9 else None for p in present]
    if with_ratings:
        response = [0 if x is None else x for x in response]
    rt = [None if x is None else int(rng.uniform(400, 1500)) for x in response]
    rel, utc, trig = _trial_lists(ev["start"], ev["end"], 10, present, [x is not None for x in response],
                                  rng, trig_offset)
    d = {"decibels": [int(x) for x in db], "endTimestamp": _iso(ev["end"]), "response": response,
         "responseTime": rt, "startTimestamp": _iso(ev["start"]), "stimulusInfo": _stimulus_info(valence, word),
         "timestamps": rel, "timestampsUTC": utc, "triggerTS": trig}
    if with_ratings:
        d["ratings"] = [int(rng.integers(4, 6)) if x == 1 else int(rng.integers(1, 6)) for x in response]
        d["ratingsRaw"] = [int(v * 258 + rng.integers(-15, 16)) for v in d["ratings"]]
    return d


def _staircase(threshold, rng):
    db, out_db, out_r, out_rt = 58.0, [], [], []
    for i in range(N_Q):
        p = 1 / (1 + np.exp(-(db - threshold) / 1.5))
        r = int(rng.random() < p)
        out_db.append(float(db)); out_r.append(r); out_rt.append(int(rng.uniform(400, 1400)))
        db = db - (4.0 / (1 + i / 4)) if r else db + (2.0 / (1 + i / 4))
    return out_db, out_r, out_rt


def _trial_q(ev, rng):
    thr = rng.uniform(38, 46)
    d1, r1, t1 = _staircase(thr, rng)
    d2, r2, t2 = _staircase(thr + rng.normal(0, 1), rng)
    est = [round(float(np.mean(d1[-10:])), 3), round(float(np.mean(d2[-10:])), 3)]
    return {"decibels_q1": d1, "decibels_q2": d2, "endTimestamp": _iso(ev["end"]),
            "processedData": {"beta": 3.5, "intensities": est,
                              "parameters": {"threshold": round(float(np.mean(est)), 3), "slope": 3.5}},
            "q1": {"params": {"tGuess": 58, "tGuessSd": 10, "pThreshold": 0.82}},
            "q2": {"params": {"tGuess": 58, "tGuessSd": 10, "pThreshold": 0.82}},
            "responseTime_q1": t1, "responseTime_q2": t2, "response_q1": r1, "response_q2": r2,
            "startTimestamp": _iso(ev["start"])}


def _mist_round(ev, stress, practice, rng):
    n_items = 4 if practice else int(rng.integers(12, 20))
    timed_out = 0 if (practice or not stress) else int(rng.integers(6, 13))
    p_correct = 1.0 if practice else (0.8 if stress else 0.92)
    items = []
    for _ in range(n_items):
        a, b = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        ans = int(rng.integers(0, 10))
        c = a + b - ans
        q = f"{a} + {b} - {c} = ?" if c > 0 else f"{a} + {b} = ?"
        true = a + b - c if c > 0 else a + b
        ok = rng.random() < p_correct
        items.append({"answer": int(true % 10) if ok else int((true + rng.integers(1, 9)) % 10),
                      "isCorrect": "Correct" if ok else "Incorrect", "question": q,
                      "time": int(rng.uniform(1500, 6000))})
    n_ok = sum(i["isCorrect"] == "Correct" for i in items)
    return {"correct": n_ok, "endTimestamp": _ms(ev["end"]), "incorrect": n_items - n_ok,
            "manipulationValence": "negative" if stress else "positive",
            "results": {"current": items}, "startTimestamp": _ms(ev["start"]), "timedOut": timed_out}


def _session_json(pid, timeline, run_types, run_valence, run_words, trials_by_block, rng):
    db_top = rng.uniform(42, 46)
    db_levels = [db_top - 6.8, db_top - 3.0, db_top]
    trig_offset = rng.uniform(1000, 1500)
    recs = []
    for ev in timeline:
        if not ev["logged"]:
            continue
        name, run_id = ev["name"], ev["run_id"]
        lag = 0.0
        if name == "Start-Session":
            data = {"allRunIds": list(RUN_IDS), "includeNeutralNeutral": True, "lang": "fr",
                    "researchAssistant": "RA-placeholder", "skipAchPracticeConfirmWord": True,
                    "skipLastAchBlock": True, "skipNeutralWords": True, "triggerType": "EEG"}
        elif name in ("RestBaseline", "NWT") or name.startswith("MIST-Feedback"):
            data = {"endTimestamp": _ms(ev["end"]), "startTimestamp": _ms(ev["start"])}
            if name == "NWT":
                data["displayText"] = LOREM
                data["typedResponse"] = LOREM.lower()[: int(rng.integers(40, len(LOREM)))]
            lag = rng.uniform(20, 60) if name == "RestBaseline" else rng.uniform(0.5, 3)
        elif name == "Setup":
            data = {"whiteNoiseDb": 80}
        elif name == "IMS":
            data = {"endTimestamp": _iso(ev["end"]),
                    "responses": [{"emotion": f"Scale {k + 1} low - Scale {k + 1} high", "emotionId": k,
                                   "rating": int(rng.integers(1, 8))} for k in range(15)],
                    "startTimestamp": _iso(ev["start"])}
        elif name.startswith("VAS"):
            data = {"arousal": _num_or_str(rng.uniform(0.4, 0.9), rng), "endTimestamp": _iso(ev["end"]),
                    "pleasure": _num_or_str(rng.uniform(0.4, 0.8), rng), "startTimestamp": _iso(ev["start"])}
        elif name.startswith("ACH-Trial-Practice"):
            data = _practice(ev, "neutral", "word00", trig_offset, rng, with_ratings=name.endswith("Rating"))
        elif name == "ACH-WordConfirm":
            data = {"endTimestamp": _iso(ev["end"]), "numPlayed": 1, "startTimestamp": _iso(ev["start"])}
        elif name == "ACH-TrialQ":
            data = _trial_q(ev, rng)
        elif name.startswith("Trial-TT"):
            data = _trial_tt(ev, trials_by_block[(run_id, name)], run_valence[run_id], run_words[run_id],
                             db_levels, trig_offset, rng)
        elif name.startswith("MIST-Round"):
            data = _mist_round(ev, run_types.get(run_id) == "stress", name.endswith("practice"), rng)
            lag = rng.uniform(0.5, 3)
        else:
            data = {}
        recs.append({"createdAt": _iso(ev["end"] + lag), "data": data, "eventName": name,
                     "runId": run_id, "sessionId": "session-1", "userId": pid})
    return recs


# --------------------------------------------------------------------------
# One participant
# --------------------------------------------------------------------------
def make_participant(pid, out_dir, group, nu, run_types, rng, inject_flatline=False, inject_missed=True):
    """run_types: {run_id: "stress" | "non-stress"} for the MIST runs."""
    out_dir = Path(out_dir)
    timeline = _timeline(rng)
    duration = timeline[-1]["end"] + 60
    n = int(duration * FS)
    tt = np.arange(n) / FS

    # physiological phases: baseline (RestBaseline), task (MIST rounds + feedback), post
    baseline = next(e for e in timeline if e["name"] == "RestBaseline")
    phases = [("baseline", baseline["start"], baseline["end"], None)]
    for run_id in MIST_RUNS:
        ev = [e for e in timeline if e["run_id"] == run_id and e["name"] in MIST_TASK]
        s, e = ev[0]["start"], ev[-1]["end"]
        phases.append((run_types[run_id], s, e, run_id))
        phases.append(("post", e, e + POST_S, run_id))

    def phase_of(x):
        return next(((p, s, e, r) for p, s, e, r in phases if s <= x < e), (None, 0, 0, None))

    # beats with phase-specific HRV
    base_rmssd = rng.normal(45, 8)
    drop = 0.55 if group == "patient" else 0.75
    beats, cur = [], 0.0
    while cur < duration:
        ph, s, e, run_id = phase_of(cur)
        mean_rr, rm = 850, base_rmssd
        if ph == "stress":
            mean_rr, rm = 720, base_rmssd * drop
        elif ph == "non-stress":
            mean_rr, rm = 800, base_rmssd * 0.92
        elif ph == "post":
            task_drop = drop if run_types[run_id] == "stress" else 0.92
            frac = min(1, (cur - s) / (POST_S * (0.4 if group == "control" else 0.6)))
            mean_rr, rm = 760 + 90 * frac, base_rmssd * (task_drop + (1 - task_drop) * frac)
        for r in _rr_series(10, mean_rr, rm, rng)[:10]:
            cur += r / 1000
            beats.append(cur)
    beats = np.array([b for b in beats if b < duration - 1])
    artifacts = []
    analysis = [(s, e) for _, s, e, _ in phases]

    def random_time(margin=8):
        s, e = analysis[rng.integers(len(analysis))]
        return rng.uniform(s + margin, e - margin)

    # 1. missed beats
    beats_drawn = beats
    if inject_missed:
        idx = [int(np.argmin(np.abs(beats - random_time()))) for _ in range(4)]
        beats_drawn = np.delete(beats, idx)
        artifacts += [{"signal": "ecg", "type": "missed_beat", "start_s": beats[i], "end_s": beats[i]} for i in idx]
    ecg = _ecg_from_beats(beats_drawn, n, FS, rng)

    # 2. false peaks
    for _ in range(2):
        i = int(np.argmin(np.abs(beats - random_time())))
        ts = beats[i] + 0.45 * (beats[i + 1] - beats[i])
        k = np.abs(tt - ts) < 0.03
        ecg[k] += 1.5 * np.exp(-0.5 * ((tt[k] - ts) / 0.005) ** 2)
        artifacts.append({"signal": "ecg", "type": "false_peak", "start_s": ts, "end_s": ts})

    # 3. movement bursts; one long one in run-2's task for *902 (>20% of that segment unusable)
    bursts = [(random_time(15), rng.uniform(3, 6))]
    if pid.endswith("902"):
        s = next(p[1] for p in phases if p[3] == "run-2" and p[0] != "post")
        bursts.append((s + 60, 70.0))
    for b0, dur in bursts:
        k = (tt >= b0) & (tt < b0 + dur)
        ecg[k] += 0.9 * rng.standard_normal(k.sum()) + 1.5 * np.sin(2 * np.pi * 1.7 * tt[k])
        artifacts.append({"signal": "ecg", "type": "movement_burst", "start_s": b0, "end_s": b0 + dur})

    # 4. signal loss in one participant: 70 s inside the first post-task window (> 20% of 300 s)
    if inject_flatline:
        s = next(p[1] for p in phases if p[0] == "post")
        a = int((s + 40) * FS)
        ecg[a:a + 70 * FS] = ecg[a] + 0.001 * rng.standard_normal(70 * FS)
        artifacts.append({"signal": "ecg", "type": "signal_loss", "start_s": s + 40, "end_s": s + 110})

    # EDA
    eda = np.full(n, rng.normal(5, 1)) + 0.0002 * tt
    for ph, s, e, run_id in phases:
        m = (tt >= s) & (tt < e)
        if ph == "stress":
            eda[m] += 1.2 * (1 - np.exp(-(tt[m] - s) / 20))
        elif ph == "non-stress":
            eda[m] += 0.3 * (1 - np.exp(-(tt[m] - s) / 20))
        elif ph == "post":
            eda[m] += (1.0 if run_types[run_id] == "stress" else 0.25) * np.exp(-(tt[m] - s) / 40)
    true_scrs, onset = [], rng.exponential(30)
    while onset < duration - 10:
        k = tt >= onset
        a = rng.uniform(0.05, 1.0)
        eda[k] += a * (np.exp(-(tt[k] - onset) / 4) - np.exp(-(tt[k] - onset) / 0.75))
        true_scrs.append({"onset_s": onset, "peak_s": onset + 1.545, "amplitude_us": a * 0.553})
        onset += rng.exponential(60 / SCR_PER_MIN.get(phase_of(onset)[0], 2.0))
    eda += 0.005 * rng.standard_normal(n)
    # movement jump inside the first MIST task
    t_art = phases[1][1] + 60
    amp, dur = rng.uniform(3, 9), rng.uniform(0.5, 2.0)
    eda[int(t_art * FS):int((t_art + dur) * FS)] += amp
    artifacts.append({"signal": "eda", "type": "movement_jump", "start_s": t_art, "end_s": t_art + dur})
    if pid.endswith("903"):
        t_lc = random_time(20)
        eda[(tt >= t_lc) & (tt < t_lc + 8)] = 0.02
        artifacts.append({"signal": "eda", "type": "lost_contact", "start_s": t_lc, "end_s": t_lc + 8})
    if rng.random() < 0.6:
        t_scr = random_time(20)
        k = tt >= t_scr
        eda[k] += 7 * (np.exp(-(tt[k] - t_scr) / 6) - np.exp(-(tt[k] - t_scr) / 0.12))
        artifacts.append({"signal": "eda", "type": "genuine_large_scr", "start_s": t_scr, "end_s": t_scr + 3})
        true_scrs.append({"onset_s": t_scr, "peak_s": t_scr + 0.48, "amplitude_us": 6.3})

    # behaviour: three Trial-TT blocks of 45 trials per run
    run_valence = {"run-1": "neutral"}
    run_words = {"run-1": "word00"}
    for i, run_id in enumerate(MIST_RUNS):
        run_valence[run_id] = "negative" if i % 2 == 0 else "positive"
        run_words[run_id] = f"word{i + 1:02d}"
    trials_by_block, true_trials = {}, []
    for run_id in RUN_IDS:
        sim = hgf.simulate(n_trials=3 * N_TT, nu=nu, seed=int(rng.integers(1e6)))
        for b in range(3):
            name = f"Trial-TT-{b + 1}"
            block = sim.iloc[b * N_TT:(b + 1) * N_TT].reset_index(drop=True)
            trials_by_block[(run_id, name)] = block
            logged = any(e["logged"] for e in timeline if e["run_id"] == run_id and e["name"] == name)
            true_trials.append(block.assign(run_id=run_id, block=name, trial=np.arange(N_TT),
                                            valence=run_valence[run_id], logged=logged))

    # ground truth (outside the data files, so the pipeline never sees it)
    gt = out_dir / "_ground_truth"
    gt.mkdir(exist_ok=True)
    pd.DataFrame({"time_s": beats}).to_csv(gt / f"{pid}_true_beats.csv", index=False)
    pd.DataFrame(artifacts).to_csv(gt / f"{pid}_injected_artifacts.csv", index=False)
    pd.DataFrame(true_scrs).sort_values("onset_s").to_csv(gt / f"{pid}_true_scrs.csv", index=False)
    pd.concat(true_trials, ignore_index=True).drop(columns="silent").to_csv(gt / f"{pid}_true_trials.csv", index=False)
    pd.DataFrame([{"run_id": r, "run_type": run_types.get(r, "none"), "valence": run_valence[r]} for r in RUN_IDS]
                 ).to_csv(gt / f"{pid}_true_runs.csv", index=False)
    pd.DataFrame([{"phase": p, "run_id": r, "start_s": s, "end_s": e} for p, s, e, r in phases]
                 ).to_csv(gt / f"{pid}_true_segments.csv", index=False)

    # recording: blockStart = code 1 (m0), blockEnd = code 2 (m1), 10 ms pulses, for every timed event
    m = np.zeros((n, 4), np.uint8)
    for ev in timeline:
        if ev["end"] > ev["start"] and ev["name"] != "Setup":
            m[int(ev["start"] * FS):int(ev["start"] * FS) + 2, 0] = 5
            m[int(ev["end"] * FS):int(ev["end"] * FS) + 2, 1] = 5
    df = pd.DataFrame({"ECG": ecg.astype(np.float32), "EDA": eda.astype(np.float32),
                       "m0": m[:, 0], "m1": m[:, 1], "m2": m[:, 2], "m3": m[:, 3]})
    path = out_dir / f"{pid}_session-1_physio.csv"
    with open(path, "w") as fh:
        fh.write(f"# sampling_rate={FS}\n")
    df.to_csv(path, mode="a", index=False, float_format="%.4f")

    recs = _session_json(pid, timeline, run_types, run_valence, run_words, trials_by_block, rng)
    (out_dir / f"{pid}_session-1_sessiondata.json").write_text(json.dumps(recs, indent=4))

    return {"participant_id": pid, "group": group, "true_nu": nu, "true_base_rmssd_ms": base_rmssd,
            "true_mist_rmssd_ratio": drop,
            "run_order": ";".join(f"{r}:{run_types[r]}" for r in MIST_RUNS),
            "missing_blocks": ";".join(f"{e['run_id']}:{e['name']}" for e in timeline if not e["logged"]),
            "injected_artifacts": "; ".join(f"{a['signal']}:{a['type']}@{a['start_s']:.1f}s" for a in artifacts)}


def make_dataset(out_dir, n_per_group=4, seed=42):
    """Write n_per_group patients + n_per_group controls. Also writes ground_truth.csv
    (the values used to generate the data) so results can be checked against them."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, truth = [], []
    k = 900
    for group in ("patient", "control"):
        for j in range(n_per_group):
            k += 1
            pid = f"ACHI_2026_{k}"
            nu = float(np.clip(rng.normal(0.65 if group == "patient" else 0.4, 0.08), 0.05, 0.95))
            order = ["stress", "stress", "non-stress", "non-stress"]
            if j % 2:
                order = order[::-1]
            run_types = dict(zip(MIST_RUNS, order))
            truth.append(make_participant(pid, out_dir, group, nu, run_types, rng, inject_flatline=(k == 901)))
            rows.append({"participant_id": pid, "group": group, "age": int(rng.integers(18, 35)),
                         "antipsychotic": ("yes" if group == "patient" and j % 2 == 0 else "no")})
    pd.DataFrame(rows).to_csv(out_dir / "participants.csv", index=False)
    pd.DataFrame(truth).to_csv(out_dir / "ground_truth.csv", index=False)
    return out_dir
