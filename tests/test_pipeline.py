"""Run with:  .venv/bin/python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from achi_pipeline import config as C  # noqa: E402
from achi_pipeline import ecg, eda, io_utils, measures, pipeline, review, stats, synthetic, triggers  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    C.set_paths(root / "data", root / "derived")
    C.ALLOW_UNREVIEWED = False
    synthetic.make_dataset(C.DATA_DIR, n_per_group=3)
    pipeline.run_all_preprocessing()
    return root


def _event(name, run, start, end, data=None, created=None):
    d = {"startTimestamp": start, "endTimestamp": end, **(data or {})}
    return {"createdAt": created or "2026-03-02T10:00:00.000Z", "eventName": name, "runId": run,
            "sessionId": "session-1", "userId": "ACHI_2026_001", "data": d}


# ---------------------------------------------------------------- unit rules
def test_rmssd_known_value():
    rr = pd.DataFrame({"t0": [0, 1, 2, 3], "t1": [1, 2, 3, 4], "rr_ms": [800, 820, 780, 800], "ok": True})
    # diffs 20, -40, 20 -> sqrt((400+1600+400)/3)
    assert measures.rmssd(rr) == pytest.approx(np.sqrt(800))


def test_rmssd_skips_differences_across_bad_intervals():
    rr = pd.DataFrame({"t0": [0, 1, 2, 3, 4], "t1": [1, 2, 3, 4, 5], "rr_ms": [800, 820, 5000, 800, 810],
                       "ok": [True, True, False, True, True]})
    # only 820-800 and 810-800 are usable
    assert measures.rmssd(rr) == pytest.approx(np.sqrt((400 + 100) / 2))


def test_flatline_detected_and_not_filled():
    fs = 200
    rng = np.random.default_rng(0)
    beats = np.cumsum(np.full(400, 0.8))
    sig = synthetic._ecg_from_beats(beats, int(330 * fs), fs, rng)
    sig[100 * fs:130 * fs] = 0.0
    res = ecg.process_ecg(sig, fs)
    assert len(res["flatline"]) == 1
    f = res["flatline"].iloc[0]
    assert 99 <= f["start_s"] <= 101 and 129 <= f["end_s"] <= 131
    t = res["peaks_auto"] / fs
    assert not ((t > f["start_s"] + 0.5) & (t < f["end_s"] - 0.5)).any(), "beats invented inside signal loss"


def test_eda_artifact_rules():
    fs = 100
    x = np.full(60 * fs, 5.0)
    x[30 * fs:30 * fs + 20] = 20.0            # fast jump
    x[50 * fs:] = 0.01                        # lost contact
    res = eda.process_eda(x, fs)
    d = res["eda"]
    assert d.loc[(d.time_s > 26) & (d.time_s < 34), "artifact"].all()     # padded by 5 s
    assert not d.loc[(d.time_s > 5) & (d.time_s < 20), "artifact"].any()
    assert d.loc[d.time_s > 51, "artifact"].all()


def test_trigger_decoding_and_alignment():
    fs = 100
    n = 100 * fs
    rec = pd.DataFrame({"m0": np.zeros(n, np.uint8), "m1": np.zeros(n, np.uint8), "m2": np.zeros(n, np.uint8)})
    for s, e in [(10, 30), (40, 70)]:
        rec.loc[s * fs:s * fs + 1, "m0"] = 1
        rec.loc[e * fs:e * fs + 1, "m1"] = 1
    ev = triggers.decode_events(rec, fs)
    bl = triggers.trigger_blocks(ev)
    assert list(bl["start_s"]) == [10, 40] and list(bl["end_s"]) == [30, 70]
    # JSON epoch clock = recording clock + 1e9 s; "Setup" (no duration) has no trigger block
    t0 = 1e9
    sess = pd.DataFrame({"eventName": ["Setup", "RestBaseline", "MIST-Round-1"], "runId": ["run-1", "run-1", "run-2"],
                         "start_epoch_s": [t0 + 5, t0 + 10.4, t0 + 40.2], "end_epoch_s": [t0 + 5, t0 + 30.3, t0 + 70.1],
                         "time_source": "data"})
    out, al = triggers.align_session_to_blocks(sess, bl)
    assert al["n_matched"] == 2 and al["offset_s"] == pytest.approx(-t0, abs=0.5)
    assert list(out["event_name"]) == ["RestBaseline", "MIST-Round-1"]
    runs = pd.DataFrame({"run_id": ["run-1", "run-2"], "has_mist": [False, True]})
    old = C.POST_TASK_S
    C.POST_TASK_S = 20
    try:
        seg = triggers.build_segments("x", sess, out, al, runs, 100.0)
    finally:
        C.POST_TASK_S = old
    seg = seg.set_index("segment")
    assert (seg.loc["baseline", "run_id"], seg.loc["baseline", "start_s"]) == ("run-1", 10)
    assert seg.loc["task", "source"] == "trigger" and seg.loc["task", "end_s"] == 70
    # no MIST-Feedback-2 in this log -> the post window starts at the end of the task
    assert (seg.loc["post", "start_s"], seg.loc["post", "end_s"]) == (70, 90)


def test_recovery_time():
    C.RECOVERY_WINDOW_S, C.RECOVERY_STEP_S, C.RECOVERY_SUSTAIN_WINDOWS = 30.0, 5.0, 3
    rng = np.random.default_rng(1)

    def rr_df(t_start, dur, rm_fn):
        t, rows = t_start, []
        while t < t_start + dur:
            r = 0.8 + rm_fn(t - t_start) / 1000 * rng.standard_normal() / np.sqrt(2)
            rows.append((t, t + r, r * 1000))
            t += r
        return pd.DataFrame(rows, columns=["t0", "t1", "rr_ms"]).assign(ok=True)

    base = rr_df(0, 180, lambda x: 50)
    post = rr_df(200, 180, lambda x: 15 if x < 90 else 50)   # recovers at 90 s
    t, reached, _ = measures.recovery_time(base, (0, 180), post, (200, 380))
    assert reached and 90 <= t <= 135
    post_never = rr_df(200, 180, lambda x: 15)
    t2, reached2, _ = measures.recovery_time(base, (0, 180), post_never, (200, 380))
    assert not reached2 and np.isnan(t2)
    C.RECOVERY_SUSTAIN_WINDOWS = 1


def test_scr_counting():
    fs = 50
    t = np.arange(0, 60, 1 / fs)
    x = np.full(len(t), 5.0)
    for onset, a in [(10, 0.5), (12, 0.5), (30, 0.04), (45, 0.3)]:   # two overlapping, one below 0.05 uS
        k = t >= onset
        x[k] += a / 0.553 * (np.exp(-(t[k] - onset) / 4) - np.exp(-(t[k] - onset) / 0.75))
    scrs = eda.find_scrs(x, fs)
    assert len(scrs) == 3                                      # overlap counted twice, small one dropped
    assert scrs["amplitude_us"].iloc[0] == pytest.approx(0.5, abs=0.02)
    df = pd.DataFrame({"time_s": t})
    bad = (t >= 40) & (t < 50)                                 # 10 s artefact hides the SCR at 45 s
    n, minutes, freq = measures.scr_frequency(scrs, df, fs, 0, 60, bad)
    assert n == 2 and minutes == pytest.approx(50 / 60, abs=0.01) and freq == pytest.approx(2.4, abs=0.05)


def test_phasic_area_per_min():
    fs = 50
    t = np.arange(0, 60, 1 / fs)
    scr = np.full(len(t), 0.2)
    scr[(t >= 20) & (t < 30)] = -0.5                           # below zero: adds nothing, does not subtract
    df = pd.DataFrame({"time_s": t, "scr": scr})
    bad = (t >= 40) & (t < 50)                                 # 10 s artefact left out of area and minutes
    # 40 s at 0.2 uS = 8 uS*s, over 50 artefact-free seconds
    assert measures.phasic_area_per_min(df, fs, 0, 60, bad) == pytest.approx(8 / (50 / 60), rel=0.01)
    assert np.isnan(measures.phasic_area_per_min(df, fs, 41, 49, bad))   # no clean signal -> missing
    df["scr"] = 0.0
    assert measures.phasic_area_per_min(df, fs, 0, 60, bad) == 0.0    # no activity is 0, not missing


def test_phasic_area_keeps_overlapping_scrs_that_the_count_merges():
    fs = 50
    t = np.arange(0, 60, 1 / fs)

    def bump(onset):
        k = t >= onset
        y = np.zeros(len(t))
        y[k] = 0.3 / 0.553 * (np.exp(-(t[k] - onset) / 4) - np.exp(-(t[k] - onset) / 0.75))
        return y
    one, two = bump(10), bump(10) + bump(11)                  # onsets 1 s apart merge into one SCR
    bad = np.zeros(len(t), bool)
    for x in (one, two):
        assert len(eda.find_scrs(5.0 + x, fs)) == 1
    area = [measures.phasic_area_per_min(pd.DataFrame({"time_s": t, "scr": x}), fs, 0, 60, bad) for x in (one, two)]
    assert area[1] == pytest.approx(2 * area[0], rel=0.02)


def test_scr_notch_does_not_split_a_response():
    # two 0.04 uS rises 2 s apart: each is below 0.05 uS, together they rise ~0.07 uS
    fs = 50
    t = np.arange(0, 40, 1 / fs)
    x = np.full(len(t), 5.0)
    for onset in (10, 12):
        k = t >= onset
        x[k] += 0.04 / 0.553 * (np.exp(-(t[k] - onset) / 4) - np.exp(-(t[k] - onset) / 0.75))
    scrs = eda.find_scrs(x, fs)
    assert len(scrs) == 1                                      # one SCR, not zero
    assert scrs["onset_s"].iloc[0] <= 10                      # onset before the first rise, not at the notch
    assert scrs["amplitude_us"].iloc[0] == pytest.approx(x.max() - 5.0, abs=0.002)


def test_ecg_20pct_is_share_of_duration():
    fs = 100
    peaks = np.arange(0, 100.5, 1.0) * fs                      # one beat per second, 0..100 s
    corr = pd.DataFrame({"time_s": [10.0, 50.0], "source": "auto"})
    flat = pd.DataFrame({"start_s": [70.0], "end_s": [80.0]})
    seg = pd.DataFrame({"run_id": ["r"], "segment": ["task"], "start_s": [0.0], "end_s": [100.0]})
    qc = review.segment_qc(seg, {"peaks": peaks, "corrections": corr, "bad_spans": flat, "fs": fs})
    # each corrected beat covers the 2 s of RR intervals around it, plus 10 s signal loss
    assert qc["pct_ecg_bad"].iloc[0] == pytest.approx(14.0)


def test_fp_rates_and_participant_collapse():
    trials = pd.DataFrame({"run_id": "run-1", "silent": [True] * 6 + [False] * 2,
                           "valence": ["negative"] * 3 + ["positive"] * 3 + ["negative"] * 2,
                           "response_yes": [1, 0, 0, 1, 1, np.nan, 1, 1]})
    fp = measures.fp_rates(trials)
    assert (fp["n_yes_neg"], fp["n_silent_neg"], fp["n_yes_pos"], fp["n_silent_pos"]) == (1, 3, 2, 2)
    assert fp["fp_rate_all"] == pytest.approx(3 / 5)
    runs = pd.DataFrame({"run_type": ["stress", "non-stress", "stress"], "rmssd_baseline": [45, 45, 45],
                         "rmssd_task": [30, 55, 20], "rmssd_post": [45, 58, 35], "scl_baseline": [5, 5, 5],
                         "scl_task": [7, 6, 9], "scr_freq_baseline": [2, 2, 2], "scr_freq_task": [6, 3, 4],
                         "phasic_area_baseline": [3, 3, 3], "phasic_area_task": [10, 6, np.nan],
                         "rmssd_recovery_time_s": [60, np.nan, np.nan]})
    for seg in ("baseline", "task", "post"):
        runs[f"duration_s_{seg}"], runs[f"pct_ecg_bad_{seg}"], runs[f"ecg_keep_{seg}"] = 100.0, 0.0, True
        runs[f"pct_eda_removed_{seg}"], runs[f"eda_keep_{seg}"] = 0.0, True
    runs.loc[1, "ecg_keep_task"] = False
    r = measures.participant_row("x", runs)
    assert r["rmssd_stress"] == 25 and r["rmssd_nonstress"] == 55 and r["rmssd_baseline"] == 45
    assert r["delta_rmssd_stress"] == -30 and r["delta_rmssd"] == -20 and r["delta_scl_stress"] == 2
    # the one session baseline (on every row) counts once: 1 + 3 task + 3 post = 700 s, 100 s excluded
    assert r["rmssd_recovery_time_s"] == 60 and r["pct_excluded_ecg"] == pytest.approx(100 / 7)
    # a stress run without phasic area (no usable EDA) is skipped, not averaged in as 0
    assert r["phasic_area_stress"] == pytest.approx(10) and r["phasic_area_nonstress"] == pytest.approx(6)


def test_binomial_model_uses_counts():
    import statsmodels.api as sm
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"participant_id": [f"p{i}" for i in range(30)], "x": rng.normal(size=30),
                      "n": rng.integers(20, 40, 30)})
    d["yes"] = rng.binomial(d["n"], 1 / (1 + np.exp(1 - 0.5 * d["x"])))
    r = stats.binomial("x", d, "yes", "n", "x", "t", "primary")
    ref = sm.GLM(np.c_[d["yes"], d["n"] - d["yes"]], sm.add_constant(d["x"]), family=sm.families.Binomial()).fit()
    assert r["estimate"] == pytest.approx(ref.params["x"]) and r["se"] == pytest.approx(ref.bse["x"])
    p = stats.paired(pd.DataFrame({"participant_id": list("abcde"), "a": [3.0, 4, 5, 6, 8], "b": [1.0, 2, 4, 4, 5]}),
                     "a", "b", "t", "primary")
    assert p["estimate"] == pytest.approx(2.0) and p["model"] == "paired t-test"


# ---------------------------------------------------------------- integration
def test_unreviewed_participants_are_refused(synth):
    assert pipeline.compute("ACHI_2026_901") is None


def test_review_decisions_and_20pct_rule(synth):
    pid = "ACHI_2026_901"
    d = review.review_dir(pid)
    for kind in ("ecg", "eda"):
        f = pd.read_csv(d / f"flags_{kind}.csv", keep_default_na=False)
        f["decision"] = "accept"
        f.to_csv(d / f"flags_{kind}.csv", index=False)
    # 901 = order stress, stress, non-stress, non-stress; the reviewer excludes run-5's task ECG
    seg = pd.read_csv(d / "segment_decisions.csv", keep_default_na=False)
    m = (seg["run_id"] == "run-5") & (seg["segment"] == "task")
    seg.loc[m, "ecg_decision"] = "exclude"
    seg.loc[m, "reason"] = "electrode came off"
    seg.to_csv(d / "segment_decisions.csv", index=False)
    out = pipeline.compute(pid)
    rows, person = out["runs"].set_index("run_id"), out["participant"].iloc[0]
    assert list(rows["run_type"]) == ["none", "stress", "stress", "non-stress", "non-stress"]
    r5 = rows.loc["run-5"]
    assert np.isnan(r5["rmssd_task"]) and not r5["ecg_keep_task"]
    assert np.isfinite(person["rmssd_nonstress"]) and np.isfinite(person["delta_rmssd_stress"])  # run-7 remains
    # one session baseline: identical on every run row
    assert rows["rmssd_baseline"].nunique() == 1 and person["rmssd_baseline"] == rows["rmssd_baseline"].iloc[0]
    assert np.isnan(rows.loc["run-1", "rmssd_task"])                       # run-1 has no MIST
    # the 70-s ECG dropout covers > 20% of run-2's 300-s post-task window -> excluded automatically
    r2 = rows.loc["run-2"]
    assert r2["pct_ecg_bad_post"] > 20 and not r2["ecg_keep_post"] and np.isnan(r2["rmssd_post"])
    old = C.ECG_EXCLUDE_PCT
    C.ECG_EXCLUDE_PCT = 30
    try:
        r2 = pipeline.compute(pid)["runs"].set_index("run_id").loc["run-2"]
        assert r2["ecg_keep_post"] and np.isfinite(r2["rmssd_post"])
    finally:
        C.ECG_EXCLUDE_PCT = old


def test_manual_peak_removal(synth):
    pid = "ACHI_2026_902"
    d = review.review_dir(pid)
    f = pd.read_csv(d / "flags_ecg.csv", keep_default_na=False)
    f["decision"] = "accept"
    f.to_csv(d / "flags_ecg.csv", index=False)
    e = pd.read_csv(d / "flags_eda.csv", keep_default_na=False)
    e["decision"] = "accept"
    e.to_csv(d / "flags_eda.csv", index=False)
    base = pipeline.compute(pid)["runs"]
    peaks = pd.read_csv(pipeline.participant_dir(pid) / "rpeaks_final.csv")
    target = peaks[(peaks.time_s > 100) & (peaks.time_s < 200)].iloc[5]["time_s"]
    f = pd.concat([f, pd.DataFrame([{"flag_id": 999, "type": "manual", "time_s": target, "end_s": "",
                                     "decision": "remove_peak", "new_time_s": "", "notes": "test"}])])
    f.to_csv(d / "flags_ecg.csv", index=False)
    after = pipeline.compute(pid)["runs"]
    peaks2 = pd.read_csv(pipeline.participant_dir(pid) / "rpeaks_final.csv")
    assert len(peaks2) == len(peaks) - 1
    assert after["rmssd_baseline"].iloc[0] != base["rmssd_baseline"].iloc[0]


def test_full_analysis(synth):
    # like real use: every participant is reviewed before measures are computed
    from achi_pipeline import simulated_review
    for pid in pipeline.discover_participants():
        simulated_review.apply(pid, error_rate=0.05)
    pipeline.run_all_compute()
    res = pipeline.analyse()
    assert res["table"]["reviewed"].astype(bool).all()
    assert len(res["table"]) == 6 and set(res["runs"]["run_type"]) == {"none", "stress", "non-stress"}
    assert res["table"]["fp_rate_all"].notna().all() and res["table"]["nu"].notna().all()
    st = res["statistics"].set_index("test")
    h1 = st.loc["H1: RMSSD stress vs non-stress MIST (expect <0)"]
    assert h1["model"] == "paired t-test" and h1["estimate"] < 0 and h1["p"] < 0.05
    assert st.loc["H1: SCL stress vs non-stress MIST (expect >0)", "estimate"] > 0
    assert st.loc["SCR frequency stress vs non-stress MIST", "estimate"] > 0
    assert st.loc["H2: FP rate (all words) ~ dRMSSD + group (expect <0)", "model"].startswith(("binomial", "quasi"))
    for f in ("participant_table", "run_table", "statistics", "descriptive_summary", "descriptive_scr_by_run_type"):
        assert (C.DERIVED_DIR / "results" / f"{f}.csv").exists()
    log = pd.read_csv(C.DERIVED_DIR / "logs" / "ecg_corrected_beats.csv")
    assert len(log) == 6 * (1 + 4 * 2) and "pct_beats_corrected" in log      # 1 baseline + 4 x (task, post)


def test_simulated_reviewer_is_perfect_without_errors_and_improves_rmssd(synth):
    from achi_pipeline import simulated_review
    pid = "ACHI_2026_904"
    for kind in ("ecg", "eda"):          # start from an unreviewed sheet
        path = review.review_dir(pid) / f"flags_{kind}.csv"
        f = pd.read_csv(path, keep_default_na=False)
        f["decision"] = ""
        if "new_time_s" in f:
            f["new_time_s"] = ""
        f.to_csv(path, index=False)
    C.ALLOW_UNREVIEWED = True
    try:
        before = pipeline.compute(pid)["runs"]
    finally:
        C.ALLOW_UNREVIEWED = False
    log = simulated_review.apply(pid, error_rate=0.0)
    assert not log["simulated_mistake"].any()
    sc = simulated_review.score(pid)
    assert sc["correct"].all()
    assert (log["decision"] == "exclude_span").any()          # the movement burst is excluded
    after = pipeline.compute(pid)["runs"]
    truth = simulated_review.true_rmssd(pid, pd.read_csv(pipeline.participant_dir(pid) / "segments.csv"))

    def err(t):
        m = t.melt(id_vars="run_id", value_vars=["rmssd_baseline", "rmssd_task", "rmssd_post"], var_name="segment",
                   value_name="est").assign(segment=lambda d: d["segment"].str.replace("rmssd_", ""))
        m = m.merge(truth, on=["run_id", "segment"]).dropna(subset=["est"])
        return (m["est"] - m["true_rmssd"]).abs().mean()

    assert err(after) < err(before)
    assert err(after) < 3


def test_simulated_mistakes_are_logged(synth):
    from achi_pipeline import simulated_review
    log = simulated_review.apply("ACHI_2026_905", error_rate=1.0)
    needed = log["correct_decision"] != "accept"
    assert log.loc[needed, "simulated_mistake"].all()
    assert (log.loc[needed, "decision"] == "accept").all()


def test_missing_eda_channel_skips_only_eda(synth, tmp_path):
    """A recording without EDA still gets ECG measures (outline: skip the EDA steps only)."""
    old = (C.DATA_DIR, C.DERIVED_DIR, C.PARTICIPANTS_FILE)
    src = C.DATA_DIR
    try:
        C.DATA_DIR, C.DERIVED_DIR = tmp_path / "data", tmp_path / "derived"
        C.PARTICIPANTS_FILE = old[2]
        dst = C.DATA_DIR
        dst.mkdir(parents=True)
        for f in src.glob("ACHI_2026_906_*"):
            if f.name.endswith("_physio.csv"):
                head = f.open().readline()
                rec = pd.read_csv(f, comment="#").drop(columns="EDA")
                (dst / f.name).write_text(head)
                rec.to_csv(dst / f.name, mode="a", index=False)
            else:
                (dst / f.name).write_bytes(f.read_bytes())
        assert pipeline.preprocess("ACHI_2026_906")
        assert review.signals_recorded("ACHI_2026_906") == {"ecg": True, "eda": False}
        assert "flags_eda" not in review.review_status("ACHI_2026_906")
        f = review.review_dir("ACHI_2026_906") / "flags_ecg.csv"
        flags = pd.read_csv(f, keep_default_na=False)
        flags["decision"] = "accept"
        flags.to_csv(f, index=False)
        p = pipeline.compute("ACHI_2026_906")["participant"].iloc[0]
        assert np.isfinite(p["rmssd_stress"]) and np.isnan(p["scl_stress"]) and p["pct_removed_eda"] == 100
        log = (C.DERIVED_DIR / "logs" / "missing_and_errors.txt").read_text()
        assert "EDA channel not found; EDA steps skipped" in log
    finally:
        C.DATA_DIR, C.DERIVED_DIR, C.PARTICIPANTS_FILE = old


# ---------------------------------------------------------------- session JSON format
def test_session_json_times_and_runs(tmp_path):
    t = 1772446800000.0                                     # epoch ms
    recs = [
        _event("Start-Session", "run-1", None, None, {"allRunIds": ["run-1", "run-2", "run-3"]},
               created="2026-03-02T10:00:00.000Z"),
        _event("RestBaseline", "run-1", t, t + 240000),                                   # epoch ms
        _event("IMS", "run-1", "2026-03-02T10:05:00.000Z", "2026-03-02T10:06:00.500Z"),   # ISO strings
        _event("MIST-Round-practice", "run-2", t + 900000, t + 915000, {"correct": 4, "incorrect": 0, "timedOut": 0}),
        _event("MIST-Round-1", "run-2", t + 920000, t + 1030000, {"correct": 10, "incorrect": 3, "timedOut": 5}),
        _event("MIST-Round-2", "run-2", t + 1050000, t + 1145000, {"correct": 9, "incorrect": 2, "timedOut": 4}),
        _event("MIST-Round-1", "run-3", t + 2000000, t + 2110000, {"correct": 15, "incorrect": 1, "timedOut": 0}),
        _event("MIST-Round-2", "run-3", t + 2130000, t + 2225000, {"correct": 14, "incorrect": 1, "timedOut": 0}),
    ]
    path = tmp_path / "ACHI_2026_001_session-1_sessiondata.json"
    path.write_text(json.dumps(recs))
    s = io_utils.load_session_json(path)
    ims = s[s["eventName"] == "IMS"].iloc[0]
    assert ims["end_epoch_s"] - ims["start_epoch_s"] == pytest.approx(60.5)
    base = s[s["eventName"] == "RestBaseline"].iloc[0]
    assert base["end_epoch_s"] - base["start_epoch_s"] == pytest.approx(240)
    assert s.loc[s["eventName"] == "Start-Session", "time_source"].iloc[0] == "createdAt"

    old_file = C.RUN_TYPES_FILE
    try:
        C.RUN_TYPES_FILE = tmp_path / "run_types.csv"
        runs = io_utils.session_runs(s, "ACHI_2026_001").set_index("run_id")
        assert list(runs["run_type"]) == ["none", "unknown", "unknown"]     # no manipulationValence yet
        assert runs.loc["run-2", "mist_timed_out"] == 9 and runs.loc["run-2", "mist_correct"] == 19  # practice excluded
        assert runs.loc["run-3", "mist_number"] == 2
        pd.DataFrame({"participant_id": ["ACHI_2026_001"], "run_id": ["run-3"], "run_type": ["Stress"]}
                     ).to_csv(C.RUN_TYPES_FILE, index=False)
        assert io_utils.session_runs(s, "ACHI_2026_001").set_index("run_id").loc["run-3", "run_type"] == "stress"
    finally:
        C.RUN_TYPES_FILE = old_file


def test_run_type_from_manipulation_valence_not_word_valence(tmp_path):
    t = 1772446800000.0

    def tt(run, words, start):                              # word valence comes from the audio file
        return _event("Trial-TT-1", run, start, start + 1000,
                      {"stimulusInfo": {"audioFile": f"/ACH/audio/fr/{words}/w_final.wav"}, "decibels": [0]})

    def mist(run, name, start, manip, timed_out):
        return _event(name, run, start, start + 100000,
                      {"correct": 10, "incorrect": 2, "timedOut": timed_out, "manipulationValence": manip})
    recs = [
        _event("Start-Session", "run-1", None, None, {"allRunIds": ["run-1", "run-2", "run-3", "run-5", "run-7"]}),
        _event("RestBaseline", "run-1", t, t + 240000),
        # stress MIST with POSITIVE words, non-stress MIST with NEGATIVE words: run type follows the MIST
        tt("run-2", "positive", t + 300000), mist("run-2", "MIST-Round-1", t + 400000, "Negative", 6),
        mist("run-2", "MIST-Round-2", t + 600000, "negative", 5),
        tt("run-3", "negative", t + 900000), mist("run-3", "MIST-Round-1", t + 1000000, "positive", 0),
        mist("run-3", "MIST-Round-2", t + 1200000, "positive", 0),
        # non-stress label but questions timed out: cross-check flags it
        mist("run-5", "MIST-Round-1", t + 1500000, "positive", 3),
        # events of one run disagree: unknown
        mist("run-7", "MIST-Round-1", t + 2000000, "negative", 4), mist("run-7", "MIST-Round-2", t + 2200000, "positive", 0),
    ]
    path = tmp_path / "ACHI_2026_001_session-1_sessiondata.json"
    path.write_text(json.dumps(recs))
    runs = io_utils.session_runs(io_utils.load_session_json(path)).set_index("run_id")
    assert list(runs["run_type"]) == ["none", "stress", "non-stress", "non-stress", "unknown"]
    assert runs.loc["run-2", "manipulation_valence"] == "negative"
    assert runs.loc["run-7", "manipulation_valence"] == "negative|positive"
    assert list(runs["timeout_check"]) == ["not checked", "agrees", "agrees", "disagrees", "not checked"]
    # synthetic sessions carry the field on every MIST round and match the true run types
    synth_dir = tmp_path / "synth"
    synthetic.make_dataset(synth_dir, n_per_group=1)
    for js in sorted(synth_dir.glob("*_sessiondata.json")):
        pid = js.name.split("_session")[0]
        got = io_utils.session_runs(io_utils.load_session_json(js)).set_index("run_id")["run_type"]
        truth = pd.read_csv(synth_dir / "_ground_truth" / f"{pid}_true_runs.csv").set_index("run_id")["run_type"]
        assert (got == truth.loc[got.index]).all()


def test_trials_from_session_json(tmp_path):
    def tt(name, run, valence, start, db, resp):
        return _event(name, run, start, start, {
            "decibels": db, "response": resp, "responseTime": [500] * len(db), "ratings": [3] * len(db),
            "stimulusInfo": {"audioFile": f"/ACH/audio/fr/{valence}/w_final.wav"}})
    recs = [tt("Trial-TT-2", "run-2", "negative", "2026-03-02T10:20:00.000Z", [0, 40.5, 0, 44.0], [1, 1, None, 0]),
            tt("Trial-TT-1", "run-2", "negative", "2026-03-02T10:10:00.000Z", [37.2, 0, 0, 40.5], [1, 0, 0, 1]),
            tt("Trial-TT-1", "run-1", "neutral", "2026-03-02T10:00:00.000Z", [0, 0, 44.0, 0], [0, 1, 1, 0])]
    path = tmp_path / "x_sessiondata.json"
    path.write_text(json.dumps(recs))
    s = io_utils.load_session_json(path)
    t = io_utils.load_trials(s, ["run-1", "run-2"])
    assert list(t.drop_duplicates(["run_id", "block"])[["run_id", "block"]].itertuples(index=False, name=None)) == \
        [("run-1", "Trial-TT-1"), ("run-2", "Trial-TT-1"), ("run-2", "Trial-TT-2")]       # session order
    assert t["silent"].sum() == 7 and t["response_yes"].isna().sum() == 1
    assert sorted(t.loc[~t["silent"], "intensity"].unique()) == [0.25, 0.5, 0.75]       # 3 dB levels
    fp = measures.fp_rates(t)
    assert (fp["n_yes_neu"], fp["n_silent_neu"]) == (1, 3)
    assert (fp["n_yes_neg"], fp["n_silent_neg"]) == (1, 3)                          # null response left out
    assert fp["n_silent_all"] == 3                                                  # neutral not in "all"
    runs = pd.DataFrame({"run_id": ["run-1", "run-2"]})
    assert io_utils.missing_trial_blocks(s, runs) == ["run-1:Trial-TT-2", "run-1:Trial-TT-3", "run-2:Trial-TT-3"]


def test_discovery_flat_folder(tmp_path):
    old = (C.DATA_DIR, C.DERIVED_DIR)
    try:
        C.set_paths(tmp_path, tmp_path / "derived")
        for name in ("ACHI_2026_001_session-1_sessiondata.json", "ACHI_2026_002_session-1_sessiondata.json",
                     "ACHI_2026_001_session-1.acq", "notes.txt"):
            (tmp_path / name).write_text("[]")
        (tmp_path / "_ground_truth").mkdir()
        (tmp_path / "_ground_truth" / "ACHI_2026_003_session-1_sessiondata.json").write_text("[]")
        found = io_utils.discover_participants()
        assert list(found) == ["ACHI_2026_001", "ACHI_2026_002"]
        assert [p.name for p in found["ACHI_2026_001"]["acq"]] == ["ACHI_2026_001_session-1.acq"]
        assert found["ACHI_2026_002"]["acq"] == []
    finally:
        C.set_paths(*old)


def test_participant_without_recording_still_gets_behaviour(synth, tmp_path):
    old = (C.DATA_DIR, C.DERIVED_DIR, C.PARTICIPANTS_FILE)
    try:
        src = C.DATA_DIR
        C.set_paths(tmp_path / "data", tmp_path / "derived")
        C.PARTICIPANTS_FILE = old[2]
        C.DATA_DIR.mkdir(parents=True)
        f = next(src.glob("ACHI_2026_903_*sessiondata.json"))
        (C.DATA_DIR / f.name).write_bytes(f.read_bytes())
        assert pipeline.preprocess("ACHI_2026_903")
        p = pipeline.compute("ACHI_2026_903")["participant"].iloc[0]
        assert np.isnan(p["rmssd_baseline"]) and np.isfinite(p["fp_rate_all"]) and np.isfinite(p["nu"])
        assert "file not found; ECG and EDA steps skipped" in (C.DERIVED_DIR / "logs" / "missing_and_errors.txt").read_text()
    finally:
        C.DATA_DIR, C.DERIVED_DIR, C.PARTICIPANTS_FILE = old
        C.set_paths(C.DATA_DIR, C.DERIVED_DIR)
