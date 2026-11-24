"""All tunable settings for the ACHI physiology pipeline.

Edit this file (or override attributes at runtime, e.g. ``config.DATA_DIR = ...``)
rather than changing values inside the processing modules. Every number here
comes from the script outline (script_outline_REVISED.md).
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Real data: the flat Data/ folder next to ACHI_analysis/ (no per-participant folders):
#   Data/ACHI_2026_XXX_session-1_sessiondata.json   session log (trials are inside it)
#   Data/ACHI_2026_XXX_....acq                       BIOPAC recording (when available)
REAL_DATA_DIR = PROJECT_ROOT.parent / "Data"
REAL_DERIVED_DIR = PROJECT_ROOT / "derived"
DATA_DIR = REAL_DATA_DIR
DERIVED_DIR = REAL_DERIVED_DIR
# participant_id, group (patient / control), age, antipsychotic (yes / no)
PARTICIPANTS_FILE = DATA_DIR / "participants.csv"
# Optional: participant_id, nu  -- nu values fitted elsewhere (e.g. TAPAS). Overrides hgf.py.
NU_ESTIMATES_FILE = DATA_DIR / "nu_estimates.csv"
# Optional: participant_id, run_id, run_type (stress / non-stress). Overrides RUN_TYPE_RULE.
RUN_TYPES_FILE = DATA_DIR / "run_types.csv"
# Optional manual segment fixes: participant_id, run_id, segment, start_s, end_s
SEGMENT_OVERRIDES_FILE = DERIVED_DIR / "segment_overrides.csv"

# Files are matched to a participant by the ID in their file name (searched in sub-folders too).
PARTICIPANT_ID_REGEX = r"ACHI_\d{4}_\d+"
ACQ_GLOB = "**/*.acq"
PHYSIO_CSV_GLOB = "**/*_physio.csv"     # synthetic stand-in for .acq
SESSION_JSON_GLOB = "**/*sessiondata.json"

# --------------------------------------------------------------------------
# Channels (matched case-insensitively by substring, first match wins)
# --------------------------------------------------------------------------
ECG_CHANNEL_CANDIDATES = ["ECG - ECG", "ECG"]
EDA_CHANNEL_CANDIDATES = ["GSR - EDA", "EDA", "GSR"]
DIGITAL_CHANNEL_PREFIX = "Digital output"
DIGITAL_HIGH_THRESHOLD_V = 4.0           # 0 / 5 V lines
# Digital lines m0, m1, m2 are bits 1, 2, 4. m3 unused.
EVENT_CODE_MAP = {
    1: "blockStart", 2: "blockEnd", 3: "trialStart", 4: "trialEnd",
    5: "stim", 6: "resp", 7: "rating",
}

# --------------------------------------------------------------------------
# Session log -> segments
# --------------------------------------------------------------------------
# Session JSON = list of {createdAt, eventName, runId, sessionId, userId, data}. Event times
# come from data.startTimestamp / data.endTimestamp (epoch ms or ISO, both handled);
# createdAt is only used when an event has neither (it can lag the real end by a minute).
#
# ONE RestBaseline for the whole session (in run-1) = the baseline for every run.
BASELINE_EVENT = "RestBaseline"
# Task segment of a run = from the start of the first to the end of the last of these
# events (MIST-Feedback-1 between the rounds is therefore inside it). Runs without them
# (run-1) have no task segment.
TASK_EVENTS = ["MIST-Round-1", "MIST-Round-2"]
# Post-task window: POST_TASK_S seconds starting at the end of POST_TASK_AFTER_EVENT
# (falls back to the end of the task segment). None = no post-task window.
# NOTE: there is no rest after the MIST in this design; this window overlaps VAS3 and
# Trial-TT-2. See decisions_and_changes.md.
POST_TASK_AFTER_EVENT = "MIST-Feedback-2"
POST_TASK_S = 300
# Behaviour: word-detection blocks (each data has decibels / response / ratings / stimulusInfo).
TRIAL_BLOCK_PREFIX = "Trial-TT-"
TRIALS_PER_BLOCK = 45
MIST_EVENT_PREFIX = "MIST-Round-"       # correct / incorrect / timedOut summed per run (practice excluded)

# --------------------------------------------------------------------------
# Stress vs non-stress runs (decision 13)
# --------------------------------------------------------------------------
# The session JSON field manipulationValence of a run's events says which MIST version
# was run: negative = stress MIST, positive = non-stress MIST. This is the MIST
# manipulation, NOT the word valence of the Trial-TT blocks (that comes from the audio
# file name and is never used here).
MANIPULATION_FIELD = "manipulationValence"
MANIPULATION_TO_RUN_TYPE = {"negative": "stress", "positive": "non-stress"}


# RUN_TYPE_RULE(run_id, mist, manipulation) -> "stress" | "non-stress" | "unknown"
#   mist: DataFrame of the run's MIST-Round-1 / -2 events with columns eventName,
#         correct, incorrect, timedOut, start_epoch_s, end_epoch_s (empty for run-1).
#   manipulation: the run's manipulationValence ("" when absent, "a|b" when the run's
#         events disagree).
# Runs without MIST events always get run_type "none". RUN_TYPES_FILE, if present,
# overrides this rule. "unknown" runs are logged and get no stress / non-stress values.
# Cross-check (io_utils.session_runs, column timeout_check): only the stress MIST has a
# time limit, so stress runs should have timed-out questions and non-stress runs none.
def RUN_TYPE_RULE(run_id, mist, manipulation):          # noqa: N802
    return MANIPULATION_TO_RUN_TYPE.get(str(manipulation).strip().lower(), "unknown")


# Max seconds between a JSON event and a trigger block (start AND end) to call it a match.
ALIGN_TOLERANCE_S = 3.0
# Alternative when the recording has no usable triggers: participant_id, offset_s
# (recording seconds = JSON epoch seconds + offset_s).
CLOCK_OFFSETS_FILE = DERIVED_DIR / "clock_offsets.csv"
# Optional cap on every segment's length (e.g. 300) to equalise durations.
# Baseline keeps its LAST N s; task and post keep their FIRST N s.
MAX_SEGMENT_S = None

# --------------------------------------------------------------------------
# ECG (outline 3.2)
# --------------------------------------------------------------------------
ECG_HIGHPASS_HZ = 0.5          # applied by nk.ecg_clean(method="neurokit")
POWERLINE_HZ = 60              # Canada
FLATLINE_WINDOW_S = 1.0
FLATLINE_RANGE_MV = 0.05       # peak-to-peak below this = signal loss
KUBIOS_ALPHA = 5.2             # Lipponen & Tarvainen threshold multiplier
KUBIOS_WINDOW_BEATS = 91
RR_MIN_MS, RR_MAX_MS = 300, 2000
RR_JUMP_MS = 250

# --------------------------------------------------------------------------
# EDA (outline 3.3)
# --------------------------------------------------------------------------
EDA_LOWPASS_HZ = 1.0
EDA_TARGET_FS = 50             # downsample after low-pass (1 Hz content), saves memory
EDA_MIN_US = 0.05             # below = lost contact
EDA_MAX_US = None             # optional upper bound (Kleckner et al. use 60); None = not checked
EDA_MAX_SLOPE_US_PER_S = 10.0
EDA_ARTIFACT_PAD_S = 5.0
EDA_DECOMPOSITION = "highpass"  # nk.eda_phasic method ("highpass", "cvxeda", "smoothmedian")
SCR_MIN_AMPLITUDE_US = 0.05     # an SCR counts if it rises at least this much from its onset
SCR_MAX_RISE_S = 5.0            # onset = lowest point in the 5 s before the peak (or since the previous peak)

# --------------------------------------------------------------------------
# Review / exclusion (outline 4)
# --------------------------------------------------------------------------
# ECG: exclude a segment when artefacts / corrected (ectopic) beats cover more than this
# % of its duration (Silveri et al., 2021). Time covered = the RR intervals touching each
# corrected beat, plus signal loss and spans the reviewer excluded.
ECG_EXCLUDE_PCT = 20.0
# EDA: flagged portions are removed; the segment is kept unless the reviewer excludes it.
# Set a number (e.g. 50) to also exclude segments automatically above that % removed.
EDA_EXCLUDE_PCT = None
ALLOW_UNREVIEWED = False        # True = compute measures without a review file (marked reviewed=False)

# --------------------------------------------------------------------------
# Recovery time (outline 5.2)
# --------------------------------------------------------------------------
RECOVERY_WINDOW_S = 30.0
RECOVERY_STEP_S = 5.0
RECOVERY_SD_BELOW = 0.0         # 0 = back at baseline RMSSD (mean of baseline 30-s windows);
                                # 1 = within 1 SD below it (more lenient)
# 1 = the FIRST window back at baseline counts (as in the outline); 3 = must stay there
# for 3 windows in a row (more robust to one noisy 30-s window).
RECOVERY_SUSTAIN_WINDOWS = 1

# --------------------------------------------------------------------------
# Behaviour (outline 5.3)
# --------------------------------------------------------------------------
# Trials come from the Trial-TT-n events in the session JSON:
#   decibels[k] = 0 -> silent trial; > 0 -> word played at that level
#   response[k] = 1 / 0 / null (null = not recorded: left out of FP rates and the nu fit)
#   valence     = /neutral/, /negative/ or /positive/ in stimulusInfo.audioFile
VALENCES = ("neutral", "negative", "positive")
# Which word valences make up the "all words" false-positive rate. run-1 (neutral words,
# no MIST) is left out by default. Add "neutral" to include it.
FP_ALL_VALENCES = ("negative", "positive")
# Word intensity for the HGF (0-1): "rank" = the participant's distinct non-zero dB
# levels in order, spaced evenly (3 levels -> 0.25 / 0.5 / 0.75); "present" = 1 for
# every word (intensity ignored).
INTENSITY_FROM_DB = "rank"

# --------------------------------------------------------------------------
# HGF (outline 5.3)
# --------------------------------------------------------------------------
# The HGF always learns from the full trial sequence. "silent" = only silent-trial
# responses enter the fit (as in the outline); "all" = every response (ν is recovered
# much better in simulation: r ~ 0.9 vs ~ 0.65).
HGF_LIKELIHOOD_TRIALS = "silent"
HGF_INPUT = "stimulus_present"  # what the learner observes each trial
HGF_SENSORY_CLIP = 0.05         # silent trial -> sensory p = 0.05
HGF_N_STARTS = 8
HGF_RESET_EACH_RUN = False
HGF_SEED = 0

# --------------------------------------------------------------------------
# Statistics (outline 6)
# --------------------------------------------------------------------------
ALPHA = 0.05
HOLM_CORRECTION = False         # True = Holm across the primary tests (H1 x2, H2 ANCOVA, H2 FP all)
WELCH_TTEST = False             # independent-samples t-tests: False = Student's, True = Welch's
OVERDISPERSION_LIMIT = 1.5      # binomial models: Pearson chi2/df above this -> quasi-binomial SEs


# --------------------------------------------------------------------------
# Switch between real and synthetic data
# --------------------------------------------------------------------------
def use_synthetic(on: bool = True) -> None:
    """Point all paths at synthetic_data/ and synthetic_derived/ (or back to the real ones).

    Synthetic and real outputs never mix: each has its own derived folder.
    """
    global DATA_DIR, DERIVED_DIR
    if on:
        DATA_DIR = PROJECT_ROOT / "synthetic_data"
        DERIVED_DIR = PROJECT_ROOT / "synthetic_derived"
    else:
        DATA_DIR = REAL_DATA_DIR
        DERIVED_DIR = REAL_DERIVED_DIR
    set_paths(DATA_DIR, DERIVED_DIR)


def set_paths(data_dir, derived_dir) -> None:
    """Point the pipeline at another data / derived folder (files derived from them follow)."""
    global DATA_DIR, DERIVED_DIR, PARTICIPANTS_FILE, NU_ESTIMATES_FILE, RUN_TYPES_FILE
    global SEGMENT_OVERRIDES_FILE, CLOCK_OFFSETS_FILE
    DATA_DIR, DERIVED_DIR = Path(data_dir), Path(derived_dir)
    PARTICIPANTS_FILE = DATA_DIR / "participants.csv"
    NU_ESTIMATES_FILE = DATA_DIR / "nu_estimates.csv"
    RUN_TYPES_FILE = DATA_DIR / "run_types.csv"
    SEGMENT_OVERRIDES_FILE = DERIVED_DIR / "segment_overrides.csv"
    CLOCK_OFFSETS_FILE = DERIVED_DIR / "clock_offsets.csv"
