# ACHI analysis pipeline

Python code for the ACHI stress-physiology analysis described in `script_outline_REVISED.md`. It takes each participant's BIOPAC recording, session log and behavioural responses and produces:
- cleaned heart-rate variability (RMSSD), skin conductance level (SCL) and SCR frequency for the baseline, stress MIST, non-stress MIST and post-task windows;
- change scores (stress vs non-stress MIST, stress MIST vs baseline) and recovery time;
- false-positive rates (negative, positive and all words) and prior weighting (ν);
- a participant table (one row per participant), a run table (one row per run) and the hypothesis tests (H1–H3).

**Still to decide:** the open choices, and the edits your script needs to match the code, are in `decisions_and_changes.md`.

## Setting up (once, on your own computer)

You only do this once. Type the commands in a **terminal**: *Terminal* on Mac, *Anaconda Prompt* or *PowerShell* on Windows.

1. **Install Python 3.13** from https://www.python.org/downloads/. On Windows, tick *"Add Python to PATH"* during installation.
2. **Go into this folder** in the terminal. Type `cd ` (with a space), drag the `ACHI_analysis` folder into the terminal window, and press Enter.
3. **Create a private Python environment and install the packages** (takes a few minutes):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Check it works:** `python -m pytest tests -q` should end with `21 passed`.

**Every time you come back:** open a terminal, `cd` into this folder, and run `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`). Then you can type `python ...` commands, or `python -m jupyter lab` to open the notebooks. (If you move this folder, the plain `jupyter` command inside `.venv` stops working; `python -m jupyter lab` still works, or recreate `.venv`.)

## Quick start

```bash
python make_synthetic_data.py     # 1. make 10 fake participants (already done once)
python run_synthetic_demo.py      # 2. run the whole pipeline on them, with a simulated reviewer (~15 s)
python -m pytest tests -q         # 3. run the 21 automated checks
```

Then open **`phsyiological_data_analysis.ipynb`** in Jupyter Lab. With `USE_SYNTHETIC = True` (the default) it runs on the fake data, so you can try every step, including the interactive review, safely.

## What's in this folder

```
ACHI_analysis/
├── README.md                          this file
├── requirements.txt                   list of Python packages to install
├── script_outline_REVISED.md          the analysis plan the code follows (your outline + the choices the code made)
├── decisions_and_changes.md          what you still have to decide, and what to change in your script
├── phsyiological_data_analysis.ipynb  main notebook: run the analysis step by step
├── make_synthetic_data.py             creates synthetic_data/
├── run_synthetic_demo.py              runs everything on synthetic_data/, with a simulated reviewer
├── achi_pipeline/                     the code
├── tests/test_pipeline.py             automated checks
├── synthetic_data/                    fake participants (input)
└── synthetic_derived/                 outputs from the fake participants
```
Real data are read from the `Data/` folder next to `ACHI_analysis/` (never written to); real outputs go to `derived/`. Real and synthetic outputs never mix.

## How the analysis runs

| Stage | Function | What happens | Outline step |
|---|---|---|---|
| 0 | `pipeline.inspect_participant(pid)` | Prints the files found, every session-log event per run, the decoded triggers, and how blocks were matched. Use it to set up `config.py`. | 1 |
| 1 | `pipeline.run_all_preprocessing()` | Loads each recording; cleans ECG and EDA (a missing channel skips only that signal); detects signal loss, beats and artifacts; cuts baseline / MIST / post-task windows per run; writes review sheets and `qc_overview.png`. | 1–3 |
| 2 | `review.ReviewApp(pid).show("ecg")` / `.show("eda")` | **You** check every flag against the raw signal and decide accept / add / remove / move peak / exclude span (ECG) or artefact / not artefact (EDA). You can also exclude whole segments with a reason. | 4 |
| 3 | `pipeline.run_all_compute()` | Applies your decisions and calculates all measures per run, then combines runs into one row per participant. Participants with unreviewed flags are skipped. | 5 |
| 4 | `pipeline.analyse()` | Builds the participant and run tables, the ECG log, and runs the statistics, descriptives and demographics. | 6–7 |

Every skipped file, run or segment is written to `derived/logs/missing_and_errors.txt` with the reason.

## The code: `achi_pipeline/`

**Suggested reading order** is the "Read #" column. It follows the data from raw files to results. `synthetic.py` and `simulated_review.py` only make and check the practice data, so they can stay a black box. For `hgf.py`, understanding *what* it estimates matters more than the maths.

After reading each file, fill in the last column: one or two sentences on what it does, as you'd explain it to a classmate. Keep your questions in a list to bring to our meetings.

| Read # | File | Responsibility | In my own words *(you fill this in)* |
|---|---|---|---|
| 1 | `config.py` | **All settings:** paths, channel names, filter cut-offs, artifact thresholds, the 20% ECG exclusion rule, SCR cut-off, recovery-time rules, which baseline to use, behavioural column names, HGF and statistics options. `use_synthetic(True/False)` switches between fake and real data. | |
| 2 | `pipeline.py` | Runs the stages, plus `inspect_participant()`, the QC plot and the participant- and run-table column lists. | |
| 3 | `io_utils.py` | Finds each participant's files in the flat data folder (by the ID in the file name), loads recordings (`.acq`; `_physio.csv` for synthetic data), the session-log JSON (event times, runs, run types, MIST performance), the word-detection trials inside it, and `participants.csv`. Logs problems. | |
| 4 | `triggers.py` | Turns the 3 digital trigger lines into events, pairs block start/end, **matches blocks to session-log events by timestamp**, builds the segment windows, and applies manual overrides. | |
| 5 | `ecg.py` | ECG filtering, flatline (signal-loss) detection, R-peak detection, Lipponen & Tarvainen beat correction run separately between flatlines, the list of corrected beats, and the RR table with flags. | |
| 6 | `eda.py` | 1 Hz low-pass, artifact flags (< 0.05 µS, slope > 10 µS/s, ±5 s padding), SCL/SCR split, SCR detection (onset-to-peak ≥ 0.05 µS). | |
| 7 | `review.py` | Writes the review sheets, the interactive `ReviewApp`, applies decisions, % of each segment unusable (20% ECG rule), keep/exclude. | |
| 8 | `measures.py` | RMSSD (no differences across bad data), mean SCL, SCR frequency, phasic area per minute, change scores, recovery time, false-positive rates and counts; combines runs into one row per participant. | |
| 9 | `stats.py` | H1 and manipulation check (paired t-tests), SCR, H2 (ANCOVA, t-tests, binomial and linear regressions), H3, order effects, descriptive and demographic tables. | |
| 10 | `hgf.py` | Fits ν per participant (3-level binary HGF + prior-weighting response model), plus `parameter_recovery()`. | |
| – | `synthetic.py` | Generates the fake participants and records the ground truth (true heartbeats, every injected artefact). | |
| – | `simulated_review.py` | **Synthetic data only.** Works out what a careful reviewer would decide for every flag (`expected_decisions`), fills the review sheets with realistic occasional mistakes (`apply`), and scores any review, including yours, against that answer key (`score`). Also computes the true RMSSD per segment (`true_rmssd`). | |

## Synthetic data

`synthetic_data/` has the same flat layout and JSON format as the real `Data/` folder (`ACHI_2026_901` … `910`; 901–905 patients, 906–910 controls). Per participant:
- `ACHI_2026_9XX_session-1_sessiondata.json`: the session log, in the real format (see below). The word-detection trials are inside it;
- `ACHI_2026_9XX_session-1_physio.csv`: ECG, EDA and trigger lines at 200 Hz, a stand-in for the `.acq` file (real ones are 1000 Hz, and their trigger coding is not known yet).

The folder also has `participants.csv`, plus the answers, which the pipeline never reads:
- **`ground_truth.csv`**: per-participant true ν, true baseline RMSSD, MIST effect, run order, list of injected artefacts.
- **`_ground_truth/<pid>_true_beats.csv`**: every true heartbeat.
- **`_ground_truth/<pid>_injected_artifacts.csv`**: every injected problem, with its time.
- **`_ground_truth/<pid>_true_scrs.csv`**: every true skin-conductance response, with its amplitude.
- **`_ground_truth/<pid>_true_runs.csv`**, **`_true_segments.csv`**, **`_true_trials.csv`**: true run types, physiological phases and every simulated trial (including blocks left out of the log).

**Session structure (as in the real study).** Runs `run-1, run-2, run-3, run-5, run-7` (listed in `Start-Session` `data.allRunIds`):
- **run-1**: Start-Session, **RestBaseline** (~4 min, the *one* baseline for the whole session), Setup, IMS, VAS1–4, ACH practice, ACH-WordConfirm, ACH-TrialQ (staircase), NWT and Trial-TT-1/2/3 with **neutral** words. No MIST.
- **runs 2–7**: IMS, VAS1–4, ACH-WordConfirm, ACH-TrialQ, Trial-TT-1, **MIST-Round-practice, MIST-Round-1, MIST-Feedback-1, MIST-Round-2, MIST-Feedback-2**, then Trial-TT-2/3. Words alternate **negative / positive** across runs.
- Each Trial-TT block has 45 trials (`decibels` 0 = silent), responses 1/0/null; about 15% of runs are missing one Trial-TT block.
- 2 stress + 2 non-stress MIST runs, order counterbalanced. The run type comes from the session field `manipulationValence` (negative = stress MIST, positive = non-stress MIST; `config.RUN_TYPE_RULE`), not from the word valence. Cross-check: only the stress MIST has a time limit, so stress runs should have `timedOut > 0` and non-stress runs none (`timeout_check` in the run table; disagreements are logged).

**Segments the pipeline cuts:** baseline = RestBaseline; task = MIST-Round-1 start → MIST-Round-2 end (Feedback-1 included); post-task = 300 s after MIST-Feedback-2.

**Built-in effects:**
- The stress MIST lowers RMSSD and raises SCL and SCR frequency (6/min vs 2/min at rest).
- The non-stress MIST has small effects (RMSSD 92% of baseline, 3 SCRs/min).
- Patients drop more (RMSSD falls to 55% of baseline vs 75% in controls) and recover more slowly (full recovery after ~108 s vs ~72 s).
- Patients have higher ν (~0.65 vs ~0.40).
- No *direct* link is built in between physiology and false positives. But patients have both a bigger RMSSD drop **and** higher ν (so more false positives), so false-positive rate and ν correlate with ΔRMSSD **through group**. The H2 models include group, which partly separates this, but with few participants the confound is worth discussing. Nothing is built in for run order.

**Built-in problems**, each needing a different review decision:

| Problem | Where | Right decision |
|---|---|---|
| Missed beats (R wave not drawn) | 4 per participant | `accept`: keep the beat the algorithm interpolated |
| Sharp spikes between beats | 2 per participant | `remove_peak` if one became a peak |
| Peaks the algorithm shifted off the R wave | varies | `move_peak` back to the R wave |
| Real R wave without a peak | occasional | `add_peak` |
| Movement burst, 3–6 s of unreadable ECG | 1 per participant | `exclude_span` with `start-end` |
| 70-s movement burst | 902, run-2 MIST | `exclude_span`; the segment then exceeds 20% and is excluded |
| 70-s ECG dropout (flat line) | 901, run-2 post-task | `accept` (already excluded); segment just over 20% → excluded |
| EDA jump (3–9 µS, 0.5–2 s) | 1 per participant | `accept` if flagged |
| EDA lost contact (8 s at 0.02 µS) | 903 | `accept` |
| Genuine large, fast SCR | ~60% of participants | `not_artifact` (physiological; keep it in SCL) |

## Simulating the review

The visual check is never skipped. `run_synthetic_demo.py` uses a **simulated reviewer** (`simulated_review.py`) in its place. It makes the decision a careful person would make from the plot (table above), writes a note for each, and overlooks about 5% of needed corrections (`--error-rate`), as a tired reviewer would.

```bash
python run_synthetic_demo.py                    # simulated reviewer, 5% mistakes
python run_synthetic_demo.py --error-rate 0     # perfect reviewer
python run_synthetic_demo.py --review manual    # stops after stage 1 so YOU review
```

**To practise reviewing yourself:**
1. Run `--review manual`, or run stage 1 in the notebook.
2. Review in the notebook's stage 2 cells.
3. Run `simulated_review.score(pid)` to see which decisions differ from the answer key and why.
4. The "practice mode" cell lets the simulated reviewer handle the participants you don't do by hand.

**What the demo reports:**
- **Review workload:** number of flags per participant, and roughly how long reviewing them takes at 15–30 s per flag. Expect more with real 1000 Hz, hour-long recordings.
- **Decisions:** mostly accepts, plus corrections: exclude span, move, add and remove peak, not_artifact.
- **Why the review matters:** RMSSD error against the true heartbeats before review (automatic correction only) and after review. A few unreviewed movement bursts can be enough to reverse the main findings.
- **Segments excluded:** 901 run-2 post-task (70-s dropout) and 902 run-2 MIST (70-s movement burst), both by the 20% rule.
- **Artefacts the automatic flags missed:** small EDA jumps (under ~4 µS) rise slower than 10 µS/s once the signal is 1 Hz low-passed, so they're never flagged and a reviewer never sees them. **Worth discussing:** check the rate on the raw signal, or scan `qc_overview.png` for jumps during review.
- **SCR frequency** against the true SCRs, **recovery time** reached in how many stress runs, the **statistics**, and **ν** against the true ν.

Exact numbers change with `--seed` in `make_synthetic_data.py`.

## Using real data

The real files sit in the flat **`Data/`** folder next to `ACHI_analysis/` (`config.REAL_DATA_DIR`). The pipeline only reads it.
```
Data/ACHI_2026_XXX_session-1_sessiondata.json   session log, with the trials inside (one per participant)
Data/ACHI_2026_XXX_....acq                      BIOPAC recording (not available yet: ECG/EDA are skipped and logged)
Data/participants.csv                           participant_id, group (patient/control), age, antipsychotic (yes/no)
Data/run_types.csv                              optional: participant_id, run_id, run_type (stress / non-stress)
Data/nu_estimates.csv                           optional: participant_id, nu   (fitted in TAPAS; overrides hgf.py)
```
1. In the notebook set `USE_SYNTHETIC = False`.
2. Run `inspect_participant()` on one participant. It prints every event per run with its duration, the run types, missing Trial-TT blocks and a trial summary per run and valence.
3. **Fill in the stress vs non-stress rule**: `RUN_TYPE_RULE` in `config.py` (it receives the run's MIST rounds with correct / incorrect / timedOut), or list the run types in `Data/run_types.csv`. Until then every MIST run is `unknown`, which is logged. H1–H3 and the order comparison stay empty, but FP rates, ν and MIST performance are still computed.
4. Check the other real-data settings in `config.py`: `TASK_EVENTS`, `POST_TASK_AFTER_EVENT` / `POST_TASK_S`, `FP_ALL_VALENCES`, `INTENSITY_FROM_DB` (see `decisions_and_changes.md`).
5. When the `.acq` files arrive: check `EVENT_CODE_MAP` and the channel names, run stage 1, and look at `alignment.json` (how many trigger blocks matched the session log), `qc_overview.png` and `segments.csv` for the first few participants. Fix wrong windows with `derived/segment_overrides.csv` (`participant_id, run_id, segment, start_s, end_s`). If a recording has no usable triggers, give the clock offset in `derived/clock_offsets.csv` (`participant_id, offset_s`, where recording seconds = JSON epoch seconds + offset_s).
6. Review every participant (stage 2), then run stages 3–4.

## Outputs (`derived/` or `synthetic_derived/`)

| File | Contents |
|---|---|
| `results/participant_table.csv` | One row per participant: RMSSD, SCL, SCR frequency and phasic area per minute (µS·s per artefact-free minute) for baseline / stress MIST / non-stress MIST (RMSSD also post-task), the four change scores, recovery time, ν, FP rates and counts (negative, positive, neutral, all), % excluded ECG, % removed EDA, reviewed. |
| `results/run_table.csv` | One row per run (run-1 has type `none`): run type and number, MIST correct / incorrect / timed out, per-run measures, FP rates and counts, % bad and keep/exclude per segment (used for the order comparison). |
| `results/statistics.csv` | Every test: model used, n, estimate (with its scale), 95% CI, p, notes (e.g. overdispersion). |
| `results/descriptive_summary.csv` | Mean, SD and n participants by group. |
| `results/descriptive_scr_by_run_type.csv` | SCR frequency and phasic area per minute by group × run type. |
| `logs/ecg_corrected_beats.csv` | % corrected beats (automatic + manual) per participant and segment, % of duration unusable, keep/exclude. |
| `results/supplementary_demographics.csv` | Age and antipsychotic use by group. |
| `participants/<pid>/qc_overview.png` | Whole session: RR intervals and EDA with segments and signal loss shaded. |
| `participants/<pid>/review/*.csv` | Review sheets: the decisions you entered. |
| `participants/<pid>/…` | Intermediate files: `session_events.csv` (every logged event with its times), `runs.csv`, `trials.csv` (every word-detection trial), segments, alignment, peaks, corrections, cleaned signals. |
| `logs/missing_and_errors.txt` | Everything skipped, and why. |

## Known limitations
- **ν model:** `hgf.py` is an approximation of the Powers et al. (2017) model and hasn't been validated against TAPAS. Run `hgf.parameter_recovery()` on a design like yours, or supply `nu_estimates.csv`. Fitting only silent-trial responses (the default, as in the outline) recovers ν much less well than fitting all responses.
- **Recovery time:** "back to baseline RMSSD" is strict for noisy 30-s windows; about half of stress runs may never reach it, which leaves H3 with too few participants. See `RECOVERY_SD_BELOW`.
- **Segment matching** has only been tested on synthetic session logs and synthetic triggers (one block per timed event). How the real `.acq` triggers are coded is not known yet. Check `alignment.json` for the first real participants.
- **No rest after the MIST:** the post-task window (recovery time, RMSSD post-task) overlaps VAS3 and the next word-detection block.
- **Synthetic data** are simplified (clean ECG shapes, 200 Hz, short segments). Passing on them shows the code works, not that the thresholds suit the real recordings.
- **EDA rate rule** misses small jumps (see above). Only flagged periods are shown to the reviewer.
- **The simulated reviewer** knows the truth. A real reviewer judges from the plots and will disagree with it in genuinely ambiguous places, so treat `score()` as a training aid, not a gold standard.
