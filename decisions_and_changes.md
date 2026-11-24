# ACHI script: decisions and changes

14 Nov 2026, updated 17 Nov 2026 for the real session-file format, 20 Nov 2026 for the SCR notch fix, and 21 Nov 2026 for phasic area per minute (see **Code changes log**)

## How to use this doc

The code in `achi_pipeline/` now follows your latest script outline. Where the outline left something open, the code picked a default; those choices are listed under **Decisions to make**. Where the code does something your outline doesn't say yet, the wording to add is under **Changes to make in the script**.

- Write your choice under *Your decision*. Every default can be changed with one setting in `achi_pipeline/config.py`.
- Tick a change off (`[x]`) once your script says it.
- Put anything you're unsure about in the notes section at the end.

## Decisions to make

Sixteen choices are open (item 13, run types, was settled on 21 Nov). The code currently runs with the default shown; items 1–4 change results the most.

**1. Which responses enter the ν fit**
- Current default: silent-trial responses only, as your script says. The model still learns from every trial. (`HGF_LIKELIHOOD_TRIALS = "silent"`)
- Other options: `"all"` = every response. In simulation, ν was recovered at r ≈ 0.91 with all responses vs r ≈ 0.65 with silent only. Or fit ν in TAPAS and supply `nu_estimates.csv`.
- Your decision:

**2. What "back to baseline" means for recovery time**
- Current default: the first time rolling 30-s RMSSD (5-s steps) reaches the baseline mean, as your outline says ("the first time RMSSD reaches its baseline"). (`RECOVERY_SD_BELOW = 0.0`, `RECOVERY_SUSTAIN_WINDOWS = 1`; changed from 3 on 17 Nov to match the outline)
- Other options: `RECOVERY_SUSTAIN_WINDOWS = 3` = must stay at baseline for 3 windows in a row, so one noisy window can't count as recovery (then say so in the script). `RECOVERY_SD_BELOW = 0.5` or `1.0` = within that many SDs below baseline. The strict version left about half of stress runs with no recovery time, so H3 may not run. The lenient version also hid the slower recovery in patients in the test data.
- Your decision:

**3. Which baseline is "the" baseline** *(settled by the design)*
- The session has one RestBaseline (in run-1); it is the baseline for every run's change scores and recovery time. (`BASELINE_EVENT = "RestBaseline"`; the old `BASELINE_FROM` setting is gone.)

**4. Multiple-comparison correction**
- Current default: none (your script names none). (`HOLM_CORRECTION = False`)
- Other options: `True` = Holm across the 4 primary tests (H1 RMSSD, H1 SCL, H2 ANCOVA, H2 FP all words).
- Your decision:

**5. How "20% of duration" is measured**
- Current default: each corrected beat covers the RR intervals on both sides of it; signal loss and spans you exclude in review also count. (`ECG_EXCLUDE_PCT = 20.0`)
- Other options: count only corrected beats, or only time without a readable QRS.
- Your decision:

**6. Automatic EDA exclusion**
- Current default: none. Flagged parts are removed, and a segment is excluded only if you exclude it, with a reason. (`EDA_EXCLUDE_PCT = None`)
- Other options: a number, e.g. `50` = exclude a segment when more than 50% is removed.
- Your decision:

**7. Upper EDA limit**
- Current default: not checked (your script only names < 0.05 µS). (`EDA_MAX_US = None`)
- Other options: `60`, as in Kleckner et al. (2018), to catch saturation.
- Your decision:

**8. Independent-samples t-tests (H2)**
- Current default: Student's t-test. (`WELCH_TTEST = False`)
- Other options: `True` = Welch's, safer when group variances differ.
- Your decision:

**9. Predictors in the ν regression (H2)**
- Current default: ΔRMSSD + group, to match the binomial model. (in `stats.py`)
- Other options: ΔRMSSD only.
- Your decision:

**10. Overdispersion in binomial models**
- Current default: if Pearson χ²/df > 1.5, standard errors are scaled (quasi-binomial). (`OVERDISPERSION_LIMIT = 1.5`)
- Other options: always binomial; always quasi-binomial; beta-binomial.
- Your decision:

**11. Order comparison models**
- Current default: ΔRMSSD uses a mixed model with a random intercept, adjusting for run type. FP rate uses a binomial model with participant-clustered SEs, adjusting for run type. (in `stats.py`)
- Other options: paired t-test of run 1 vs run 2.
- Your decision:

**12. Keep the steps your script doesn't mention?**
- Current default: yes. The 60 Hz power-line filter, flatline (signal-loss) detection before beat correction, and review of remaining RR flags (< 300 ms, > 2000 ms, jumps > 250 ms). (`POWERLINE_HZ`, `FLATLINE_RANGE_MV`, `RR_*`)
- Other options: drop them. Not recommended: without flatline detection, dropouts are filled with invented beats.
- Your decision:

**13. Stress vs non-stress runs** *(settled 21 Nov 2026)*
- Rule: the session field `manipulationValence` of the run: negative = stress MIST, positive = non-stress MIST (`RUN_TYPE_RULE`, `MANIPULATION_TO_RUN_TYPE` in `config.py`). This is the MIST manipulation, not the word valence of the Trial-TT blocks. A run with no value, another value, or events that disagree is `unknown` and logged. `Data/run_types.csv` still overrides it per run if needed.
- Cross-check: only the stress MIST has a time limit, so stress runs should have timed-out questions and non-stress runs none. The run table has `manipulation_valence` and `timeout_check` (agrees / disagrees / not checked); a disagreement is logged.
- To confirm on the first real file: where the field sits. The code looks for it in any event of the run (including nested data), so it works wherever it is stored per run. If it is stored only once per session (e.g. in Start-Session as a list), every run will come out `unknown` and the code needs a small change.

**14. Task window and post-task window**
- Current default: task = MIST-Round-1 start → MIST-Round-2 end, so MIST-Feedback-1 is included and the practice round and Feedback-2 are not. Post-task = 300 s after MIST-Feedback-2 ends. (`TASK_EVENTS`, `POST_TASK_AFTER_EVENT`, `POST_TASK_S`)
- Points to decide: does the feedback belong to the stress manipulation? And there is **no rest after the MIST** in this design: the post-task window overlaps VAS3 and Trial-TT-2. That affects RMSSD post-task and recovery time (H3). Options: shorter window (e.g. only until Trial-TT-2 starts), or drop the post-task measures.
- Your decision:

**15. Which words make up "all words" for the FP rate**
- Current default: negative + positive (the MIST runs). run-1's neutral words get their own rate (`fp_rate_neu`) but are not in "all". (`FP_ALL_VALENCES`)
- Other options: add `"neutral"`.
- Your decision:

**16. Word intensity for the HGF**
- Current default: each participant's distinct non-zero dB levels in order, spaced evenly on 0–1 (3 levels → 0.25 / 0.5 / 0.75). (`INTENSITY_FROM_DB = "rank"`)
- Other options: `"present"` = every word counts as fully present (ignores loudness); or scale dB relative to the participant's ACH-TrialQ threshold (not built yet; check what the real dB values look like first).
- Your decision:

**17. SCR frequency vs phasic area per minute** *(added 20 Nov 2026, changed 21 Nov 2026)*
- Current default: both are computed. Phasic area per minute = area of the phasic component above zero (µS·s) ÷ artefact-free minutes, over the same artefact-free samples as SCL. It does not use SCR detection, so overlapping SCRs that the count merges still add their area (your lab manager's MIST concern). Reported in both tables and the descriptive summaries only; no statistical test uses it yet.
- Points to decide: (a) which measure to report, or both, after comparing them on 2–3 real participants (`checks/scr_overlap/`); plan is to confirm this with your lab manager. (b) Whether phasic area gets a test like SCR frequency (paired stress vs non-stress MIST, secondary) or stays descriptive. (c) The split method. With the current 0.05 Hz high-pass split (`EDA_DECOMPOSITION = "highpass"`), the phasic signal keeps only the fast part of each SCR and swings below zero after it. So the area is not additive: two SCRs 3–6 s apart give only 65–75% of their separate areas, while SCRs < 1 s apart (counted as one) keep ~95%. `"cvxeda"` (Greco et al., 2016) gives a non-negative phasic driver without this undershoot, but it is slower and also changes SCL; try it on the same participants before deciding.
- Your decision:

## Changes to make in the script

These edits make your outline describe what the code does. Most add detail rather than change the method; those that depend on a decision above say which one.

### Inputs
- [ ] Name `participants.csv` (participant_id, group, age, antipsychotic) as the source of group, age and antipsychotic use.
- [ ] Describe the session file: one JSON per participant (`ACHI_2026_XXX_session-1_sessiondata.json`) holding every event with its start and end time, and the word-detection trials (Trial-TT blocks: dB level, 0 = silent; response; rating; valence from the audio file). There is no separate behavioural file.
- [ ] Describe the session: run-1 (RestBaseline, practice, neutral word blocks, no MIST) and four MIST runs (run-2, -3, -5, -7) with negative or positive words; two stress and two non-stress (rule: decision 13).
- [ ] Say how files are matched to the session log: the triggers only mark when a block starts and ends, not which task it is, so each block is matched to a session-log event by its start and end time. Wrong windows can be fixed by hand (`segment_overrides.csv`).
- [ ] Add: a Trial-TT block missing from the log is logged; its trials are simply absent.

### Participant table
- [ ] Define **percent excluded ECG**: share of the participant's analysed ECG time not used. Excluded segments count in full; kept segments count their unusable share.
- [ ] Define **percent signal removed EDA** the same way, over the baseline and MIST segments.
- [ ] Say that RMSSD post-task comes from the post-task window after the stress MIST.
- [ ] Expand the run table: besides run type and run number, it holds MIST performance (correct, incorrect, timed out), per-run RMSSD, SCL, SCR frequency, ΔRMSSD (MIST minus that run's own baseline), FP rates and counts. The order comparison needs these.
- [ ] Optional: the table also stores the number of stress and non-stress runs, where ν came from, and whether the participant was reviewed.

### Load ECG and EDA
- [ ] Add: a missing session log skips the participant. A missing recording skips the ECG and EDA steps (FP rates and ν are still computed from the session log). All are logged.
- [ ] Add: missing trigger channels skip the ECG and EDA steps.

### Clean ECG
- [ ] Add the 60 Hz power-line filter (the 0.5 Hz high-pass is a 5th-order Butterworth).
- [ ] Add signal-loss detection before beat correction: 1-s windows with peak-to-peak amplitude < 0.05 mV. Beat correction runs separately on each stretch of good signal and never fills in across signal loss (decision 12).
- [ ] Reword "interpolate single bad beats": the method corrects by inserting, removing or moving beats, never by deleting intervals.
- [ ] Add: remaining suspicious RR intervals (< 300 ms, > 2000 ms, or a jump > 250 ms) are also flagged for review.
- [ ] Say when the ECG log is written: after the visual check, so % corrected beats includes your manual corrections. File: `logs/ecg_corrected_beats.csv`.

### Clean EDA
- [ ] Say the rate-of-change check uses the 1 Hz low-passed signal.
- [ ] Say flagged samples are left out, not interpolated.
- [ ] Add the upper limit if you keep it (decision 7).
- [ ] Define an SCR: a rise of at least 0.05 µS from onset to peak in the low-passed signal; onset = lowest point in the 5 s before the peak (or since the previous counted SCR, if that is closer). Any dip separates two SCRs; there is no minimum dip depth. *(Wording changed 20 Nov: "previous peak" → "previous counted SCR", see the code changes log.)*
- [ ] Add: an SCR counts only if its onset and peak are both outside flagged regions. SCR frequency = SCRs ÷ artefact-free minutes.

### Cut segments
- [ ] Describe the windows: one session baseline (RestBaseline, run-1); per MIST run a task window (decision 14) and a post-task window (decision 14). The four windows are then: baseline, stress MIST, non-stress MIST, and post-task after the stress MIST.
- [ ] Add: when a participant has more than one run of a type, those runs are averaged (your statistics section says this; the measures section should too).

### Visual check
- [ ] List the ECG review options: accept, add a peak, remove a peak, move a peak, or exclude a span.
- [ ] Define what counts toward the 20% (decision 5).
- [ ] EDA: say each flagged region is marked as artefact (removed) or not an artefact (kept, e.g. a genuine large SCR).
- [ ] Add: participants with unreviewed flags are not processed.

### Calculate measures
- [ ] RMSSD: add that successive differences are only taken between adjacent valid intervals, never across excluded periods.
- [ ] SCL: add that flagged samples are left out of the mean.
- [ ] Add phasic area per minute: area of the phasic component above zero (µS·s) divided by artefact-free minutes, per segment, with flagged samples left out (decision 17). Add it to the participant-table and run-table column lists.

### Change scores
- [ ] Replace the recovery-time sentence with the exact rule from decision 2: rolling 30-s windows in 5-s steps, the threshold, how many windows in a row, and that time is measured to the end of the first window that meets it.

### Prior weighting
- [ ] Reword the HGF step: the model learns from the full trial sequence of all runs in session order; the fit uses silent-trial responses (or all responses, decision 1).
- [ ] Attribute ν to Powers et al. (2017); keep Mathys et al. (2014) for the HGF itself.
- [ ] Add: trials with no response are left out of the FP rates.

### Statistics
- [ ] H1: reword to "paired t-tests of stress vs non-stress MIST values". A paired test on ΔRMSSD stress is the same as testing whether ΔRMSSD stress differs from 0; the current wording says both.
- [ ] Manipulation check: name the test (paired t-tests).
- [ ] H2 ν regression: name its predictors (decision 9).
- [ ] H2 binomial models: say how overdispersion is handled (decision 10).
- [ ] Secondary order comparison: name the models (decision 11).
- [ ] State the multiple-comparison approach (decision 4).
- [ ] Name the t-test type for the group comparisons (decision 8).
- [ ] Add: a model is not fitted when there are too few participants for its parameters (fewer than 2 residual degrees of freedom). This can affect H3 if many recovery times are missing.

### Descriptive summary
- [ ] Say SCR frequency is also summarised by group × run type in a separate table (your H1 section already says this).
- [ ] Add phasic area per minute to the descriptive summary and to the group × run type table.

## Code changes log

**21 Nov 2026**
- **Run types from `manipulationValence`** (decision 13). `RUN_TYPE_RULE` now reads the run's `manipulationValence` (negative = stress, positive = non-stress); the old placeholder and the synthetic-only "time-outs > 0" rule are gone. New run-table columns `manipulation_valence` and `timeout_check`; disagreements with the time-outs are logged. The synthetic session files now carry the field on every MIST round (regenerated with the same seed; recordings, ground truth and all other JSON content are unchanged). On the 10 synthetic participants all 40 MIST runs match the true run types and agree with the time-outs, including the 20 runs where word valence and manipulation differ. This agreement is built into the fake data, so repeat the check on real files. `synthetic_derived/*/runs.csv` gets the new columns on the next preprocessing run.
- **Tests:** added `test_run_type_from_manipulation_valence_not_word_valence` (stress MIST with positive words, non-stress with negative words, a non-stress run with time-outs, a run whose events disagree, and synthetic sessions against their true run types). All 25 tests pass.
- **Mean SCR amplitude removed; phasic area per minute added instead.** Your lab manager asked for SCR frequency and a measure that reflects how long responses stay high, so mean SCR amplitude (added 20 Nov) is gone from the code, tables and tests. New columns `phasic_area_baseline` / `phasic_area_task` (run table) and `phasic_area_baseline` / `phasic_area_stress` / `phasic_area_nonstress` (participant table), also in both descriptive tables (`measures.phasic_area_per_min`). Unit: µS·s per artefact-free minute. A segment with no usable EDA gives a missing value; a segment with no phasic activity gives 0. Results already in `synthetic_derived/` get the new columns on the next compute run.
- **Overlap check now compares the two measures** (`checks/scr_overlap/`, output `count_vs_area.csv`; `count_vs_amplitude.csv` deleted). Two 0.3 µS SCRs 1 s apart count as 1 SCR but give twice the area of one SCR, so the area picks up what the count misses. Limitation: with the high-pass split the area is not additive (65–75% for SCRs 3–6 s apart; decision 17c). On synthetic participants 901 and 902 the two measures ranked the segments similarly; synthetic data only.
- **Tests:** added `test_phasic_area_per_min` and `test_phasic_area_keeps_overlapping_scrs_that_the_count_merges`; removed the amplitude checks. All 24 tests pass.

**20 Nov 2026**
- **Mean SCR amplitude added** *(removed 21 Nov, see above)*: mean onset-to-peak amplitude of the counted SCRs, per segment.
- **SCR notch fix** (`eda.find_scrs`). Before, the onset search restarted after every local peak, even a rejected one. A small notch on a rise therefore split one response into two parts below 0.05 µS, and neither was counted: four test responses with total rises of 0.053–0.076 µS were counted as 0. Now the search only restarts after a counted SCR, so these count as 1. On synthetic participants 901 and 902 the SCR counts did not change; one mean amplitude moved by 0.002 µS.
- **Overlap check** (`checks/scr_overlap/`, rerun after the fix). Two overlapping SCRs are counted separately once their onsets are ≥ 2 s apart (0.3 + 0.3 µS), ≥ 1.5 s (0.1 then 0.3), ≥ 4.75 s (0.3 then 0.1) or ≥ 5.25 s (0.06 + 0.06); closer than that they merge into one. Noise of 0.02 µS SD did not change this. Counting in the phasic component separates slightly earlier but undershoots below zero before large SCRs, which supports counting in the low-passed signal. Synthetic data only; repeat on 1–2 real recordings.
- **Tests:** added `test_scr_notch_does_not_split_a_response` and amplitude checks; all 22 tests pass.

## Citations and references

- [ ] **Silveri et al. (2021)**: check that it actually states a 20% exclusion rule. It is a conference paper on detecting ischemic heart disease with machine learning, so it may not be the right source for an HRV editing threshold.
- [ ] **ν**: cite Powers, A. R., Mathys, C., & Corlett, P. R. (2017). Pavlovian conditioning–induced hallucinations result from overweighting of perceptual priors. *Science*, 357(6351), 596–600. https://doi.org/10.1126/science.aan3458
- [ ] **HGF equations**: optionally add Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011). A Bayesian foundation for individual learning under uncertainty. *Frontiers in Human Neuroscience*, 5, 39. https://doi.org/10.3389/fnhum.2011.00039
- [ ] **Boucsein et al. (2012)**: "roth WT" → "Roth WT".
- [ ] **Kleckner et al. (2018)**: if you keep the 60 µS upper limit (decision 7), cite it there too.
- [ ] **Welch's t-test** (only if you choose it, decision 8): add a source, e.g. Delacre, Lakens & Leys (2017).

## Notes and questions for meetings

- 
- 
- 
