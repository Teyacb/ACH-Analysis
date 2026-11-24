# Script Outline (revised)

Your current script outline (version of 14 Nov 2026), written out in the detail the code needs. Text marked **[CODE]** is something the code does that your outline doesn't say yet. Text marked **[DECIDE]** is a default the code picked where your outline leaves the choice open; the setting to change it is in `achi_pipeline/config.py`. The code in `achi_pipeline/` and `phsyiological_data_analysis.ipynb` implements this version.

---

## Choices the code makes

| # | Topic | What the code does | Setting |
|---|-------|--------------------|---------|
| 1 | ν fit | The HGF learns from the full trial sequence; only silent-trial responses enter the fit, as in the outline. **[DECIDE]** In simulation ν is recovered far better with all responses (r ≈ 0.91 vs ≈ 0.65). | `HGF_LIKELIHOOD_TRIALS = "silent"` |
| 2 | Recovery time | Rolling 30-s RMSSD in 5-s steps over the post-task window. Recovered = the first window at or above the baseline mean (mean of baseline 30-s windows). Time = end of the first such window minus post-task start. **[DECIDE]** About half of stress runs never reach this, so H3 may not run. | `RECOVERY_SD_BELOW = 0.0`, `RECOVERY_SUSTAIN_WINDOWS = 1` |
| 3 | Baseline and post-task | One baseline for the whole session (`RestBaseline`, in run-1), used for every run. Post-task = the 300 s after `MIST-Feedback-2` of the stress run. There is no rest after the MIST, so this window overlaps VAS3 and Trial-TT-2. **[DECIDE]** Keep it (and say so) or shorten it. | `BASELINE_EVENT`, `POST_TASK_AFTER_EVENT`, `POST_TASK_S = 300` |
| 4 | Multiple comparisons | None, as the outline names none. **[DECIDE]** Holm across the 4 primary tests is available. | `HOLM_CORRECTION = False` |
| 5 | 20% ECG rule | % of segment duration covered by the RR intervals on both sides of each corrected beat (automatic + manual), signal loss, and spans excluded in review. | `ECG_EXCLUDE_PCT = 20.0` |
| 6 | EDA exclusion | Flagged portions are removed; a segment is excluded only by the reviewer, with a reason. | `EDA_EXCLUDE_PCT = None` |
| 7 | EDA range | Below 0.05 µS is flagged. No upper limit (Kleckner et al. use 60 µS). **[DECIDE]** | `EDA_MAX_US = None` |
| 8 | Group t-tests | Student's independent-samples t-test. **[DECIDE]** | `WELCH_TTEST = False` |
| 9 | ν regression (H2) | ν ~ ΔRMSSD + group, to match the binomial model. **[DECIDE]** | `stats.py` |
| 10 | Binomial models | "yes" count out of silent trials, logit link. If Pearson χ²/df > 1.5, SEs are scaled (quasi-binomial). | `OVERDISPERSION_LIMIT = 1.5` |
| 11 | Order comparison | Run table. ΔRMSSD (that run's MIST − session baseline): mixed model, random intercept for participant, run type as covariate. FP rate: binomial with participant-clustered SEs, run type as covariate. | `stats.py` |
| 12 | Too few data | A model is not fitted with fewer than 2 residual degrees of freedom; a paired test needs 3 participants with both values. | `stats.py` |

---

## Revised outline

**1. Inputs.** All files sit in one `Data/` folder. Each participant's raw BIOPAC file (`.acq`) and session log (`ACHI_2026_XXX_session-1_sessiondata.json`) are matched by the participant ID in the file name (`ACHI_2026_XXX`). The trial-by-trial responses are inside the session log, so there is no separate behavioural file. **[CODE]** Group, age and antipsychotic use come from `participants.csv`. The triggers mark only when a block starts and ends, so each trigger block is matched to its session-log event by timestamp (best constant offset); event names are mapped to baseline / stress MIST / non-stress MIST / post-task in `SEGMENT_EVENTS`. Wrong windows can be fixed in `segment_overrides.csv`.

**2. Participant table.** One row per participant. Columns: participant ID, group, age, antipsychotic use, RMSSD baseline / stress MIST / non-stress MIST / post-task, SCL baseline / stress MIST / non-stress MIST, SCR frequency baseline / stress MIST / non-stress MIST (per minute), ΔRMSSD stress (stress − non-stress MIST), ΔSCL stress, ΔRMSSD (stress MIST − baseline), ΔSCL (stress MIST − baseline), RMSSD recovery time (s), ν, false-positive rate for negative, positive and all words, "yes" and silent-trial counts for each, percent excluded ECG, percent signal removed EDA. **[CODE]** Also: number of stress and non-stress runs, source of ν, reviewed (yes/no).
- Percent excluded ECG = share of the participant's analysed ECG time not used: excluded segments count in full, kept segments count their unusable share. Percent signal removed EDA = the same over the baseline and MIST segments.
- Age and antipsychotic use are recorded for the supplementary section (Clamor et al., 2016; Boucsein et al., 2012).
- A second table, one row per run, holds run type and run number, **[CODE]** plus that run's RMSSD, SCL, SCR frequency, ΔRMSSD and ΔSCL (that run's MIST − session baseline), recovery time, and FP rates and counts, for the order comparison.

**3. Per participant.**
1. **Load ECG and EDA.** Read the raw ECG, EDA and digital trigger channels. Log any missing file or channel (participant ID, file name, reason) to `logs/missing_and_errors.txt`. A missing ECG channel skips the ECG steps; a missing EDA channel skips the EDA steps. **[CODE]** A missing recording or missing trigger channels skip both, but FP rates and ν are still computed from the session log; a missing session log skips the participant, because runs, segments and trials all come from it.
2. **Clean ECG.** 0.5 Hz high-pass filter (5th-order Butterworth) to remove baseline drift (Quintana et al., 2016) **[CODE]** and a 60 Hz power-line filter. **[CODE]** Detect signal loss first: 1-s windows with peak-to-peak amplitude < 0.05 mV. Detect heartbeats, then correct abnormal beats with the Lipponen & Tarvainen (2019) method (each interval compared with the surrounding 91 beats; deviations > 5.2 × the local quartile deviation flagged), separately within each stretch of good signal and never across signal loss. Beats are corrected by inserting, removing or moving them, not by deleting intervals, because deleting beats distorts RMSSD (Peltola, 2012). **[CODE]** Remaining suspicious RR intervals (< 300 ms, > 2000 ms, or a jump > 250 ms) are flagged for review. The percentage of corrected beats per segment (automatic + manual, after review) is written to `logs/ecg_corrected_beats.csv`, one row per participant and segment.
3. **Clean EDA.** 1 Hz low-pass filter on the whole recording (Posada-Quintero & Chon, 2020). Flag samples below 0.05 µS (lost contact) and samples where the low-passed signal changes faster than 10 µS/s; extend each flag by 5 s on each side (Kleckner et al., 2018). Flagged samples are left out, not interpolated. Split the signal into SCL (tonic) and SCR (phasic). **[CODE]** An SCR is a rise of at least 0.05 µS from onset to peak in the low-passed signal, with onset = lowest point in the 5 s before the peak (or since the previous peak), so overlapping responses count separately (Boucsein et al., 2012). An SCR counts only if its onset and peak are both unflagged.
4. **Cut segments.** **[CODE]** The session has one baseline (`RestBaseline`, in run-1), used for every run. Each later run contains a MIST: the stress version in stress runs and the non-stress version in non-stress runs (Dedovic et al., 2005). The MIST segment runs from the start of MIST-Round-1 to the end of MIST-Round-2. The four windows are: baseline, stress MIST, non-stress MIST, and post-task = the 300 s after MIST-Feedback-2 of the stress run (choice 3). There is no rest period after the MIST, so the post-task window overlaps VAS3 and Trial-TT-2. Runs of the same type are averaged. Filtering is done on the whole recording before cutting. Run type and run number are recorded in the run table.

**4. Visual check.** Automated detection followed by visual inspection (Quigley et al., 2024).
- **ECG:** review every corrected beat, remaining RR flag and signal-loss period against the raw and cleaned ECG. For each: accept, add a peak, remove a peak, move a peak, or exclude a span. Record keep/exclude with a reason for each segment. A segment is always excluded when artefacts or corrected beats cover more than 20% of its duration (Silveri et al., 2021; see choice 5).
- **EDA:** review each flagged region (lost contact, sudden jumps) in the baseline and MIST segments against the raw EDA. Mark it as an artefact (removed) or not an artefact (kept, e.g. a genuine large SCR). Record keep/exclude with a reason for each segment.
- **[CODE]** Participants with unreviewed flags are not processed further.

**5. Measures (per run, then per participant).**
1. **ECG:** RMSSD (ms) per segment from the final beat intervals, using only differences between adjacent valid intervals (none across excluded periods). RMSSD is the main vagal index (Singh et al., 2018), less affected by breathing rate than spectral measures (Penttilä et al., 2001; Quigley et al., 2024). Post-task: RMSSD only.
2. **EDA:** for baseline and MIST segments, mean SCL (flagged samples left out; Boucsein et al., 2012) and SCR frequency = SCRs ÷ minutes of artefact-free signal.
3. **Change scores:** ΔRMSSD stress and ΔSCL stress (stress − non-stress MIST); ΔRMSSD and ΔSCL (stress MIST − baseline).
4. **Recovery time:** seconds from the start of the post-task window until RMSSD is back at baseline (choice 2). If not reached within the post-task window, it is missing. Longer = slower recovery (Bamert & Inauen, 2022).
5. **Prior weighting and false positives.** Fit the Hierarchical Gaussian Filter (Mathys et al., 2014) once per participant. **[CODE]** The model learns from the full trial sequence of all runs in session order; the fit uses silent-trial responses (choice 1). The response model weights prior expectation against sensory evidence with ν (Powers et al., 2017); higher ν = greater reliance on priors. The Python model is an approximation (see below); ν values fitted in TAPAS can be supplied in `nu_estimates.csv`. False-positive rate = silent trials answered "yes" ÷ silent trials, for negative words, positive words and all words. **[CODE]** Trials without a response are left out.

**6. Statistics.** On the participant table (one row per participant); the run table is used only for the order comparison. Runs of the same type are averaged first. Significance p < .05, two-sided; exact p-values and 95% CIs reported. Multiple comparisons: choice 4.
1. **H1:** paired t-tests of stress vs non-stress MIST on RMSSD and SCL (equivalently, whether ΔRMSSD stress < 0 and ΔSCL stress > 0). Manipulation check: paired t-tests of stress MIST vs baseline on RMSSD (lower) and SCL (higher).
2. **SCR frequency** (secondary, descriptive): paired t-test of stress vs non-stress MIST, uncorrected; summarised by group × run type.
3. **H2:** ANCOVA of ΔRMSSD (stress MIST − baseline) on group, adjusting for baseline RMSSD (primary; Quigley et al., 2024). Also the unadjusted independent-samples t-test of ΔRMSSD, and a t-test of baseline RMSSD between groups (choice 8). False positives: binomial regression (logit link), "yes" count out of silent trials = intercept + ΔRMSSD + group (Warton & Hui, 2011); all words primary, negative and positive words secondary (choice 10). ν: linear regression on ΔRMSSD + group (secondary; choice 9). Expected: negative association.
4. **H3 (exploratory):** linear regression of ν and binomial regression of the overall FP rate on recovery time, group and recovery × group. Expected: positive association, stronger in patients.
5. **Secondary:** run 1 vs later runs on ΔRMSSD and the false-positive rate (choice 11).

**Supplementary.** Age and antipsychotic use by group; their possible influence is a limitation. Respiration was not recorded; breathing rate and depth, and speech during the task, can change HRV independently of vagal activity, so within-person RMSSD changes are interpreted with caution (Penttilä et al., 2001; Quigley et al., 2024).

**7. Descriptive summary.** Mean, SD and number of participants by group for each segment's RMSSD, SCL and SCR frequency and each change score. SCR frequency is also summarised by group × run type. The tests use the individual rows, not these averages.

---

## HGF / ν implementation note

`achi_pipeline/hgf.py` implements:
- the standard 3-level binary HGF update equations (Mathys et al., 2011, 2014; same form as TAPAS `tapas_hgf_binary`). ω₂ is estimated; κ = 1 and ω₃ = −6 are fixed.
- a response model for detection reports: `logit P(yes) = ζ · [ν · logit(μ̂₁) + (1 − ν) · sensory_logodds]`, where μ̂₁ is the trial's prior expectation that a word is present and `sensory_logodds` is the clipped logit of stimulus intensity (silent → logit(0.05)).
- MAP estimation of (ω₂, ν, ζ) with weak Gaussian priors and several random starts.

This captures the idea in Powers et al. (2017) but is **not** their exact model and has not been validated against TAPAS. Before reporting ν, check parameter recovery with `hgf.parameter_recovery()` (it uses the current `HGF_LIKELIHOOD_TRIALS`), or fit ν in TAPAS and supply `nu_estimates.csv` (`participant_id, nu`).

## References (additions and corrections to your list)

Boucsein W, Fowles DC, Grimnes S, Ben-Shakhar G, **Roth WT**, Dawson ME, Filion DL; Society for Psychophysiological Research Ad Hoc Committee on Electrodermal Measures. Publication recommendations for electrodermal measurements. *Psychophysiology*. 2012;49(8):1017–34.

**[ADD]** Powers, A. R., Mathys, C., & Corlett, P. R. (2017). Pavlovian conditioning–induced hallucinations result from overweighting of perceptual priors. *Science*, 357(6351), 596–600. https://doi.org/10.1126/science.aan3458

**[ADD]** Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011). A Bayesian foundation for individual learning under uncertainty. *Frontiers in Human Neuroscience*, 5, 39. https://doi.org/10.3389/fnhum.2011.00039

**[CHECK]** Silveri et al. (2021) is a conference paper on detecting ischemic heart disease with machine learning. Check that it states the 20% exclusion rule before citing it for that.

All other references are as in your outline.
