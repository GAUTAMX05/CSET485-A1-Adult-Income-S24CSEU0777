# CHANGELOG

Assignment 1, Adult Census Income, seed 777, roll no S24CSEU0777.
This file explains, in plain language, every change between the version I submitted first
(commit "CSET485 A1: Adult Income stress-testing seed 777", referred to below as **v1**) and this
corrected version (**v2**). Nothing here is meant to disguise anything — it is a list of
methodological fixes and what each one did to the numbers.

---

## 1. Preprocessing leakage (task 2)

**What was wrong.** In v1 I read the whole 48842-row dataset, imputed the missing values, computed
the outlier caps and built the dummy columns *before* splitting into train and test. That means the
test rows were used to decide the train rows' imputed values and clipping bounds — classic
train/test leakage. The train and test numbers were not independent.

**What I changed.** I now split the raw frame first (`train_test_split(..., test_size=0.30,
random_state=777, stratify=...)`), then fit only on the training split:
- missing-value modes: workclass `Private`, occupation `Prof-specialty`, native-country `United-States`
  (these happen to be the same values the full dataset produced, so this change is mostly about method)
- outlier caps for `capital-gain`, `capital-loss`, `fnlwgt` from the **train 99.5th percentile**
- `hours-per-week` (80) and `age` (75) stay as fixed a-priori domain caps, because they are not
  taken from any data
- dummy columns are fixed from the training split and forced onto the test split
- the scaler is fit on train only and applied to test (this was already true in v1)

**A finding worth knowing about.** The train 99.5th percentile of `capital-gain` is 99999, which is
also the dataset maximum, because about 0.5% of rows carry that single value. So the fitted cap for
capital-gain does nothing. I left it as fitted rather than quietly choosing a tighter cap, because
picking a cap after looking at the test data would be data-snooping. The consequence is that the
extreme `capital-gain` rows are still in the model, which is why the capital-gain coefficient is
now large (2.33) and why the OLS range is wider. That is a limitation I state in the report.

**fnlwgt.** `fnlwgt` is the CPS **sampling weight** — how many people in the population the row
represents, not a property of the person. I **kept** it, for two reasons: the brief asks for the
variables as given, and dropping it would change every other coefficient and make v1 and v2
incomparable. Its coefficient is small, and it is recorded explicitly in `results.json`
(`"fnlwgt_kept": true`) with the note that it is a weight, not a personal trait.

**Numbers that moved because of this:** OLS median 0.2141 → **0.2194**, mismatch 4488 → **4514**,
model A accuracy 71.18% → **70.92%**, model B 85.13% → **85.09%**.

---

## 2. Logistic regression penalty (task 3)

**What was wrong.** v1 used `LogisticRegression()` everywhere, which scikit-learn gives a default
**L2 penalty** (`C=1.0`). L2 shrinks the coefficients toward zero, so those coefficients are not
maximum-likelihood estimates. Meanwhile I compared them against **statsmodels**, which fits plain
maximum likelihood and has no penalty. So I was comparing two different models.

**What I changed.** Every logistic fit used for coefficient interpretation (Part 2.1 Simpson, Part 2.2
omitted-variable bias, and the headline Part 1 model) is now `LogisticRegression(penalty=None,
max_iter=1000, solver="lbfgs")`. That is the unpenalised MLE, so it line up with the statsmodels
confidence intervals.

**A note for the viva.** sklearn with `penalty=None` still stops at `tol=1e-4`, so its coefficients are
*approximately* MLE: `age` comes out 0.3188 / 3.3553 / 0.1065 (age <30 / >=30) while statsmodels gives
0.3181 / 3.3251 / 0.1072 — about 1% apart. I use the sklearn numbers in the coefficient tables and
statsmodels for the standard errors and CIs, because only statsmodels computes standard errors.

**Numbers that moved:** most coefficients got larger in magnitude because the shrinkage is gone —
e.g. `Married-AF-spouse` 1.804 → 2.753, `capital-gain` 1.195 → 2.330,
`Priv-house-serv` -1.390 → -2.115.

---

## 3. Part 2.1 — confidence intervals, degenerate columns, race reference (task 4)

**Added.** statsmodels 95% CIs for `age` in three models, printed as a real table with coefficient,
standard error, CI and p-value, plus an explicit CI-overlap test:

| model  | coef | SE   | CI 95%                | p        |
|--------|------|------|-----------------------|----------|
| Full   | 0.3181 | 0.0214 | [0.2763, 0.3600]   | 3.94e-50 |
| Age<30 | 3.3251 | 0.3758 | [2.5886, 4.0616]   | 8.85e-19 |
| Age>=30| 0.1072 | 0.0250 | [0.0582, 0.1563]   | 1.81e-05 |

None of the three intervals overlap, including Full vs Age<30 and Age<30 vs Age>=30. So the Simpson
pattern is not sampling noise.

**Removed.** The v1 claim "*relationship_Wife: full +1.147 → Male subgroup -0.065 (flips to
zero/negative)*". Inside the Male-only subgroup `sex_Male` is constant, so `relationship_Wife` (which
is essentially the male counterpart) becomes collinear with it and has no meaningful coefficient. The
notebook now **detects and lists the dropped degenerate columns** instead of quoting them:
Male drops `sex_Male`; Female drops `sex_Male` and `occupation_Armed-Forces`; the age splits drop
nothing.

**Clarified.** All `race_*` coefficients are relative to the **reference category**. Because
`drop_first=True` drops the first level alphabetically, that reference is
`Amer-Indian-Eskimo` (470 rows overall) — not White, and not a generic baseline. Stated in the report
preprocessing section and in `results.json` as `race_reference`.

**Numbers that moved:** `age` full 0.314 → 0.3188 (sklearn) and 0.3181 (statsmodels);
Age<30 2.907 → 3.3553 / 3.3251; Age>=30 0.104 → 0.1065 / 0.1072;
`Self-emp-not-inc` <30 0.261 → 0.3455.

---

## 4. Part 2.3 — exact statistics and a calibration check (task 5)

**Removed every "~"**. v1 reported approximate values like white share "~85.5%" and hours<=40 "~58%".
All are now exact, computed as numerator/denominator:

| statistic | 50-row subset | full test set |
|---|---|---|
| male | 40/50 = 80.00% | 9763/14653 = 66.63% |
| white | 44/50 = 88.00% | 85.36% |
| black | 6.00% | 9.85% |
| hours<=40 | 56.00% | 70.91% |
| capital-gain==0 | 86.00% | 91.97% |
| education<=9 | 62.00% | 45.01% |
| never-married | 12.00% | 32.78% |
| age<30 | 14.00% | 30.15% |
| age mean / median | 41.84 / 38 | 38.4451 / 37 |

Note that the v1 headline "94% have capital-gain 0" is now **86%**, because the split changed which
50 rows seed 777 picks out of the error set.

**Replaced the "over-confident" claim with a calibration check.** I now report:
- band `P<=0.25`: n=9505, observed positive rate 0.0630 (so labelling all of them "poor" is wrong
  6.30% of the time)
- band `P>=0.75`: n=1291, observed negative rate 0.1185
- a reliability table by predicted-probability decile: mean predicted vs observed rate per decile,
  largest gap 0.0270 at decile 8 (mean 0.6038 vs observed 0.5768). The model is broadly well
  calibrated; it is not systematically over-confident, so the v1 wording was wrong and is removed.

**Verified the "young Exec-managerial P~0.85" claim.** Among the 276 under-30 Exec-managerial rows in
the test set: max P = 0.9422, **mean P = 0.2200**, median P = 0.0893, and only 17 rows reach P>=0.75.
So the claim is false as a general statement and has been deleted; the actual numbers are reported.

**Numbers that moved:** high-confidence errors 754 → **752** (FP 152 → **153**, FN 602 → **599**).

---

## 5. Part 2.4 — fairness group order (task 6)

**What was wrong.** v1 set Group A = Male and Group B = Female. Since men are selected far more
often, every ratio came out above 1 — DP from 2.21 to 3.95 — which does not fit the assignment's
0–1.5 axes, so the chart clipped or ran off the view.

**What I changed.** **Group A = Female, Group B = Male.** All ratios now fall inside 0–1.5 and the
direction of the gap is clear (women are selected less often). I also print the two equalized-odds
components separately so the `min()` is transparent instead of being a black box:

| t | DP | EO | TPR ratio | FPR ratio | PP | acc |
|---|---|---|---|---|---|---|
| 0.05 | 0.4531 | 0.4136 | 0.9419 | 0.4136 | 0.7672 | 0.6354 |
| 0.15 | 0.2972 | 0.2270 | 0.8273 | 0.2270 | 1.0275 | 0.7534 |
| 0.25 | 0.2794 | 0.2029 | 0.8012 | 0.2029 | 1.0583 | 0.8062 |
| 0.35 | 0.2748 | 0.1870 | 0.8014 | 0.1870 | 1.0764 | 0.8378 |
| 0.45 | 0.2754 | 0.1896 | 0.7858 | 0.1896 | 1.0533 | 0.8484 |
| 0.55 | 0.2741 | 0.1766 | 0.7871 | 0.1766 | 1.0597 | 0.8494 |
| 0.65 | 0.2799 | 0.1958 | 0.7774 | 0.1958 | 1.0250 | 0.8421 |
| 0.75 | 0.2824 | 0.2221 | 0.7650 | 0.2221 | 0.9997 | 0.8280 |
| 0.85 | 0.2920 | 0.2136 | 0.7950 | 0.2136 | 1.0048 | 0.8063 |
| 0.95 | 0.2413 | 0.3133 | 0.6418 | 0.3133 | 0.9815 | 0.7869 |

**Which threshold is closest to 1 (read off the table, not assumed):**
- DP closest: **t=0.05** (0.4531)
- EO closest: **t=0.05** (0.4136)
- PP closest: **t=0.75** (0.9997)

**Chart.** Regenerated on 0–1.5 × 0–1.5 axes, dashed reference lines at 1.0, every point labelled with
its threshold, and a check printed in the notebook confirming **10/10 points are inside the view**,
so the chart is not empty.

**Trade-off examples recomputed.** Example: improve DP from 3.597 → 2.213 equivalent (t=0.45 → 0.05)
gains 38.47% toward parity but costs 25.23% accuracy (84.84% → 63.54%) and pushes PP from 1.0533 to
0.7672. With the F/M orientation: moving t=0.35 → 0.05 changes EO 0.1870 → 0.4136 while accuracy
falls 83.78% → 63.54%.

---

## 6. Q1, Q6 and the agreement chart (tasks 7 and 8)

**Q1 fixed.** v1 asserted that all 4488 mismatching rows had an OLS score of at least 0.5, and that
A-low/B-high was exactly 0. Neither is true in v2. The notebook now prints the distribution:
in [median, 0.5) = **4410**, >= 0.5 = **88**, < median = **16**. So **16** cases are A-low/B-high
(the model flagged high by logistic but low by OLS median) — the confusion matrix is
`[[7310, 16], [4498, 2829]]`, not all lower-triangular. The claim "zero cases of the opposite" is
deleted.

**Q6 fixed.** v1 described two invented people ("Anil" and "Meena") with made-up scores. I now use
two **real test rows** chosen deterministically: same race, occupation and marital status, and equal
age / education / hours:

- Female, test index 24765: **P = 0.2830**, true y = 0
- Male, test index 20355: **P = 0.2655**, true y = 1

Both are age 37, education-num 9, 40 hours, Adm-clerical, Married-civ-spouses, White, capital-gain 0.
Their predictions: t=0.05 → 1 and 1; t=0.35 → 0 and 0; t=0.75 → 0 and 0.

The honest reading, which I state rather than spin: for two people with nearly identical features the
model gives nearly the same score (and the female's is actually *higher*). So at the individual level
the model does not discriminate between them — the disparity is an **aggregate** one in flag rates
and FPRs, not something that shows up between a matched pair. The old "Anil/Meena" narrative
described a disparity the matched-pair data does not show.

**Agreement chart fixed.** v1's y-axis label "A high (>=median)" was clipped off the top of the image.
The label is now two lines (`"A high\n(>=median)"`), the figure is 6.4×4.3 inches instead of 5×4, and
a colour bar is added. Same for the x-axis label.

---

## 7. Two bugs found and fixed while verifying (worth mentioning in the viva)

1. **The fairness sweep printed an undefined variable** (`point or 0`) and crashed the cell, which
   also broke the results-saving cell downstream. Fixed to use the DP value it was meant to print.
2. **v1's `adversarial.common` block computed a helper frame `ft`** from the error subset and then
   never used it — dead code that implied a comparison existed when it did not. Removed.
3. **v1's Part 2.3 print statement had a `if False else` leftover** that did nothing. Cleaned.

---

## 8. Files added / changed

- `CSET485_A1_Gautam_S24CSEU0777.ipynb` — rewritten; the source of truth. Runs top to bottom in a
  fresh environment with no errors. Markdown explanation added before every section.
- `full_analysis.py` — **auto-generated from the notebook's code cells** by `sync_script.py`, so the
  script and notebook cannot drift apart. Verified: running the script reproduces `results.json`
  byte-identical to the notebook's.
- `numbers_cited.json` — every number cited in the report, machine-readable.
- `report_checklist.md` — maps each group of numbers to the notebook cell that produces it, plus the
  list of claims I deleted because they could not be verified.
- `v1_vs_v2_diff.json` — old value vs new value for each number that changed.
- `CHANGELOG.md` — this file.
- `README.md` — exact reproduction instructions and the exact library versions used.
- `requirements.txt` — now pinned to the versions actually used.

## 9. What I could NOT verify

- **The assignment brief PDF was not available to me.** I worked from the task description given in
  the prompt and from the existing notebook. In particular the exact definitions of "demographic
  parity", "equalized odds", "the 0–1.5 axes", "the brief's threshold set" and the word limits are
  taken from that description, not read from the brief file. **Check these against the real
  `Assignment_1_-_CSET485.pdf` before submitting** — if the brief defines EO differently (for
  example as a single reportable number rather than `min(TPR ratio, FPR ratio)`), only the EO column
  and its "closest" line need to change.
- **Oral assessment / AI-content checks** are outside what I can see.
- **The underlying fairness of the Adult dataset** (1994 US census extract) is not something my
  numbers speak to; the 0.27 female/male DP reflects the data, and whether that gap is a property of
  the population or of period-specific labour patterns is a domain question I have not claimed to
  settle.
