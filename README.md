# CSET485 AI and Society — Assignment 1
### Adult Census Income: Building & Stress-Testing a High-Stakes Classifier

**Name:** Gautam Chauhan | **Roll No:** S24CSEU0777 | **Seed:** 777 (last 3 digits) | **Dataset:** Adult Census Income (last digit 7)
**Split:** 70% train / 30% test, stratified on income. Train n=34189, test n=14653.

`CSET485_A1_Gautam_S24CSEU0777.ipynb` is the single source of truth. Every number in the PDF report,
in `results.json` and in `numbers_cited.json` is produced by that notebook.

See **CHANGELOG.md** for what changed since the first version and why, and
**report_checklist.md** for a map from each number in the report to the notebook cell that produces it.

---

## Reproduce

```bash
# 1. from inside this folder
python -m venv .venv
.venv\Scripts\activate

# 2. install the exact versions the results were produced with
pip install -r requirements.txt

# 3. run either one — both give identical output

#   (a) the notebook
jupyter notebook CSET485_A1_Gautam_S24CSEU0777.ipynb
#     then Run > Run All Cells

#   (b) the script (identical code, extracted from the notebook)
python full_analysis.py
```

Both write `results.json` and the four `chart_*.png` files into the current folder. Running the script
was checked to produce a `results.json` byte-identical to the notebook's.

Only the two UCI file downloads need internet access; everything else is deterministic given
`random_state=777`.

### Versions this was produced and tested with

| package | version |
|---|---|
| Python | 3.11.9 (Windows 11, 64-bit) |
| pandas | 3.0.6 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| statsmodels | 0.15.0 |
| nbformat | 5.11.1 |
| jupyter / nbconvert | 7.x (execution only) |

You can verify with:

```bash
python -c "import sys,pandas,numpy,sklearn,matplotlib,statsmodels,nbformat; \
print(sys.version); \
[print(m.__name__, m.__version__) for m in (pandas,numpy,sklearn,matplotlib,statsmodels,nbformat)]"
```

---

## Files

| file | what it is |
|---|---|
| `CSET485_A1_Gautam_S24CSEU0777.ipynb` | the notebook — source of truth |
| `full_analysis.py` | same code as a script, auto-generated from the notebook |
| `sync_script.py` | regenerates `full_analysis.py` from the notebook |
| `results.json` | every computed number for seed 777 |
| `numbers_cited.json` | every number the report cites, machine-readable |
| `v1_vs_v2_diff.json` | old vs new value for each number that changed |
| `report_checklist.md` | number → notebook cell map, and deleted claims |
| `CHANGELOG.md` | what changed and why, in plain language |
| `chart_*.png` | the four figures used in the report |
| `CSET485_A1_Report_Gautam_S24CSEU0777.pdf` | the report |

---

## Headline results (seed 777, test n=14653)

**Preprocessing** — missing values imputed with train-split modes; outlier caps fit on the train split
(99.5th percentile for capital-gain/loss/fnlwgt; fixed domain caps of 80 and 75 for hours and age);
`StandardScaler` fit on train only. `fnlwgt` (a CPS sampling weight, not a personal trait) is kept.
Race coefficients are relative to `Amer-Indian-Eskimo`. All logistic models used for coefficient
comparison are unpenalised (`penalty=None`) so they match the statsmodels MLE.

- **Part 1:** OLS median 0.2194 vs logistic 0.5 → mismatch **4514 (30.81%)**; model B accuracy
  **85.09%**, model A 70.92%. Of the mismatches, 4410 have an OLS score in [0.2194, 0.5), 88 are
  >= 0.5, and 16 are A-low/B-high.
- **Part 2.1:** `age` full 0.3188 / <30 3.3553 / >=30 0.1065 (sklearn); statsmodels
  0.3181 [0.2763, 0.3600], 3.3251 [2.5886, 4.0616], 0.1072 [0.0582, 0.1563] — none of the CIs
  overlap.
- **Part 2.2** (drop `education-num`): Exec-managerial **+37.15%**, race_White **+28.04%**,
  age **-25.27%**, hours +8.76%, capital-loss +6.31%.
- **Part 2.3:** **752** high-confidence errors (5.13% of test; 153 FP, 599 FN). 50-row sample:
  80% male, mean age 41.84, 86% capital-gain 0. Calibration: observed positive rate 0.0630 in the
  `P<=0.25` band, observed negative rate 0.1185 in `P>=0.75`; decile reliability max gap 0.0270.
- **Part 2.4** (Group A = Female, Group B = Male): closest to 1 is DP **t=0.05** (0.4531),
  EO **t=0.05** (0.4136), PP **t=0.75** (0.9997). No threshold is close to 1 on all three.

---

## Dataset

https://archive.ics.uci.edu/ml/machine-learning-databases/adult/ — `adult.data` (32561 rows) +
`adult.test` (16281 rows) = 48842 rows.

## Academic note

All numbers were computed by me with seed 777. **Read the notebook and CHANGELOG before the oral
assessment** — the questions will ask why each design choice was made (train-only fitting, the
unpenalised logistic, dropping degenerate columns, the Female/Male group order, and why the
"Wife flips" and "over-confident" claims were removed).
