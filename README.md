# CSET485 AI and Society — Assignment 1
### Adult Census Income: Building & Stress-Testing a High-Stakes Classifier

**Name:** Gautam Chauhan | **Roll No:** S24CSEU0777 | **Seed:** 777 | **Dataset:** Adult Census Income
**Split:** 70% train / 30% test, stratified on income. Train n=34189, test n=14653.

`CSET485_A1_Gautam_S24CSEU0777.ipynb` and `full_analysis.py` are the reproducible analysis sources. The PDF is built only from `results.json` by `build_report.py`.

## Reproduce

```bash
pip install -r requirements.txt
python full_analysis.py
python augment_results.py
python build_report.py
```

The full workflow also executes the notebook, compares its `results.json` with the script output, and stops if they disagree. The two UCI Adult files require internet access during execution.

## Headline results

- **Part 1:** OLS median 0.2194; mismatch **4514 (30.81%)**; logistic accuracy **85.09%**; OLS-median classifier accuracy **70.92%**.
- **Part 2.1:** statsmodels age coefficient: full **0.3181**, Age<30 **3.3251**, Age>=30 **0.1072**; the three 95% CIs do not overlap.
- **Part 2.2:** the largest omitted-variable shift is `occupation_Exec-managerial` **+37.15%** after dropping education-num.
- **Part 2.3:** **752** high-confidence errors (**153 FP, 599 FN**); 50-row sample selected with seed 777.
- **Part 2.4:** Group A = Female, Group B = Male; closest DP and EO are both at t=0.05; closest PP is t=0.75.
- **Sex base rates on test:** Female 11.19%, Male 30.31%.
- **Chosen report threshold:** t=0.35, defended in Q7 with a counterargument.

## Files

| file | purpose |
|---|---|
| `CSET485_A1_Gautam_S24CSEU0777.ipynb` | notebook source of truth |
| `full_analysis.py` | script equivalent of the notebook |
| `augment_results.py` | adds report-specific metrics to results.json |
| `results.json` | computed values used by the report |
| `results_from_nb.json` | notebook execution result used for consistency checking |
| `build_report.py` | builds the PDF only from results.json |
| `report_audit.md` | number-to-key mapping, word counts and prefinish checks |
| `CSET485_A1_Report_Gautam_S24CSEU0777.pdf` | final assignment report |
| `chart_*.png` | four regenerated figures |
| `CHANGELOG.md` | methodological corrections from the earlier version |

## Dataset and methodology

The UCI Adult dataset is combined before the assignment's own stratified split. Missing-value modes, 99.5th-percentile numeric caps and the scaler are fitted on the training split only. `fnlwgt` is retained as the CPS sampling weight. Logistic coefficient comparisons use unpenalised `penalty=None` fits, while statsmodels supplies confidence intervals and p-values.

The fairness sweep uses Female/Male ratios so the direction of the gap is visible on the 0–1.5 chart. Q6 uses the real matched pair stored in `results.json`; Q7 and Q8 are written from the current sweep rather than the previous report.
