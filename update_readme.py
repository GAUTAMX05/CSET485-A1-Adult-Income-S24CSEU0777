from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
R=json.loads((ROOT/'results.json').read_text(encoding='utf-8'))
def g(p):
 x=R
 for k in p.split('.'): x=x[k]
 return x
pct=lambda x:f'{100*x:.2f}%'
fair={round(float(r['t']),2):r for r in R['fairness']}
sex=R['sex_base_rates']['test']
text=f'''# CSET485 AI and Society — Assignment 1
### Adult Census Income: Building & Stress-Testing a High-Stakes Classifier

**Name:** Gautam Chauhan | **Roll No:** S24CSEU0777 | **Seed:** {g('seed')} | **Dataset:** Adult Census Income
**Split:** 70% train / 30% test, stratified on income. Train n={g('train_n')}, test n={g('test_n')}.

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

- **Part 1:** OLS median {g('modelA.median'):.4f}; mismatch **{g('mismatch')} ({g('mismatch_pct'):.2f}%)**; logistic accuracy **{pct(g('modelB.acc'))}**; OLS-median classifier accuracy **{pct(g('modelA.acc'))}**.
- **Part 2.1:** statsmodels age coefficient: full **{g('sm_age.Full.coef'):.4f}**, Age<30 **{g('sm_age.Age<30.coef'):.4f}**, Age>=30 **{g('sm_age.Age>=30.coef'):.4f}**; the three 95% CIs do not overlap.
- **Part 2.2:** the largest omitted-variable shift is `occupation_Exec-managerial` **{next(x['pct'] for x in R['ovb'] if x['feat']=='occupation_Exec-managerial'):+.2f}%** after dropping education-num.
- **Part 2.3:** **{g('adversarial.hc_count')}** high-confidence errors (**{g('adversarial.fp')} FP, {g('adversarial.fn')} FN**); 50-row sample selected with seed {g('seed')}.
- **Part 2.4:** Group A = Female, Group B = Male; closest DP and EO are both at t={min(R['fairness'],key=lambda r:abs(r['DP']-1))['t']:.2f}; closest PP is t={min(R['fairness'],key=lambda r:abs(r['PP']-1))['t']:.2f}.
- **Sex base rates on test:** Female {pct(sex['Female']['positive_rate'])}, Male {pct(sex['Male']['positive_rate'])}.
- **Chosen report threshold:** t={g('report_config.chosen_threshold'):.2f}, defended in Q7 with a counterargument.

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
'''
(ROOT/'README.md').write_text(text,encoding='utf-8')
print('README regenerated from results.json')
