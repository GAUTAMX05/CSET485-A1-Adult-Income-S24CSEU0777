# Report -> notebook cell checklist

Every number quoted in the PDF, and the notebook cell that produces it. All numbers come from `results.json` / `numbers_cited.json`, which the notebook writes itself.

Notebook cell indices are the code cells of `CSET485_A1_Gautam_S24CSEU0777.ipynb` (cell 1 = imports, cell 3 = load, cell 5 = preprocessing, cell 7 = scaling, cell 9 = Part 1, cell 11 = 2.1 sklearn, cell 13 = 2.1 statsmodels CIs, cell 15 = 2.2 OVB, cell 17 = 2.3 adversarial, cell 19 = 2.4 fairness + Q6 pair, cell 21 = results.json + charts).


## Preprocessing (missing values, modes, caps, fnlwgt, race reference)

Produced by: **cell 5**

- workclass 2799 / occupation 2809 / native-country 857 missing
- train-fitted modes: workclass=Private, occupation=Prof-specialty, native-country=United-States
- caps: capital-gain 99999 (no-op), capital-loss 2258, fnlwgt 590223, hours 80, age 75
- 43 features; race reference category Amer-Indian-Eskimo

## Part 1 (OLS vs logistic)

Produced by: **cell 9**

- median 0.2194, mean 0.2393, min -0.4093, max 1.6342
- model B acc 0.8509, model A acc 0.7092
- agreement matrix [[7310,16],[4498,2829]], mismatch 4514 = 30.81%
- Q1 band: [median,0.5)=4410, >=0.5=88, <median=16
- top/bottom coefficients (penalty=None)

## Part 2.1 (Simpson + statsmodels CIs)

Produced by: **cells 11 + 13**

- age sklearn: full 0.3188, <30 3.3553, >=30 0.1065
- age statsmodels: 0.3181 [0.2763,0.3600] p=3.94e-50; 3.3251 [2.5886,4.0616] p=8.85e-19; 0.1072 [0.0582,0.1563] p=1.81e-05
- all three CIs do not overlap
- subgroup sizes Male 22887 / Female 11302 / Age<30 10097 / Age>=30 24092
- dropped degenerate columns: Male [sex_Male], Female [sex_Male, occupation_Armed-Forces]
- workclass_Self-emp-not-inc <30 0.3455 vs >=30 -1.0639

## Part 2.2 (OVB dropping education-num)

Produced by: **cell 15**

- Exec-managerial +37.15%
- race_White +28.04%
- age -25.27%
- hours +8.76%
- capital-loss +6.31%
- sex_Male +3.22%
- capital-gain +0.96%
- Married-civ +0.32%

## Part 2.3 (adversarial + calibration)

Produced by: **cell 17**

- 752 high-confidence errors = 5.13% (153 FP, 599 FN)
- 50-row subset: 40/50 male vs 9763/14653 = 66.63% test; 44/50 = 88% white vs 85.36% test
- gain==0 86% vs 91.97%; edu<=9 62% vs 45.01%; never-married 12% vs 32.78%; age mean 41.84 vs 38.4451; median 38 vs 37; hours<=40 56% vs 70.91%
- calibration: P<=0.25 band n=9505 observed positive rate 0.0630; P>=0.75 band n=1291 observed negative rate 0.1185
- reliability table by decile (max |gap| 0.0270 at decile 8)
- young Exec-managerial: n=276, max P 0.9422, mean P 0.2200, 17 rows P>=0.75

## Part 2.4 (fairness, A=Female B=Male)

Produced by: **cell 19**

- sweep table t=0.05..0.95 with DP, EO, PP, TPR_ratio, FPR_ratio, acc
- closest DP=1: t=0.05 (0.4531); closest EO=1: t=0.05 (0.4136); closest PP=1: t=0.75 (0.9997)
- Q6 matched pair: Female idx 24765 P=0.2830 y=0; Male idx 20355 P=0.2655 y=1; both same age 37, edu 9, hours 40, Adm-clerical, Married-civ-spouse, White; predicted 1/1 at t=0.05, 0/0 at t=0.35 and t=0.75

## Charts

Produced by: **cell 21**

- chart_modelA_hist.png
- chart_agreement.png (multi-line labels, 'A high' not cropped)
- chart_fairness_tradeoff.png (0-1.5 axes, all 10 points labelled)
- chart_ovb.png

## Claims removed because they could not be verified

- 'Wife flips in the Male subgroup' - `relationship_Wife` is degenerate once `sex_Male` is constant inside the Male subgroup, so no coefficient exists there to compare. Removed.

- 'Over-confident' as a blanket claim for the adversarial subset - replaced by the decile reliability table and the two band error rates above.

- 'Young Exec-managerial P~0.85' - replaced by the actual distribution above (max 0.9422, mean 0.2200, 17 of 276 at or above 0.75).

- 'OldMale 45+ age -0.453' - this subgroup is not computed in the notebook, so it is not quoted anywhere.

- Approximate values marked '~' in v1 (white share, hours<=40 share, etc.) - all now exact.


## Note on the two optimizers

sklearn `LogisticRegression(penalty=None)` is unpenalised but stops at `tol=1e-4`, so its age coefficients (0.3188 / 3.3553 / 0.1065) differ from statsmodels MLE (0.3181 / 3.3251 / 0.1072) by about 1%. Coefficient comparison tables use the sklearn numbers; confidence intervals and p-values use statsmodels, because only statsmodels computes standard errors.
