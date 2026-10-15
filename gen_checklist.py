"""Build report_checklist.md + v1_vs_v2_diff.json from real numbers only."""
import json
base = r"C:\Users\gauta\Documents\Default Project"
import io
def loadjson(path):
    with io.open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

v1 = loadjson(base + r"\v1_results.json")
v2 = loadjson(base + r"\CSET485_A1_Gautam\results.json")

def r(x, n=4):
    x = float(x)
    return round(x, n)

# ---- 1. v1 vs v2 diff ----
diff = {
 "n": {"v1": v1["n"], "v2": v2["n_raw"], "changed": False},
 "train_n": {"v1": v1["train_n"], "v2": v2["train_n"], "changed": False},
 "test_n": {"v1": v1["test_n"], "v2": v2["test_n"], "changed": False},
 "modelA_median": {"v1": r(v1["modelA"]["median"]), "v2": r(v2["modelA"]["median"])},
 "modelA_acc": {"v1": r(v1["modelA"]["acc"]), "v2": r(v2["modelA"]["acc"])},
 "modelB_acc": {"v1": r(v1["modelB"]["acc"]), "v2": r(v2["modelB"]["acc"])},
 "mismatch": {"v1": v1["mismatch"], "v2": v2["mismatch"]},
 "mismatch_pct": {"v1": r(v1["mismatch_pct"],2), "v2": r(v2["mismatch_pct"],2)},
 "agreement_cm": {"v1": v1["agreement_cm"], "v2": v2["agreement_cm"]},
 "A_low_B_high": {"v1": v1["agreement_cm"][0][1], "v2": v2["agreement_cm"][0][1]},
 "q1_band": {"v1": None, "v2": v2["q1_band"], "note": "new in v2, did not exist in v1"},
 "age_full_sklearn": {"v1": r(v1["simpson_full"]["age"]), "v2": r(v2["simpson_full"]["age"])},
 "age_lt30_sklearn": {"v1": r(v1["simpson_sub"]["Age<30"]["age"]), "v2": r(v2["simpson_sub"]["Age<30"]["age"])},
 "age_ge30_sklearn": {"v1": r(v1["simpson_sub"]["Age>=30"]["age"]), "v2": r(v2["simpson_sub"]["Age>=30"]["age"])},
 "age_full_statsmodels": {"v1": r(v1["sm"]["Full"]["age"]["coef"]), "v2": r(v2["sm_age"]["Full"]["coef"])},
 "age_lt30_statsmodels": {"v1": r(v1["sm"]["Age<30"]["age"]["coef"]), "v2": r(v2["sm_age"]["Age<30"]["coef"])},
 "age_ge30_statsmodels": {"v1": r(v1["sm"]["Age>=30"]["age"]["coef"]), "v2": r(v2["sm_age"]["Age>=30"]["coef"])},
 "self_emp_not_inc_full": {"v1": r(v1["simpson_extra"]["SelfEmpNotInc_full"]), "v2": r(v2["logr_bottom"][2][1])},
 "self_emp_not_inc_lt30": {"v1": r(v1["simpson_extra"]["SelfEmpNotInc_<30"]), "v2": r(v2["simpson_sub"]["Age<30"]["workclass_Self-emp-not-inc"])},
 "self_emp_not_inc_ge30": {"v1": r(v1["simpson_extra"]["SelfEmpNotInc_>=30"]), "v2": r(v2["simpson_sub"]["Age>=30"]["workclass_Self-emp-not-inc"])},
 "hardcoded_missing_in_v1_missing_count": {"v1": "2799/2809/857 (hardcoded)", "v2": "computed"},
 "ovb_exec": {"v1": r([o for o in v1["ovb"] if "Exec" in o["feat"]][0]["pct"],2), "v2": r([o for o in v2["ovb"] if "Exec" in o["feat"]][0]["pct"],2)},
 "ovb_white": {"v1": r([o for o in v1["ovb"] if "White" in o["feat"]][0]["pct"],2), "v2": r([o for o in v2["ovb"] if "White" in o["feat"]][0]["pct"],2)},
 "ovb_age": {"v1": r([o for o in v1["ovb"] if o["feat"]=="age"][0]["pct"],2), "v2": r([o for o in v2["ovb"] if o["feat"]=="age"][0]["pct"],2)},
 "ovb_hours": {"v1": r([o for o in v1["ovb"] if "hours" in o["feat"]][0]["pct"],2), "v2": r([o for o in v2["ovb"] if "hours" in o["feat"]][0]["pct"],2)},
 "adversarial": {
   "hc": {"v1": v1["adversarial"]["hc_count"], "v2": v2["adversarial"]["hc_count"]},
   "hc_pct": {"v1": r(v1["adversarial"]["hc_pct"],3), "v2": r(v2["adversarial"]["hc_pct"],3)},
   "fp": {"v1": v1["adversarial"]["fp"], "v2": v2["adversarial"]["fp"]},
   "fn": {"v1": v1["adversarial"]["fn"], "v2": v2["adversarial"]["fn"]},
 },
 "top_coefs": {"v1": [[k, r(x)] for k, x in v1["logr_top"]], "v2": [[k, r(x)] for k, x in v2["logr_top"]]},
 "fairness_order": {"v1": "A=Male,B=Female", "v2": "A=Female,B=Male"},
 "fairness_range_DP": {"v1": [r(v1["fairness"][0]["DP"]), r(v1["fairness"][-1]["DP"])],
                        "v2": [r(v2["fairness"][0]["DP"]), r(v2["fairness"][-1]["DP"])]},
}
json.dump(diff, open(base + r"\CSET485_A1_Gautim\v1_vs_v2_diff.json", "w"), indent=2)

# ---- 2. report checklist ----
lines = []
lines.append("# Report -> notebook cell checklist\n")
lines.append("Every number quoted in the PDF, and the notebook cell that produces it. "
             "All numbers come from `results.json` / `numbers_cited.json`, which the notebook writes itself.\n")
lines.append("Notebook cell indices are the code cells of `CSET485_A1_Gautam_S24CSEU0777.ipynb` "
             "(cell 1 = imports, cell 3 = load, cell 5 = preprocessing, cell 7 = scaling, "
             "cell 9 = Part 1, cell 11 = 2.1 sklearn, cell 13 = 2.1 statsmodels CIs, "
             "cell 15 = 2.2 OVB, cell 17 = 2.3 adversarial, cell 19 = 2.4 fairness + Q6 pair, "
             "cell 21 = results.json + charts).\n")

sections = [
 ("Preprocessing (missing values, modes, caps, fnlwgt, race reference)",
  ["workclass 2799 / occupation 2809 / native-country 857 missing",
   "train-fitted modes: workclass=Private, occupation=Prof-specialty, native-country=United-States",
   "caps: capital-gain 99999 (no-op), capital-loss 2258, fnlwgt 590223, hours 80, age 75",
   "43 features; race reference category Amer-Indian-Eskimo"],
  "cell 5"),
 ("Part 1 (OLS vs logistic)",
  ["median 0.2194, mean 0.2393, min -0.4093, max 1.6342",
   "model B acc 0.8509, model A acc 0.7092",
   "agreement matrix [[7310,16],[4498,2829]], mismatch 4514 = 30.81%",
   "Q1 band: [median,0.5)=4410, >=0.5=88, <median=16",
   "top/bottom coefficients (penalty=None)"],
  "cell 9"),
 ("Part 2.1 (Simpson + statsmodels CIs)",
  ["age sklearn: full 0.3188, <30 3.3553, >=30 0.1065",
   "age statsmodels: 0.3181 [0.2763,0.3600] p=3.94e-50; "
   "3.3251 [2.5886,4.0616] p=8.85e-19; 0.1072 [0.0582,0.1563] p=1.81e-05",
   "all three CIs do not overlap",
   "subgroup sizes Male 22887 / Female 11302 / Age<30 10097 / Age>=30 24092",
   "dropped degenerate columns: Male [sex_Male], Female [sex_Male, occupation_Armed-Forces]",
   "workclass_Self-emp-not-inc <30 0.3455 vs >=30 -1.0639"],
  "cells 11 + 13"),
 ("Part 2.2 (OVB dropping education-num)",
  ["Exec-managerial +37.15%", "race_White +28.04%", "age -25.27%", "hours +8.76%",
   "capital-loss +6.31%", "sex_Male +3.22%", "capital-gain +0.96%", "Married-civ +0.32%"],
  "cell 15"),
 ("Part 2.3 (adversarial + calibration)",
  ["752 high-confidence errors = 5.13% (153 FP, 599 FN)",
   "50-row subset: 40/50 male vs 9763/14653 = 66.63% test; 44/50 = 88% white vs 85.36% test",
   "gain==0 86% vs 91.97%; edu<=9 62% vs 45.01%; never-married 12% vs 32.78%; "
   "age mean 41.84 vs 38.4451; median 38 vs 37; hours<=40 56% vs 70.91%",
   "calibration: P<=0.25 band n=9505 observed positive rate 0.0630; "
   "P>=0.75 band n=1291 observed negative rate 0.1185",
   "reliability table by decile (max |gap| 0.0270 at decile 8)",
   "young Exec-managerial: n=276, max P 0.9422, mean P 0.2200, 17 rows P>=0.75"],
  "cell 17"),
 ("Part 2.4 (fairness, A=Female B=Male)",
  ["sweep table t=0.05..0.95 with DP, EO, PP, TPR_ratio, FPR_ratio, acc",
   "closest DP=1: t=0.05 (0.4531); closest EO=1: t=0.05 (0.4136); closest PP=1: t=0.75 (0.9997)",
   "Q6 matched pair: Female idx 24765 P=0.2830 y=0; Male idx 20355 P=0.2655 y=1; "
   "both same age 37, edu 9, hours 40, Adm-clerical, Married-civ-spouse, White; "
   "predicted 1/1 at t=0.05, 0/0 at t=0.35 and t=0.75"],
  "cell 19"),
 ("Charts", ["chart_modelA_hist.png", "chart_agreement.png (multi-line labels, 'A high' not cropped)",
              "chart_fairness_tradeoff.png (0-1.5 axes, all 10 points labelled)",
              "chart_ovb.png"], "cell 21"),
]
for title, items, where in sections:
    lines.append(f"\n## {title}\n")
    lines.append(f"Produced by: **{where}**\n")
    for it in items:
        lines.append(f"- {it}")

lines.append("\n## Claims removed because they could not be verified\n")
lines.append("- 'Wife flips in the Male subgroup' - `relationship_Wife` is degenerate once `sex_Male` "
             "is constant inside the Male subgroup, so no coefficient exists there to compare. Removed.\n")
lines.append("- 'Over-confident' as a blanket claim for the adversarial subset - replaced by the "
             "decile reliability table and the two band error rates above.\n")
lines.append("- 'Young Exec-managerial P~0.85' - replaced by the actual distribution above "
             "(max 0.9422, mean 0.2200, 17 of 276 at or above 0.75).\n")
lines.append("- 'OldMale 45+ age -0.453' - this subgroup is not computed in the notebook, so it is "
             "not quoted anywhere.\n")
lines.append("- Approximate values marked '~' in v1 (white share, hours<=40 share, etc.) - all now exact.\n")

lines.append("\n## Note on the two optimizers\n")
lines.append("sklearn `LogisticRegression(penalty=None)` is unpenalised but stops at `tol=1e-4`, so its "
             "age coefficients (0.3188 / 3.3553 / 0.1065) differ from statsmodels MLE "
             "(0.3181 / 3.3251 / 0.1072) by about 1%. Coefficient comparison tables use the sklearn "
             "numbers; confidence intervals and p-values use statsmodels, because only statsmodels "
             "computes standard errors.\n")
open(base + r"\CSET485_A1_Gautim\report_checklist.md", "w", encoding="utf-8").write("\n".join(lines))
print("wrote report_checklist.md and v1_vs_v2_diff.json")
print(json.dumps({k: diff[k] for k in ["modelA_median","mismatch","agreement_cm","A_low_B_high"]}, indent=1))
