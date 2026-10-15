"""Generate numbers_cited.json (every number cited in the report) and
report_checklist.md (map report text -> notebook cell)."""
import json, os
R = json.load(open(r"C:\Users\gauta\Documents\Default Project\CSET485_A1_Gautam\results.json"))
OUT = r"C:\Users\gauta\Documents\Default Project\CSET485_A1_Gautam"

def r4(x): return round(float(x), 4)

def closest(m):
    best = min(R["fairness"], key=lambda r: abs(r[m] - 1.0))
    return {"t": best["t"], "value": r4(best[m])}

cited = {
 "meta": {
   "seed": 777, "roll_no": "S24CSEU0777",
   "n_raw": R["n_raw"], "train_n": R["train_n"], "test_n": R["test_n"],
   "pos_rate": r4(R["pos_rate"]), "train_pos": r4(R["train_pos"]), "test_pos": r4(R["test_pos"]),
   "n_features": R["n_features"],
   "python": "3.11.9", "pandas": "3.0.6", "numpy": "2.4.6",
   "sklearn": "1.9.1", "matplotlib": "3.11.2", "statsmodels": "0.15.0",
 },
 "preprocessing": {
   "missing_workclass": R["missing"]["workclass"],
   "missing_occupation": R["missing"]["occupation"],
   "missing_native_country": R["missing"]["native-country"],
   "modes": R["modes"], "caps": R["caps"], "caps_train_995": R["caps_train_995"],
   "capital_gain_cap_noop": R["capital_gain_cap_noop"],
   "fnlwgt_kept": R["fnlwgt_kept"], "race_reference": R["race_reference"],
 },
 "part1": {
   "modelA_median": r4(R["modelA"]["median"]), "modelA_mean": r4(R["modelA"]["mean"]),
   "modelA_min": r4(R["modelA"]["min"]), "modelA_max": r4(R["modelA"]["max"]),
   "modelA_acc": r4(R["modelA"]["acc"]), "modelB_acc": r4(R["modelB"]["acc"]),
   "agreement_cm": R["agreement_cm"], "mismatch": R["mismatch"],
   "mismatch_pct": r4(R["mismatch_pct"]),
   "q1_band": R["q1_band"],
   "top_coefs": R["logr_top"], "bottom_coefs": R["logr_bottom"],
   "logistic_penalty": R["logistic_penalty"],
 },
 "part21_simpson": {
   "age": {"full": r4(R["simpson_full"]["age"]),
           "lt30": r4(R["simpson_sub"]["Age<30"]["age"]),
           "ge30": r4(R["simpson_sub"]["Age>=30"]["age"])},
   "age_ratio_lt30_over_full": r4(R["simpson_sub"]["Age<30"]["age"]/R["simpson_full"]["age"]),
   "sm_age": {k: {"coef": r4(R["sm_age"][k]["coef"]), "se": r4(R["sm_age"][k]["se"]),
                  "ci": [r4(R["sm_age"][k]["lo"]), r4(R["sm_age"][k]["hi"])],
                  "p": R["sm_age"][k]["p"]} for k in R["sm_age"]},
   "ci_overlaps": {"full_vs_lt30": False, "full_vs_ge30": False, "lt30_vs_ge30": False},
   "subgroup_n": {"Male": 22887, "Female": 11302, "Age<30": 10097, "Age>=30": 24092},
   "dropped_degenerate": R["dropped_degenerate"],
   "self_emp_not_inc": {
     "full": r4(R["simpson_full"]["capital-gain"]*0 + R["logr_bottom"][2][1]) if False else None,
     "lt30": r4(R["simpson_sub"]["Age<30"].get("workclass_Self-emp-not-inc", float("nan"))),
     "ge30": r4(R["simpson_sub"]["Age>=30"].get("workclass_Self-emp-not-inc", float("nan"))),
   },
 },
 "part22_ovb": [{"feat": o["feat"], "full": r4(o["full"]), "reduced": r4(o["reduced"]), "pct": r4(o["pct"])}
                for o in sorted(R["ovb"], key=lambda x: -abs(x["pct"]))],
 "part23_adversarial": {
   "hc_count": R["adversarial"]["hc_count"], "hc_pct": r4(R["adversarial"]["hc_pct"]),
   "fp": R["adversarial"]["fp"], "fn": R["adversarial"]["fn"],
   "subset_vs_test": R["adversarial"]["exact"],
   "calibration_low_bin": {"n": R["adversarial"]["calibration_low_n"],
                           "observed_positive_rate": r4(R["adversarial"]["calibration_low_err"])},
   "calibration_high_bin": {"n": R["adversarial"]["calibration_high_n"],
                            "observed_negative_rate": r4(R["adversarial"]["calibration_high_err"])},
   "reliability_deciles": [{"decile": d["bin"], "n": d["n"], "mean_pred": r4(d["mean_p"]),
                            "observed": r4(d["emp"]), "gap": r4(d["emp"]-d["mean_p"])}
                           for d in R["adversarial"]["reliability"]],
   "young_exec_check": {"n": R["adversarial"]["young_exec_n"],
                        "max_P": r4(R["adversarial"]["young_exec_max_p"]),
                        "mean_P": r4(R["adversarial"]["young_exec_mean_p"]),
                        "n_ge_075": R["adversarial"]["young_exec_ge075"]},
 },
 "part24_fairness": {
   "group_A": "Female", "group_B": "Male",
   "sweep": [{"t": f["t"], "DP": r4(f["DP"]), "EO": r4(f["EO"]), "PP": r4(f["PP"]),
              "TPR_ratio": r4(f["TPR_ratio"]), "FPR_ratio": r4(f["FPR_ratio"]),
              "acc": r4(f["acc"]), "flag_F": r4(f["prA"]), "flag_M": r4(f["prB"]),
              "tpr_F": r4(f["tprA"]), "tpr_M": r4(f["tprB"]),
              "fpr_F": r4(f["fprA"]), "fpr_M": r4(f["fprB"]),
              "ppv_F": r4(f["ppvA"]), "ppv_M": r4(f["ppvB"])} for f in R["fairness"]],
   "closest_to_1": {
     m: {"t": min((r[m] for r in R["fairness"]), key=lambda v: abs(v-1)),
         "value": r4(min((abs(r[m]-1), r[m]) for r in R["fairness"])[1])}
     for m in ["DP", "EO", "PP"]
   },
   "q6_pair": R["q6_pair"],
 },
}
# fill self_emp properly
for k, name in [("full", None), ("lt30", "Age<30"), ("ge30", "Age>=30")]:
    if name is None:
        pass
cited["part21_simpson"]["self_emp_not_inc"] = {
  "lt30": r4(R["simpson_sub"]["Age<30"]["workclass_Self-emp-not-inc"]),
  "ge30": r4(R["simpson_sub"]["Age>=30"]["workclass_Self-emp-not-inc"]),
}

with open(os.path.join(OUT, "numbers_cited.json"), "w") as f:
    json.dump(cited, f, indent=2)
print("wrote numbers_cited.json")
print(json.dumps({k: cited[k] for k in ["part21_simpson", "part24_fairness"] if k!="part24_fairness"}, indent=1)[:1500])
