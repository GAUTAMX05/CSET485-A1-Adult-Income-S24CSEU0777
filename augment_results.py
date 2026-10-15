"""Append report-only metrics to results.json after full_analysis.py runs.
All values are recomputed from the same seed-777 train/test/model objects created by full_analysis.py.
"""
import json
import os
import runpy
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
# Run the exact analysis script as the source of truth, then reuse its objects.
ns = runpy.run_path(os.path.join(HERE, "full_analysis.py"), run_name="__main__")

with open(os.path.join(HERE, "results.json"), encoding="utf-8") as f:
    results = json.load(f)

# Configuration used by the report. Stored here so build_report.py never hard-codes
# report-facing numeric values.
results["report_config"] = {
    "ci_level": 0.95,
    "parity_target": 1.0,
    "high_confidence_positive_threshold": 0.75,
    "high_confidence_negative_threshold": 0.25,
    "fairness_thresholds": [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.1)],
    "chosen_threshold": 0.35,
    "alternative_threshold": 0.75,
    "seed": 777,
}

# Test-set base positive rate by sex and Male+Wife count.
test_raw = ns["test_raw"]
df_raw = ns["df_raw"]
for frame_name, frame in [("full", df_raw), ("train", ns["train_raw"]), ("test", test_raw)]:
    by_sex = {}
    for sex in ["Female", "Male"]:
        s = frame[frame["sex"] == sex]
        by_sex[sex] = {
            "n": int(len(s)),
            "positive_n": int(s["income_binary"].sum()),
            "positive_rate": float(s["income_binary"].mean()),
            "relationship_Wife_n": int(((s["relationship"] == "Wife")).sum()),
        }
    results.setdefault("sex_base_rates", {})[frame_name] = by_sex
results["male_relationship_wife_count"] = {
    "full": int(((df_raw["sex"] == "Male") & (df_raw["relationship"] == "Wife")).sum()),
    "train": int(((ns["train_raw"]["sex"] == "Male") & (ns["train_raw"]["relationship"] == "Wife")).sum()),
    "test": int(((test_raw["sex"] == "Male") & (test_raw["relationship"] == "Wife")).sum()),
}

# Statsmodels CIs/p-values for workclass_Self-emp-not-inc in the same three models.
Xtr = ns["Xtr"]
y_train = ns["y_train"]
masks = ns["masks"]
sm_selfemp = {}
for name, mask in [("Full", np.ones(len(Xtr), dtype=bool)), ("Age<30", masks["Age<30"]), ("Age>=30", masks["Age>=30"])]:
    X = Xtr[mask]
    y = y_train[mask]
    res = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=100)
    lo, hi = res.conf_int().loc["workclass_Self-emp-not-inc"]
    sm_selfemp[name] = {
        "coef": float(res.params["workclass_Self-emp-not-inc"]),
        "se": float(res.bse["workclass_Self-emp-not-inc"]),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(res.pvalues["workclass_Self-emp-not-inc"]),
        "n": int(mask.sum()),
    }
results["sm_selfemp"] = sm_selfemp

# FN/FP composition of the exact 50-row seed-777 subset already selected by full_analysis.py.
meta_test = ns["meta_test"].copy()
proba = ns["proba"]
y_true = ns["y_true"]
sel = ns["sel"]
adv = meta_test.iloc[sel].copy()
adv["P"] = proba[sel]
adv["y"] = y_true[sel]
adv["error_type"] = np.where((adv["P"] >= 0.75) & (adv["y"] == 0), "FP", "FN")

def profile(frame):
    return {
        "n": int(len(frame)),
        "male_pct": float((frame["sex"] == "Male").mean() * 100) if len(frame) else None,
        "white_pct": float((frame["race"] == "White").mean() * 100) if len(frame) else None,
        "gain0_pct": float((frame["capital-gain"] == 0).mean() * 100) if len(frame) else None,
        "edu_le9_pct": float((frame["education-num"] <= 9).mean() * 100) if len(frame) else None,
        "never_married_pct": float((frame["marital-status"] == "Never-married").mean() * 100) if len(frame) else None,
        "hours_le40_pct": float((frame["hours-per-week"] <= 40).mean() * 100) if len(frame) else None,
        "age_mean": float(frame["age"].mean()) if len(frame) else None,
        "age_median": float(frame["age"].median()) if len(frame) else None,
    }
results["adversarial"]["subset_error_split"] = {
    "selected_n": int(len(adv)),
    "fp_n": int((adv["error_type"] == "FP").sum()),
    "fn_n": int((adv["error_type"] == "FN").sum()),
    "fp_profile": profile(adv[adv["error_type"] == "FP"]),
    "fn_profile": profile(adv[adv["error_type"] == "FN"]),
    "overall_profile": profile(adv),
}

# Real Q6 pair: enrich the existing deterministic pair with relationship and coefficients.
pair = results["q6_pair"]
fi = int(pair["female_idx"])
mi = int(pair["male_idx"])
frow = meta_test.loc[fi]
mrow = meta_test.loc[mi]
coef_series = pd.Series(ns["logr"].coef_[0], index=ns["Xtr"].columns)
pair["female_relationship"] = str(frow["relationship"])
pair["male_relationship"] = str(mrow["relationship"])
pair["sex_Male_coef"] = float(coef_series["sex_Male"])
pair["relationship_Wife_coef"] = float(coef_series["relationship_Wife"])
pair["relationship_Wife_offsets_sex_Male"] = bool(frow["relationship"] == "Wife" and mrow["sex"] == "Male" and coef_series["relationship_Wife"] > coef_series["sex_Male"])
results["q6_pair"] = pair

# Explicit DP/EO trade-off for t=0.75 -> t=0.85.
fair = {round(float(r["t"]), 2): r for r in ns["fair"]}
r75, r85 = fair[0.75], fair[0.85]
dp75, dp85 = float(r75["DP"]), float(r85["DP"])
eo75, eo85 = float(r75["EO"]), float(r85["EO"])
parity = results["report_config"]["parity_target"]
trade = {
    "from_t": 0.75,
    "to_t": 0.85,
    "from": {k: float(r75[k]) for k in ["DP", "EO", "PP", "acc"]},
    "to": {k: float(r85[k]) for k in ["DP", "EO", "PP", "acc"]},
    "dp_distance_from_parity_before": abs(dp75 - parity),
    "dp_distance_from_parity_after": abs(dp85 - parity),
    "dp_distance_change": abs(dp85 - parity) - abs(dp75 - parity),
    "eo_distance_from_parity_before": abs(eo75 - parity),
    "eo_distance_from_parity_after": abs(eo85 - parity),
    "eo_distance_change": abs(eo85 - parity) - abs(eo75 - parity),
    "accuracy_loss_pp": (float(r75["acc"]) - float(r85["acc"])) * 100,
    "fairness_gain": bool(abs(dp85 - parity) < abs(dp75 - parity) or abs(eo85 - parity) < abs(eo75 - parity)),
}
results["fairness_tradeoff_075_085"] = trade

# A compact audit map used by build_report.py and its validation checks.
results["report_audit_map"] = {
    "dataset_rows": "n_raw",
    "train_rows": "train_n",
    "test_rows": "test_n",
    "positive_rate": "pos_rate",
    "test_positive_rate": "test_pos",
    "modelA_median": "modelA.median",
    "modelB_accuracy": "modelB.acc",
    "modelA_accuracy": "modelA.acc",
    "mismatch_count": "mismatch",
    "mismatch_percent": "mismatch_pct",
    "age_full": "sm_age.Full.coef",
    "age_young": "sm_age.Age<30.coef",
    "age_old": "sm_age.Age>=30.coef",
    "selfemp_full": "sm_selfemp.Full.coef",
    "selfemp_young": "sm_selfemp.Age<30.coef",
    "selfemp_old": "sm_selfemp.Age>=30.coef",
    "q6_female_probability": "q6_pair.female_P",
    "q6_male_probability": "q6_pair.male_P",
    "q6_sex_male_coef": "q6_pair.sex_Male_coef",
    "q6_wife_coef": "q6_pair.relationship_Wife_coef",
    "chosen_threshold": "report_config.chosen_threshold",
    "tradeoff_accuracy_loss_pp": "fairness_tradeoff_075_085.accuracy_loss_pp",
}

with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Augmented results.json with report metrics.")
