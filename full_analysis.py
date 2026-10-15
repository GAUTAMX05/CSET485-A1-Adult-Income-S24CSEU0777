"""CSET485 Assignment 1 - Adult Census Income, seed 777
Auto-generated from CSET485_A1_Gautam_S24CSEU0777.ipynb (the source of truth).
Run: python full_analysis.py   (it writes results.json and 4 charts next to itself)
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---- notebook cell ----
import pandas as pd, numpy as np, json, os, sys, platform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

SEED = 777
OUT = os.path.dirname(os.path.abspath("__file__")) if os.path.exists("__file__") else os.getcwd()
print("python", sys.version.split()[0], "|", platform.platform())
for lib in [pd, np, __import__("sklearn"), __import__("matplotlib"), __import__("statsmodels")]:
    print(lib.__name__, lib.__version__)

# ---- notebook cell ----
cols = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
        "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
tr = pd.read_csv(URL+"adult.data",  header=None, names=cols, sep=",", skipinitialspace=True)
te = pd.read_csv(URL+"adult.test", header=None, names=cols, sep=",", skipinitialspace=True, skiprows=1)
te["income"] = te["income"].str.replace(".", "", regex=False)
df_raw = pd.concat([tr, te], ignore_index=True)

for c in df_raw.select_dtypes(include="object").columns:
    df_raw[c] = df_raw[c].astype(str).str.strip()
df_raw.replace("?", np.nan, inplace=True)
df_raw["income_binary"] = (df_raw["income"]==">50K").astype(int)

print("shape", df_raw.shape)
print(df_raw["income"].value_counts())
print("positives: %d (%.4f)" % (df_raw["income_binary"].sum(), df_raw["income_binary"].mean()))
print("\nsex:\n", df_raw["sex"].value_counts())
print("\nrace:\n", df_raw["race"].value_counts())
print("\nmissing:\n", df_raw.isna().sum()[lambda s: s>0])

# ---- 70/30 stratified split on the RAW frame ----
train_raw, test_raw = train_test_split(df_raw, test_size=0.30, random_state=SEED,
                                       stratify=df_raw["income_binary"])
print("\nraw split: train %s, test %s" % (train_raw.shape, test_raw.shape))

# ---- notebook cell ----
# ---- FIT on TRAIN ONLY, then APPLY to both splits ----
modes = {c: str(train_raw[c].mode()[0]) for c in ["workclass","occupation","native-country"]}
caps_fit = {c: float(train_raw[c].quantile(0.995)) for c in ["capital-gain","capital-loss","fnlwgt"]}
caps = {"capital-gain": int(caps_fit["capital-gain"]), "capital-loss": int(caps_fit["capital-loss"]),
        "hours-per-week": 80, "age": 75, "fnlwgt": int(caps_fit["fnlwgt"])}
print("train-fitted modes:", modes)
print("train-fitted 99.5th percentiles:", caps_fit)
print("caps actually applied:", caps)

num_cols  = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
cat_cols  = ["workclass","marital-status","occupation","relationship","race","native-country-grouped"]

def preprocess(df_in, is_train, dummy_cols=None):
    df = df_in.copy()
    # 1. impute with TRAIN-fitted modes
    for c in ["workclass","occupation","native-country"]:
        df[c] = df[c].fillna(modes[c])
    # 2. derive
    df["native-country-grouped"] = np.where(df["native-country"]=="United-States","United-States","Non-US")
    # 3. clip with TRAIN-fitted / fixed caps
    for c, cap in caps.items():
        df[c] = df[c].clip(upper=cap)
    df["sex_Male"] = (df["sex"]=="Male").astype(int)
    # 4. one-hot; on test, force the TRAIN-determined columns
    dummies = pd.get_dummies(df[cat_cols], drop_first=True, dtype=int)
    if is_train:
        dummy_cols = list(dummies.columns)
    else:
        dummies = dummies.reindex(columns=dummy_cols, fill_value=0)
    X = pd.concat([df[num_cols], df[["sex_Male"]], dummies], axis=1)
    meta = df[["age","sex","race","education-num","hours-per-week","capital-gain","capital-loss",
               "occupation","marital-status","relationship","workclass"]].copy()
    return X, df["income_binary"], meta, dummy_cols

X_train_raw, y_train, meta_train, dummy_cols = preprocess(train_raw, is_train=True)
X_test_raw,  y_test,  meta_test,  _           = preprocess(test_raw,  is_train=False, dummy_cols=dummy_cols)
print("X_train", X_train_raw.shape, "| X_test", X_test_raw.shape, "| dummies", len(dummy_cols))
print("train pos rate %.4f | test pos rate %.4f" % (y_train.mean(), y_test.mean()))
print("race reference category (dropped by drop_first=True):",
      sorted(train_raw["race"].dropna().unique())[0])

# ---- notebook cell ----
scaler = StandardScaler()
Xtr = X_train_raw.copy(); Xte = X_test_raw.copy()
Xtr[num_cols] = scaler.fit_transform(X_train_raw[num_cols])
Xte[num_cols] = scaler.transform(X_test_raw[num_cols])
print("scaled train mean ~0:", np.round(Xtr[num_cols].mean().values, 6))
print("scaled train std  ~1:", np.round(Xtr[num_cols].std().values, 3))

# ---- notebook cell ----
lin = LinearRegression(); lin.fit(Xtr, y_train)
pred_a  = lin.predict(Xte)
median_a = float(np.median(pred_a))
high_a  = (pred_a >= median_a).astype(int)

logr = LogisticRegression(penalty=None, max_iter=1000, solver="lbfgs")
logr.fit(Xtr, y_train)
proba  = logr.predict_proba(Xte)[:, 1]
high_b = (proba >= 0.5).astype(int)

cm = confusion_matrix(high_a, high_b, labels=[0,1])
n_test = len(y_test)
mismatch = int((high_a != high_b).sum())
acc_b = float((high_b == y_test.values).mean())
acc_a = float((high_a == y_test.values).mean())

print("Model A OLS: median %.4f  mean %.4f  min %.4f  max %.4f" % (median_a, pred_a.mean(), pred_a.min(), pred_a.max()))
print("Accuracy B (t=0.5)  %.4f" % acc_b)
print("Accuracy A (median) %.4f" % acc_a)
print("\nAgreement (rows = A at median, cols = B at 0.5):\n", cm)
print("mismatch %d / %d = %.2f%%" % (mismatch, n_test, mismatch/n_test*100))
print("flagged high: A=%d (%.1f%%)  B=%d (%.1f%%)" % (high_a.sum(), high_a.mean()*100, high_b.sum(), high_b.mean()*100))

# Q1: where exactly do the mismatching OLS scores sit?
mis  = pred_a[high_a != high_b]
in_band = int(((mis >= median_a) & (mis < 0.5)).sum())
ge_half = int((mis >= 0.5).sum())
below   = int((mis < median_a).sum())
print("\nQ1 mismatch OLS score distribution:")
print("  in [median, 0.5) : %d" % in_band)
print("  >= 0.5           : %d" % ge_half)
print("  < median         : %d" % below)
print("  check sum        : %d" % (in_band+ge_half+below))
print("  A-low/B-high     : %d  (in v1 this was 0; train-only caps make 16 cases flip the other way)" % int(((high_a==0)&(high_b==1)).sum()))

print("\nTop coefficients (penalty=None):")
print(pd.Series(logr.coef_[0], index=Xtr.columns).sort_values(ascending=False).head(8))
print(pd.Series(logr.coef_[0], index=Xtr.columns).sort_values().head(8))

# ---- notebook cell ----
def fit_nopenalty(Xd, yd):
    m = LogisticRegression(penalty=None, max_iter=1000, solver="lbfgs")
    m.fit(Xd, yd)
    return pd.Series(m.coef_[0], index=Xd.columns)

def nondegenerate(Xsub):
    keep = Xsub.columns[Xsub.nunique() > 1].tolist()
    dropped = [c for c in Xsub.columns if c not in keep]
    return keep, dropped

full_mask = np.ones(len(Xtr), dtype=bool)
c_full = fit_nopenalty(Xtr, y_train)

masks = {"Male": (meta_train["sex"]=="Male").values,
         "Female": (meta_train["sex"]=="Female").values,
         "Age<30": (meta_train["age"]<30).values,
         "Age>=30": (meta_train["age"]>=30).values}

sub_coefs, dropped_info = {}, {}
for k, m in masks.items():
    keep, dropped = nondegenerate(Xtr[m])
    dropped_info[k] = dropped
    print("%-8s n=%d   dropped degenerate: %s" % (k, m.sum(), dropped if dropped else "none"))
    sub_coefs[k] = fit_nopenalty(Xtr[m][keep], y_train[m])

print("\n%-22s %8s" % ("predictor", "full"), "".join("%9s" % k for k in masks))
for feat in ["age","education-num","hours-per-week","capital-gain",
             "workclass_Self-emp-not-inc","race_Black","marital-status_Married-civ-spouse"]:
    row = "%8.4f" % c_full[feat]
    line = "%-22s %8s" % (feat, row)
    for k in masks:
        line += "%9.4f" % (sub_coefs[k][feat] if feat in sub_coefs[k] else float("nan"))
    print(line)

# ---- notebook cell ----
import statsmodels.api as sm

# --- statsmodels 95% CIs for `age`, same three models, same data ---
ci_rows = []
sm_results = {}
for name, mask in [("Full", full_mask), ("Age<30", masks["Age<30"]), ("Age>=30", masks["Age>=30"])]:
    X = Xtr[mask] if name != "Full" else Xtr
    y = y_train[mask] if name != "Full" else y_train
    res = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=100)
    r, se = res.params["age"], res.bse["age"]
    lo, hi = res.conf_int().loc["age"]
    sm_results[name] = {"coef": float(r), "se": float(se), "lo": float(lo), "hi": float(hi), "p": float(res.pvalues["age"])}
    ci_rows.append([name, "%.4f" % r, "%.4f" % se, "[%.4f, %.4f]" % (lo, hi), "%.3g" % res.pvalues["age"], int(mask.sum())])

print("95%% CIs for `age` (statsmodels MLE, unpenalised):")
print("%-9s %8s %8s %-18s %10s %6s" % ("model","coef","SE","CI 95%","p-value","n"))
for r in ci_rows: print("%-9s %8s %8s %-18s %10s %6d" % tuple(r))

def ci_overlap(a, b):
    return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])
print("\nCI overlap tests (age):")
print("  Full vs Age<30  overlap:", ci_overlap(sm_results["Full"],  sm_results["Age<30"]))
print("  Full vs Age>=30 overlap:", ci_overlap(sm_results["Full"],  sm_results["Age>=30"]))
print("  Age<30 vs Age>=30 overlap:", ci_overlap(sm_results["Age<30"], sm_results["Age>=30"]))

# reference CIs for the other headline predictors on the FULL model only
res_full = sm.Logit(y_train, sm.add_constant(Xtr, has_constant="add")).fit(disp=0, maxiter=100)
print("\nFull-model reference (education-num, hours-per-week, capital-gain):")
for k in ["education-num","hours-per-week","capital-gain"]:
    lo, hi = res_full.conf_int().loc[k]
    print("  %-18s %.4f  SE %.4f  CI [%.4f, %.4f]  p=%.2g"
          % (k, res_full.params[k], res_full.bse[k], lo, hi, res_full.pvalues[k]))

# ---- notebook cell ----
cols_drop = [c for c in Xtr.columns if c != "education-num"]
mf = LogisticRegression(penalty=None, max_iter=1000); mf.fit(Xtr, y_train)
mr = LogisticRegression(penalty=None, max_iter=1000); mr.fit(Xtr[cols_drop], y_train)
cf = pd.Series(mf.coef_[0], index=Xtr.columns)
cr = pd.Series(mr.coef_[0], index=cols_drop)

OVB_FEATS = ["sex_Male","hours-per-week","age","capital-gain","occupation_Exec-managerial",
             "marital-status_Married-civ-spouse","race_White","capital-loss"]
ovb = []
for feat in OVB_FEATS:
    b, a = float(cf[feat]), float(cr[feat])
    pct = (a - b) / b * 100
    ovb.append({"feat": feat, "full": b, "reduced": a, "pct": float(pct)})
    print("%-34s %8.4f -> %8.4f  (%+.2f%%)" % (feat, b, a, pct))

print("\nRanked by size of shift:")
for o in sorted(ovb, key=lambda x: -abs(x["pct"])):
    print("  %-34s %+.2f%%" % (o["feat"], o["pct"]))

# ---- notebook cell ----
y_true = y_test.values
hc_fp = (proba >= 0.75) & (y_true == 0)
hc_fn = (proba <= 0.25) & (y_true == 1)
hc = hc_fp | hc_fn
hc_idx = np.where(hc)[0]
print("high-confidence errors: %d / %d = %.2f%%   (FP %d, FN %d)"
      % (hc.sum(), len(y_true), hc.mean()*100, hc_fp.sum(), hc_fn.sum()))

rng = np.random.RandomState(SEED)
sel = rng.choice(hc_idx, size=min(50, len(hc_idx)), replace=False)
adv = meta_test.iloc[sel]

def p(x): return float(np.mean(x) * 100)
exact = {
 "adv_n": int(len(sel)),
 "adv_male_n": int((adv["sex"]=="Male").sum()), "adv_male_pct": p(adv["sex"]=="Male"),
 "test_male_n": int((meta_test["sex"]=="Male").sum()), "test_male_pct": p(meta_test["sex"]=="Male"),
 "adv_white_n": int((adv["race"]=="White").sum()), "adv_white_pct": p(adv["race"]=="White"),
 "test_white_pct": p(meta_test["race"]=="White"),
 "adv_black_pct": p(adv["race"]=="Black"), "test_black_pct": p(meta_test["race"]=="Black"),
 "adv_hours_le40_pct": p(adv["hours-per-week"]<=40), "test_hours_le40_pct": p(meta_test["hours-per-week"]<=40),
 "adv_gain0_pct": p(adv["capital-gain"]==0), "test_gain0_pct": p(meta_test["capital-gain"]==0),
 "adv_edu_le9_pct": p(adv["education-num"]<=9), "test_edu_le9_pct": p(meta_test["education-num"]<=9),
 "adv_nevermarried_pct": p(adv["marital-status"]=="Never-married"),
 "test_nevermarried_pct": p(meta_test["marital-status"]=="Never-married"),
 "adv_age_lt30_pct": p(adv["age"]<30), "test_age_lt30_pct": p(meta_test["age"]<30),
 "adv_age_mean": float(adv["age"].mean()), "test_age_mean": float(meta_test["age"].mean()),
 "adv_age_median": float(adv["age"].median()), "test_age_median": float(meta_test["age"].median()),
}
print("\n50-row subset vs full test set (exact, no rounding to '~'):")
for k in ["adv_male_n","adv_male_pct","test_male_pct","adv_white_n","adv_white_pct","test_white_pct",
          "adv_black_pct","test_black_pct","adv_hours_le40_pct","test_hours_le40_pct",
          "adv_gain0_pct","test_gain0_pct","adv_edu_le9_pct","test_edu_le9_pct",
          "adv_nevermarried_pct","test_nevermarried_pct","adv_age_lt30_pct","test_age_lt30_pct",
          "adv_age_mean","test_age_mean","adv_age_median","test_age_median"]:
    print("  %-26s %s" % (k, exact[k]))

# ---- calibration / reliability ----
low, high = proba <= 0.25, proba >= 0.75
cal_low  = float((y_true[low]==1).mean())      # empirical positive rate in the low band
cal_high = float((y_true[high]==0).mean())     # empirical negative rate in the high band
print("\nCalibration check:")
print("  band P<=0.25 : n=%-6d observed P(y=1)=%.4f  (error if we label all 0)" % (low.sum(), cal_low))
print("  band P>=0.75 : n=%-6d observed P(y=0)=%.4f  (error if we label all 1)" % (high.sum(), cal_high))

edges = np.quantile(proba, np.linspace(0, 1, 11))
rel = []
print("\nReliability table (prob decile):  mean_pred  observed   n   gap")
for i in range(10):
    lo_, hi_ = edges[i], edges[i+1]
    m = (proba >= lo_) & (proba <= hi_) if i == 9 else (proba >= lo_) & (proba < hi_)
    mp, obs = float(proba[m].mean()), float(y_true[m].mean())
    rel.append({"bin": i, "n": int(m.sum()), "mean_p": mp, "emp": obs})
    print("  decile %d: %8.4f  %8.4f  %5d  %+.4f" % (i, mp, obs, m.sum(), obs-mp))

# ---- verify the 'young Exec-managerial P~0.85' claim ----
cand = meta_test[(meta_test["age"] < 30) & (meta_test["occupation"] == "Exec-managerial")]
cand_p = proba[[meta_test.index.get_loc(i) for i in cand.index]] if len(cand) else np.array([])
print("\nYoung (<30) Exec-managerial in test: n=%d" % len(cand))
if len(cand):
    print("  P: max=%.4f  mean=%.4f  median=%.4f  #P>=0.75 = %d"
          % (cand_p.max(), cand_p.mean(), np.median(cand_p), int((cand_p>=0.75).sum())))

# ---- notebook cell ----
thresholds = [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.1)]
fair = []
for t in thresholds:
    pred = (proba >= t).astype(int)
    gA = (meta_test["sex"]=="Female").values   # Group A = Female
    gB = (meta_test["sex"]=="Male").values     # Group B = Male
    def stats(m):
        yp, yy = pred[m], y_true[m]
        tp = int(((yp==1)&(yy==1)).sum()); fp = int(((yp==1)&(yy==0)).sum())
        tn = int(((yp==0)&(yy==0)).sum()); fn = int(((yp==0)&(yy==1)).sum())
        return {"pr": float(yp.mean()),
                "tpr": tp/(tp+fn) if tp+fn else 0.0,
                "fpr": fp/(fp+tn) if fp+tn else 0.0,
                "ppv": tp/(tp+fp) if tp+fp else 0.0}
    A, B = stats(gA), stats(gB)
    tpr_r = A["tpr"]/B["tpr"] if B["tpr"] else float("nan")
    fpr_r = A["fpr"]/B["fpr"] if B["fpr"] else float("nan")
    dp = A["pr"]/B["pr"] if B["pr"] else float("nan")
    fair.append({"t": t, "DP": float(dp), "EO": float(min(tpr_r, fpr_r)), "PP": float(A["ppv"]/B["ppv"] if B["ppv"] else float("nan")),
                 "TPR_ratio": float(tpr_r), "FPR_ratio": float(fpr_r),
                 "prA": A["pr"], "prB": B["pr"], "tprA": A["tpr"], "tprB": B["tpr"],
                 "fprA": A["fpr"], "fprB": B["fpr"], "ppvA": A["ppv"], "ppvB": B["ppv"],
                 "acc": float((pred==y_true).mean())})
    print("t=%.2f  DP=%.4f  EO=%.4f (TPRr %.4f | FPRr %.4f)  PP=%.4f  acc=%.4f"
          % (t, fair[-1]["DP"], fair[-1]["EO"], fair[-1]["TPR_ratio"], fair[-1]["FPR_ratio"], fair[-1]["PP"], fair[-1]["acc"]))

print("\nClosest to 1.0 in this sweep (read off the table):")
for metric in ["DP", "EO", "PP"]:
    top = sorted([(abs(r[metric]-1), r["t"], r[metric]) for r in fair])[:3]
    print("  %-4s -> %s" % (metric, ", ".join("t=%.2f (%s=%.4f)" % (t, metric, v) for _, t, v in top)))

# ---- Q6: one real Female and one real Male test row with matching features ----
test_df = meta_test.copy(); test_df["P"] = proba; test_df["y"] = y_true
best = None
for occ in test_df["occupation"].unique():
    for mar in ["Married-civ-spouse", "Never-married"]:
        for race in test_df["race"].unique():
            sub = test_df[(test_df["occupation"]==occ) & (test_df["marital-status"]==mar) & (test_df["race"]==race)]
            M, F = sub[sub["sex"]=="Male"], sub[sub["sex"]=="Female"]
            if len(M)==0 or len(F)==0: continue
            for fi, frow in F.iterrows():
                if not (0.15 <= frow["P"] <= 0.6): continue
                d = ((M["age"]-frow["age"])**2 + ((M["education-num"]-frow["education-num"])*3)**2
                     + ((M["hours-per-week"]-frow["hours-per-week"])/5)**2)
                mi = d.idxmin(); mrow = M.loc[mi]
                score = float(d.min() + abs(frow["P"]-mrow["P"])*2)
                if best is None or score < best[0]:
                    best = (score, fi, mi, frow, mrow)
fi, mi, frow, mrow = best[1], best[2], best[3], best[4]

print("\nQ6 matched pair (same race, occupation, marital status; age/edu/hours equal):")
print("  Female idx %s: P=%.4f  y=%d" % (fi, frow["P"], frow["y"]))
print("    %s" % {k: str(frow[k]) for k in ["age","sex","race","education-num","hours-per-week","capital-gain","occupation","marital-status"]})
print("  Male idx %s: P=%.4f  y=%d" % (mi, mrow["P"], mrow["y"]))
print("    %s" % {k: str(mrow[k]) for k in ["age","sex","race","education-num","hours-per-week","capital-gain","occupation","marital-status"]})
for t in [0.05, 0.35, 0.75]:
    print("  t=%.2f -> Female predicted %d, Male predicted %d" % (t, int(frow["P"]>=t), int(mrow["P"]>=t)))

# ---- notebook cell ----
sub_out = {k: {i: float(v) for i, v in s.items()} for k, s in sub_coefs.items()}

results = {
 "seed": SEED, "n_raw": int(len(df_raw)), "train_n": int(len(train_raw)), "test_n": int(n_test),
 "pos_rate": float(df_raw["income_binary"].mean()),
 "train_pos": float(y_train.mean()), "test_pos": float(y_test.mean()),
 "missing": {c: int(df_raw[c].isna().sum()) for c in ["workclass","occupation","native-country"]},
 "modes": modes, "caps": caps, "caps_train_995": caps_fit, "n_features": int(Xtr.shape[1]),
 "fnlwgt_kept": True,
 "fnlwgt_note": "CPS sampling weight, kept unchanged from v1; small coefficient, not a personal trait",
 "capital_gain_cap_noop": True,
 "capital_gain_cap_note": "train 99.5th percentile of capital-gain is 99999 = dataset max, so the fitted cap does not clip anything",
 "logistic_penalty": "None (unpenalised MLE for all coefficient comparisons, matches statsmodels)",
 "race_reference": "Amer-Indian-Eskimo (first alphabetical race level dropped by drop_first=True)",
 "modelA": {"median": median_a, "mean": float(pred_a.mean()), "min": float(pred_a.min()),
            "max": float(pred_a.max()), "acc": acc_a},
 "modelB": {"acc": acc_b}, "agreement_cm": cm.tolist(), "mismatch": mismatch,
 "mismatch_pct": float(mismatch/n_test*100),
 "q1_band": {"in_median_to_half": in_band, "ge_half": ge_half, "below_median": below},
 "logr_top":  [[k, float(v)] for k, v in pd.Series(logr.coef_[0], index=Xtr.columns).sort_values(ascending=False).head(8).items()],
 "logr_bottom":[[k, float(v)] for k, v in pd.Series(logr.coef_[0], index=Xtr.columns).sort_values().head(8).items()],
 "simpson_full": {k: float(c_full[k]) for k in ["age","education-num","hours-per-week","capital-gain"]},
 "simpson_sub": sub_out, "dropped_degenerate": dropped_info,
 "sm_age": sm_results,
 "ovb": ovb,
 "adversarial": {"hc_count": int(hc.sum()), "hc_pct": float(hc.mean()*100), "fp": int(hc_fp.sum()),
                 "fn": int(hc_fn.sum()), "sel_n": int(len(sel)), "exact": exact,
                 "calibration_low_n": int(low.sum()), "calibration_low_err": cal_low,
                 "calibration_high_n": int(high.sum()), "calibration_high_err": cal_high,
                 "reliability": rel, "young_exec_n": int(len(cand)),
                 "young_exec_max_p": float(cand_p.max()), "young_exec_mean_p": float(cand_p.mean()),
                 "young_exec_ge075": int((cand_p>=0.75).sum())},
 "fairness": fair,
 "q6_pair": {"female_idx": str(fi), "female_P": float(frow["P"]), "female_y": int(frow["y"]),
             "female_feat": {k: str(frow[k]) for k in ["age","sex","race","education-num","hours-per-week","capital-gain","occupation","marital-status"]},
             "male_idx": str(mi), "male_P": float(mrow["P"]), "male_y": int(mrow["y"]),
             "male_feat": {k: str(mrow[k]) for k in ["age","sex","race","education-num","hours-per-week","capital-gain","occupation","marital-status"]}},
}

# ---- charts ----
plt.figure(figsize=(6,4))
plt.hist(pred_a, bins=40, color="#4a6fa5", edgecolor="white")
plt.axvline(median_a, color="red", linestyle="--", label=f"median {median_a:.4f}")
plt.axvline(0.5, color="black", linestyle=":", label="logistic 0.5")
plt.xlabel("Model A OLS predicted value"); plt.ylabel("Count")
plt.title("Model A predictions with thresholds"); plt.legend()
plt.tight_layout(); plt.savefig("chart_modelA_hist.png", dpi=150); plt.close()

plt.figure(figsize=(6.4,4.3))
mat = np.array(cm)
plt.imshow(mat, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(mat[i,j]), ha="center", va="center", fontsize=11, fontweight="bold")
plt.xticks([0,1], ["B low\n(P<0.5)", "B high\n(P>=0.5)"], fontsize=9)
plt.yticks([0,1], ["A low\n(<median)", "A high\n(>=median)"], fontsize=9)
plt.title(f"Agreement A vs B (mismatch {mismatch}, {mismatch/n_test*100:.2f}%)", fontsize=10)
plt.colorbar(); plt.tight_layout(); plt.savefig("chart_agreement.png", dpi=150); plt.close()

plt.figure(figsize=(6.2,4.6))
dps = [r["DP"] for r in fair]; eos = [r["EO"] for r in fair]
plt.scatter(dps, eos, c=[r["t"] for r in fair], cmap="viridis", s=90, edgecolor="black", zorder=3)
for r in fair:
    plt.annotate(f"{r['t']:.2f}", (r["DP"], r["EO"]), fontsize=8,
                 xytext=(5,5), textcoords="offset points")
plt.axhline(1.0, color="grey", linestyle="--"); plt.axvline(1.0, color="grey", linestyle="--")
plt.xlim(0, 1.5); plt.ylim(0, 1.5)
plt.xlabel("Demographic Parity F/M (0-1.5)")
plt.ylabel("Equalized Odds min(TPRr,FPRr) (0-1.5)")
plt.title("Fairness trade-off: DP vs EO by threshold (A=Female, B=Male)")
plt.colorbar(label="threshold"); plt.grid(alpha=0.2)
plt.tight_layout(); plt.savefig("chart_fairness_tradeoff.png", dpi=150); plt.close()
print("trade-off chart points inside the 0-1.5 view:", sum(1 for d,e in zip(dps,eos) if 0<=d<=1.5 and 0<=e<=1.5), "/", len(dps))

plt.figure(figsize=(7.2,3.8))
feats = [o["feat"] for o in ovb]; pcts = [o["pct"] for o in ovb]
plt.barh(feats, pcts, color=["#c0392b" if abs(x)==max(abs(np.array(pcts))) else "#4a6fa5" for x in pcts])
plt.xlabel("% change after dropping education-num (penalty=None)")
plt.title("Omitted-variable bias: % change")
plt.tight_layout(); plt.savefig("chart_ovb.png", dpi=150); plt.close()

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results.json and 4 charts. Headline: median", round(median_a,4),
      "| mismatch", mismatch, "(%.2f%%)" % (mismatch/n_test*100),
      "| accB", round(acc_b,4))

