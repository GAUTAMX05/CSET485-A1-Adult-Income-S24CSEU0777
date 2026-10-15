"""Full reproducible analysis - Adult Income, seed 777, Gautam Chauhan S24CSEU0777"""
import pandas as pd, numpy as np, json, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUT = r"C:\Users\gauta\Documents\Default Project\CSET485_A1_Gautam"
os.makedirs(OUT, exist_ok=True)
SEED = 777

cols = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
tr = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, names=cols, sep=",", skipinitialspace=True)
te = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", header=None, names=cols, sep=",", skipinitialspace=True, skiprows=1)
te["income"] = te["income"].str.replace(".", "", regex=False)
df_raw = pd.concat([tr, te], ignore_index=True)

df = df_raw.copy()
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()
df.replace("?", np.nan, inplace=True)
miss = df.isna().sum()
modes = {}
for c in ["workclass","occupation","native-country"]:
    modes[c] = str(df[c].mode()[0])
    df[c] = df[c].fillna(modes[c])
df["native-country-grouped"] = np.where(df["native-country"]=="United-States","United-States","Non-US")
df["income_binary"] = (df["income"]==">50K").astype(int)

num_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
caps = {"capital-gain":41310,"capital-loss":2258,"hours-per-week":80,"age":75,"fnlwgt":589809}
for c,cap in caps.items():
    df[c] = df[c].clip(upper=cap)

df["sex_Male"] = (df["sex"]=="Male").astype(int)
cat_cols = ["workclass","marital-status","occupation","relationship","race","native-country-grouped"]
dummies = pd.get_dummies(df[cat_cols], drop_first=True, dtype=int)
X_full = pd.concat([df[num_cols], df[["sex_Male"]], dummies], axis=1)
y = df["income_binary"]
meta = df[["age","sex","race","education-num","hours-per-week","capital-gain","capital-loss","occupation","marital-status","relationship","workclass"]].copy()

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(X_full, y, meta, test_size=0.30, random_state=SEED, stratify=y)
scaler = StandardScaler()
Xtr = X_train.copy(); Xte = X_test.copy()
Xtr[num_cols] = scaler.fit_transform(X_train[num_cols])
Xte[num_cols] = scaler.transform(X_test[num_cols])

# Part1
lin = LinearRegression(); lin.fit(Xtr, y_train)
pred_a = lin.predict(Xte)
median_a = float(np.median(pred_a))
high_a = (pred_a >= median_a).astype(int)
logr = LogisticRegression(max_iter=1000); logr.fit(Xtr, y_train)
proba = logr.predict_proba(Xte)[:,1]
high_b = (proba >= 0.5).astype(int)
cm = confusion_matrix(high_a, high_b, labels=[0,1]).tolist()
mismatch = int((high_a!=high_b).sum())
n_test = int(len(y_test))
acc_b = float((high_b==y_test.values).mean())
acc_a = float((high_a==y_test.values).mean())

# Simpson - logistic coefs full + subgroups
def fitc(mask):
    m = LogisticRegression(max_iter=1000); m.fit(Xtr[mask], y_train[mask])
    return pd.Series(m.coef_[0], index=Xtr.columns)

full_mask = np.ones(len(Xtr), dtype=bool)
c_full = fitc(full_mask)
masks = {
 "Male": (meta_train["sex"]=="Male").values,
 "Female": (meta_train["sex"]=="Female").values,
 "Age<30": (meta_train["age"]<30).values,
 "Age>=30": (meta_train["age"]>=30).values,
}
sub_coefs = {}
for k,m in masks.items():
    sub_coefs[k] = fitc(m)
# statsmodels CIs
import statsmodels.api as sm
sm_rows = {}
for name in ["Full","Male","Female","Age<30","Age>=30"]:
    if name=="Full":
        Xs = sm.add_constant(Xtr, has_constant='add'); ys=y_train; cols_sm=list(Xtr.columns)
    elif name in ["Male","Female"]:
        cols_sm=[c for c in Xtr.columns if c!="sex_Male"]
        mk = masks[name]; Xs=sm.add_constant(Xtr.loc[mk,cols_sm], has_constant='add'); ys=y_train[mk]
    else:
        cols_sm=list(Xtr.columns); mk=masks[name]; Xs=sm.add_constant(Xtr.loc[mk], has_constant='add'); ys=y_train[mk]
    try:
        res=sm.Logit(ys,Xs).fit(disp=0, maxiter=100)
        d={}
        for k in ["education-num","hours-per-week","age","capital-gain","sex_Male"]:
            if k in res.params.index:
                ci=res.conf_int().loc[k]
                d[k]={"coef":float(res.params[k]),"se":float(res.bse[k]),"ci_low":float(ci[0]),"ci_high":float(ci[1]),"p":float(res.pvalues[k])}
        sm_rows[name]=d
    except Exception as e:
        sm_rows[name]={"error":str(e)}

# OVB - drop education-num
cols_nodrop=list(Xtr.columns)
cols_drop=[c for c in Xtr.columns if c!="education-num"]
m_full=LogisticRegression(max_iter=1000); m_full.fit(Xtr, y_train)
m_red=LogisticRegression(max_iter=1000); m_red.fit(Xtr[cols_drop], y_train)
cf=pd.Series(m_full.coef_[0], index=cols_nodrop)
cr=pd.Series(m_red.coef_[0], index=cols_drop)
ovb=[]
for k in ["sex_Male","hours-per-week","age","capital-gain","occupation_Exec-managerial","marital-status_Married-civ-spouse","race_White","capital-loss"]:
    b=float(cf[k]); a=float(cr[k]); pct=float((a-b)/b*100) if b!=0 else 0
    ovb.append({"feat":k,"full":b,"reduced":a,"pct":pct})
# biggest shift
ovb_sorted=sorted(ovb, key=lambda x: abs(x["pct"]), reverse=True)

# Adversarial
proba_test=proba; y_true=y_test.values
hc_fp=(proba_test>=0.75)&(y_true==0)
hc_fn=(proba_test<=0.25)&(y_true==1)
hc=hc_fp|hc_fn
hc_idx=np.where(hc)[0]
hc_count=int(hc.sum()); hc_pct=float(hc.mean()*100)
fp_c=int(hc_fp.sum()); fn_c=int(hc_fn.sum())
rng=np.random.RandomState(SEED)
sel=rng.choice(hc_idx, size=min(50,len(hc_idx)), replace=False) if len(hc_idx)>0 else np.array([],dtype=int)
adv_meta=meta_test.iloc[sel] if len(sel)>0 else meta_test.iloc[0:0]
# demographics
def desc(s):
    return {"mean":float(s.mean()),"median":float(s.median()),"std":float(s.std())}
adv_age=desc(adv_meta["age"]) if len(sel)>0 else {}
test_age=desc(meta_test["age"])
adv_sex=adv_meta["sex"].value_counts(normalize=True).to_dict() if len(sel)>0 else {}
test_sex=meta_test["sex"].value_counts(normalize=True).to_dict()
adv_race=adv_meta["race"].value_counts(normalize=True).to_dict() if len(sel)>0 else {}
# common characteristic search
common={}
if len(sel)>0:
    # check several candidate traits
    ft=meta_test.iloc[hc_idx] if len(hc_idx)>0 else meta_test.iloc[0:0]
    # traits
    common["never_married_pct"]=float(((adv_meta["marital-status"]=="Never-married").mean())*100)
    common["own_child_pct"]=float(((adv_meta["relationship"]=="Own-child").mean())*100)
    common["hours_le40_pct"]=float(((adv_meta["hours-per-week"]<=40).mean())*100)
    common["edu_le9_pct"]=float(((adv_meta["education-num"]<=9).mean())*100)
    common["capital_gain_zero_pct"]=float(((adv_meta["capital-gain"]==0).mean())*100)
    common["age_lt30_pct"]=float(((adv_meta["age"]<30).mean())*100)
    # full test baselines
    common["never_married_base"]=float(((meta_test["marital-status"]=="Never-married").mean())*100)
    common["capital_gain_zero_base"]=float(((meta_test["capital-gain"]==0).mean())*100)
    common["edu_le9_base"]=float(((meta_test["education-num"]<=9).mean())*100)

# Fairness - sex Male vs Female
thresholds=[round(x,2) for x in np.arange(0.05,1.0,0.1)]
fair=[]
for t in thresholds:
    pred=(proba_test>=t).astype(int)
    yt=y_true
    # groups
    gA=(meta_test["sex"]=="Male").values  # Group A Male
    gB=(meta_test["sex"]=="Female").values
    def grp(m):
        yp=pred[m]; yy=yt[m]
        pr=float(yp.mean()) if len(yp)>0 else 0
        tp=int(((yp==1)&(yy==1)).sum()); fp=int(((yp==1)&(yy==0)).sum()); tn=int(((yp==0)&(yy==0)).sum()); fn=int(((yp==0)&(yy==1)).sum())
        tpr=tp/(tp+fn) if (tp+fn)>0 else 0
        fpr=fp/(fp+tn) if (fp+tn)>0 else 0
        ppv=tp/(tp+fp) if (tp+fp)>0 else 0
        acc=(tp+tn)/len(yp) if len(yp)>0 else 0
        return {"pr":pr,"tpr":tpr,"fpr":fpr,"ppv":ppv,"acc":acc,"n":int(m.sum())}
    A=grp(gA); B=grp(gB)
    dp=A["pr"]/B["pr"] if B["pr"]>0 else float("nan")
    # EO as min(TPR_A/TPR_B, FPR_A/FPR_B) per spec; guard div0
    def safe_div(a,b):
        if b==0:
            return float("nan") if a!=0 else 1.0
        return a/b
    eo=min(safe_div(A["tpr"],B["tpr"]), safe_div(A["fpr"],B["fpr"]))
    pp=A["ppv"]/B["ppv"] if B["ppv"]>0 else float("nan")
    fair.append({"t":t,"DP":float(dp),"EO":float(eo),"PP":float(pp),
                 "prA":A["pr"],"prB":B["pr"],"tprA":A["tpr"],"tprB":B["tpr"],"fprA":A["fpr"],"fprB":B["fpr"],"ppvA":A["ppv"],"ppvB":B["ppv"],"acc":float((pred==yt).mean())})

# Charts
plt.figure(figsize=(6,4))
plt.hist(pred_a, bins=40, color="#4a6fa5", edgecolor="white")
plt.axvline(median_a, color="red", linestyle="--", label=f"median {median_a:.3f}")
plt.axvline(0.5, color="black", linestyle=":", label="logistic 0.5")
plt.xlabel("Model A OLS predicted value"); plt.ylabel("Count"); plt.title("Model A predictions with thresholds")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT,"chart_modelA_hist.png"), dpi=150); plt.close()

plt.figure(figsize=(5,4))
mat=np.array(cm)
plt.imshow(mat, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j,i,str(mat[i,j]),ha="center",va="center",fontsize=12, fontweight="bold")
plt.xticks([0,1],["B low (P<0.5)","B high (P>=0.5)"]); plt.yticks([0,1],["A low (<median)","A high (>=median)"])
plt.title(f"Agreement A vs B (mismatch {mismatch}, {mismatch/n_test*100:.1f}%)")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"chart_agreement.png"), dpi=150); plt.close()

plt.figure(figsize=(6,4.5))
dps=[r["DP"] for r in fair]; eos=[r["EO"] for r in fair]
plt.scatter(dps, eos, c=[r["t"] for r in fair], cmap="viridis", s=80, edgecolor="black")
for r in fair:
    plt.annotate(f"{r['t']:.2f}", (r["DP"], r["EO"]), fontsize=8, xytext=(4,4), textcoords="offset points")
plt.axhline(1.0,color="grey",linestyle="--"); plt.axvline(1.0,color="grey",linestyle="--")
plt.xlim(0,1.6); plt.ylim(0,1.6)
plt.xlabel("Demographic Parity (Male/Female)"); plt.ylabel("Equalized Odds min(TPRr,FPRr)")
plt.title("Fairness trade-off: DP vs EO by threshold"); plt.colorbar(label="threshold")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"chart_fairness_tradeoff.png"), dpi=150); plt.close()

# OVB bar
plt.figure(figsize=(7,3.5))
feats=[o["feat"] for o in ovb]; pcts=[o["pct"] for o in ovb]
plt.barh(feats, pcts, color=["#c0392b" if abs(x)==max(abs(np.array(pcts))) else "#4a6fa5" for x in pcts])
plt.xlabel("% change after dropping education-num"); plt.title("Omitted-variable bias: % change")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"chart_ovb.png"), dpi=150); plt.close()

results={
 "seed":SEED, "n_raw":int(len(df_raw)), "n":int(len(df)),
 "pos_rate":float(y.mean()), "train_n":int(len(X_train)), "test_n":n_test,
 "missing":{"workclass":2799,"occupation":2809,"native-country":857},
 "modes":modes, "caps":caps, "n_features":int(X_full.shape[1]),
 "modelA":{"median":median_a,"mean":float(pred_a.mean()),"min":float(pred_a.min()),"max":float(pred_a.max()),"acc":acc_a},
 "modelB":{"acc":acc_b},
 "agreement_cm":cm, "mismatch":mismatch, "mismatch_pct":float(mismatch/n_test*100),
 "logr_top":[(k,float(c_full[k])) for k in c_full.sort_values(ascending=False).head(8).index],
 "logr_bottom":[(k,float(c_full[k])) for k in c_full.sort_values().head(8).index],
 "simpson_full":{k:float(c_full[k]) for k in ["education-num","hours-per-week","age","capital-gain","sex_Male"]},
 "simpson_sub":{k:{kk:float(v[kk]) for kk in ["education-num","hours-per-week","age","capital-gain","sex_Male"] if kk in v} for k,v in sub_coefs.items()},
 "simpson_extra":{
   "SelfEmpNotInc_full":float(c_full["workclass_Self-emp-not-inc"]),
   "SelfEmpNotInc_<30":float(sub_coefs["Age<30"]["workclass_Self-emp-not-inc"]),
   "SelfEmpNotInc_>=30":float(sub_coefs["Age>=30"]["workclass_Self-emp-not-inc"]),
   "Wife_full":float(c_full["relationship_Wife"]),
   "Wife_Male":float(sub_coefs["Male"]["relationship_Wife"]),
   "Wife_Female":float(sub_coefs["Female"]["relationship_Wife"]),
   "Black_full":float(c_full["race_Black"]) if "race_Black" in c_full else 0,
   "Black_Female":float(sub_coefs["Female"]["race_Black"]) if "race_Black" in sub_coefs["Female"] else 0,
 },
 "sm":sm_rows,
 "ovb":ovb, "ovb_sorted":ovb_sorted,
 "adversarial":{"hc_count":hc_count,"hc_pct":hc_pct,"fp":fp_c,"fn":fn_c,"sel_n":int(len(sel)),
   "adv_age":adv_age,"test_age":test_age,"adv_sex":adv_sex,"test_sex":test_sex,"adv_race":adv_race,"common":common},
 "fairness":fair,
}
with open(os.path.join(OUT,"results.json"),"w") as f:
    json.dump(results,f,indent=2)
print(json.dumps({k:results[k] for k in ["modelA","modelB","mismatch","mismatch_pct","simpson_full","simpson_sub","ovb","adversarial","fairness"]}, indent=2)[:8000])
print("SAVED results.json + charts")
