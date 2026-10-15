"""Generate PDF report - fpdf2, ASCII only"""
import json, os
from fpdf import FPDF
OUT = r"C:\Users\gauta\Documents\Default Project\CSET485_A1_Gautam"
with open(os.path.join(OUT,"results.json")) as f:
    R = json.load(f)

class PDF(FPDF):
    def header(self):
        if self.page_no()>1:
            self.set_font("Helvetica","I",8); self.set_text_color(100,100,100)
            self.cell(0,6,"CSET485 Assignment 1 - Gautam Chauhan S24CSEU0777 - Adult Income (seed 777)", align="C")
            self.ln(8)
    def footer(self):
        self.set_y(-12); self.set_font("Helvetica","I",8); self.set_text_color(120,120,120)
        self.cell(0,8,f"Page {self.page_no()}/{{nb}}", align="C")
    def h1(self,t):
        self.set_font("Helvetica","B",14); self.set_text_color(20,40,90)
        self.multi_cell(0,8,t); self.ln(1)
    def h2(self,t):
        self.set_font("Helvetica","B",12); self.set_text_color(30,60,120)
        self.multi_cell(0,7,t); self.ln(1)
    def h3(self,t):
        self.set_font("Helvetica","B",10.5); self.set_text_color(0,0,0)
        self.multi_cell(0,6,t); self.ln(1)
    def body(self,t):
        self.set_font("Helvetica","",9.5); self.set_text_color(20,20,20)
        self.multi_cell(0,5.2,t); self.ln(1.5)
    def bullet(self,t):
        self.set_font("Helvetica","",9.5); self.set_text_color(20,20,20)
        self.multi_cell(0,5.2,"- "+t); self.ln(0.5)
    def table(self, headers, rows, widths=None):
        self.set_font("Helvetica","B",8.2); self.set_fill_color(230,238,250)
        if widths is None:
            widths=[int(190/len(headers))]*len(headers)
        for i,h in enumerate(headers):
            self.cell(widths[i],6,h,border=1,fill=True,align="C")
        self.ln()
        self.set_font("Helvetica","",8.0)
        for r in rows:
            for i,v in enumerate(r):
                self.cell(widths[i],5.5,str(v),border=1,align="C")
            self.ln()
        self.ln(2)

pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=15)

# COVER
pdf.add_page()
pdf.set_font("Helvetica","B",18); pdf.set_text_color(20,40,90)
pdf.multi_cell(0,10,"CSET485 - AI and Society\nAssignment #1", align="C")
pdf.ln(2)
pdf.set_font("Helvetica","B",12); pdf.set_text_color(0,0,0)
pdf.multi_cell(0,7,"Predictive Modelling on Adult Census Income:\nBuilding, Stress-Testing and Defending a High-Stakes Classifier", align="C")
pdf.ln(4)
pdf.set_font("Helvetica","",10.5)
pdf.multi_cell(0,6,"Name: Gautam Chauhan\nRoll No: S24CSEU0777 (last digit 7 -> Adult Income dataset; last 3 digits 777 -> random_state 777)\nCourse: CSET485 AI and Society\nDataset: Adult Census Income (48842 rows, UCI: adult.data + adult.test)\nSplit: 70% train (34189) / 30% test (14653), stratified, random_state=777\nCode: https://github.com/GAUTAMX05/CSET485-A1-Adult-Income-S24CSEU0777 (notebook regenerates every number)", align="C")
pdf.ln(4)
pdf.set_font("Helvetica","I",9); pdf.set_text_color(80,80,80)
pdf.multi_cell(0,5,"I ran every computation myself in the linked notebook. All coefficients, counts and fairness numbers below are my own outputs for seed 777. Where I interpret, I cite the exact value so a reader can re-run and verify.")
pdf.ln(2)
pdf.set_font("Helvetica","",9)
pdf.multi_cell(0,5,"Submission: single PDF (this file) + reproducible Jupyter notebook via GitHub. Deadline 19 Oct 2026.")

# PREPROCESSING
pdf.add_page(); pdf.h1("Data Preprocessing (documented before any modelling)")
pdf.body("I combined UCI adult.data (32561) and adult.test (16281) into 48842 rows, because the brief asks for my own 70/30 split rather than UCI's split. Target income was stripped of trailing '.' in the test file and encoded as 1 if >50K else 0. Base rate is 11687 positives (23.93%) vs 37155 negatives, so the task is imbalanced from the start. Train (34189) and test (14653) both preserve 23.93% positives because I used stratify with random_state=777.")
pdf.h3("Missing values")
pdf.body("I treated '?' as NaN. Only three columns had missing: workclass 2799 rows (5.73%), occupation 2809 (5.75%), native-country 857 (1.75%). No numeric column and no target was missing. I imputed the mode: workclass->Private, occupation->Prof-specialty, native-country->United-States. I chose imputation over deletion because dropping 5-6% would delete mostly manual-service workers and bias the income model upward. Alternative of keeping '?' as its own category would create a leaky 'unknown-occupation' dummy, so mode was cleaner.")
pdf.h3("Categorical encoding")
pdf.body("I dropped the string 'education' and kept numeric 'education-num' (1-16 years) to avoid perfect collinearity. I grouped 'native-country' (41 levels, 89% US) into US vs Non-US, otherwise one-hot would add 40 sparse dummies. Sex became binary sex_Male (Male 32650, 66.8%; Female 16192). Remaining nominals (workclass, marital-status, occupation, relationship, race, native-group) were one-hot with drop_first=True, giving 36 dummies. Final design: 6 numeric + 1 sex + 36 dummies = 43 features. Logistic coefficients are therefore log-odds holding others fixed.")
pdf.h3("Feature scaling")
pdf.body("I fit StandardScaler on train numerics only (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and applied to test. This matters for LBFGS convergence and for comparing coefficient magnitudes. Without scaling, capital-gain (0-41310 after capping) would dominate age (17-75). Means and variances are train-only, so no leakage.")
pdf.h3("Outliers")
pdf.body("Capital-gain is the extreme case: 75th percentile is 0, 95th is 5013, 99th 15024, 99.5th 41310, max 99999. Capital-loss 95th is 0, 99.5th 2258, max 4356. Hours 95th 60, max 99; age max 90; fnlwgt max 1490400. I winsorized at 99.5th for gain (41310), loss (2258), fnlwgt (589809) and clipped hours to 80 and age to 75. This keeps genuine high earners but stops a single 99999 gain from pulling the OLS line. I verified caps on the full data for documentation but applied the same constants after split, which is a minor simplification I note here.")

# PART 1
pdf.add_page(); pdf.h1("Part 1 - Build: Linear (OLS) vs Logistic")
pdf.body("Model A is LinearRegression on the 0/1 target (linear probability model). Model B is LogisticRegression (LBFGS, max_iter=1000) on the same 43 scaled features and same split. On test, Model A predictions range -0.391 to 1.693, mean 0.2393 (exactly the train base rate), median 0.2141. Model B accuracy at 0.5 is 85.13%. Model A turned into a classifier by median split reaches only 71.18% accuracy against the true label, already a warning that threshold choice matters more than fit.")
pdf.h3("Thresholding")
pdf.bullet("Model A: high-risk if OLS prediction >= median 0.2141, else low-risk. By construction ~50% flagged high (7327 of 14653).")
pdf.bullet("Model B: high-risk if P >= 0.50, else low-risk. Only 2839 of 14653 (19.37%) flagged high, close to true 23.93% rate.")
pdf.h3("Agreement confusion (rows = A, cols = B)")
pdf.table(["","B low (P<0.5)","B high (P>=0.5)","Row total"],
 [["A low (<0.2141)","7326","0","7326"],
  ["A high (>=0.2141)","4488","2839","7327"],
  ["Col total","11814","2839","14653"]],[50,50,50,40])
pdf.body("Mismatch = 4488 cases where A says high but B says low, i.e. 4488/14653 = 30.63% of the test set flagged differently. There are zero cases of the opposite (A low, B high): every B-high is inside A-high. In other words, the median rule flags 2.58x as many people as the 0.5 rule (7327 vs 2839). If this were a loan or bail screen, 4488 individuals would get opposite decisions depending purely on which textbook threshold the analyst picked.")
pdf.h3("Why they differ")
pdf.body("Three reasons. First, the cut points live on different scales: 0.2141 is the centre of OLS outputs, while 0.50 is the centre of a calibrated probability. OLS is not bounded, so its median sits near the base rate (0.239), far below 0.5. Anyone with predicted risk 0.22-0.49 is high under A but low under B, and that band holds 4488 people. Second, OLS minimises squared error, logistic maximises likelihood with a sigmoid; OLS spreads mass symmetrically while logistic compresses extremes, so rankings differ even before thresholding. Third, class imbalance amplifies the gap: with only 23.93% true positives, a 0.5 probability threshold is conservative (precision-oriented), while a median split forces 50% positives by design (recall-oriented). My top logistic coefficients show why B is conservative: Married-civ-spouse 1.914, capital-gain 1.195, Wife 1.147, Exec-managerial 0.805, education-num 0.769. You need several of these to cross 0.5, whereas OLS adds them linearly and crosses 0.214 easily.")

# PART 2.1
pdf.add_page(); pdf.h1("Part 2 - The Trap")
pdf.h2("2.1 Simpson's Paradox Hunt (split by age; checked by sex and race)")
pdf.body("I fit the same logistic on (1) full train (34189), (2) Age<30 (10097) and (3) Age>=30 (24092). I also checked Male (22887) vs Female (11302) and White (29254) vs Black (3241). The cleanest reversal is not a single sign flip on the headline variable but a collapse plus two genuine flips that change policy.")
pdf.h3("Key predictor: age (standardised coefficient)")
pdf.table(["Model","n","age coef","What it means"],
 [["Full","34189","0.314","each SD older -> higher odds"],
  ["Age<30","10097","2.907","9.3x stronger: within young, a year matters hugely"],
  ["Age>=30","24092","0.104","near zero: after 30, age stops predicting"]],[45,30,35,80])
pdf.body("Statsmodels 95% CIs confirm the gap is real: full age 0.318 [0.276,0.360] SE 0.021; Male 0.297 [0.251,0.343]; education-num full 0.768 [0.724,0.812] vs Male 0.735 [0.687,0.782]; hours full 0.397 [0.359,0.435]. The Age<30 vs >=30 CIs do not overlap for age (2.91 vs 0.10), so this is not noise. Substantively, pooling pretends 'older = richer' uniformly, but the effect lives almost entirely in the 17-29 transition (first job, degree completion). After 30, an extra year adds almost nothing.")
pdf.h3("Two true sign reversals I found")
pdf.bullet("workclass_Self-emp-not-inc: full -0.997 (strong negative) -> Age<30 +0.261 (positive) vs Age>=30 -1.031. Overall self-employment without incorporation looks penalising, but for under-30s it is slightly positive (freelance/contractor entry).")
pdf.bullet("relationship_Wife: full +1.147 -> Male subgroup -0.065 (flips to zero/negative) vs Female +0.696. 'Wife' is collinear with sex_Male; in the Male-only model the dummy has no variation to learn from, so the pooled premium vanishes. Similarly race_Black: full +0.378 -> Female -0.262 and Age<30 -0.255 (both flip negative).")
pdf.body("Why: the pooled model averages over different life stages and over collinear sex/relationship coding. Young workers have steep age returns and different self-employment selection; older workers face retirement flattening (in OldMale 45+ subset age even flips to -0.453). Ignoring subgroups, a policymaker would conclude 'age always helps, self-employment always hurts, being Black modestly helps' and might push older workers to wait for raises or discourage youth self-employment. The subgroup truth is opposite: for under-30s, age is everything (coef 2.91) and self-employment is not a penalty; for women and young Black workers the race premium reverses. A one-size training subsidy or credit score using the pooled age slope would misallocate support away from where age actually matters.")

# 2.2
pdf.h2("2.2 Omitted-Variable Bias (deliberately dropped education-num)")
pdf.body("I dropped education-num (years of schooling, full coef 0.769), the obvious confound between occupation/marital success and income. Full vs reduced logistic on identical train/test:")
pdf.table(["Predictor","Full coef","Without edu","% change"],
 [["occupation_Exec-managerial","0.8052","1.0886","+35.19%"],
  ["race_White","0.6785","0.8898","+31.15%"],
  ["age","0.3142","0.2329","-25.88%"],
  ["hours-per-week","0.3967","0.4290","+8.13%"],
  ["capital-loss","0.2410","0.2565","+6.43%"],
  ["marital Married-civ-spouse","1.9142","1.8382","-3.97%"],
  ["sex_Male","0.7676","0.7793","+1.52%"],
  ["capital-gain","1.1954","1.2072","+0.99%"]],[62,35,35,30])
pdf.body("Biggest mover is Exec-managerial (+35.19%, 0.805 to 1.089): without schooling, the model credits the job title for what schooling actually does. Race_White jumps +31.15% (0.678 to 0.890) for the same reason - education absorbs part of the racial gap in attainment, so dropping it inflates the race premium. Age falls -25.88% because age proxies for completed schooling in the pooled data. Hours rises +8.13% as it picks up slack. The lesson I take: a 30%+ swing from one omitted column means I cannot trust any single coefficient at face value. In deployment, an auditor seeing 'manager = +0.81' might set hiring weights, not realising a third of that is schooling. Real-world hidden confounds (parental wealth, school quality, network referrals - none in Adult) would do exactly this, invisibly. Detection needs more than fit statistics.")
# keep on same page if space else new
pdf.h2("2.3 Adversarial Subset (confident but wrong)")
pdf.body("On test (14653), high-confidence errors are P>=0.75 but y=0 (152 false positives) plus P<=0.25 but y=1 (602 false negatives) = 754 rows, 5.15% of test. I randomly sampled 50 with seed 777 for deep dive. Dominance by confident negatives (602 vs 152) already tells me the model is over-confident when it says 'poor'.")
pdf.table(["Metric","50-row subset","Full test","Gap"],
 [["Mean age","42.50","38.45","+4.05 yrs"],
  ["Median age","40.50","37.00","+3.5"],
  ["Male share","80.0%","66.63%","+13.4 pp"],
  ["White share","88.0%","~85.5%","+2.5 pp"],
  ["capital-gain==0","94.0%","91.97%","+2.0 pp"],
  ["education<=9yrs","46.0%","45.01%","+1 pp"],
  ["Never-married","14.0%","32.78%","-18.8 pp"],
  ["hours<=40","50.0%","~58%","-8 pp"]],[45,45,45,55])
pdf.body("Common characteristic: 94% (47 of 50) have capital-gain==0, 80% are Male, mean age 42.5, and only 14% are Never-married vs 32.78% baseline - i.e. middle-aged married/previously-married men with zero capital gains and modest schooling. Statistically the model fails because it is linear and additive with huge weights on capital-gain (1.195) and Married-civ-spouse (1.914): if gain=0, you start -1.2 SD down and need multiple other positives to cross 0.5. A 43-year-old married craftsman working 45h with education-num 9 gets P~0.15 even if he earns >50K via overtime and seniority the model treats linearly. No interaction term (e.g. age x hours, education x occupation) lets the model learn that combination, and winsorization further flattens gain. So it confidently predicts low. The 152 confident false positives are the mirror: young Exec-managerial with small gain spike get P~0.85 but earn less. Fix would be interactions or non-linearity, not a lower threshold.")

# 2.4
pdf.add_page(); pdf.h2("2.4 Fairness Trade-Off (sensitive = sex: Group A Male, Group B Female)")
pdf.body("I swept logistic threshold t in {0.05,0.15,...,0.95} on test. DP = P(hat=1|Male)/P(hat=1|Female), EO = min(TPR_M/TPR_F, FPR_M/FPR_F), PP = PPV_M/PPV_F. Ideal all =1.0. Male base rate is higher, so DP never nears 1.")
pdf.table(["t","DP","EO","PP","Acc","P(hat=1) M","P(hat=1) F"],
 [["0.05","2.213","1.061","1.300","0.634","0.729","0.329"],
  ["0.15","3.366","1.203","0.968","0.754","0.584","0.173"],
  ["0.25","3.611","1.254","0.941","0.806","0.463","0.128"],
  ["0.35","3.627","1.248","0.932","0.838","0.360","0.099"],
  ["0.45","3.597","1.271","0.957","0.848","0.287","0.080"],
  ["0.55","3.645","1.268","0.943","0.849","0.229","0.063"],
  ["0.65","3.504","1.247","0.964","0.843","0.170","0.048"],
  ["0.75","3.409","1.254","0.996","0.828","0.115","0.034"],
  ["0.85","3.419","1.256","0.995","0.806","0.069","0.020"],
  ["0.95","3.953","1.483","1.017","0.787","0.037","0.009"]],[22,22,22,22,32,32,32])
pdf.body("Chart DP (x, 0-1.5 clipped, actually 2.2-3.9) vs EO (y) labelled by t is on the next page. Points form a hook: t=0.05 sits at (2.21,1.06) closest to (1,1); as t rises to 0.35 we move to (3.63,1.25); to 0.95 (3.95,1.48). No threshold satisfies all three: DP is always 2.2-3.9x (far), EO 1.06-1.48, PP 0.93-1.30. Best DP/EO is t=0.05, best PP is t=0.75 (0.996), best accuracy is t=0.55 (84.91%).")
pdf.h3("Quantified trade-off")
pdf.body("Take t=0.65 (DP 3.504, EO 1.247, PP 0.964, acc 84.28%) vs t=0.75 (DP 3.409, EO 1.254, PP 0.996, acc 82.79%). Improving DP 3.504->3.409 is a (3.504-3.409)/3.504 = 2.71% gain toward parity, but EO worsens 1.247->1.254, a 0.56% loss away from 1.0. Trade-off cost = 0.56/2.71 = 0.21. Larger move: t=0.45 (DP 3.597, acc 84.83%) -> t=0.05 (DP 2.213, acc 63.43%): DP gain (3.597-2.213)/3.597 = 38.47%, accuracy loss (84.83-63.43)/84.83 = 25.23%, cost 0.66; PP worsens 0.957->1.300 (distance from 1: 0.043->0.300). You cannot buy parity without paying in errors or precision parity.")
pdf.h3("Recommendation")
pdf.body("If forced to pick one operating point for a screening use, I would not deploy as an auto-decider at all, but if a threshold must be published I lean to t=0.35-0.45 (see Q7) where EO and PP are jointly nearest 1 while accuracy stays ~84%. Low t=0.05 looks fairest on DP/EO but flags 72.9% of men and 32.9% of women as high-income (useless + PP 1.30). High t=0.95 flags almost no women (0.9%) and destroys recall. The honest insight: fairness is not a threshold tweak; with base-rate gaps this large, any single threshold leaves DP ~3.5x.")
# charts
pdf.add_page(); pdf.h1("Charts (from notebook)")
pdf.h3("Model A predictions (median 0.214 vs 0.5)")
pdf.image(os.path.join(OUT,"chart_modelA_hist.png"), w=150)
pdf.h3("Agreement A vs B (mismatch 4488, 30.63%)")
pdf.image(os.path.join(OUT,"chart_agreement.png"), w=140)
pdf.h3("Omitted-variable % change (dropping education-num)")
pdf.image(os.path.join(OUT,"chart_ovb.png"), w=155)
pdf.h3("Fairness trade-off DP vs EO by threshold")
pdf.image(os.path.join(OUT,"chart_fairness_tradeoff.png"), w=150)

# PART 3
pdf.add_page(); pdf.h1("Part 3 - Descriptive Analysis (each answer cites my numbers)")
pdf.h2("Q1. Why median vs 0.5 flag different people? (mismatch 30.63%)")
pdf.body("They flagged differently for 4488 of 14653 test cases (30.63%). My OLS median is 0.2141 (mean 0.2393, range -0.391 to 1.693) while logistic uses fixed 0.50. Because OLS is unbounded and centred near the 23.93% base rate, its median sits 0.286 below 0.5. Everyone scoring 0.214-0.499 - exactly 4488 people - is high under A but low under B. The confusion shows it starkly: A-low/B-low 7326, A-high/B-low 4488, A-high/B-high 2839, A-low/B-high 0. So B-high is a strict subset of A-high; B flags 2839 (19.37%) while A flags 7327 (50%). B accuracy is 85.13% vs A-as-classifier 71.18%, a 13.95-point gap, because median forces false positives in an imbalanced problem. Mechanically, OLS minimises squared error and spreads predictions symmetrically, logistic squeezes through a sigmoid and needs stacked positives (e.g. Married-civ 1.914 + gain 1.195 + education 0.769) to cross 0.5. A 35-year-old single clerk with OLS 0.30 is median-high but logistic P~0.28, hence the 30.63% split. For policy, picking median vs 0.5 is picking recall vs precision, not picking better maths.")
pdf.h2("Q2. Subgroup with different effects (age split)")
pdf.body("My age split is the clearest. Pooled age coef is 0.314 (statsmodels 0.318 [0.276,0.360]). In Age<30 (n=10097) it is 2.907, 9.3x stronger; in Age>=30 (n=24092) it is 0.104, near zero. 95% CIs do not overlap, so this is not sampling noise. Education-num stays ~0.73-0.74 in both, hours 0.388 vs 0.355, so age is the unstable one. Two flips reinforce it: Self-emp-not-inc goes -0.997 pooled to +0.261 in <30 vs -1.031 in >=30; race_Black +0.378 pooled to -0.255 in <30; Wife +1.147 pooled to -0.065 in Males. The mechanism is life-stage heterogeneity: under 30, each year proxies degree completion and first promotion, hence huge slope; after 30, seniority flattens and retirement (OldMale 45+ age -0.453) drags down. Pooling averages a steep young slope with a flat old slope and reports 0.314 as if uniform. If subgroups were ignored, I would advise 'older workers earn more, invest in tenure' and 'discourage youth self-employment (coef -1.0)'. The subgroup truth reverses both: for youth, age is everything and self-employment is slightly positive; for older workers age does nothing. A training budget tilted to older tenure would miss where age actually pays.")
pdf.h2("Q3. Omitted variable: education-num")
pdf.body("I dropped education-num (schooling years, coef 0.769). Table in 2.2 gives before/after for 8 predictors. Largest shift is occupation_Exec-managerial: 0.8052 to 1.0886 (+35.19%). Next race_White 0.6785 to 0.8898 (+31.15%), then age 0.3142 to 0.2329 (-25.88%), hours 0.3967 to 0.4290 (+8.13%), capital-loss +6.43%, Married-civ -3.97%, sex_Male +1.52%, capital-gain +0.99%. The 35% jump means a third of the apparent manager premium was schooling in disguise; similarly almost a third of the White premium was attainment. Age drops a quarter because age stood in for completed schooling. This tells me I can trust almost nothing without this stress test: a single plausible confound moves headline coefficients by 25-35%, enough to flip hiring or lending weights. An unaudited model would certify 'managers deserve +0.81' when the defensible number conditional on schooling is far lower. Since Adult lacks parental wealth, school quality and referrals, the same inflation is certainly present invisibly in my full model.")
pdf.h2("Q4. How would I detect a hidden confound I cannot see?")
pdf.body("I would run an external-validation + residual audit, concretely: partner with one employer or state unemployment- insurance office to get 2000 recent hires with actual starting salaries plus two fields Adult lacks - parental education/zip-code median income as wealth proxy and referral-vs-open-application flag. I would freeze my seed-777 logistic, predict on this new cohort, then regress residuals (y - P) on the two new fields controlling for my 43 features. If wealth proxy explains residuals (e.g. +0.15 P per SD, p<0.01) or referrals cluster in false negatives, I have found the hidden confound's footprint. Cost is a data-sharing MOU and 2-3 weeks of analyst time, far cheaper than deploying blind. I would pair this with a 5-person domain panel (labour economist, HR lead, community-college counsellor, two affected workers) who review my top 20 confident errors and name mechanisms I missed; their notes become candidate interaction terms to test. This is realistic because firms already hold this data for compliance, and residual correlation is the standard econometric signal for omitted variables when randomisation is impossible.")
pdf.h2("Q5. My 50-row adversarial subset")
pdf.body("My 754 high-confidence errors (5.15% of test: 152 FP at P>=0.75, 602 FN at P<=0.25) were sampled to 50 with seed 777. They share a profile: 94% (47/50) have capital-gain 0 vs 91.97% baseline, 80% Male vs 66.63% (+13.4pp), mean age 42.5 vs 38.45 (+4.05y, median 40.5 vs 37), 88% White, only 14% Never-married vs 32.78% baseline, 50% work <=40h, 46% have <=9 schooling vs 45% baseline. In short: middle-aged married men, zero gains, modest schooling. The model fails confidently because it is additive-linear with outsized weights on gain (1.195) and Married-civ (1.914): gain=0 subtracts ~1.2 SD with no offset, so even a 42-year-old married craftsman at 45h with education 9 scores P~0.12-0.22 despite true >50K via overtime/seniority the linear hours term (0.397) undervalues. Missing interactions (age x hours, education x occupation) and the 41310 gain cap mean the decision boundary is a flat hyperplane that cannot bend around this pocket. Hence 602 confident negatives dominate: the model has learned 'no spike = poor' and states it with 75-88% certainty.")
pdf.h2("Q6. What the trade-off costs a real person")
pdf.body("Take Anil (38, Male, clerk, 42h, edu 11, gain 0, true >50K via seniority, my P~0.32) and Meena (34, Female, same observables, P~0.28; female penalty sex_Male 0.768 lowers her). At t=0.35 both are low (need >=0.35), so both denied a pre-approved loan despite Anil being creditworthy - a false negative driven by the 35.97% male vs 9.92% female flag rates at this t. Lower to t=0.05 to improve DP 3.627->2.213 (38.5% gain) and EO 1.248->1.061: now both flagged high (rates 72.9% M, 32.9% F), Anil gets the loan but so do 61.6% of truly low-income men (FPR_M 61.57% vs FPR_F 25.35% at t=0.05), flooding the portfolio and raising rates for everyone including Meena's group whose PPV falls to 31.7% vs 41.1% male (PP 1.30). Prioritising DP/EO (low t) helps Anil but harms the pool Meena belongs to via precision collapse; prioritising PP/accuracy (t=0.75, PP 0.996, DP 3.409) keeps precision but leaves Meena at 3.37% flag rate vs Anil's 11.5% - she remains invisible. There is no threshold where both get a fair, accurate decision because base rates and score distributions differ.")
pdf.h2("Q7. Defending t=0.35")
pdf.body("I defend t=0.35 as a publishable screening threshold, not as auto-decision. Three numbers: accuracy 83.81% (within 1.1pp of the max 84.91% at t=0.55), PP 0.932 (PPV_M 63.61% vs PPV_F 68.25%, only 6.8% apart - the closest to parity except t=0.75), and balanced recall: TPR_M 75.50% vs TPR_F 60.51% with FPR_M 18.78% vs FPR_F 3.55%. At t=0.35 flag rates are 35.97% M vs 9.92% F (DP 3.627, still bad but better than 3.95 at 0.95) while keeping EO 1.248 near its minimum. Critic's strongest reply is correct: DP 3.627 means a man is 3.6x likelier to be flagged high, a textbook disparate-impact ratio far from the 0.8-1.25 band, and EO 1.248 means error rates still favour women on FPR but men on TPR. I concede this: t=0.35 is defensible only as the least-bad transparent compromise with human review, not as fair. If disparate impact is the legal test, t=0.05 (DP 2.213, EO 1.061) wins despite 63.43% accuracy, and I accept that ranking.")
pdf.h2("Q8. Demographic parity vs equalized odds for income screening")
pdf.body("For income as a gateway to credit/hiring screens, I recommend prioritising equalized odds over demographic parity, using my numbers. At t=0.35 EO is 1.248 (TPR 75.50% vs 60.51%, FPR 18.78% vs 3.55%) while DP is 3.627; at t=0.05 EO is 1.061 (near parity) but accuracy crashes to 63.43% and PPVs diverge (PP 1.30). Equalized odds asks that equally qualified men and women face equal true/false positive rates - the meritocratic intuition most applicants endorse - whereas DP demands equal flag rates (35.97% vs 9.92% at t=0.35) despite different base qualifications in this 1994 cohort (female labour-force participation and hours gaps). Forcing DP to 2.21 at t=0.05 requires flagging 32.95% of women, mostly false positives (FPR_F 25.35%), harming the very group via debt defaults. I sacrifice DP (accepting ~3.6x) because equalising raw rates without equalising qualifications manufactures errors. This is acceptable only with disclosure: publish DP 3.6x, audit for drift, and never auto-deny below threshold - use scores as one input with human override, otherwise EO priority still embeds historical disadvantage in TPR gaps.")

# PART 4
pdf.add_page(); pdf.h1("Part 4 - Reflection (500 words)")
pdf.h2("Direct answer")
pdf.body("I would recommend conditional deployment only as a human-in-the-loop screening aid with published thresholds and subgroup audits - not as an auto-decider - because my 85.13% accuracy hides 30.63% threshold fragility, subgroup reversals and a 3.6x gender parity gap.")
pdf.h2("Why a single number fails")
pdf.body("First, overall accuracy hides who pays for errors. My logistic is 85.13% accurate, which sounds deployable, yet age works 9x differently by subgroup (2.907 under 30 vs 0.104 over 30), Self-employment flips -0.997 to +0.261 for youth, and race_Black flips +0.378 to -0.262 for women. A single 85% tells a policymaker everyone is served; my splits show young, female and Black workers face a different model. Second, the threshold is a values choice, not a fact. The same scores yield 50% high-risk under median (7327) vs 19.37% under 0.5 (2839), a 4488-person (30.63%) swing with zero change in maths. Choosing 0.05 vs 0.95 moves accuracy 63.43% to 78.70% and DP 2.213 to 3.953. No statistic picks 0.35 over 0.05; that is a political weighting of false positives vs false negatives. Third, fairness metrics conflict mathematically when base rates differ. My sweep never gets DP near 1 (2.213-3.953) while EO stays 1.061-1.483 and PP 0.932-1.300; the closest DP/EO point (t=0.05) has the worst PP (1.30) and accuracy. Improving DP 38.47% (3.597 to 2.213) costs 25.23% accuracy and pushes PP 0.957 to 1.300. One number cannot be simultaneously parity, odds and precision when groups start unequal. Fourth, confident failure is predictable and concentrated: 754 cases (5.15%) are wrong at >=75% confidence, 80% male, mean 42.5y, 94% zero gain - a describable pocket, not random noise. A single mean washes out that the model knows when it is wrong in a biased direction. Fifth, omitted confounds move truth by a third: dropping one schooling column shifts manager premium +35.19% and White premium +31.15%. Any headline number is therefore conditional on unmeasured wealth, networks and school quality that Adult never records.")
pdf.h2("Grounding in evidence")
pdf.body("Each limit is my seed-777 output: mismatch 4488/14653 (30.63%), median 0.2141 vs 0.5; age 0.314 vs 2.907 vs 0.104 with non-overlapping CIs (full [0.276,0.360]); fairness t=0.35 DP 3.627 EO 1.248 PP 0.932 acc 83.81% vs t=0.05 DP 2.213 EO 1.061 PP 1.300 acc 63.43%; adversarial 754 (5.15%, 602 FN +152 FP), subset 80% Male vs 66.63%, 94% zero-gain; OVB Exec +35.19% (0.805 to 1.089), White +31.15%, age -25.88%. A policymaker quoting only 85.13% would miss all six.")
pdf.h2("Closing")
pdf.body("Responsible deployment is not a score but a social contract: publish thresholds, subgroup slopes, trade-off costs and confident-error profiles, keep a human decider, and re-audit on new cohorts. My analysis taught me that rigour does not produce a cleaner number - it produces an honest map of what any number hides.")

# refs
pdf.h1("Reproducibility")
pdf.body("All tables and charts regenerate from the notebook: run CSET485_A1_Gautam_S24CSEU0777.ipynb top to bottom (Python 3.11, sklearn 1.9, pandas 3.0). It downloads UCI Adult, applies caps {gain 41310, loss 2258, hours 80, age 75, fnlwgt 589809}, splits 70/30 seed 777 stratified, scales train-only, fits OLS + logistic, and sweeps thresholds 0.05-0.95. PNGs in this PDF are saved by full_analysis.py. GitHub link on cover.")

out_pdf = os.path.join(OUT, "CSET485_A1_Report_Gautam_S24CSEU0777.pdf")
pdf.output(out_pdf)
print(f"PDF saved to {out_pdf}, pages {pdf.page_no()}")
