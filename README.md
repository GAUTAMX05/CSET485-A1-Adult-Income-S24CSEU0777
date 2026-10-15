# CSET485 AI and Society — Assignment 1
### Adult Census Income: Building & Stress-Testing a High-Stakes Classifier
**Name:** Gautam Chauhan | **Roll No:** S24CSEU0777 | **Seed:** 777 | **Dataset:** Adult Income (last digit 7)

Every number in the PDF report regenerates from the notebook.

## Files
- `CSET485_A1_Gautam_S24CSEU0777.ipynb` — fully reproducible notebook (run top to bottom)
- `CSET485_A1_Report_Gautam_S24CSEU0777.pdf` — final report (preprocessing + Parts 1-4)
- `full_analysis.py` — standalone script that reproduces `results.json` + charts
- `results.json` — all computed numbers (seed 777)
- `chart_*.png` — figures embedded in PDF

## Reproduce
```bash
pip install -r requirements.txt
jupyter notebook CSET485_A1_Gautam_S24CSEU0777.ipynb
# or
python full_analysis.py
```

## Key results (seed 777, 70/30 split, test n=14653)
- Model A OLS median 0.2141 vs Model B 0.5 → mismatch 4488 (30.63%), B acc 85.13%, A-as-classifier 71.18%
- Simpson: age coef full 0.314 → <30 2.907 vs >=30 0.104; Self-emp-not-inc -0.997 → +0.261 (<30)
- OVB (drop education-num): Exec-managerial +35.19%, race_White +31.15%, age -25.88%
- Adversarial: 754 high-conf wrong (5.15%, 152 FP + 602 FN); 50-sample: 80% Male, mean 42.5y, 94% gain==0
- Fairness (Male/Female): t=0.35 → DP 3.627, EO 1.248, PP 0.932, acc 83.81%; no threshold satisfies all three

## Dataset
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/ (adult.data + adult.test = 48842 rows)

## Academic note
Work computed for S24CSEU0777 with personal seed 777. Understand the notebook before viva — oral assessment tests your explanation of these numbers.
