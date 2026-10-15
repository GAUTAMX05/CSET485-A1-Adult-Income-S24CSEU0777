# Report prefinish audit

- PDF: `CSET485_A1_Report_Gautam_S24CSEU0777.pdf`
- PDF pages: 11
- Forbidden old-value scan: PASS

## Number -> results.json key mapping

| report item | results.json key |
|---|---|
| dataset_rows | `n_raw` |
| train_rows | `train_n` |
| test_rows | `test_n` |
| positive_rate | `pos_rate` |
| test_positive_rate | `test_pos` |
| modelA_median | `modelA.median` |
| modelB_accuracy | `modelB.acc` |
| modelA_accuracy | `modelA.acc` |
| mismatch_count | `mismatch` |
| mismatch_percent | `mismatch_pct` |
| age_full | `sm_age.Full.coef` |
| age_young | `sm_age.Age<30.coef` |
| age_old | `sm_age.Age>=30.coef` |
| selfemp_full | `sm_selfemp.Full.coef` |
| selfemp_young | `sm_selfemp.Age<30.coef` |
| selfemp_old | `sm_selfemp.Age>=30.coef` |
| male_base_rate_test | `sex_base_rates.test.Male.positive_rate` |
| female_base_rate_test | `sex_base_rates.test.Female.positive_rate` |
| male_relationship_wife_count | `male_relationship_wife_count.test` |
| q6_female_probability | `q6_pair.female_P` |
| q6_male_probability | `q6_pair.male_P` |
| q6_sex_Male_coef | `q6_pair.sex_Male_coef` |
| q6_relationship_Wife_coef | `q6_pair.relationship_Wife_coef` |
| chosen_threshold | `report_config.chosen_threshold` |
| tradeoff_accuracy_loss_pp | `fairness_tradeoff_075_085.accuracy_loss_pp` |

## Word counts

| section | words |
|---|---:|
| Q1 | 290 |
| Q2 | 265 |
| Q3 | 264 |
| Q4 | 273 |
| Q5 | 271 |
| Q6 | 300 |
| Q7 | 278 |
| Q8 | 299 |
| Reflection - Direct answer | 37 |
| Reflection - Why a single number fails | 340 |
| Reflection - Grounding in evidence | 79 |
| Reflection - Closing | 55 |
| Reflection - Reproducibility | 49 |

## Threshold

- chosen threshold: `0.35`
- Q7 explicitly defends the chosen threshold and includes a counterargument.
- Q6 uses the stored real matched pair and explains the relationship_Wife offset against sex_Male; it does not claim a sex penalty.
- Q8 recommends equalized odds as the primary fairness audit and reports demographic parity plus the sex base-rate ratio.