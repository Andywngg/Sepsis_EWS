# Model Card: Sepsis Early-Warning System

## Intended Use
- Research prototype for early sepsis warning using ICU vitals/labs.
- Not approved for clinical use.

## Data
- PhysioNet/CinC Challenge 2019 ICU time-series.

## Model
- Tree-based gradient boosting classifier with engineered time-series features.
- Early-warning policy optimized by official utility.

## Performance (Held-out)
- AUROC: 0.847
- AUPRC: 0.128
- Patient-level AUROC: 0.860
- Patient-level AUPRC: 0.444
- Official utility: 0.132
- Early detection rate: 0.260
- False alert rate: 0.022
- Median lead time (hours): 40.50

## Calibration
- Brier score: 0.017
- Brier score (raw): 0.017
- Calibration method: sigmoid (200 patients)
- Best sweep policy: sigmoid (utility 0.966, brier 0.011, max_patients 200)

## Error Analysis
- Error analysis subset size: 500
- Patient-level TP/FP/FN/TN: 7/11/1/92
- Sensitivity: 0.875
- Specificity: 0.893
- Alert burden (alerts per patient-day): 2.466

## Statistical Confidence (bootstrap)
- Bootstrap patients: 40,336 (full), n_bootstrap: 10
- AUROC 95% CI: [0.820, 0.888]
- AUPRC 95% CI: [0.096, 0.166]
- Official utility 95% CI: [0.081, 0.184]

## Alert Policies
- Sensitive: threshold 0.05, early detection 0.927, alerts/patient-day 1.981
- Conservative: threshold 0.50, early detection 0.512, alerts/patient-day 0.388

## Uncertainty-Aware Triage (conformal)
- Conformal selective prediction on 20,000 patients (test split).
- Alpha=0.02: coverage 0.366, utility 0.232, early detection 0.397, false alert 0.053, alerts/day 0.962.
- Alpha=0.10: coverage 0.591, utility 0.391, early detection 0.644, false alert 0.259.

## Dynamic Threshold Ablation
- Trend-adjusted thresholding tested; best k=0.0 (no improvement over static).

## Cross-site Generalization
- Cross-site results use patient-level normalization to mitigate domain shift.
- Train A -> Test B official utility: 0.592, AUROC: 0.967
- Train B -> Test A official utility: 0.627, AUROC: 0.965

## Prospective (Real-Time) Simulation
- 500 held-out patients (test split).
- AUROC: 0.877 | AUPRC: 0.144 | Official utility: 0.309
- Early detection: 0.486 | False alert: 0.062 | Median lead time: 25.0 hours

## Subgroup Analysis (500 patients, test split)
- Age 40-59 AUROC 0.907, utility 0.341 (n=162)
- Age 60-79 AUROC 0.885, utility 0.277 (n=222)
- Age >=80 AUROC 0.805, utility 0.214 (n=79)
- Age <40 AUROC 0.853, utility 0.432 (n=37)

## Missingness Stress Test (1,000 patients, test split)
- Additional missingness 10%: AUROC 0.844, utility 0.276
- Additional missingness 20%: AUROC 0.847, utility 0.282
- Additional missingness 30%: AUROC 0.841, utility 0.295

## Measurement Delay Stress Test (1,000 test patients)
- Delay 1 hour: AUROC 0.847, utility 0.272
- Delay 2 hours: AUROC 0.850, utility 0.297
- Delay 3 hours: AUROC 0.849, utility 0.310

## Case Studies
- Early detection: p000009 (lead time 190h)
- Late detection: p000765 (alert 7h after onset)
- False alarm: p000978 (non-sepsis, 30 alerts)

## Limitations
- Retrospective evaluation on ICU time-series; no prospective validation.
- Domain shift across hospitals remains a challenge.
- Missing data patterns can bias alerts.

## Ethical Considerations
- Intended for clinical decision support only; should not replace clinician judgment.
- Requires monitoring for bias across sites and patient populations.


