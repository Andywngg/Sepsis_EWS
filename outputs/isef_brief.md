# ISEF Brief: Sepsis Early-Warning System

## Research question
Can a utility-optimized early warning system predict sepsis hours before onset using ICU vitals and labs?

## Methods
- Time-series features from ICU vitals/labs
- Risk model with early-warning policy
- Utility-based threshold selection

## Results (held-out patients)
- AUROC: 0.847
- AUPRC: 0.128
- Patient-level AUROC: 0.860
- Patient-level AUPRC: 0.444
- Brier score: 0.017
- Brier score (raw): 0.017
- Calibration: sigmoid (200 patients)
- Utility score: 0.132
- Official utility: 0.132
- Accuracy: 0.971
- F-measure: 0.165
- Early detection rate: 0.260
- False alert rate: 0.022
- Median lead time (hours): 40.50

- Alerts per patient-day: 0.392
- Alerts per non-sepsis patient-day: 0.081
- Mean alerts per patient: 0.630

## Baseline comparison
- Baseline AUROC: 0.768 vs improved 0.847
- Baseline AUPRC: 0.067 vs improved 0.128
- Baseline utility: 0.048 vs improved 0.132
- Baseline official utility: 0.048 vs improved 0.132

## Statistical Confidence (bootstrap)
- Bootstrap patients: 40,336 (full), n_bootstrap: 10
- AUROC 95% CI: [0.820, 0.888]
- AUPRC 95% CI: [0.096, 0.166]
- Official utility 95% CI: [0.081, 0.184]

## Alert Policies (clinical tradeoff)
- Sensitive policy: threshold 0.05, early detection 0.927, alerts/patient-day 1.981
- Conservative policy: threshold 0.50, early detection 0.512, alerts/patient-day 0.388

## Uncertainty-Aware Triage (conformal)
- Conformal selective prediction on 20,000 patients (test split).
- Example operating point (alpha=0.02): coverage 0.366, utility 0.232, early detection 0.397, false alert 0.053, alerts/day 0.962.
- Higher coverage increases early detection but raises alert burden (alpha=0.1 utility 0.391, false alert 0.259).

## Dynamic Threshold Ablation
- Trend-adjusted dynamic thresholding was evaluated and did not improve utility.
- Best k=0.0, indicating a static threshold is optimal for this model/data.

## Novelty
Unlike standard sepsis classifiers that optimize AUROC, this system optimizes a clinical utility score and learns an early-warning policy that balances early detection against false alarms. It also evaluates uncertainty-aware triage (conformal prediction) and dynamic policy ablations to quantify tradeoffs and robustness.

## Cross-site Generalization (true A/B, full, patient-normalized)
- Train A -> Test B official utility: 0.592, AUROC: 0.967
- Train B -> Test A official utility: 0.627, AUROC: 0.965

## Prospective (Real-Time) Simulation
- Simulated streaming evaluation on 500 held-out patients (test split).
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
- Alert burden remains ~1.23-1.27 alerts per patient-day at threshold 0.10

## Measurement Delay Stress Test (1,000 test patients)
- Delay 1 hour: AUROC 0.847, utility 0.272
- Delay 2 hours: AUROC 0.850, utility 0.297
- Delay 3 hours: AUROC 0.849, utility 0.310
- Alert burden remains ~1.23-1.26 alerts per patient-day at threshold 0.10

## Deployment Framing
- Thresholds are selected to optimize utility under a target alert budget.
- Conformal triage enables coverage control for staffing limits.
- A deployment policy and monitoring plan are documented in outputs/deployment_policy.md.

## External Validation Plan
- A step-by-step plan for MIMIC-IV/eICU external validation is documented in external/external_validation_plan.md.

## Case Studies
- Early detection: p000009 (lead time 190h)
- Late detection: p000765 (alert 7h after onset)
- False alarm: p000978 (non-sepsis, 30 alerts)



