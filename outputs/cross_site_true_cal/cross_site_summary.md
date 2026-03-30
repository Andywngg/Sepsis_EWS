# Cross-Site Generalization

- Model: hgb
- Feature set: enhanced
- Utility: official
- Alert k: 1
- Max patients per site: 200
- Target calibration: sigmoid
- Calibration fraction: 0.1
- Calibration max patients: 50

## Train A -> Test B
{
  "metrics": {
    "auroc": 0.7301998383660275,
    "auprc": 0.049549202208334135
  },
  "patient_level_metrics": {
    "auroc": 0.7005988023952096,
    "auprc": 0.32334959238460736
  },
  "best_threshold": 0.1,
  "utility_score": 0.04287343215507411,
  "utility_kind": "official",
  "alert_k": 1,
  "official_utility": 0.04287343215507411,
  "custom_utility": -0.1072222222222222,
  "accuracy": 0.971114555445963,
  "f_measure": 0.07239819004524888,
  "early_warning": {
    "early_detection_rate": 0.15384615384615385,
    "false_alert_rate": 0.005988023952095809,
    "median_lead_time_hours": 33.5
  },
  "test_patients": 180,
  "calibration": {
    "method": "sigmoid",
    "patients": 20,
    "eval_patients": 180
  }
}

## Train B -> Test A
{
  "metrics": {
    "auroc": 0.5,
    "auprc": 0.028181289947704823
  },
  "patient_level_metrics": {
    "auroc": 0.5,
    "auprc": 0.1111111111111111
  },
  "best_threshold": 0.1,
  "utility_score": 0.0,
  "utility_kind": "official",
  "alert_k": 1,
  "official_utility": 0.0,
  "custom_utility": -0.2222222222222222,
  "accuracy": 0.9718187100522951,
  "f_measure": 0.0,
  "early_warning": {
    "early_detection_rate": 0.0,
    "false_alert_rate": 0.0,
    "median_lead_time_hours": 0.0
  },
  "test_patients": 180,
  "calibration": {
    "method": "sigmoid",
    "patients": 20,
    "eval_patients": 180
  }
}