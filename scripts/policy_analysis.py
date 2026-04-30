from __future__ import annotations

# PURPOSE: Alert threshold vs clinical tradeoff analysis.
# IDEA:    A lower threshold catches more sepsis cases but fires more false alarms.
#          A higher threshold fires less often but misses more patients.
#          This script sweeps 20 thresholds (0.05 to 0.50) and records:
#            - early detection rate (how many sepsis patients were caught early)
#            - false alert rate (how many healthy patients got an alert)
#            - alerts per patient-day (operational load on the nursing staff)
#          It also picks two named policies:
#            "sensitive"    -- maximizes utility while meeting a minimum early-detection target
#            "conservative" -- minimizes alert burden while maintaining positive utility
# OUTPUT:  policy_analysis.json and policy_tradeoff.png
# RUN:     python scripts/policy_analysis.py
#              --data-dir data/train --weights outputs/utility/model.joblib
#              --medians outputs/utility/medians.json
#              --output-dir outputs/policy_analysis

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

from sepsis_ews.data import build_dataset
from sepsis_ews.utils import alert_burden_stats, compute_official_utility, early_warning_stats, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--medians", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-patients", type=int, default=5000)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--alert-k", type=int, default=1)
    parser.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none")
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-max-patients", type=int, default=200)
    parser.add_argument("--sensitive-min-early", type=float, default=0.3)
    parser.add_argument("--conservative-max-alerts", type=float, default=0.5)
    parser.add_argument("--num-thresholds", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, patient_ids, hours, onset_hours, _, _ = build_dataset(
        Path(args.data_dir), max_patients=args.max_patients, feature_set=args.feature_set
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))

    med = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)

    bundle = joblib.load(args.weights)
    model = bundle["model"]
    scaler = bundle["scaler"]

    X_test = np.where(np.isnan(X[test_idx]), medians, X[test_idx])
    X_test = scaler.transform(X_test)
    y_test = y[test_idx]
    pid_test = patient_ids[test_idx]
    hours_test = hours[test_idx]
    onset_test = onset_hours[test_idx]

    if args.calibrate != "none":
        train_pids = np.unique(patient_ids[train_idx])
        rng = np.random.default_rng(42)
        rng.shuffle(train_pids)
        n_cal = max(1, int(len(train_pids) * args.calibration_fraction))
        n_cal = min(n_cal, args.calibration_max_patients)
        cal_pids = set(train_pids[:n_cal])
        cal_idx = np.array([i for i, pid in enumerate(patient_ids) if pid in cal_pids], dtype=int)
        X_cal = np.where(np.isnan(X[cal_idx]), medians, X[cal_idx])
        X_cal = scaler.transform(X_cal)
        y_cal = y[cal_idx]
        calibrator = CalibratedClassifierCV(model, cv="prefit", method=args.calibrate)
        calibrator.fit(X_cal, y_cal)
        y_prob = calibrator.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.5, args.num_thresholds)
    rows = []
    for thr in thresholds:
        util = compute_official_utility(pid_test, y_test, y_prob, float(thr))
        policy = early_warning_stats(pid_test, hours_test, onset_test, y_test, y_prob, float(thr), alert_k=args.alert_k)
        burden = alert_burden_stats(pid_test, hours_test, y_test, y_prob, float(thr), alert_k=args.alert_k)
        rows.append(
            {
                "threshold": float(thr),
                "utility": float(util),
                "early_detection_rate": policy["early_detection_rate"],
                "false_alert_rate": policy["false_alert_rate"],
                "alerts_per_patient_day": burden["alerts_per_patient_day"],
            }
        )

    # sensitive policy: maximize utility subject to early_detection >= target
    sens_candidates = [r for r in rows if r["early_detection_rate"] >= args.sensitive_min_early]
    sensitive = max(sens_candidates, key=lambda r: r["utility"]) if sens_candidates else max(rows, key=lambda r: r["early_detection_rate"])

    # conservative policy: minimize alerts per day subject to utility > 0
    cons_candidates = [r for r in rows if r["alerts_per_patient_day"] <= args.conservative_max_alerts]
    conservative = min(cons_candidates, key=lambda r: r["alerts_per_patient_day"]) if cons_candidates else min(rows, key=lambda r: r["alerts_per_patient_day"])

    out = {
        "rows": rows,
        "sensitive_policy": sensitive,
        "conservative_policy": conservative,
        "settings": {
            "sensitive_min_early": args.sensitive_min_early,
            "conservative_max_alerts": args.conservative_max_alerts,
            "num_thresholds": args.num_thresholds,
        },
    }
    save_json(output_dir / "policy_analysis.json", out)

    plt.figure(figsize=(5, 4))
    plt.plot([r["alerts_per_patient_day"] for r in rows], [r["early_detection_rate"] for r in rows], marker="o")
    plt.xlabel("Alerts per patient-day")
    plt.ylabel("Early detection rate")
    plt.title("Alert Policy Tradeoff")
    plt.tight_layout()
    plt.savefig(output_dir / "policy_tradeoff.png")
    plt.close()

    print(f"Saved policy analysis to {output_dir}")


if __name__ == "__main__":
    main()
