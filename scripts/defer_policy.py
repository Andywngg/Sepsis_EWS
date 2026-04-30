from __future__ import annotations

# Selective prediction: skip patients the model is uncertain about and defer them
# to clinical judgment rather than firing or suppressing an alert.
#
# For each patient, we compute a "confidence score" = the average distance between
# the model's predicted probability and 0.5 across all their ICU hours.
# If the average prediction was 0.9 or 0.1 (far from 0.5), the model is confident.
# If the average was near 0.5 every hour, the model was repeatedly on the fence.
#
# We can then choose to only act on the high-confidence patients and defer the rest.
# Deferring low-confidence patients reduces false alarms on cases the model cannot
# handle well, at the cost of less coverage (some patients get no automated decision).
#
# This is a coverage-utility tradeoff: higher coverage = more patients get a decision,
# but lower precision because some uncertain cases are included.
#
# Run: python scripts/defer_policy.py
#      --data-dir data/train --weights outputs/utility/model.joblib
#      --medians outputs/utility/medians.json --output-dir outputs/defer

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

from sepsis_ews.data import build_dataset
from sepsis_ews.utils import compute_official_utility, early_warning_stats, alert_burden_stats, save_json


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

    # Optionally calibrate the probabilities before computing confidence.
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

    # Compute per-patient confidence: the mean absolute deviation of the probability from 0.5.
    # A patient where the model predicted 0.9 every hour has confidence close to 0.4 (high).
    # A patient where the model predicted 0.5 every hour has confidence 0.0 (uncertain).
    unique_pids = np.unique(pid_test)
    conf = {}
    for pid in unique_pids:
        mask = pid_test == pid
        conf[pid] = float(np.mean(np.abs(y_prob[mask] - 0.5)))

    # Sweep over different coverage levels from 50% to 100% of patients.
    # At each level, compute the confidence threshold that retains exactly that fraction,
    # then evaluate utility only on those patients.
    coverages = [round(c, 2) for c in np.linspace(0.5, 1.0, 11)]
    rows = []

    for cov in coverages:
        # The confidence threshold for this coverage level: the (1-cov) quantile.
        # Example: coverage=0.8 means keep the top 80% by confidence.
        # We set the threshold at the 20th percentile of confidence values.
        threshold = np.quantile(list(conf.values()), 1 - cov)
        keep_pids = {pid for pid, v in conf.items() if v >= threshold}
        mask = np.array([pid in keep_pids for pid in pid_test], dtype=bool)
        if not np.any(mask):
            continue
        cov_actual = len(keep_pids) / len(unique_pids)

        util = compute_official_utility(pid_test[mask], y_test[mask], y_prob[mask], 0.1)
        policy = early_warning_stats(
            pid_test[mask], hours_test[mask], onset_test[mask], y_test[mask], y_prob[mask], 0.1, alert_k=args.alert_k
        )
        burden = alert_burden_stats(
            pid_test[mask], hours_test[mask], y_test[mask], y_prob[mask], 0.1, alert_k=args.alert_k
        )

        rows.append(
            {
                "coverage": cov_actual,
                "confidence_threshold": float(threshold),
                "utility": util,
                "early_detection_rate": policy["early_detection_rate"],
                "false_alert_rate": policy["false_alert_rate"],
                "alerts_per_patient_day": burden["alerts_per_patient_day"],
            }
        )

    save_json(output_dir / "defer_policy.json", {"rows": rows})

    # Plot how utility and alert burden change as we raise the coverage requirement.
    # At coverage=1.0 (right side) all patients are included. At coverage=0.5 only the
    # most confident half are included. The sweet spot is usually somewhere in between.
    if rows:
        plt.figure(figsize=(5, 4))
        plt.plot([r["coverage"] for r in rows], [r["utility"] for r in rows], marker="o", label="Utility")
        plt.plot([r["coverage"] for r in rows], [r["alerts_per_patient_day"] for r in rows], marker="o", label="Alerts/day")
        plt.xlabel("Coverage (fraction of patients kept)")
        plt.ylabel("Metric")
        plt.title("Selective Prediction Tradeoff")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "defer_policy.png")
        plt.close()

    print(f"Saved defer policy to {output_dir}")


if __name__ == "__main__":
    main()
