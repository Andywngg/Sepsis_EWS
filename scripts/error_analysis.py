from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV

from sepsis_ews.data import build_dataset
from sepsis_ews.utils import apply_alert_policy, alert_burden_stats, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--medians", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--patient-normalize", action="store_true")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    feature_set = metrics.get("feature_set", "basic")
    threshold = float(metrics.get("best_threshold", 0.5))
    alert_k = int(metrics.get("alert_k", 1))

    X, y, patient_ids, hours, onset_hours, quality, _ = build_dataset(
        Path(args.data_dir),
        max_patients=args.max_patients,
        feature_set=feature_set,
        patient_normalize=args.patient_normalize,
    )

    split_file = Path(args.weights).parent / "test_patients.json"
    if split_file.exists():
        split = json.loads(split_file.read_text(encoding="utf-8"))
        test_patients = set(split.get("patient_ids", []))
        test_idx = np.array([i for i, pid in enumerate(patient_ids) if pid in test_patients], dtype=int)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, test_idx = next(splitter.split(X, y, groups=patient_ids))

    X_test = X[test_idx]
    y_test = y[test_idx]
    pid_test = patient_ids[test_idx]
    hours_test = hours[test_idx]
    onset_test = onset_hours[test_idx]
    quality_test = quality[test_idx]

    med = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_test = np.where(np.isnan(X_test), medians, X_test)

    bundle = joblib.load(args.weights)
    model = bundle["model"]
    scaler = bundle["scaler"]
    X_test = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    calib = metrics.get("calibration", {})
    if calib and calib.get("method") and calib.get("method") != "none":
        if split_file.exists():
            train_idx = np.array([i for i, pid in enumerate(patient_ids) if pid not in test_patients], dtype=int)
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, _ = next(splitter.split(X, y, groups=patient_ids))
        train_pids = np.unique(patient_ids[train_idx])
        rng = np.random.default_rng(42)
        rng.shuffle(train_pids)
        n_cal = max(1, int(len(train_pids) * float(calib.get("fraction", 0.1))))
        n_cal = min(n_cal, int(calib.get("max_patients", 200)))
        cal_pids = train_pids[:n_cal]
        cal_idx = np.array([i for i, pid in enumerate(patient_ids) if pid in cal_pids], dtype=int)
        X_cal = np.where(np.isnan(X[cal_idx]), medians, X[cal_idx])
        X_cal = scaler.transform(X_cal)
        y_cal = y[cal_idx]
        calibrator = CalibratedClassifierCV(model, cv="prefit", method=calib.get("method", "sigmoid"))
        calibrator.fit(X_cal, y_cal)
        y_prob = calibrator.predict_proba(X_test)[:, 1]

    patients = np.unique(pid_test)
    tp = fp = fn = tn = 0
    group_quality = {"tp": [], "fp": [], "fn": [], "tn": []}
    group_hours = {"tp": [], "fp": [], "fn": [], "tn": []}
    lead_times = []

    for pid in patients:
        mask = pid_test == pid
        has_sepsis = np.any(y_test[mask] == 1)
        preds = apply_alert_policy(y_prob[mask], threshold, alert_k=alert_k)
        predicted = np.any(preds == 1)
        mean_quality = float(np.mean(quality_test[mask]))
        hours_len = int(len(hours_test[mask]))

        if has_sepsis and predicted:
            tp += 1
            group_quality["tp"].append(mean_quality)
            group_hours["tp"].append(hours_len)
            onset = onset_test[mask][0]
            alert_idx = np.where(preds == 1)[0]
            if onset >= 0 and len(alert_idx):
                lead_times.append(float(onset - hours_test[mask][alert_idx[0]]))
        elif has_sepsis and not predicted:
            fn += 1
            group_quality["fn"].append(mean_quality)
            group_hours["fn"].append(hours_len)
        elif (not has_sepsis) and predicted:
            fp += 1
            group_quality["fp"].append(mean_quality)
            group_hours["fp"].append(hours_len)
        else:
            tn += 1
            group_quality["tn"].append(mean_quality)
            group_hours["tn"].append(hours_len)

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    summary = {
        "threshold": threshold,
        "alert_k": alert_k,
        "patient_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "mean_quality": {k: float(np.mean(v)) if v else 0.0 for k, v in group_quality.items()},
        "mean_hours": {k: float(np.mean(v)) if v else 0.0 for k, v in group_hours.items()},
        "lead_time": {
            "mean": float(np.mean(lead_times)) if lead_times else 0.0,
            "median": float(np.median(lead_times)) if lead_times else 0.0,
            "count": int(len(lead_times)),
        },
        "alert_burden": alert_burden_stats(
            pid_test, hours_test, y_test, y_prob, threshold, alert_k=alert_k
        ),
        "max_patients": args.max_patients,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "error_analysis.json", summary)

    md = []
    md.append("# Error Analysis\n\n")
    md.append(f"- Threshold: {threshold}\n")
    md.append(f"- Alert k: {alert_k}\n")
    md.append(f"- TP/FP/FN/TN: {tp}/{fp}/{fn}/{tn}\n")
    md.append(f"- Sensitivity: {summary['sensitivity']:.3f}\n")
    md.append(f"- Specificity: {summary['specificity']:.3f}\n")
    md.append(f"- Mean lead time: {summary['lead_time']['mean']:.2f} hours\n")
    md.append(f"- Median lead time: {summary['lead_time']['median']:.2f} hours\n\n")
    md.append("## Data Quality (mean per patient)\n")
    md.append(f"- TP: {summary['mean_quality']['tp']:.3f}\n")
    md.append(f"- FP: {summary['mean_quality']['fp']:.3f}\n")
    md.append(f"- FN: {summary['mean_quality']['fn']:.3f}\n")
    md.append(f"- TN: {summary['mean_quality']['tn']:.3f}\n\n")
    md.append("## Alert Burden\n")
    md.append(
        f"- Alerts per patient-day: {summary['alert_burden']['alerts_per_patient_day']:.3f}\n"
    )
    md.append(
        f"- Alerts per non-sepsis patient-day: {summary['alert_burden']['alerts_per_nonsepsis_patient_day']:.3f}\n"
    )
    md.append(
        f"- Mean alerts per patient: {summary['alert_burden']['mean_alerts_per_patient']:.3f}\n"
    )
    (output_dir / "error_analysis.md").write_text("".join(md), encoding="utf-8")
    print(f"Saved error analysis to {output_dir}")


if __name__ == "__main__":
    main()
