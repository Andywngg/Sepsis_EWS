from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss

from sepsis_ews.data import build_dataset
from sepsis_ews.utils import (
    compute_basic_metrics,
    compute_official_utility,
    compute_patient_level_metrics,
    compute_utility,
    save_json,
    select_threshold_by_utility,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--medians", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-patients", type=int, default=500)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--utility", choices=["official", "custom"], default="official")
    parser.add_argument("--alert-k", type=int, default=1)
    parser.add_argument("--patient-normalize", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, patient_ids, hours, onset_hours, _, _ = build_dataset(
        Path(args.data_dir),
        max_patients=args.max_patients,
        feature_set=args.feature_set,
        patient_normalize=args.patient_normalize,
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    pid_test = patient_ids[test_idx]

    med = json_load(Path(args.medians))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)

    bundle = joblib.load(args.weights)
    model = bundle["model"]
    scaler = bundle["scaler"]
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_prob_raw = model.predict_proba(X_test)[:, 1]

    configs = [
        {"method": "none", "fraction": 0.0, "max_patients": 0},
        {"method": "sigmoid", "fraction": 0.05, "max_patients": 200},
        {"method": "sigmoid", "fraction": 0.1, "max_patients": 500},
        {"method": "sigmoid", "fraction": 0.2, "max_patients": 1000},
        {"method": "isotonic", "fraction": 0.05, "max_patients": 200},
        {"method": "isotonic", "fraction": 0.1, "max_patients": 500},
        {"method": "isotonic", "fraction": 0.2, "max_patients": 1000},
    ]

    results = []
    thresholds = np.linspace(0.1, 0.9, 33)

    for cfg in configs:
        if cfg["method"] == "none":
            y_prob = y_prob_raw
            cal_patients = 0
        else:
            train_pids = np.unique(patient_ids[train_idx])
            rng = np.random.default_rng(42)
            rng.shuffle(train_pids)
            n_cal = max(1, int(len(train_pids) * cfg["fraction"]))
            n_cal = min(n_cal, cfg["max_patients"])
            cal_pids = train_pids[:n_cal]
            cal_idx = np.array([i for i, pid in enumerate(patient_ids) if pid in cal_pids], dtype=int)
            X_cal = np.where(np.isnan(X[cal_idx]), medians, X[cal_idx])
            X_cal = scaler.transform(X_cal)
            y_cal = y[cal_idx]
            calibrator = CalibratedClassifierCV(model, cv="prefit", method=cfg["method"])
            calibrator.fit(X_cal, y_cal)
            y_prob = calibrator.predict_proba(X_test)[:, 1]
            cal_patients = int(len(np.unique(cal_pids)))

        metrics = compute_basic_metrics(y_test, y_prob)
        patient_metrics = compute_patient_level_metrics(pid_test, y_test, y_prob)
        brier = float(brier_score_loss(y_test, y_prob))
        brier_raw = float(brier_score_loss(y_test, y_prob_raw))
        best_thr, best_util = select_threshold_by_utility(
            pid_test,
            hours[test_idx],
            onset_hours[test_idx],
            y_test,
            y_prob,
            thresholds,
            utility_kind=args.utility,
            alert_k=args.alert_k,
        )
        official_util = compute_official_utility(pid_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
        custom_util = compute_utility(
            pid_test, hours[test_idx], onset_hours[test_idx], y_test, y_prob, best_thr, alert_k=args.alert_k
        )

        results.append(
            {
                "method": cfg["method"],
                "cal_fraction": cfg["fraction"],
                "cal_patients": cal_patients,
                "max_patients": args.max_patients,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "patient_auroc": patient_metrics["auroc"],
                "patient_auprc": patient_metrics["auprc"],
                "brier": brier,
                "brier_raw": brier_raw,
                "best_threshold": best_thr,
                "utility": best_util,
                "official_utility": official_util,
                "custom_utility": custom_util,
            }
        )

    results.sort(key=lambda r: r["official_utility"], reverse=True)
    save_json(output_dir / "calibration_sweep.json", {"results": results})

    csv_path = output_dir / "calibration_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved calibration sweep to {output_dir}")


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
