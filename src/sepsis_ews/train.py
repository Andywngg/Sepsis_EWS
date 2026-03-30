from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .data import build_dataset
from .utils import (
    apply_alert_policy,
    compute_accuracy_f_measure,
    compute_basic_metrics,
    compute_official_utility,
    compute_utility,
    early_warning_stats,
    save_json,
    select_threshold_by_utility,
)


def compute_sample_weights(y: np.ndarray, hours: np.ndarray, onset_hours: np.ndarray, utility_weighted: bool) -> np.ndarray:
    weights = np.ones_like(y, dtype=float)
    if not utility_weighted:
        return weights
    for i in range(len(y)):
        if onset_hours[i] < 0:
            continue
        if y[i] == 1:
            lead = onset_hours[i] - hours[i]
            if lead >= 0 and lead <= 6:
                weights[i] = 2.0
            elif lead < 0 and lead >= -3:
                weights[i] = 1.5
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", choices=["logreg", "hgb"], default="logreg")
    parser.add_argument("--utility-weighted", action="store_true")
    parser.add_argument("--utility", choices=["official", "custom"], default="official")
    parser.add_argument("--alert-k", type=int, default=1)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="basic")
    parser.add_argument("--patient-normalize", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, patient_ids, hours, onset_hours, quality, feature_names = build_dataset(
        data_dir,
        max_patients=args.max_patients,
        feature_set=args.feature_set,
        patient_normalize=args.patient_normalize,
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    hours_train, hours_test = hours[train_idx], hours[test_idx]
    onset_train, onset_test = onset_hours[train_idx], onset_hours[test_idx]
    pid_train, pid_test = patient_ids[train_idx], patient_ids[test_idx]

    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sample_weights = compute_sample_weights(y_train, hours_train, onset_train, args.utility_weighted)

    if args.model == "logreg":
        model = LogisticRegression(max_iter=200, n_jobs=1)
    else:
        model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_basic_metrics(y_test, y_prob)

    thresholds = np.linspace(0.1, 0.9, 33)
    best_thr, best_util = select_threshold_by_utility(
        pid_test,
        hours_test,
        onset_test,
        y_test,
        y_prob,
        thresholds,
        utility_kind=args.utility,
        alert_k=args.alert_k,
    )
    policy = early_warning_stats(
        pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=args.alert_k
    )

    preds = np.zeros_like(y_test)
    for pid in np.unique(pid_test):
        mask = pid_test == pid
        preds[mask] = apply_alert_policy(y_prob[mask], best_thr, alert_k=args.alert_k)
    accuracy, f_measure = compute_accuracy_f_measure(y_test, preds)

    official_util = compute_official_utility(pid_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
    custom_util = compute_utility(
        pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=args.alert_k
    )

    report = {
        "model": args.model,
        "utility_weighted": bool(args.utility_weighted),
        "metrics": metrics,
        "best_threshold": best_thr,
        "utility_score": best_util,
        "utility_kind": args.utility,
        "alert_k": int(args.alert_k),
        "official_utility": float(official_util),
        "custom_utility": float(custom_util),
        "accuracy": float(accuracy),
        "f_measure": float(f_measure),
        "early_warning": policy,
        "feature_count": int(len(feature_names)),
        "max_patients": args.max_patients,
        "patient_normalize": bool(args.patient_normalize),
    }

    joblib.dump({"model": model, "scaler": scaler}, output_dir / "model.joblib")
    save_json(output_dir / "medians.json", {"medians": medians.tolist(), "feature_names": feature_names})
    report["feature_set"] = args.feature_set
    save_json(output_dir / "metrics.json", report)
    save_json(output_dir / "split_sizes.json", {"train": int(len(train_idx)), "test": int(len(test_idx))})
    save_json(output_dir / "test_patients.json", {"patient_ids": sorted(list(set(pid_test.tolist())))})

    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    main()
