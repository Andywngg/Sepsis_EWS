from __future__ import annotations

# PURPOSE: Hyperparameter grid search for the HistGradientBoosting model.
# PARAMETERS SEARCHED:
#   max_depth:       [4, 6, 8]    -- depth of each tree
#   learning_rate:   [0.03, 0.05, 0.10] -- how much each tree contributes
#   max_leaf_nodes:  [31, 63]     -- max number of leaf nodes per tree
# TOTAL CONFIGS:     3 x 3 x 2 = 18 combinations
# RANKED BY:         official PhysioNet utility score on the validation set
# NOTE:  Uses a fresh train/test split (does not load a saved model).
#        Intended to be run BEFORE train.py to find good hyperparameters.
# OUTPUT:  tuning_results.json and tuning_results.csv -- all 18 configs ranked.
# RUN:     python scripts/tune_hgb.py
#              --data-dir data/train --feature-set enhanced --utility-weighted
#              --output-dir outputs/tuning --max-patients 500

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from sepsis_ews.data import build_dataset
from sepsis_ews.train import compute_sample_weights
from sepsis_ews.utils import (
    apply_alert_policy,
    compute_accuracy_f_measure,
    compute_basic_metrics,
    compute_official_utility,
    compute_utility,
    save_json,
    select_threshold_by_utility,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-patients", type=int, default=500)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--utility-weighted", action="store_true")
    parser.add_argument("--utility", choices=["official", "custom"], default="official")
    parser.add_argument("--alert-k", type=int, default=1)
    parser.add_argument("--patient-normalize", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, patient_ids, hours, onset_hours, _, feature_names = build_dataset(
        Path(args.data_dir),
        max_patients=args.max_patients,
        feature_set=args.feature_set,
        patient_normalize=args.patient_normalize,
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    hours_test, onset_test = hours[test_idx], onset_hours[test_idx]
    pid_test = patient_ids[test_idx]

    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sample_weights = compute_sample_weights(y_train, hours[train_idx], onset_hours[train_idx], args.utility_weighted)

    grid = []
    for max_depth in (4, 6, 8):
        for learning_rate in (0.03, 0.05, 0.1):
            for max_leaf_nodes in (31, 63):
                grid.append(
                    {
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "max_leaf_nodes": max_leaf_nodes,
                    }
                )

    results = []
    thresholds = np.linspace(0.1, 0.9, 33)

    for params in grid:
        model = HistGradientBoostingClassifier(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"],
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_basic_metrics(y_test, y_prob)
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

        preds = np.zeros_like(y_test)
        for pid in np.unique(pid_test):
            mask = pid_test == pid
            preds[mask] = apply_alert_policy(y_prob[mask], best_thr, alert_k=args.alert_k)
        accuracy, f_measure = compute_accuracy_f_measure(y_test, preds)

        official_util = compute_official_utility(pid_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
        custom_util = compute_utility(
            pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=args.alert_k
        )

        results.append(
            {
                **params,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "best_threshold": best_thr,
                "utility_score": best_util,
                "official_utility": official_util,
                "custom_utility": custom_util,
                "accuracy": accuracy,
                "f_measure": f_measure,
            }
        )

    results.sort(key=lambda r: r["utility_score"], reverse=True)

    save_json(output_dir / "tuning_results.json", {"results": results, "feature_names": feature_names})

    csv_path = output_dir / "tuning_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved tuning results to {output_dir}")


if __name__ == "__main__":
    main()
