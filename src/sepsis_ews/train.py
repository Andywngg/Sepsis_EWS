from __future__ import annotations

# PURPOSE: Train the sepsis classifier and save the model + metrics to disk.
# PIPELINE: load data → split by patient → impute → scale → weight → fit → threshold search → save
# RUN:  python -m sepsis_ews.train --data-dir data/train --model hgb --utility-weighted
#                                  --feature-set enhanced --utility official --output-dir outputs/utility

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
    # -----------------------------------------------------------------------
    # UTILITY-WEIGHTED TRAINING
    # Upweights samples in the clinically critical pre-onset window so the
    # model learns to prioritize early detection over raw accuracy.
    #
    #   weight = 2.0  → 0-6 hours BEFORE onset  (most important: catch it early)
    #   weight = 1.5  → 0-3 hours AFTER onset   (late alert still has value)
    #   weight = 1.0  → everything else          (normal importance)
    # -----------------------------------------------------------------------
    weights = np.ones_like(y, dtype=float)
    if not utility_weighted:
        return weights
    for i in range(len(y)):
        if onset_hours[i] < 0:
            continue  # -1 sentinel = no sepsis, skip
        if y[i] == 1:
            lead = onset_hours[i] - hours[i]  # positive = hours before onset
            if lead >= 0 and lead <= 6:
                weights[i] = 2.0
            elif lead < 0 and lead >= -3:
                weights[i] = 1.5
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",         required=True)
    parser.add_argument("--model",            choices=["logreg", "hgb"], default="logreg")
    parser.add_argument("--utility-weighted", action="store_true")
    parser.add_argument("--utility",          choices=["official", "custom"], default="official")
    parser.add_argument("--alert-k",          type=int, default=1)
    parser.add_argument("--max-patients",     type=int, default=None)
    parser.add_argument("--feature-set",      choices=["basic", "enhanced"], default="basic")
    parser.add_argument("--patient-normalize",action="store_true")
    parser.add_argument("--output-dir",       required=True)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- STEP 1: LOAD DATA ---
    # Returns one row per patient-hour across all ~40k patients
    X, y, patient_ids, hours, onset_hours, quality, feature_names = build_dataset(
        data_dir,
        max_patients=args.max_patients,
        feature_set=args.feature_set,
        patient_normalize=args.patient_normalize,
    )

    # --- STEP 2: TRAIN/TEST SPLIT BY PATIENT (prevents data leakage) ---
    # GroupShuffleSplit ensures all hours from a patient go to ONE side only.
    # Never split individual patient hours across train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))
    X_train, y_train   = X[train_idx], y[train_idx]
    X_test,  y_test    = X[test_idx],  y[test_idx]
    hours_train, hours_test   = hours[train_idx],       hours[test_idx]
    onset_train, onset_test   = onset_hours[train_idx], onset_hours[test_idx]
    pid_train,   pid_test     = patient_ids[train_idx], patient_ids[test_idx]

    # --- STEP 3: IMPUTATION (fill missing values with training-set medians) ---
    # Medians are computed from TRAINING data only — never from test data.
    # Same medians are applied to test set to avoid information leakage.
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test  = np.where(np.isnan(X_test),  medians, X_test)

    # --- STEP 4: SCALING (StandardScaler — zero mean, unit variance) ---
    # fit_transform on train: learns mean/std FROM training data, then scales.
    # transform on test: applies the SAME learned mean/std (no recomputing).
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --- STEP 5: SAMPLE WEIGHTS ---
    sample_weights = compute_sample_weights(y_train, hours_train, onset_train, args.utility_weighted)

    # --- STEP 6: MODEL SELECTION ---
    # logreg: simple linear baseline for comparison
    # hgb:    HistGradientBoosting — main model, captures non-linear interactions,
    #         natively handles NaN, scales well to 40k patients
    if args.model == "logreg":
        model = LogisticRegression(max_iter=200, n_jobs=1)
    else:
        model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # --- STEP 7: PREDICT PROBABILITIES ON TEST SET ---
    # predict_proba returns [[prob_class0, prob_class1], ...]
    # [:, 1] takes the sepsis probability column only
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_basic_metrics(y_test, y_prob)

    # --- STEP 8: GRID SEARCH FOR BEST ALERT THRESHOLD ---
    # Try 33 thresholds from 0.1 to 0.9; pick the one maximizing clinical utility.
    # Threshold converts probability → binary alert (fire if prob >= threshold).
    thresholds = np.linspace(0.1, 0.9, 33)
    best_thr, best_util = select_threshold_by_utility(
        pid_test, hours_test, onset_test, y_test, y_prob,
        thresholds, utility_kind=args.utility, alert_k=args.alert_k,
    )

    policy   = early_warning_stats(pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
    preds    = np.zeros_like(y_test)
    for pid in np.unique(pid_test):
        mask = pid_test == pid
        preds[mask] = apply_alert_policy(y_prob[mask], best_thr, alert_k=args.alert_k)
    accuracy, f_measure = compute_accuracy_f_measure(y_test, preds)
    official_util = compute_official_utility(pid_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
    custom_util   = compute_utility(pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=args.alert_k)

    report = {
        "model":            args.model,
        "utility_weighted": bool(args.utility_weighted),
        "metrics":          metrics,
        "best_threshold":   best_thr,
        "utility_score":    best_util,
        "utility_kind":     args.utility,
        "alert_k":          int(args.alert_k),
        "official_utility": float(official_util),
        "custom_utility":   float(custom_util),
        "accuracy":         float(accuracy),
        "f_measure":        float(f_measure),
        "early_warning":    policy,
        "feature_count":    int(len(feature_names)),
        "max_patients":     args.max_patients,
        "patient_normalize":bool(args.patient_normalize),
        "feature_set":      args.feature_set,
    }

    # --- STEP 9: SAVE MODEL, MEDIANS, AND METRICS ---
    # model.joblib contains both the trained model AND the scaler bundled together.
    # medians.json is needed at eval/inference time to impute missing values.
    # test_patients.json records which patients are held out — used by eval.py.
    joblib.dump({"model": model, "scaler": scaler}, output_dir / "model.joblib")
    save_json(output_dir / "medians.json",    {"medians": medians.tolist(), "feature_names": feature_names})
    save_json(output_dir / "metrics.json",    report)
    save_json(output_dir / "split_sizes.json",{"train": int(len(train_idx)), "test": int(len(test_idx))})
    save_json(output_dir / "test_patients.json", {"patient_ids": sorted(list(set(pid_test.tolist())))})

    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    main()
