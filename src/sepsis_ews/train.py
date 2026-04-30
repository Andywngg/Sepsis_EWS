from __future__ import annotations

# Trains the sepsis classifier and saves the model plus a metrics report to disk.
# Run this first before eval.py or any of the analysis scripts.
# Example: python -m sepsis_ews.train --data-dir data/train --model hgb --utility-weighted
#          --feature-set enhanced --utility official --output-dir outputs/utility

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
    # When utility_weighted is False, every training sample gets equal weight (all 1.0).
    # When True, we up-weight hours in the most clinically important window around onset.
    # The idea: the model should pay extra attention to predicting correctly in the
    # hours just before and just after sepsis onset, because those predictions have the
    # highest real-world impact on patient outcomes.
    weights = np.ones_like(y, dtype=float)
    if not utility_weighted:
        return weights

    for i in range(len(y)):
        # Skip patients who never developed sepsis (onset_hour = -1 sentinel).
        if onset_hours[i] < 0:
            continue
        if y[i] == 1:
            # lead is how many hours BEFORE onset this sample is.
            # Positive lead = the sample is in the future relative to this hour.
            lead = onset_hours[i] - hours[i]
            if lead >= 0 and lead <= 6:
                # The 6 hours right before onset are the highest-value prediction window.
                # Catching sepsis here gives the most time to intervene.
                weights[i] = 2.0
            elif lead < 0 and lead >= -3:
                # The 3 hours after onset still have clinical value (patient can still be helped).
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

    # Load all patient files and convert them into a single stacked matrix.
    # Each row of X is one patient-hour. y is the sepsis label for that hour.
    # patient_ids tells us which patient each row belongs to (needed for the split below).
    X, y, patient_ids, hours, onset_hours, quality, feature_names = build_dataset(
        data_dir,
        max_patients=args.max_patients,
        feature_set=args.feature_set,
        patient_normalize=args.patient_normalize,
    )

    # Split patients into training (80%) and test (20%) groups.
    # GroupShuffleSplit keeps all hours from the same patient on the same side.
    # If we used a regular train_test_split instead, it would split individual hours
    # from the same patient across both groups. The model would then see some of that
    # patient's hours during training and be tested on other hours from the same patient,
    # giving an inflated accuracy because it has already "seen" that patient.
    # The groups= argument is what enforces the patient-level boundary.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))
    X_train, y_train   = X[train_idx], y[train_idx]
    X_test,  y_test    = X[test_idx],  y[test_idx]
    hours_train, hours_test   = hours[train_idx],       hours[test_idx]
    onset_train, onset_test   = onset_hours[train_idx], onset_hours[test_idx]
    pid_train,   pid_test     = patient_ids[train_idx], patient_ids[test_idx]

    # Fill in missing values using the median of each column from the training data.
    # We compute medians from training data only and apply those same values to the test
    # set, so the test set cannot influence imputation in any way.
    # The np.where line at the end replaces any remaining NaN that survived (happens
    # when an entire column is missing across all training patients, making nanmedian
    # return NaN itself).
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test  = np.where(np.isnan(X_test),  medians, X_test)

    # Scale features so every column has mean 0 and standard deviation 1.
    # This prevents variables with large numeric ranges (like blood pressure in mmHg)
    # from dominating variables with small ranges (like lactate in mmol/L).
    # fit_transform on training data: learn the mean and std, then apply them.
    # transform on test data: apply the SAME learned mean and std without re-computing.
    # Re-computing on test data would be a leakage error.
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Assign a weight to each training sample based on its clinical importance.
    # See compute_sample_weights() above for the weighting logic.
    sample_weights = compute_sample_weights(y_train, hours_train, onset_train, args.utility_weighted)

    # Choose the classifier. LogisticRegression is the simple baseline; it draws a single
    # linear boundary in feature space. HistGradientBoosting (HGB) is the main model:
    # it builds many small decision trees sequentially, each one correcting the errors
    # of the previous. It handles non-linear interactions between features and naturally
    # tolerates missing values, which matters given how sparse this dataset is.
    if args.model == "logreg":
        model = LogisticRegression(max_iter=200, n_jobs=1)
    else:
        model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # predict_proba returns a two-column array: [prob_not_sepsis, prob_sepsis].
    # We take column index 1, which is the probability the model assigns to sepsis.
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_basic_metrics(y_test, y_prob)

    # The model outputs a continuous probability (0 to 1). To trigger an alert in the
    # ICU, we need to convert that to a yes/no decision using a threshold.
    # Different thresholds lead to different tradeoffs: a low threshold catches more
    # sepsis but fires more false alarms; a high threshold is more specific but misses cases.
    # We try 33 evenly spaced values and pick the one with the best clinical utility score.
    thresholds = np.linspace(0.1, 0.9, 33)
    best_thr, best_util = select_threshold_by_utility(
        pid_test, hours_test, onset_test, y_test, y_prob,
        thresholds, utility_kind=args.utility, alert_k=args.alert_k,
    )

    # Compute final performance statistics using the best threshold we found.
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

    # Save everything needed to reproduce predictions later.
    # model.joblib contains both the trained model and its scaler bundled together.
    # medians.json is needed at inference time to fill in missing values the same way.
    # test_patients.json records which patients were held out so eval.py can reproduce
    # the exact same split without re-running training.
    joblib.dump({"model": model, "scaler": scaler}, output_dir / "model.joblib")
    save_json(output_dir / "medians.json",    {"medians": medians.tolist(), "feature_names": feature_names})
    save_json(output_dir / "metrics.json",    report)
    save_json(output_dir / "split_sizes.json",{"train": int(len(train_idx)), "test": int(len(test_idx))})
    save_json(output_dir / "test_patients.json", {"patient_ids": sorted(list(set(pid_test.tolist())))})

    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    main()
