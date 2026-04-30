from __future__ import annotations

# Uncertainty-aware triage using conformal prediction.
#
# The standard model outputs a single probability and fires an alert if it crosses
# a threshold. But it cannot tell you how confident it is in that prediction.
#
# Conformal prediction takes a different approach. Instead of a single label, it
# produces a PREDICTION SET for each hour: a set of labels that are consistent with
# the data at a chosen error level. The three meaningful outcomes are:
#   {0}    = model is confident this patient does NOT have sepsis right now
#   {1}    = model is confident this patient IS developing sepsis right now
#   {0,1}  = model is uncertain (both outcomes are plausible given the data)
#
# We only fire an alert when the prediction set is exactly {1}.
# When the model is uncertain ({0,1}), we defer to a human rather than alerting.
# This trades lower coverage (some hours are skipped) for higher precision.
#
# alpha is the target error rate. alpha=0.02 means "at most 2% of the time
# the true label should NOT be in the prediction set", giving a 98% coverage guarantee.
#
# Run: python scripts/conformal_alert.py
#      --data-dir data/train --weights outputs/utility/model.joblib
#      --medians outputs/utility/medians.json --output-dir outputs/conformal

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


def _load_split(patient_ids: np.ndarray, weights_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    # Load the saved train/test split so we evaluate on the same held-out patients.
    split_file = weights_path.parent / "test_patients.json"
    if split_file.exists():
        split = json.loads(split_file.read_text(encoding="utf-8"))
        test_patients = set(split.get("patient_ids", []))
        test_idx = np.array([i for i, pid in enumerate(patient_ids) if pid in test_patients], dtype=int)
        train_idx = np.array([i for i, pid in enumerate(patient_ids) if pid not in test_patients], dtype=int)
        return train_idx, test_idx, "test_patients.json"
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(patient_ids)), np.zeros(len(patient_ids)), groups=patient_ids))
    return train_idx, test_idx, "group_shuffle_split"


def _pvals_from_scores(scores_sorted: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    # For each test score, compute a p-value: the fraction of calibration scores
    # that are at least as extreme (large) as this test score.
    # A high p-value means the test score looks normal compared to calibration.
    # A low p-value means the test score is unusually high (more extreme than calibration).
    # We use binary search (searchsorted) for efficiency instead of scanning all calibration scores.
    n = len(scores_sorted)
    idx = np.searchsorted(scores_sorted, test_scores, side="left")
    # (n - idx + 1) / (n + 1) is the standard conformal p-value formula.
    # The +1 in numerator and denominator provides finite-sample coverage guarantees.
    return (n - idx + 1) / (n + 1)


def _conformal_sets(p: np.ndarray, cal_scores_pos: np.ndarray, cal_scores_neg: np.ndarray, alpha: float) -> np.ndarray:
    # Assign a prediction set to each test hour based on alpha (the allowed error rate).
    # For each hour, we test two hypotheses: "this hour is class 1 (sepsis)" and
    # "this hour is class 0 (no sepsis)". A hypothesis is included in the prediction
    # set if its p-value is above alpha (i.e., the score is not unusually extreme).
    #
    # Nonconformity score for class 1: 1 - p (high if the model disagrees with "sepsis")
    # Nonconformity score for class 0: p     (high if the model disagrees with "no sepsis")
    score_pos = 1.0 - p
    score_neg = p
    cal_pos_sorted = np.sort(cal_scores_pos)
    cal_neg_sorted = np.sort(cal_scores_neg)
    pval_pos = _pvals_from_scores(cal_pos_sorted, score_pos)
    pval_neg = _pvals_from_scores(cal_neg_sorted, score_neg)
    # Include label 1 in the set if the model does not strongly reject "sepsis" hypothesis.
    set_pos = pval_pos > alpha
    # Include label 0 in the set if the model does not strongly reject "no sepsis" hypothesis.
    set_neg = pval_neg > alpha
    # Encode the prediction sets as integers: 0={0}, 1={1}, 2={0,1}, 3=empty set
    sets = np.zeros_like(p, dtype=int)
    sets[set_pos & ~set_neg] = 1   # confident sepsis
    sets[~set_pos & set_neg] = 0   # confident no sepsis
    sets[set_pos & set_neg] = 2    # uncertain (both labels possible)
    sets[~set_pos & ~set_neg] = 3  # empty set (should not happen with a valid alpha)
    return sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--medians", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--alert-k", type=int, default=1)
    parser.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-max-patients", type=int, default=200)
    parser.add_argument("--alphas", default="0.01,0.02,0.05,0.1,0.2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, patient_ids, hours, onset_hours, _, _ = build_dataset(
        Path(args.data_dir), max_patients=args.max_patients, feature_set=args.feature_set
    )

    train_idx, test_idx, split_source = _load_split(patient_ids, Path(args.weights))

    med = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)

    bundle = joblib.load(args.weights)
    model = bundle["model"]
    scaler = bundle["scaler"]

    # Set aside a small portion of training patients to compute conformal p-values.
    # These calibration patients give us the distribution of "typical" nonconformity
    # scores, which we compare against test scores to produce prediction sets.
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

    X_test = np.where(np.isnan(X[test_idx]), medians, X[test_idx])
    X_test = scaler.transform(X_test)
    y_test = y[test_idx]
    pid_test = patient_ids[test_idx]
    hours_test = hours[test_idx]
    onset_test = onset_hours[test_idx]

    # Optionally calibrate the probabilities before computing conformal scores.
    if args.calibrate != "none":
        calibrator = CalibratedClassifierCV(model, cv="prefit", method=args.calibrate)
        calibrator.fit(X_cal, y_cal)
        p_cal = calibrator.predict_proba(X_cal)[:, 1]
        p_test = calibrator.predict_proba(X_test)[:, 1]
    else:
        p_cal = model.predict_proba(X_cal)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

    # Separate calibration scores by true label so we have two reference distributions:
    # one for how scores look on actual sepsis hours, and one for non-sepsis hours.
    cal_scores_pos = 1.0 - p_cal[y_cal == 1]   # nonconformity scores for true positives
    cal_scores_neg = p_cal[y_cal == 0]           # nonconformity scores for true negatives

    rows = []
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    for alpha in alphas:
        # Build prediction sets for every test hour at this alpha level.
        sets = _conformal_sets(p_test, cal_scores_pos, cal_scores_neg, alpha)

        # singleton_rate: fraction of hours where the model gave a definitive answer.
        # Higher is better (less deferral). Controlled by alpha.
        singleton = (sets == 0) | (sets == 1)
        singleton_rate = float(np.mean(singleton)) if len(singleton) else 0.0

        # Only alert on hours where the prediction set is {1} (confident sepsis).
        # Hours with {0,1} are deferred; the model says "I am not sure, ask a doctor".
        p_conf = (sets == 1).astype(float)
        util = compute_official_utility(pid_test, y_test, p_conf, 0.5, alert_k=args.alert_k)
        policy = early_warning_stats(pid_test, hours_test, onset_test, y_test, p_conf, 0.5, alert_k=args.alert_k)
        burden = alert_burden_stats(pid_test, hours_test, y_test, p_conf, 0.5, alert_k=args.alert_k)

        rows.append(
            {
                "alpha": alpha,
                "singleton_rate": singleton_rate,
                "utility": util,
                "early_detection_rate": policy["early_detection_rate"],
                "false_alert_rate": policy["false_alert_rate"],
                "alerts_per_patient_day": burden["alerts_per_patient_day"],
            }
        )

    save_json(
        output_dir / "conformal_alert.json",
        {
            "rows": rows,
            "split_source": split_source,
            "calibration": {
                "method": args.calibrate,
                "fraction": args.calibration_fraction,
                "max_patients": args.calibration_max_patients,
            },
        },
    )

    # Plot how utility and alert rate change as we vary alpha.
    # Lower alpha = stricter (more deferral, fewer alerts, higher precision on what fires).
    # Higher alpha = looser (fewer deferrals, more alerts, lower precision).
    if rows:
        plt.figure(figsize=(5, 4))
        plt.plot([r["alpha"] for r in rows], [r["utility"] for r in rows], marker="o", label="Utility")
        plt.plot([r["alpha"] for r in rows], [r["alerts_per_patient_day"] for r in rows], marker="o", label="Alerts/day")
        plt.xlabel("Alpha (conformal error rate)")
        plt.ylabel("Metric")
        plt.title("Conformal Alert Tradeoff")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "conformal_alert.png")
        plt.close()

    print(f"Saved conformal alert analysis to {output_dir}")


if __name__ == "__main__":
    main()
