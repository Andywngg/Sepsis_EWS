from __future__ import annotations

# Trend-adaptive alert policy: lowers the alert threshold when the risk score is rising
# quickly, and raises it when the risk score is falling.
#
# The standard policy fires an alert whenever probability >= fixed_threshold.
# This does not distinguish between a patient whose risk jumped from 0.05 to 0.20 in
# one hour (rapidly deteriorating) versus a patient who has been stable at 0.20 for
# 12 hours (elevated but not changing). The first patient may need more urgent attention.
#
# The dynamic policy adjusts the threshold each hour based on the trend:
#   adj_threshold = base_threshold - k * (trend / trend_scale)
# where trend = p[t] - p[t-1] (how much did the risk score change this hour?).
# trend_scale normalizes the trend so that k has the same meaning across all patients.
# k controls how aggressively the threshold responds to trends.
# k is grid-searched on a held-out calibration set to find the value that maximizes utility.
#
# Run: python scripts/dynamic_policy.py
#      --data-dir data/train --weights outputs/utility/model.joblib
#      --medians outputs/utility/medians.json --output-dir outputs/dynamic_policy

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

from sepsis_ews.data import build_dataset
from sepsis_ews.utils import (
    compute_basic_metrics,
    compute_prediction_utility,
    compute_patient_level_metrics,
    save_json,
)


def _load_split(patient_ids: np.ndarray, weights_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
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


def _official_utility_from_predictions(patient_ids: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> float:
    # Computes the normalized PhysioNet utility score from pre-computed binary predictions.
    # Used internally because dynamic_policy already converts probabilities to alerts
    # before calling utility; the standard compute_official_utility takes probabilities.
    dt_early = -12
    dt_optimal = -6
    dt_late = 3
    max_u_tp = 1
    min_u_fn = -2
    u_fp = -0.05
    u_tn = 0

    patients = np.unique(patient_ids)
    observed = 0.0
    best = 0.0
    inaction = 0.0

    for pid in patients:
        mask = patient_ids == pid
        y = labels[mask]
        p = preds[mask]

        observed += compute_prediction_utility(y, p, dt_early=dt_early, dt_optimal=dt_optimal,
            dt_late=dt_late, max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn)

        # Compute the score for the perfect oracle (always alerts in the optimal window).
        best_preds = np.zeros_like(y)
        if np.any(y):
            t_sepsis = np.argmax(y) - dt_optimal
            start = max(0, t_sepsis + dt_early)
            end = min(t_sepsis + dt_late + 1, len(y))
            best_preds[start:end] = 1
        best += compute_prediction_utility(y, best_preds, dt_early=dt_early, dt_optimal=dt_optimal,
            dt_late=dt_late, max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn)

        # Compute the score for always doing nothing (never alerting).
        inaction += compute_prediction_utility(y, np.zeros_like(y), dt_early=dt_early,
            dt_optimal=dt_optimal, dt_late=dt_late, max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn)

    denom = best - inaction
    if denom == 0:
        return 0.0
    return float((observed - inaction) / denom)


def _build_index(patient_ids: np.ndarray) -> dict[str, np.ndarray]:
    # Pre-build a mapping from patient ID to row indices for fast per-patient access.
    return {pid: np.where(patient_ids == pid)[0] for pid in np.unique(patient_ids)}


def _apply_dynamic_policy(
    idx_by_pid: dict[str, np.ndarray],
    probabilities: np.ndarray,
    base_threshold: float,
    trend_scale: float,
    k: float,
    alert_k: int,
) -> np.ndarray:
    # Apply the dynamic threshold policy to all patients.
    # For each patient, compute the per-hour trend and adjust the threshold accordingly.
    preds = np.zeros_like(probabilities, dtype=int)
    for pid, idx in idx_by_pid.items():
        p = probabilities[idx]

        # trend[t] = p[t] - p[t-1]: positive when risk is rising, negative when falling.
        # trend[0] = 0 because there is no previous hour.
        trend = np.zeros_like(p)
        trend[1:] = p[1:] - p[:-1]

        # Normalize the trend by trend_scale so k has a consistent meaning.
        # np.clip prevents extreme adjustments from unusual single-hour spikes.
        adj = k * np.clip(trend / max(trend_scale, 1e-6), -1.0, 1.0)

        # Subtract the adjustment: rising trend lowers the threshold (easier to alert),
        # falling trend raises the threshold (harder to alert).
        thr = np.clip(base_threshold - adj, 0.01, 0.99)

        # Fire when the probability exceeds this patient's adjusted threshold.
        raw = (p >= thr).astype(int)

        # Optional consecutive filter: require alert_k consecutive hours above threshold.
        if alert_k <= 1:
            preds[idx] = raw
            continue
        out = np.zeros_like(raw)
        run = 0
        for i, val in enumerate(raw):
            if val:
                run += 1
            else:
                run = 0
            if run >= alert_k:
                out[i] = 1
        preds[idx] = out
    return preds


def _early_warning_from_predictions(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    onset_hours: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict:
    # Same logic as early_warning_stats in utils.py but operates on pre-computed
    # binary predictions instead of probabilities plus a threshold.
    patients = np.unique(patient_ids)
    lead_times = []
    early_hits = 0
    sepsis_count = 0
    false_alerts = 0

    for pid in patients:
        mask = patient_ids == pid
        y = labels[mask]
        p = preds[mask]
        hrs = hours[mask]
        onset = onset_hours[mask][0]
        has_sepsis = np.any(y == 1)
        alert_idx = np.where(p == 1)[0]
        if len(alert_idx) == 0:
            if not has_sepsis:
                continue
            sepsis_count += 1
            continue
        first_alert_hour = hrs[alert_idx[0]]
        if has_sepsis:
            sepsis_count += 1
            if onset >= 0:
                lead = onset - first_alert_hour
                lead_times.append(float(lead))
                if lead >= 0:
                    early_hits += 1
        else:
            false_alerts += 1

    early_rate = early_hits / sepsis_count if sepsis_count else 0.0
    false_rate = false_alerts / max(len(patients) - sepsis_count, 1)
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    return {
        "early_detection_rate": float(early_rate),
        "false_alert_rate": float(false_rate),
        "median_lead_time_hours": median_lead,
    }


def _alert_burden_from_predictions(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
) -> dict:
    # Compute alert burden from binary predictions rather than probabilities.
    patients = np.unique(patient_ids)
    total_alerts = 0.0
    total_days = 0.0
    nonsepsis_alerts = 0.0
    nonsepsis_days = 0.0
    alerts_per_patient = []

    for pid in patients:
        mask = patient_ids == pid
        p = preds[mask]
        hrs = hours[mask]
        alerts = float(np.sum(p))
        duration_days = max(len(hrs), 1) / 24.0
        total_alerts += alerts
        total_days += duration_days
        alerts_per_patient.append(alerts)

        has_sepsis = np.any(labels[mask] == 1)
        if not has_sepsis:
            nonsepsis_alerts += alerts
            nonsepsis_days += duration_days

    return {
        "alerts_per_patient_day": float(total_alerts / total_days) if total_days else 0.0,
        "alerts_per_nonsepsis_patient_day": float(nonsepsis_alerts / nonsepsis_days) if nonsepsis_days else 0.0,
        "mean_alerts_per_patient": float(np.mean(alerts_per_patient)) if alerts_per_patient else 0.0,
    }


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
    parser.add_argument("--base-threshold", type=float, default=0.1)
    parser.add_argument("--k-grid", default="0.0,0.05,0.1,0.2,0.3,0.4,0.5")
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

    # Use a slice of training patients as a calibration set for both probability
    # calibration and for tuning the k parameter.
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

    if args.calibrate != "none":
        calibrator = CalibratedClassifierCV(model, cv="prefit", method=args.calibrate)
        calibrator.fit(X_cal, y_cal)
        p_cal = calibrator.predict_proba(X_cal)[:, 1]
        p_test = calibrator.predict_proba(X_test)[:, 1]
    else:
        p_cal = model.predict_proba(X_cal)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

    # Compute the typical magnitude of hourly risk changes across calibration patients.
    # The 95th percentile of absolute trend values becomes trend_scale.
    # Dividing by this scale means k=0.1 causes the same threshold adjustment across
    # all patients regardless of the absolute scale of their probability changes.
    cal_idx_by_pid = _build_index(patient_ids[cal_idx])
    cal_trends = []
    for idx in cal_idx_by_pid.values():
        p = p_cal[idx]
        if len(p) > 1:
            cal_trends.append(np.abs(np.diff(p)))
    trend_scale = float(np.quantile(np.concatenate(cal_trends) if cal_trends else np.array([1.0]), 0.95))

    # Grid-search k on the calibration set. We test each k value, compute the utility
    # of the resulting alert policy, and keep the k with the highest score.
    k_grid = [float(k.strip()) for k in args.k_grid.split(",") if k.strip()]
    k_results = []
    for k in k_grid:
        preds_cal = _apply_dynamic_policy(
            cal_idx_by_pid, p_cal, args.base_threshold, trend_scale, k, args.alert_k
        )
        util = _official_utility_from_predictions(patient_ids[cal_idx], y_cal, preds_cal)
        k_results.append({"k": k, "utility": util})

    best = max(k_results, key=lambda r: r["utility"])
    best_k = float(best["k"])

    # Apply the best k to the held-out test set.
    test_idx_by_pid = _build_index(pid_test)
    preds_test = _apply_dynamic_policy(test_idx_by_pid, p_test, args.base_threshold, trend_scale, best_k, args.alert_k)
    util_test = _official_utility_from_predictions(pid_test, y_test, preds_test)
    policy = _early_warning_from_predictions(pid_test, hours_test, onset_test, y_test, preds_test)
    burden = _alert_burden_from_predictions(pid_test, hours_test, y_test, preds_test)

    # Compare against a simple static threshold policy as a baseline.
    static_preds = (p_test >= args.base_threshold).astype(int)
    static_util = _official_utility_from_predictions(pid_test, y_test, static_preds)

    report = {
        "split_source": split_source,
        "trend_scale": trend_scale,
        "base_threshold": args.base_threshold,
        "best_k": best_k,
        "calibration": {
            "method": args.calibrate,
            "fraction": args.calibration_fraction,
            "max_patients": args.calibration_max_patients,
        },
        "k_sweep": k_results,
        "dynamic_policy": {
            "official_utility": util_test,
            "early_warning": policy,
            "alert_burden": burden,
        },
        "static_policy": {
            "official_utility": static_util,
        },
        "patient_level_metrics": compute_patient_level_metrics(pid_test, y_test, p_test),
        "basic_metrics": compute_basic_metrics(y_test, p_test),
    }

    save_json(output_dir / "dynamic_policy.json", report)

    # Show how utility on the calibration set changed as k was varied.
    # The best k sits at the peak of this curve.
    plt.figure(figsize=(5, 4))
    plt.plot([r["k"] for r in k_results], [r["utility"] for r in k_results], marker="o")
    plt.xlabel("Trend sensitivity (k)")
    plt.ylabel("Official utility (calibration set)")
    plt.title("Dynamic Threshold Tuning")
    plt.tight_layout()
    plt.savefig(output_dir / "dynamic_policy_k_sweep.png")
    plt.close()

    print(f"Saved dynamic policy analysis to {output_dir}")


if __name__ == "__main__":
    main()
