from __future__ import annotations

# PURPOSE: Load a saved model and run full evaluation on the held-out test set.
# KEY DIFFERENCE FROM train.py: uses the exact test patient IDs saved during training
#                               to guarantee the same split is always evaluated.
# RUN: python -m sepsis_ews.eval --data-dir data/train --weights outputs/utility/model.joblib
#                                --medians outputs/utility/medians.json
#                                --feature-set enhanced --utility official
#                                --calibrate sigmoid --output-dir outputs/eval

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from .data import build_dataset
from .utils import (
    apply_alert_policy,
    compute_accuracy_f_measure,
    compute_basic_metrics,
    compute_official_utility,
    compute_utility,
    compute_patient_level_metrics,
    early_warning_stats,
    lead_time_distribution,
    alert_burden_stats,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",                required=True)
    parser.add_argument("--weights",                 required=True)
    parser.add_argument("--medians",                 required=True)
    parser.add_argument("--output-dir",              required=True)
    parser.add_argument("--max-patients",            type=int,   default=None)
    parser.add_argument("--feature-set",             choices=["basic", "enhanced"], default="basic")
    parser.add_argument("--patient-normalize",       action="store_true")
    parser.add_argument("--quality-report",          action="store_true")
    parser.add_argument("--quality-percentiles",     default="0:50:5")
    parser.add_argument("--utility",                 choices=["official", "custom"], default="official")
    parser.add_argument("--alert-k",                 type=int,   default=1)
    parser.add_argument("--calibrate",               choices=["none", "sigmoid", "isotonic"], default="none")
    parser.add_argument("--calibration-fraction",    type=float, default=0.1)
    parser.add_argument("--calibration-max-patients",type=int,   default=200)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- STEP 1: LOAD ALL DATA ---
    X, y, patient_ids, hours, onset_hours, quality, _ = build_dataset(
        data_dir,
        max_patients=args.max_patients,
        feature_set=args.feature_set,
        patient_normalize=args.patient_normalize,
    )

    # --- STEP 2: REPRODUCE EXACT SAME TEST SPLIT FROM TRAINING ---
    # test_patients.json was saved by train.py — these are the held-out patient IDs.
    # This guarantees the model is always evaluated on patients it never trained on.
    split_file = Path(args.weights).parent / "test_patients.json"
    if split_file.exists():
        split         = json.loads(split_file.read_text(encoding="utf-8"))
        test_patients = set(split.get("patient_ids", []))
        test_idx      = np.array([i for i, pid in enumerate(patient_ids) if pid in test_patients], dtype=int)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, test_idx = next(splitter.split(X, y, groups=patient_ids))

    X_test, y_test         = X[test_idx], y[test_idx]
    hours_test, onset_test = hours[test_idx], onset_hours[test_idx]
    pid_test               = patient_ids[test_idx]

    # --- STEP 3: CALIBRATION PATIENT SELECTION ---
    # Select a small subset of TRAINING patients to fit the calibration layer.
    # These patients are separate from test — never used to train the main model.
    calibration_idx, calibration_pids = None, []
    if args.calibrate != "none":
        if split_file.exists():
            train_idx      = np.array([i for i, pid in enumerate(patient_ids) if pid not in test_patients], dtype=int)
            train_pids     = np.unique(patient_ids[train_idx])
            rng            = np.random.default_rng(42)
            rng.shuffle(train_pids)
            n_cal          = max(1, int(len(train_pids) * args.calibration_fraction))
            n_cal          = min(n_cal, args.calibration_max_patients)
            calibration_pids = train_pids[:n_cal]
            calibration_idx  = np.array([i for i, pid in enumerate(patient_ids) if pid in calibration_pids], dtype=int)
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=args.calibration_fraction, random_state=43)
            _, calibration_idx = next(splitter.split(X, y, groups=patient_ids))
            calibration_pids   = np.unique(patient_ids[calibration_idx])

    # --- STEP 4: IMPUTATION AND SCALING ---
    # Use medians saved from training — same values applied here to avoid leakage.
    med     = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_test  = np.where(np.isnan(X_test), medians, X_test)

    bundle = joblib.load(args.weights)
    model  = bundle["model"]
    scaler = bundle["scaler"]
    X_test = scaler.transform(X_test)  # transform only — do NOT re-fit on test data

    # --- STEP 5: RAW PROBABILITY PREDICTIONS ---
    y_prob_raw = model.predict_proba(X_test)[:, 1]

    # --- STEP 6: OPTIONAL CALIBRATION (Platt scaling / isotonic regression) ---
    # Aligns predicted probabilities with true outcome frequencies.
    # Improves Brier score; uses a small held-out calibration set.
    calibration_info = {"method": "none", "patients": 0, "fraction": float(args.calibration_fraction), "max_patients": int(args.calibration_max_patients)}
    y_prob = y_prob_raw
    if args.calibrate != "none" and calibration_idx is not None and len(calibration_idx) > 0:
        X_cal = np.where(np.isnan(X[calibration_idx]), medians, X[calibration_idx])
        X_cal = scaler.transform(X_cal)
        y_cal = y[calibration_idx]
        # cv="prefit" = don't retrain the model, just fit a sigmoid on top
        calibrator = CalibratedClassifierCV(model, cv="prefit", method=args.calibrate)
        calibrator.fit(X_cal, y_cal)
        y_prob = calibrator.predict_proba(X_test)[:, 1]
        calibration_info = {"method": args.calibrate, "patients": int(len(np.unique(calibration_pids))), "fraction": float(args.calibration_fraction), "max_patients": int(args.calibration_max_patients)}

    # --- STEP 7: COMPUTE ALL METRICS ---
    metrics         = compute_basic_metrics(y_test, y_prob)
    patient_metrics = compute_patient_level_metrics(pid_test, y_test, y_prob)
    brier           = float(brier_score_loss(y_test, y_prob))
    brier_raw       = float(brier_score_loss(y_test, y_prob_raw))

    # Grid search for best threshold (maximizes official utility)
    thresholds = np.linspace(0.1, 0.9, 33)
    utilities  = []
    for thr in thresholds:
        if args.utility == "official":
            util = compute_official_utility(pid_test, y_test, y_prob, float(thr), alert_k=args.alert_k)
        else:
            util = compute_utility(pid_test, hours_test, onset_test, y_test, y_prob, float(thr), alert_k=args.alert_k)
        utilities.append(util)
    best_idx  = int(np.argmax(utilities))
    best_thr  = float(thresholds[best_idx])

    policy       = early_warning_stats(pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
    preds        = np.zeros_like(y_test)
    for pid in np.unique(pid_test):
        mask = pid_test == pid
        preds[mask] = apply_alert_policy(y_prob[mask], best_thr, alert_k=args.alert_k)
    accuracy, f_measure = compute_accuracy_f_measure(y_test, preds)
    official_util       = compute_official_utility(pid_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
    custom_util         = compute_utility(pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
    alert_burden        = alert_burden_stats(pid_test, hours_test, y_test, y_prob, best_thr, alert_k=args.alert_k)

    report = {
        "metrics":              metrics,
        "patient_level_metrics":patient_metrics,
        "brier_score":          brier,
        "brier_score_raw":      brier_raw,
        "best_threshold":       best_thr,
        "utility_score":        float(utilities[best_idx]),
        "utility_kind":         args.utility,
        "alert_k":              int(args.alert_k),
        "official_utility":     float(official_util),
        "custom_utility":       float(custom_util),
        "accuracy":             float(accuracy),
        "f_measure":            float(f_measure),
        "early_warning":        policy,
        "alert_burden":         alert_burden,
        "feature_set":          args.feature_set,
        "max_patients":         args.max_patients,
        "patient_normalize":    bool(args.patient_normalize),
        "calibration":          calibration_info,
    }
    save_json(output_dir / "metrics.json", report)

    # --- STEP 8: GENERATE VISUALIZATIONS ---

    # Plot 1: Utility score vs threshold — shows where peak utility occurs
    plt.figure(figsize=(5, 4))
    plt.plot(thresholds, utilities, marker="o")
    plt.xlabel("Alert threshold")
    plt.ylabel("Utility score")
    plt.title("Utility vs Threshold")
    plt.tight_layout()
    plt.savefig(output_dir / "utility_curve.png")
    plt.close()

    # Plot 2: Calibration curve — predicted probability vs actual sepsis frequency
    # Perfect calibration = diagonal line. Points above = underestimates risk.
    frac_pos,     mean_pred     = calibration_curve(y_test, y_prob,     n_bins=10, strategy="quantile")
    frac_pos_raw, mean_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 4))
    plt.plot(mean_pred_raw, frac_pos_raw, marker="o", label="Uncalibrated")
    plt.plot(mean_pred, frac_pos, marker="o", label="Calibrated" if calibration_info["method"] != "none" else "Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png")
    plt.close()

    # Plot 3: Lead time histogram — distribution of hours of warning given to clinicians
    lead_times = lead_time_distribution(pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=args.alert_k)
    if lead_times.size:
        plt.figure(figsize=(5, 4))
        plt.hist(lead_times, bins=20, color="#4c78a8")
        plt.xlabel("Lead time (hours)")
        plt.ylabel("Count")
        plt.title("Lead Time Distribution (Sepsis Patients)")
        plt.tight_layout()
        plt.savefig(output_dir / "lead_time_hist.png")
        plt.close()

    if args.quality_report:
        quality_test = quality[test_idx]
        percentiles  = _parse_percentiles(args.quality_percentiles)
        rows = []
        for p in percentiles:
            cutoff = np.percentile(quality_test, p)
            mask   = quality_test >= cutoff
            if not np.any(mask):
                continue
            q_metrics = compute_basic_metrics(y_test[mask], y_prob[mask])
            q_policy  = early_warning_stats(pid_test[mask], hours_test[mask], onset_test[mask], y_test[mask], y_prob[mask], best_thr, alert_k=args.alert_k)
            rows.append({"drop_pct": int(p), "coverage": float(mask.mean()), **q_metrics, **q_policy})
        save_json(output_dir / "quality_tradeoff.json", {"tradeoff": rows})
        _plot_quality_tradeoff(rows, output_dir / "quality_tradeoff.png")

    print(f"Saved evaluation to {output_dir}")


def _parse_percentiles(value: str) -> list[int]:
    if "," in value:
        return [int(v) for v in value.split(",")]
    if ":" in value:
        start, end, step = value.split(":")
        return list(range(int(start), int(end) + 1, int(step)))
    return [int(value)]


def _plot_quality_tradeoff(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    coverages = [r["coverage"] for r in rows]
    aurocs    = [r.get("auroc", 0.0) for r in rows]
    early     = [r.get("early_detection_rate", 0.0) for r in rows]
    plt.figure(figsize=(5, 4))
    plt.plot(coverages, aurocs, marker="o", label="AUROC")
    plt.plot(coverages, early, marker="o", label="Early detection rate")
    plt.xlabel("Coverage (fraction kept)")
    plt.ylabel("Metric")
    plt.title("Quality Gating Tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
