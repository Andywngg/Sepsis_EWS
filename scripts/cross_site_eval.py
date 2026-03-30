from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from sepsis_ews.data import build_dataset, build_dataset_from_files, list_patient_files
from sepsis_ews.train import compute_sample_weights
from sepsis_ews.utils import (
    apply_alert_policy,
    compute_accuracy_f_measure,
    compute_basic_metrics,
    compute_official_utility,
    compute_patient_level_metrics,
    compute_utility,
    early_warning_stats,
    save_json,
    select_threshold_by_utility,
)


def _fit_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray):
    if model_name == "logreg":
        model = LogisticRegression(max_iter=200, n_jobs=1)
    else:
        model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def _prepare_X(X_train: np.ndarray, X_test: np.ndarray):
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, medians, scaler


def _evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pid_test: np.ndarray,
    hours_test: np.ndarray,
    onset_test: np.ndarray,
    utility_kind: str,
    alert_k: int,
    calibrator: CalibratedClassifierCV | None = None,
):
    if calibrator is None:
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = calibrator.predict_proba(X_test)[:, 1]
    metrics = compute_basic_metrics(y_test, y_prob)
    patient_metrics = compute_patient_level_metrics(pid_test, y_test, y_prob)
    thresholds = np.linspace(0.1, 0.9, 33)
    best_thr, best_util = select_threshold_by_utility(
        pid_test,
        hours_test,
        onset_test,
        y_test,
        y_prob,
        thresholds,
        utility_kind=utility_kind,
        alert_k=alert_k,
    )
    policy = early_warning_stats(pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=alert_k)

    preds = np.zeros_like(y_test)
    for pid in np.unique(pid_test):
        mask = pid_test == pid
        preds[mask] = apply_alert_policy(y_prob[mask], best_thr, alert_k=alert_k)
    accuracy, f_measure = compute_accuracy_f_measure(y_test, preds)

    official_util = compute_official_utility(pid_test, y_test, y_prob, best_thr, alert_k=alert_k)
    custom_util = compute_utility(pid_test, hours_test, onset_test, y_test, y_prob, best_thr, alert_k=alert_k)

    return {
        "metrics": metrics,
        "patient_level_metrics": patient_metrics,
        "best_threshold": best_thr,
        "utility_score": best_util,
        "utility_kind": utility_kind,
        "alert_k": int(alert_k),
        "official_utility": float(official_util),
        "custom_utility": float(custom_util),
        "accuracy": float(accuracy),
        "f_measure": float(f_measure),
        "early_warning": policy,
        "test_patients": int(len(np.unique(pid_test))),
    }


def _run_target_eval(
    model,
    X_target: np.ndarray,
    y_target: np.ndarray,
    pid_target: np.ndarray,
    hours_target: np.ndarray,
    onset_target: np.ndarray,
    utility_kind: str,
    alert_k: int,
    calibrate_target: str,
    calibration_fraction: float,
    calibration_max_patients: int,
):
    if calibrate_target == "none":
        report = _evaluate(
            model,
            X_target,
            y_target,
            pid_target,
            hours_target,
            onset_target,
            utility_kind=utility_kind,
            alert_k=alert_k,
            calibrator=None,
        )
        report["calibration"] = {"method": "none", "patients": 0}
        return report

    target_pids = np.unique(pid_target)
    rng = np.random.default_rng(42)
    rng.shuffle(target_pids)
    n_cal = max(1, int(len(target_pids) * calibration_fraction))
    n_cal = min(n_cal, calibration_max_patients)
    cal_pids = set(target_pids[:n_cal])
    cal_idx = np.array([i for i, pid in enumerate(pid_target) if pid in cal_pids], dtype=int)
    eval_idx = np.array([i for i, pid in enumerate(pid_target) if pid not in cal_pids], dtype=int)

    if len(cal_idx) == 0 or len(eval_idx) == 0:
        report = _evaluate(
            model,
            X_target,
            y_target,
            pid_target,
            hours_target,
            onset_target,
            utility_kind=utility_kind,
            alert_k=alert_k,
            calibrator=None,
        )
        report["calibration"] = {"method": "none", "patients": 0}
        return report

    calibrator = CalibratedClassifierCV(model, cv="prefit", method=calibrate_target)
    calibrator.fit(X_target[cal_idx], y_target[cal_idx])

    report = _evaluate(
        model,
        X_target[eval_idx],
        y_target[eval_idx],
        pid_target[eval_idx],
        hours_target[eval_idx],
        onset_target[eval_idx],
        utility_kind=utility_kind,
        alert_k=alert_k,
        calibrator=calibrator,
    )
    report["calibration"] = {
        "method": calibrate_target,
        "patients": int(len(cal_pids)),
        "eval_patients": int(len(np.unique(pid_target[eval_idx]))),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir-a")
    parser.add_argument("--data-dir-b")
    parser.add_argument("--combined-dir", help="Use a single directory and split by sorted patient IDs.")
    parser.add_argument("--split-n", type=int, default=20336)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=["logreg", "hgb"], default="hgb")
    parser.add_argument("--utility-weighted", action="store_true")
    parser.add_argument("--utility", choices=["official", "custom"], default="official")
    parser.add_argument("--alert-k", type=int, default=1)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--patient-normalize", action="store_true")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--calibrate-target", choices=["none", "sigmoid", "isotonic"], default="none")
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-max-patients", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.combined_dir:
        files = list_patient_files(Path(args.combined_dir))
        files = sorted(files)
        a_files = files[: args.split_n]
        b_files = files[args.split_n :]
        if args.max_patients:
            a_files = a_files[: args.max_patients]
            b_files = b_files[: args.max_patients]
        Xa, ya, pida, ha, onsa, _, _ = build_dataset_from_files(
            a_files, feature_set=args.feature_set, patient_normalize=args.patient_normalize
        )
        Xb, yb, pidb, hb, onsb, _, _ = build_dataset_from_files(
            b_files, feature_set=args.feature_set, patient_normalize=args.patient_normalize
        )
    else:
        if not args.data_dir_a or not args.data_dir_b:
            raise ValueError("Provide --data-dir-a and --data-dir-b, or use --combined-dir.")
        Xa, ya, pida, ha, onsa, _, _ = build_dataset(
            Path(args.data_dir_a),
            max_patients=args.max_patients,
            feature_set=args.feature_set,
            patient_normalize=args.patient_normalize,
        )
        Xb, yb, pidb, hb, onsb, _, _ = build_dataset(
            Path(args.data_dir_b),
            max_patients=args.max_patients,
            feature_set=args.feature_set,
            patient_normalize=args.patient_normalize,
        )

    # Train on A, evaluate on B.
    Xa_train, Xb_test, _, _ = _prepare_X(Xa, Xb)
    weights_a = compute_sample_weights(ya, ha, onsa, args.utility_weighted)
    model_a = _fit_model(args.model, Xa_train, ya, weights_a)
    report_a_to_b = _run_target_eval(
        model_a,
        Xb_test,
        yb,
        pidb,
        hb,
        onsb,
        utility_kind=args.utility,
        alert_k=args.alert_k,
        calibrate_target=args.calibrate_target,
        calibration_fraction=args.calibration_fraction,
        calibration_max_patients=args.calibration_max_patients,
    )
    save_json(output_dir / "a_to_b_metrics.json", report_a_to_b)

    # Train on B, evaluate on A.
    Xb_train, Xa_test, _, _ = _prepare_X(Xb, Xa)
    weights_b = compute_sample_weights(yb, hb, onsb, args.utility_weighted)
    model_b = _fit_model(args.model, Xb_train, yb, weights_b)
    report_b_to_a = _run_target_eval(
        model_b,
        Xa_test,
        ya,
        pida,
        ha,
        onsa,
        utility_kind=args.utility,
        alert_k=args.alert_k,
        calibrate_target=args.calibrate_target,
        calibration_fraction=args.calibration_fraction,
        calibration_max_patients=args.calibration_max_patients,
    )
    save_json(output_dir / "b_to_a_metrics.json", report_b_to_a)

    summary = []
    summary.append("# Cross-Site Generalization\n\n")
    summary.append(f"- Model: {args.model}\n")
    summary.append(f"- Feature set: {args.feature_set}\n")
    summary.append(f"- Utility: {args.utility}\n")
    summary.append(f"- Alert k: {args.alert_k}\n")
    summary.append(f"- Patient normalization: {args.patient_normalize}\n")
    if args.max_patients:
        summary.append(f"- Max patients per site: {args.max_patients}\n")
    summary.append(f"- Target calibration: {args.calibrate_target}\n")
    if args.calibrate_target != "none":
        summary.append(f"- Calibration fraction: {args.calibration_fraction}\n")
        summary.append(f"- Calibration max patients: {args.calibration_max_patients}\n")
    summary.append("\n## Train A -> Test B\n")
    summary.append(json.dumps(report_a_to_b, indent=2))
    summary.append("\n\n## Train B -> Test A\n")
    summary.append(json.dumps(report_b_to_a, indent=2))
    (output_dir / "cross_site_summary.md").write_text("".join(summary), encoding="utf-8")

    print(f"Saved cross-site reports to {output_dir}")


if __name__ == "__main__":
    main()
