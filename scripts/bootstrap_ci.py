from __future__ import annotations

# Computes 95% confidence intervals for all key metrics using bootstrapping.
#
# Why do we need confidence intervals? A single number like AUROC=0.847 tells us
# the model's performance on this specific test set, but it does not tell us how
# stable that number is. If we had slightly different test patients, would it be 0.80?
# Or 0.89? Bootstrapping answers this by repeatedly re-sampling the test patients
# with replacement and recomputing all metrics each time.
#
# With 200 bootstrap resamples, we get 200 different AUROC values. The 2.5th percentile
# and 97.5th percentile of those 200 values form the 95% confidence interval.
# This answers: "how stable is our result if we had slightly different test patients?"
#
# Run: python scripts/bootstrap_ci.py
#      --data-dir data/train --weights outputs/utility/model.joblib
#      --medians outputs/utility/medians.json
#      --output outputs/bootstrap/ci.json --n-bootstrap 200

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

from sepsis_ews.data import build_dataset
from sepsis_ews.utils import (
    apply_alert_policy,
    compute_basic_metrics,
    compute_official_utility,
    early_warning_stats,
    alert_burden_stats,
    save_json,
)


def _apply_calibration(
    model,
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    medians: np.ndarray,
    scaler,
    method: str,
    fraction: float,
    max_patients: int,
):
    # Fit a calibration wrapper on a small subset of training patients.
    # Returns None if calibration is disabled or no patients are available.
    if method == "none":
        return None
    train_pids = np.unique(patient_ids)
    rng = np.random.default_rng(42)
    rng.shuffle(train_pids)
    # Use at most max_patients calibration patients to keep this step fast.
    n_cal = max(1, int(len(train_pids) * fraction))
    n_cal = min(n_cal, max_patients)
    cal_pids = set(train_pids[:n_cal])
    cal_idx = np.array([i for i, pid in enumerate(patient_ids) if pid in cal_pids], dtype=int)
    if len(cal_idx) == 0:
        return None
    X_cal = np.where(np.isnan(X[cal_idx]), medians, X[cal_idx])
    X_cal = scaler.transform(X_cal)
    y_cal = y[cal_idx]
    # cv="prefit" means: do not retrain the model, only fit the sigmoid on top.
    calibrator = CalibratedClassifierCV(model, cv="prefit", method=method)
    calibrator.fit(X_cal, y_cal)
    return calibrator


def bootstrap_ci(values: list[float]) -> dict:
    # Given a list of metric values from all bootstrap resamples, return the mean
    # and the 95% confidence interval bounds (2.5th and 97.5th percentiles).
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--medians", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint JSON path for resume.")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fixed-threshold", type=float, default=None)
    parser.add_argument("--max-patients", type=int, default=5000)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--alert-k", type=int, default=1)
    parser.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none")
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--calibration-max-patients", type=int, default=200)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    args = parser.parse_args()

    X, y, patient_ids, hours, onset_hours, _, _ = build_dataset(
        Path(args.data_dir), max_patients=args.max_patients, feature_set=args.feature_set
    )

    # Load the saved test patient list so we bootstrap over the exact same test set
    # that was used to evaluate the model during training.
    weights_dir = Path(args.weights).parent
    split_file = weights_dir / "test_patients.json"
    if split_file.exists():
        split = json.loads(split_file.read_text(encoding="utf-8"))
        test_patients = set(split.get("patient_ids", []))
        test_idx = np.array([i for i, pid in enumerate(patient_ids) if pid in test_patients], dtype=int)
        train_idx = np.array([i for i, pid in enumerate(patient_ids) if pid not in test_patients], dtype=int)
        split_source = "test_patients.json"
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=patient_ids))
        split_source = "group_shuffle_split"

    med = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)

    bundle = joblib.load(args.weights)
    model = bundle["model"]
    scaler = bundle["scaler"]

    X_test = np.where(np.isnan(X[test_idx]), medians, X[test_idx])
    X_test = scaler.transform(X_test)
    y_test = y[test_idx]
    pid_test = patient_ids[test_idx]
    hours_test = hours[test_idx]
    onset_test = onset_hours[test_idx]

    # Optionally fit a calibration layer on training patients.
    calibrator = _apply_calibration(
        model,
        X[train_idx],
        y[train_idx],
        patient_ids[train_idx],
        medians,
        scaler,
        args.calibrate,
        args.calibration_fraction,
        args.calibration_max_patients,
    )

    if calibrator is None:
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = calibrator.predict_proba(X_test)[:, 1]

    # Pre-build a lookup table: patient ID -> row indices in the test set.
    # This avoids re-scanning all rows on every bootstrap iteration.
    unique_pids = np.unique(pid_test)
    idx_by_pid = {pid: np.where(pid_test == pid)[0] for pid in unique_pids}

    # Containers that accumulate one value per bootstrap resample for each metric.
    metrics_samples = {
        "auroc": [],
        "auprc": [],
        "official_utility": [],
        "early_detection_rate": [],
        "false_alert_rate": [],
        "alerts_per_patient_day": [],
    }
    thresholds = np.linspace(0.1, 0.9, 33)

    # Support for checkpointing: if the run is interrupted, we can resume from the last
    # saved state rather than starting over from bootstrap iteration 0.
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    iteration_seeds: list[int]
    start_idx = 0
    if args.resume and checkpoint_path and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        iteration_seeds = checkpoint.get("iteration_seeds", [])
        metrics_samples = checkpoint.get("metrics_samples", metrics_samples)
        start_idx = checkpoint.get("completed", 0)
        if len(iteration_seeds) != args.n_bootstrap:
            raise ValueError("Checkpoint n_bootstrap does not match current run.")
    else:
        # Pre-generate all random seeds so the bootstrap is fully reproducible even if
        # we resume from a checkpoint midway through.
        rng = np.random.default_rng(args.seed)
        iteration_seeds = rng.integers(0, 2**32 - 1, size=args.n_bootstrap).tolist()
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "iteration_seeds": iteration_seeds,
                        "metrics_samples": metrics_samples,
                        "completed": start_idx,
                        "max_patients": args.max_patients,
                        "feature_set": args.feature_set,
                        "alert_k": args.alert_k,
                        "fixed_threshold": args.fixed_threshold,
                        "calibration": {
                            "method": args.calibrate,
                            "fraction": args.calibration_fraction,
                            "max_patients": args.calibration_max_patients,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    for i in range(start_idx, args.n_bootstrap):
        rng = np.random.default_rng(iteration_seeds[i])

        # Sample the patient list with replacement. Some patients appear multiple times;
        # some are left out entirely. This simulates having a slightly different cohort.
        sampled = rng.choice(unique_pids, size=len(unique_pids), replace=True)

        # Gather all the row indices for the sampled patients.
        # Append a suffix to the patient ID each time the same patient is sampled twice,
        # so patient-level metrics do not accidentally merge two copies of the same patient.
        idxs = []
        new_pids = []
        for j, pid in enumerate(sampled):
            idx = idx_by_pid[pid]
            idxs.append(idx)
            new_pids.append(np.array([f"{pid}_{j}"] * len(idx)))
        idxs = np.concatenate(idxs)
        new_pids = np.concatenate(new_pids)

        y_b = y_test[idxs]
        p_b = y_prob[idxs]
        h_b = hours_test[idxs]
        o_b = onset_test[idxs]

        metrics = compute_basic_metrics(y_b, p_b)
        if args.fixed_threshold is not None:
            best_thr = float(args.fixed_threshold)
        else:
            # Find the best threshold for this resample.
            best_thr = float(
                thresholds[
                    np.argmax(
                        [compute_official_utility(new_pids, y_b, p_b, float(t)) for t in thresholds]
                    )
                ]
            )
        policy = early_warning_stats(new_pids, h_b, o_b, y_b, p_b, float(best_thr), alert_k=args.alert_k)
        burden = alert_burden_stats(new_pids, h_b, y_b, p_b, float(best_thr), alert_k=args.alert_k)

        metrics_samples["auroc"].append(metrics["auroc"])
        metrics_samples["auprc"].append(metrics["auprc"])
        metrics_samples["official_utility"].append(compute_official_utility(new_pids, y_b, p_b, float(best_thr)))
        metrics_samples["early_detection_rate"].append(policy["early_detection_rate"])
        metrics_samples["false_alert_rate"].append(policy["false_alert_rate"])
        metrics_samples["alerts_per_patient_day"].append(burden["alerts_per_patient_day"])

        # Save progress periodically so a crash does not lose all completed work.
        if checkpoint_path and (i + 1) % args.checkpoint_every == 0:
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "iteration_seeds": iteration_seeds,
                        "metrics_samples": metrics_samples,
                        "completed": i + 1,
                        "max_patients": args.max_patients,
                        "feature_set": args.feature_set,
                        "alert_k": args.alert_k,
                        "fixed_threshold": args.fixed_threshold,
                        "calibration": {
                            "method": args.calibrate,
                            "fraction": args.calibration_fraction,
                            "max_patients": args.calibration_max_patients,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # Summarize: for each metric, compute the mean and 95% CI from all bootstrap values.
    summary = {k: bootstrap_ci(v) for k, v in metrics_samples.items()}
    summary["n_bootstrap"] = args.n_bootstrap
    summary["max_patients"] = args.max_patients
    summary["fixed_threshold"] = args.fixed_threshold
    summary["split_source"] = split_source
    summary["calibration"] = {
        "method": args.calibrate,
        "fraction": args.calibration_fraction,
        "max_patients": args.calibration_max_patients,
    }
    summary["completed"] = args.n_bootstrap

    output = Path(args.output)
    save_json(output, summary)
    print(f"Saved bootstrap CIs to {output}")


if __name__ == "__main__":
    main()
