from __future__ import annotations

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
    n = len(scores_sorted)
    # count(scores >= s) = n - idx_left
    idx = np.searchsorted(scores_sorted, test_scores, side="left")
    return (n - idx + 1) / (n + 1)


def _conformal_sets(p: np.ndarray, cal_scores_pos: np.ndarray, cal_scores_neg: np.ndarray, alpha: float) -> np.ndarray:
    score_pos = 1.0 - p
    score_neg = p
    cal_pos_sorted = np.sort(cal_scores_pos)
    cal_neg_sorted = np.sort(cal_scores_neg)
    pval_pos = _pvals_from_scores(cal_pos_sorted, score_pos)
    pval_neg = _pvals_from_scores(cal_neg_sorted, score_neg)
    set_pos = pval_pos > alpha
    set_neg = pval_neg > alpha
    # 0 = {0}, 1 = {1}, 2 = {0,1}, 3 = empty
    sets = np.zeros_like(p, dtype=int)
    sets[set_pos & ~set_neg] = 1
    sets[~set_pos & set_neg] = 0
    sets[set_pos & set_neg] = 2
    sets[~set_pos & ~set_neg] = 3
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

    cal_scores_pos = 1.0 - p_cal[y_cal == 1]
    cal_scores_neg = p_cal[y_cal == 0]

    rows = []
    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    for alpha in alphas:
        sets = _conformal_sets(p_test, cal_scores_pos, cal_scores_neg, alpha)
        singleton = (sets == 0) | (sets == 1)
        singleton_rate = float(np.mean(singleton)) if len(singleton) else 0.0

        # Conservative alert: only alert when prediction set is {1}
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

    if rows:
        plt.figure(figsize=(5, 4))
        plt.plot([r["alpha"] for r in rows], [r["utility"] for r in rows], marker="o", label="Utility")
        plt.plot([r["alpha"] for r in rows], [r["alerts_per_patient_day"] for r in rows], marker="o", label="Alerts/day")
        plt.xlabel("Alpha (conformal error)")
        plt.ylabel("Metric")
        plt.title("Conformal Alert Tradeoff")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "conformal_alert.png")
        plt.close()

    print(f"Saved conformal alert analysis to {output_dir}")


if __name__ == "__main__":
    main()
