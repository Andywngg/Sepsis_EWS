from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from sepsis_ews.data import list_patient_files, load_patient_df, build_features, compute_onset_hour
from sepsis_ews.utils import (
    compute_basic_metrics,
    compute_patient_level_metrics,
    compute_official_utility,
    early_warning_stats,
    alert_burden_stats,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--medians", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-patients", type=int, default=5000)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--patient-normalize", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--alert-k", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_patient_files(data_dir)

    split_file = Path(args.weights).parent / "test_patients.json"
    split_source = "group_shuffle_split"
    if split_file.exists():
        split = json.loads(split_file.read_text(encoding="utf-8"))
        test_patients = set(split.get("patient_ids", []))
        files = [p for p in files if p.stem in test_patients]
        split_source = "test_patients.json"
    if args.max_patients:
        files = files[: args.max_patients]

    med = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)

    bundle = joblib.load(args.weights)
    model = bundle["model"]
    scaler = bundle["scaler"]

    all_y = []
    all_p = []
    all_pid = []
    all_hours = []
    all_onset = []

    for path in files:
        df = load_patient_df(path)
        labels = df["SepsisLabel"].values.astype(int)
        feats, _ = build_features(df, feature_set=args.feature_set, patient_normalize=args.patient_normalize)
        X = np.where(np.isnan(feats), medians, feats)
        X = scaler.transform(X)
        probs = model.predict_proba(X)[:, 1]

        all_y.append(labels)
        all_p.append(probs)
        all_pid.append(np.array([path.stem] * len(labels)))
        all_hours.append(np.arange(len(labels)))
        onset = compute_onset_hour(labels)
        onset_val = -1 if onset is None else onset
        all_onset.append(np.array([onset_val] * len(labels)))

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    pid = np.concatenate(all_pid)
    hours = np.concatenate(all_hours)
    onset_hours = np.concatenate(all_onset)

    metrics = compute_basic_metrics(y, p)
    patient_metrics = compute_patient_level_metrics(pid, y, p)
    official_util = compute_official_utility(pid, y, p, args.threshold, alert_k=args.alert_k)
    policy = early_warning_stats(pid, hours, onset_hours, y, p, args.threshold, alert_k=args.alert_k)
    burden = alert_burden_stats(pid, hours, y, p, args.threshold, alert_k=args.alert_k)

    report = {
        "metrics": metrics,
        "patient_level_metrics": patient_metrics,
        "official_utility": float(official_util),
        "early_warning": policy,
        "alert_burden": burden,
        "threshold": float(args.threshold),
        "alert_k": int(args.alert_k),
        "feature_set": args.feature_set,
        "max_patients": args.max_patients,
        "patient_normalize": bool(args.patient_normalize),
        "note": "Prospective simulation using causal features (no future leakage).",
        "split_source": split_source,
    }
    save_json(output_dir / "prospective_eval.json", report)
    print(f"Saved prospective eval to {output_dir}")


if __name__ == "__main__":
    main()
