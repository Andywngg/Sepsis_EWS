from __future__ import annotations

# Fairness and equity analysis: does the model perform equally well across different
# groups of patients, or is it better for some groups than others?
#
# A model that looks good on average can still be harmful if it performs poorly for
# a specific subgroup. For example: elderly patients deteriorate faster, giving less
# time between the model's optimal alert window and actual sepsis onset. Or one ICU
# unit type might draw labs less frequently, creating sparser feature vectors.
#
# This script segments test patients by:
#   age bucket: under 40, 40-59, 60-79, 80 and over
#   gender:     encoded as 0 or 1 in the dataset
#   ICU unit:   Unit1 or Unit2 (different ward types in the PhysioNet dataset)
#
# For each subgroup it computes AUROC, utility, early detection rate, and alert burden.
#
# Run: python scripts/subgroup_analysis.py
#      --data-dir data/train --weights outputs/utility/model.joblib
#      --medians outputs/utility/medians.json --threshold 0.1
#      --output-dir outputs/subgroup

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


def age_bucket(age: float | int | None) -> str:
    # Convert a continuous age value to a discrete group label.
    if age is None or np.isnan(age):
        return "age_unknown"
    if age < 40:
        return "age_lt40"
    if age < 60:
        return "age_40_59"
    if age < 80:
        return "age_60_79"
    return "age_ge80"


def gender_bucket(gender: float | int | None) -> str:
    if gender is None or np.isnan(gender):
        return "gender_unknown"
    return f"gender_{int(gender)}"


def unit_bucket(unit1: float | int | None, unit2: float | int | None) -> str:
    # The PhysioNet dataset encodes ICU unit type as two binary columns.
    # A patient is in unit1 if the Unit1 column is 1, and in unit2 if Unit2 is 1.
    if unit1 == 1:
        return "unit1"
    if unit2 == 1:
        return "unit2"
    return "unit_unknown"


def compute_group_metrics(
    pid: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    hours: np.ndarray,
    onset_hours: np.ndarray,
    threshold: float,
    alert_k: int,
) -> dict:
    # Compute all performance metrics for a single subgroup.
    # This is the same set of metrics as the overall evaluation, just applied to a subset.
    metrics = compute_basic_metrics(y, p)
    patient_metrics = compute_patient_level_metrics(pid, y, p)
    official_util = compute_official_utility(pid, y, p, threshold, alert_k=alert_k)
    policy = early_warning_stats(pid, hours, onset_hours, y, p, threshold, alert_k=alert_k)
    burden = alert_burden_stats(pid, hours, y, p, threshold, alert_k=alert_k)
    return {
        "metrics": metrics,
        "patient_level_metrics": patient_metrics,
        "official_utility": float(official_util),
        "early_warning": policy,
        "alert_burden": burden,
        "patients": int(len(np.unique(pid))),
        "rows": int(len(y)),
    }


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

    # Load only the held-out test patient files so we never evaluate on training patients.
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
    patient_groups = {}  # maps patient_id -> {age: ..., gender: ..., unit: ...}

    for path in files:
        df = load_patient_df(path)
        labels = df["SepsisLabel"].values.astype(int)

        # Run the same preprocessing pipeline used at training time.
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

        # Read demographic columns from the first row of this patient's file.
        # Use np.nan as the default if the column is missing from this dataset.
        age = df["Age"].iloc[0] if "Age" in df.columns else np.nan
        gender = df["Gender"].iloc[0] if "Gender" in df.columns else np.nan
        unit1 = df["Unit1"].iloc[0] if "Unit1" in df.columns else np.nan
        unit2 = df["Unit2"].iloc[0] if "Unit2" in df.columns else np.nan
        patient_groups[path.stem] = {
            "age": age_bucket(age),
            "gender": gender_bucket(gender),
            "unit": unit_bucket(unit1, unit2),
        }

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    pid = np.concatenate(all_pid)
    hours = np.concatenate(all_hours)
    onset_hours = np.concatenate(all_onset)

    # Compute metrics on the full test set as the baseline for comparison.
    overall = compute_group_metrics(pid, y, p, hours, onset_hours, args.threshold, args.alert_k)

    def subset_for(group_key: str, group_value: str) -> dict:
        # Select only the rows belonging to patients in this subgroup, then compute metrics.
        keep = np.array([patient_groups[patient_id][group_key] == group_value for patient_id in pid])
        return compute_group_metrics(
            pid[keep], y[keep], p[keep], hours[keep], onset_hours[keep], args.threshold, args.alert_k
        )

    # Compute metrics for every distinct value of each grouping variable.
    groups = {"age": {}, "gender": {}, "unit": {}}
    for g in ("age", "gender", "unit"):
        values = sorted({info[g] for info in patient_groups.values()})
        for v in values:
            groups[g][v] = subset_for(g, v)

    save_json(
        output_dir / "subgroup_analysis.json",
        {
            "overall": overall,
            "groups": groups,
            "threshold": float(args.threshold),
            "alert_k": int(args.alert_k),
            "feature_set": args.feature_set,
            "max_patients": args.max_patients,
            "patient_normalize": bool(args.patient_normalize),
            "split_source": split_source,
        },
    )
    print(f"Saved subgroup analysis to {output_dir}")


if __name__ == "__main__":
    main()
