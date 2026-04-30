from __future__ import annotations

# PURPOSE: Robustness test for lab result reporting delays.
# PROBLEM: In real hospitals, lab results often don't appear in the EHR immediately.
#          A blood draw at hour 5 might not show up until hour 7 due to processing time.
#          This test checks whether the model still works if all lab values are shifted
#          1, 2, or 3 hours later (simulating reporting delay).
# METHOD:  For each delay level, use pd.DataFrame.shift(delay) on all clinical variables
#          (excluding static fields like Age and SepsisLabel). Run the trained model.
# OUTPUT:  delay_00.json, delay_01.json, ... and delay_summary.json
# RUN:     python scripts/measurement_delay_stress.py
#              --data-dir data/train --weights outputs/utility/model.joblib
#              --medians outputs/utility/medians.json --delays 0,1,2,3
#              --output-dir outputs/delay_stress

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from sepsis_ews.data import (
    list_patient_files,
    load_patient_df,
    build_features,
    compute_onset_hour,
)
from sepsis_ews.utils import (
    compute_basic_metrics,
    compute_patient_level_metrics,
    compute_official_utility,
    early_warning_stats,
    alert_burden_stats,
    save_json,
)


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_delays(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def apply_delay(df: pd.DataFrame, delay_hours: int, exclude_cols: set[str]) -> pd.DataFrame:
    if delay_hours <= 0:
        return df
    out = df.copy()
    cols = [c for c in out.columns if c not in exclude_cols]
    for c in cols:
        out[c] = out[c].shift(delay_hours)
    return out


def build_dataset_with_delay(
    files: list[Path],
    delay_hours: int,
    feature_set: str,
    patient_normalize: bool,
    exclude_cols: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_x = []
    all_y = []
    all_pid = []
    all_hour = []
    all_onset = []

    for path in files:
        df = load_patient_df(path)
        df = apply_delay(df, delay_hours, exclude_cols)
        labels = df["SepsisLabel"].values.astype(int)
        feats, _ = build_features(df, feature_set=feature_set, patient_normalize=patient_normalize)
        hours = np.arange(len(df), dtype=int)
        onset = compute_onset_hour(labels)
        onset_val = -1 if onset is None else onset
        all_x.append(feats)
        all_y.append(labels)
        all_pid.append(np.array([path.stem] * len(labels)))
        all_hour.append(hours)
        all_onset.append(np.array([onset_val] * len(labels)))

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    patient_ids = np.concatenate(all_pid, axis=0)
    hours = np.concatenate(all_hour, axis=0)
    onset_hours = np.concatenate(all_onset, axis=0)
    return X, y, patient_ids, hours, onset_hours


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--medians", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-patients", default="")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--feature-set", choices=["basic", "enhanced"], default="enhanced")
    parser.add_argument("--patient-normalize", action="store_true")
    parser.add_argument("--delays", default="0,1,2,3")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--exclude-cols",
        default="SepsisLabel,Age,Gender,Unit1,Unit2,HospAdmTime,ICULOS",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    med = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)

    bundle = joblib.load(args.weights)
    model = bundle["model"]
    scaler = bundle["scaler"]

    exclude_cols = set(_parse_list(args.exclude_cols))
    delays = _parse_delays(args.delays)

    files = list_patient_files(data_dir)
    if args.test_patients:
        split = json.loads(Path(args.test_patients).read_text(encoding="utf-8"))
        test_set = set(split.get("patient_ids", []))
        files = [p for p in files if p.stem in test_set]
    if args.max_patients:
        files = files[: args.max_patients]

    summary_rows = []
    for delay in delays:
        X, y, patient_ids, hours, onset_hours = build_dataset_with_delay(
            files,
            delay_hours=delay,
            feature_set=args.feature_set,
            patient_normalize=args.patient_normalize,
            exclude_cols=exclude_cols,
        )
        X = np.where(np.isnan(X), medians, X)
        X = scaler.transform(X)
        y_prob = model.predict_proba(X)[:, 1]

        metrics = compute_basic_metrics(y, y_prob)
        patient_metrics = compute_patient_level_metrics(patient_ids, y, y_prob)
        brier = float(brier_score_loss(y, y_prob))
        policy = early_warning_stats(
            patient_ids, hours, onset_hours, y, y_prob, args.threshold, alert_k=1
        )
        alert_burden = alert_burden_stats(
            patient_ids, hours, y, y_prob, args.threshold, alert_k=1
        )
        official_util = compute_official_utility(patient_ids, y, y_prob, args.threshold, alert_k=1)

        row = {
            "delay_hours": int(delay),
            "metrics": metrics,
            "patient_level_metrics": patient_metrics,
            "brier_score": brier,
            "official_utility": float(official_util),
            "early_warning": policy,
            "alert_burden": alert_burden,
            "threshold": float(args.threshold),
            "feature_set": args.feature_set,
            "patient_normalize": bool(args.patient_normalize),
            "patients": int(len(np.unique(patient_ids))),
        }
        save_json(output_dir / f"delay_{delay:02d}.json", row)
        summary_rows.append(row)

    save_json(output_dir / "delay_summary.json", {"rows": summary_rows})
    print(f"Saved delay stress results to {output_dir}")


if __name__ == "__main__":
    main()
