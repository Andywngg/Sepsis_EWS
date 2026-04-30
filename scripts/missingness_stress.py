from __future__ import annotations

# Robustness test: how well does the model hold up when the data is even messier than usual?
#
# The PhysioNet dataset already has about 70% missing values. In real hospitals the data
# can be worse: equipment failures, late chart entry, or less frequent lab draws all
# increase missingness. This script tests whether the model degrades gracefully by
# ARTIFICIALLY dropping an additional 10%, 20%, or 30% of the observed measurements
# at random, then running the trained model on the degraded data.
#
# The key distinction: we only drop values that WERE present. We never add
# missingness where there was already a NaN. Structural columns like Age, Gender,
# and SepsisLabel are excluded from dropping so the labels remain intact.
#
# Run: python scripts/missingness_stress.py
#      --data-dir data/train --weights outputs/utility/model.joblib
#      --medians outputs/utility/medians.json --drop-rates 0,0.1,0.2,0.3
#      --output-dir outputs/missingness_stress

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from sepsis_ews.data import load_patient_df, build_features, compute_onset_hour, list_patient_files
from sepsis_ews.utils import (
    apply_alert_policy,
    compute_basic_metrics,
    compute_official_utility,
    compute_patient_level_metrics,
    early_warning_stats,
    alert_burden_stats,
    save_json,
)


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_rates(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v.strip()]


def apply_missingness(
    df: pd.DataFrame,
    drop_rate: float,
    exclude_cols: set[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    # Randomly replace a fraction of observed values with NaN.
    # exclude_cols are never touched (labels, demographics, etc.).
    if drop_rate <= 0:
        return df
    cols = [c for c in df.columns if c not in exclude_cols]
    out = df.copy()

    # observed is a boolean mask: True where a value is NOT NaN.
    # We only drop values that already exist (never change NaN to NaN).
    observed = out[cols].notna().values
    # Generate a random draw for every cell. If the draw is less than drop_rate,
    # that cell is a candidate for dropping. Combined with "observed", we only
    # actually drop cells that had a real value.
    drop_mask = rng.random(size=observed.shape) < drop_rate
    to_drop = drop_mask & observed
    values = out[cols].values
    values[to_drop] = np.nan
    out[cols] = values
    return out


def build_dataset_with_missingness(
    files: list[Path],
    drop_rate: float,
    feature_set: str,
    patient_normalize: bool,
    exclude_cols: set[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load all patients and apply the specified drop rate to each one.
    # A different random seed is used per patient to avoid correlated dropout patterns.
    all_x = []
    all_y = []
    all_pid = []
    all_hour = []
    all_onset = []
    rng = np.random.default_rng(seed)

    for path in files:
        df = load_patient_df(path)
        df = apply_missingness(df, drop_rate, exclude_cols, rng)
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
    parser.add_argument("--drop-rates", default="0,0.1,0.2,0.3")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--exclude-cols",
        default="SepsisLabel,Age,Gender,Unit1,Unit2,HospAdmTime,ICULOS",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    drop_rates = _parse_rates(args.drop_rates)

    # Restrict evaluation to the held-out test patients if a split file is provided.
    files = list_patient_files(data_dir)
    if args.test_patients:
        split = json.loads(Path(args.test_patients).read_text(encoding="utf-8"))
        test_set = set(split.get("patient_ids", []))
        files = [p for p in files if p.stem in test_set]
    if args.max_patients:
        files = files[: args.max_patients]

    summary_rows = []
    for rate in drop_rates:
        # Build the dataset with the specified amount of artificial missingness.
        # A different seed per rate ensures the random drops differ across runs.
        X, y, patient_ids, hours, onset_hours = build_dataset_with_missingness(
            files,
            drop_rate=rate,
            feature_set=args.feature_set,
            patient_normalize=args.patient_normalize,
            exclude_cols=exclude_cols,
            seed=args.seed + int(rate * 1000),
        )

        # Apply the same imputation and scaling as during training.
        X = np.where(np.isnan(X), medians, X)
        X = scaler.transform(X)
        y_prob = model.predict_proba(X)[:, 1]

        metrics = compute_basic_metrics(y, y_prob)
        patient_metrics = compute_patient_level_metrics(patient_ids, y, y_prob)
        brier = float(brier_score_loss(y, y_prob))
        policy = early_warning_stats(patient_ids, hours, onset_hours, y, y_prob, args.threshold, alert_k=1)
        alert_burden = alert_burden_stats(patient_ids, hours, y, y_prob, args.threshold, alert_k=1)
        official_util = compute_official_utility(patient_ids, y, y_prob, args.threshold, alert_k=1)

        row = {
            "drop_rate": float(rate),
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
        save_json(output_dir / f"missingness_{rate:.2f}.json", row)
        summary_rows.append(row)

    save_json(output_dir / "missingness_summary.json", {"rows": summary_rows})
    print(f"Saved missingness stress results to {output_dir}")


if __name__ == "__main__":
    main()
