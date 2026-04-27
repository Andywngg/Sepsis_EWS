from __future__ import annotations

# PURPOSE: Load raw patient .psv files and engineer features for the model.
# ENTRY POINT: build_dataset() loads all patients into one stacked matrix.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# Container holding everything about one patient
@dataclass
class PatientSeries:
    patient_id: str
    features: np.ndarray   # shape: (num_hours, num_features)
    labels: np.ndarray     # shape: (num_hours,)  0=no sepsis, 1=sepsis
    hours: np.ndarray      # [0, 1, 2, ...] — ICU hour index
    onset_hour: int | None # first hour where label=1, or None if never septic


def list_patient_files(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.psv")])


def load_patient_df(path: Path) -> pd.DataFrame:
    # PSV = pipe-separated values; each row = one hour, each col = one variable
    return pd.read_csv(path, sep="|")


def compute_onset_hour(labels: np.ndarray) -> int | None:
    # Find the first hour where SepsisLabel=1 (sepsis onset)
    idx = np.where(labels == 1)[0]
    return int(idx[0]) if len(idx) else None


def _time_since_last_observed(series: pd.Series) -> np.ndarray:
    # Per-variable: how many hours since this measurement was last taken?
    # Captures irregular lab-draw patterns as a feature.
    last_seen = -1
    out = np.zeros(len(series), dtype=np.float32)
    for i, val in enumerate(series.values):
        if pd.notna(val):
            last_seen = i
            out[i] = 0.0
        else:
            out[i] = float(i - last_seen) if last_seen >= 0 else float(i + 1)
    return out


def compute_quality(df: pd.DataFrame) -> np.ndarray:
    # Per-hour quality score: fraction of non-missing values (1.0 = all present)
    cols = [c for c in df.columns if c != "SepsisLabel"]
    missing_rate = df[cols].isna().mean(axis=1).values.astype(np.float32)
    return 1.0 - missing_rate


def build_features(
    df: pd.DataFrame, feature_set: str = "basic", patient_normalize: bool = False
) -> Tuple[np.ndarray, List[str]]:
    # -----------------------------------------------------------------------
    # FEATURE ENGINEERING — transforms raw measurements into model inputs
    #
    # Basic (always):
    #   raw       — original measurements + hour index
    #   delta     — change from previous hour (trend / rate of change)
    #   missing   — binary flag per variable: 1=missing, 0=present
    #
    # Enhanced (--feature-set enhanced):
    #   rollmean3/6 — rolling mean over 3 and 6 hour windows
    #   rollstd3/6  — rolling std (instability signal)
    #   tslo        — time since last observed per variable
    # -----------------------------------------------------------------------

    feature_cols = [c for c in df.columns if c != "SepsisLabel"]
    raw = df[feature_cols].copy()

    # Optional: z-score each variable relative to THIS patient's own baseline
    # Reduces domain shift when the model is tested on a different hospital
    if patient_normalize:
        for c in raw.columns:
            mean = raw[c].mean(skipna=True)
            std = raw[c].std(skipna=True)
            if pd.isna(std) or std == 0:
                std = 1.0
            raw[c] = (raw[c] - mean) / std

    raw.insert(0, "Hour", np.arange(len(raw)))

    # Delta: how much did each variable change in the last hour?
    delta = raw.diff().fillna(0.0)
    delta.columns = [f"{c}_delta" for c in delta.columns]

    # Missingness indicators: the pattern of which labs are absent is informative
    missing = raw.isna().astype(int)
    missing.columns = [f"{c}_miss" for c in missing.columns]

    feat_frames = [raw, delta, missing]

    if feature_set == "enhanced":
        # Forward-fill within patient timeline (past→future only, no leakage)
        raw_filled = raw.ffill().fillna(raw.median(numeric_only=True))
        for window in (3, 6):
            roll_mean = raw_filled.rolling(window=window, min_periods=1).mean()
            roll_std  = raw_filled.rolling(window=window, min_periods=1).std().fillna(0.0)
            roll_mean.columns = [f"{c}_rollmean{window}" for c in roll_mean.columns]
            roll_std.columns  = [f"{c}_rollstd{window}"  for c in roll_std.columns]
            feat_frames.extend([roll_mean, roll_std])

        tslo = {}
        for c in raw.columns:
            tslo[f"{c}_tslo"] = _time_since_last_observed(raw[c])
        feat_frames.append(pd.DataFrame(tslo))

    # Stack all feature groups side-by-side into one wide matrix
    feat = pd.concat(feat_frames, axis=1)
    return feat.values.astype(np.float32), list(feat.columns)


def load_patient_series(path: Path, feature_set: str = "basic", patient_normalize: bool = False) -> PatientSeries:
    df = load_patient_df(path)
    labels = df["SepsisLabel"].values.astype(int)
    feats, _ = build_features(df, feature_set=feature_set, patient_normalize=patient_normalize)
    hours = np.arange(len(df), dtype=int)
    onset = compute_onset_hour(labels)
    return PatientSeries(path.stem, feats, labels, hours, onset)


def build_dataset(
    data_dir: Path,
    max_patients: int | None = None,
    feature_set: str = "basic",
    patient_normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    files = list_patient_files(data_dir)
    if max_patients:
        files = files[:max_patients]
    return build_dataset_from_files(files, feature_set=feature_set, patient_normalize=patient_normalize)


def build_dataset_from_files(
    files: List[Path],
    feature_set: str = "basic",
    patient_normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # -----------------------------------------------------------------------
    # Loop all patient files, build features, then stack into one big matrix.
    # Each row in the final matrix = one patient-hour.
    # patient_ids array tracks which patient each row belongs to.
    # -----------------------------------------------------------------------
    all_x, all_y, all_pid, all_hour, all_onset, all_quality = [], [], [], [], [], []
    feature_names: List[str] = []

    for path in files:
        series = load_patient_series(path, feature_set=feature_set, patient_normalize=patient_normalize)
        if not feature_names:
            _, feature_names = build_features(
                load_patient_df(path), feature_set=feature_set, patient_normalize=patient_normalize
            )
        df = load_patient_df(path)
        all_quality.append(compute_quality(df))
        all_x.append(series.features)
        all_y.append(series.labels)
        # Repeat patient ID for every hour row so we can track ownership after stacking
        all_pid.append(np.array([series.patient_id] * len(series.labels)))
        all_hour.append(series.hours)
        # -1 sentinel = no sepsis (None cannot be stored in an int array)
        onset_val = -1 if series.onset_hour is None else series.onset_hour
        all_onset.append(np.array([onset_val] * len(series.labels)))

    X           = np.concatenate(all_x,       axis=0)
    y           = np.concatenate(all_y,       axis=0)
    patient_ids = np.concatenate(all_pid,     axis=0)
    hours       = np.concatenate(all_hour,    axis=0)
    onset_hours = np.concatenate(all_onset,   axis=0)
    quality     = np.concatenate(all_quality, axis=0)
    return X, y, patient_ids, hours, onset_hours, quality, feature_names
