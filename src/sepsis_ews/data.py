from __future__ import annotations

# Loads raw patient files from disk and converts them into numeric feature matrices
# the model can learn from. Every other script imports build_dataset() from here.
# The main entry point is build_dataset(), which loads all patients and stacks them
# into one big matrix where each row is one patient-hour.

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# This dataclass holds everything
# about a single patient: their feature matrix, their hour-by-hour sepsis labels,
# and when (if ever) they developed sepsis.
@dataclass
class PatientSeries:
    patient_id: str
    features: np.ndarray   # shape: (num_hours, num_features)
    labels: np.ndarray     # shape: (num_hours,)  0=no sepsis, 1=sepsis
    hours: np.ndarray      # [0, 1, 2, ...] ICU hour index
    onset_hour: int | None # first hour where label flipped to 1, or None if patient never got sepsis


def list_patient_files(data_dir: Path) -> List[Path]:
    # Collect all .psv files in the data directory. Sorting ensures the same order
    # every run, which makes the train/test split reproducible.
    return sorted([p for p in data_dir.glob("*.psv")])


def load_patient_df(path: Path) -> pd.DataFrame:
    # PSV stands for "pipe-separated values". The PhysioNet dataset uses | as the delimiter
    # instead of commas. Each row is one hour of ICU measurements for this patient.
    return pd.read_csv(path, sep="|")


def compute_onset_hour(labels: np.ndarray) -> int | None:
    # Scan the label array to find the earliest hour where SepsisLabel became 1.
    # Returns None if no such hour exists (the patient never developed sepsis).
    idx = np.where(labels == 1)[0]
    return int(idx[0]) if len(idx) else None


def _time_since_last_observed(series: pd.Series) -> np.ndarray:
    # For one lab variable, compute how many hours have passed since it was last measured.
    # If WBC was drawn at hour 3 and not again until hour 7, hours 4/5/6 each get
    # a value of 1, 2, 3 respectively. This tells the model "this patient's WBC hasn't
    # been checked for a while", which is itself a clinical signal.
    last_seen = -1
    out = np.zeros(len(series), dtype=np.float32)
    for i, val in enumerate(series.values):
        if pd.notna(val):
            last_seen = i
            out[i] = 0.0        # value was observed this hour, so time-since = 0
        else:
            # If no prior observation exists, treat it as if the patient arrived late
            out[i] = float(i - last_seen) if last_seen >= 0 else float(i + 1)
    return out


def compute_quality(df: pd.DataFrame) -> np.ndarray:
    # Per-hour data quality score. A score of 1.0 means every variable was measured
    # that hour. A score of 0.3 means only 30% of variables have values.
    # Used later to filter out low-quality hours or to test robustness.
    cols = [c for c in df.columns if c != "SepsisLabel"]
    missing_rate = df[cols].isna().mean(axis=1).values.astype(np.float32)
    return 1.0 - missing_rate


def build_features(
    df: pd.DataFrame, feature_set: str = "basic", patient_normalize: bool = False
) -> Tuple[np.ndarray, List[str]]:
    # Takes a single patient's raw DataFrame and returns a numeric feature matrix.
    # Each row is one ICU hour; each column is one engineered feature.
    # The feature groups are described below. The enhanced set adds more columns
    # that capture trends and irregularity in the time series.

    feature_cols = [c for c in df.columns if c != "SepsisLabel"]
    raw = df[feature_cols].copy()

    # Patient normalization: re-center each variable around THIS patient's own average.
    # For example, if a patient's normal heart rate is 90 (above average), subtract 90
    # so the model sees deviations from their personal baseline rather than absolute values.
    # This helps when the model is deployed at a different hospital with different
    # measurement calibrations or patient demographics (called "domain shift").
    if patient_normalize:
        for c in raw.columns:
            mean = raw[c].mean(skipna=True)
            std = raw[c].std(skipna=True)
            if pd.isna(std) or std == 0:
                std = 1.0
            raw[c] = (raw[c] - mean) / std

    # Add a column for the ICU hour index (0, 1, 2, ...). The model can use this
    # to learn that risk patterns change over time in the ICU.
    raw.insert(0, "Hour", np.arange(len(raw)))

    # Delta features: how much did each variable change compared to the previous hour?
    # A sudden spike in lactate is more alarming than a lactate that has been elevated
    # for days. diff() computes row[i] - row[i-1]; fillna(0) handles the first row.
    delta = raw.diff().fillna(0.0)
    delta.columns = [f"{c}_delta" for c in delta.columns]

    # Missingness indicators: a binary (0 or 1) column for every variable.
    # 1 means the value was NOT measured that hour; 0 means it was measured.
    # CRITICAL: this must be computed before any imputation (before fillna/ffill).
    # If computed after, every cell would look observed and the signal would be lost.
    # The model learns patterns like "no lactate drawn for 20 hours in an unstable patient
    # is unusual and predicts bad outcomes", because doctors draw labs when they are worried.
    missing = raw.isna().astype(int)
    missing.columns = [f"{c}_miss" for c in missing.columns]

    feat_frames = [raw, delta, missing]

    if feature_set == "enhanced":
        # Forward-fill (ffill) propagates the last known value forward through NaN gaps.
        # We use ffill ONLY, never backward-fill (bfill). Backward-fill would mean:
        # "at hour 3, use the value that won't be measured until hour 7", which leaks
        # future data into past predictions. The model would learn from information
        # it could not have had in a real deployment.
        # The median fallback handles variables that were NEVER measured for this patient.
        raw_filled = raw.ffill().fillna(raw.median(numeric_only=True))

        for window in (3, 6):
            # Rolling mean: average of the last 3 (or 6) hours. Smooths out single
            # bad readings and gives the model a recent trend signal.
            roll_mean = raw_filled.rolling(window=window, min_periods=1).mean()
            # Rolling standard deviation: how much variation was there in the last 3/6 hours?
            # A rising std means the measurements are becoming unstable, which is a
            # physiological warning sign.
            roll_std  = raw_filled.rolling(window=window, min_periods=1).std().fillna(0.0)
            roll_mean.columns = [f"{c}_rollmean{window}" for c in roll_mean.columns]
            roll_std.columns  = [f"{c}_rollstd{window}"  for c in roll_std.columns]
            feat_frames.extend([roll_mean, roll_std])

        # Time-since-last-observed: how many hours since each variable was last drawn?
        # Captures irregular lab-draw patterns. Doctors order more labs when worried,
        # so "WBC was drawn 18 hours ago" is a signal that things look stable to the team.
        tslo = {}
        for c in raw.columns:
            tslo[f"{c}_tslo"] = _time_since_last_observed(raw[c])
        feat_frames.append(pd.DataFrame(tslo))

    # Concatenate all feature groups side-by-side into one wide matrix.
    # Each hour becomes one row with all feature groups combined into one vector.
    feat = pd.concat(feat_frames, axis=1)
    return feat.values.astype(np.float32), list(feat.columns)


def load_patient_series(path: Path, feature_set: str = "basic", patient_normalize: bool = False) -> PatientSeries:
    # Load one patient file, compute features, and package everything into a PatientSeries.
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
    # Find all patient files and optionally cap how many are loaded (useful for fast testing).
    files = list_patient_files(data_dir)
    if max_patients:
        files = files[:max_patients]
    return build_dataset_from_files(files, feature_set=feature_set, patient_normalize=patient_normalize)


def build_dataset_from_files(
    files: List[Path],
    feature_set: str = "basic",
    patient_normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # Process every patient file and stack the results into large arrays.
    # After this function, each row corresponds to one patient-hour. The patient_ids
    # array tells downstream code which patient each row belongs to.
    #
    # Returns: X (features), y (labels), patient_ids, hours, onset_hours, quality, feature_names
    all_x, all_y, all_pid, all_hour, all_onset, all_quality = [], [], [], [], [], []
    feature_names: List[str] = []

    for path in files:
        series = load_patient_series(path, feature_set=feature_set, patient_normalize=patient_normalize)

        # Capture the feature column names from the first patient we process.
        # All patients share the same feature schema.
        if not feature_names:
            _, feature_names = build_features(
                load_patient_df(path), feature_set=feature_set, patient_normalize=patient_normalize
            )

        df = load_patient_df(path)
        all_quality.append(compute_quality(df))
        all_x.append(series.features)
        all_y.append(series.labels)

        # Repeat the patient ID for every hour row so that after stacking we can still
        # tell which rows belong to the same patient. This is needed for GroupShuffleSplit
        # and for per-patient metric calculations.
        all_pid.append(np.array([series.patient_id] * len(series.labels)))
        all_hour.append(series.hours)

        # Store onset_hour as -1 when the patient never developed sepsis.
        # Python's None cannot be stored inside a NumPy integer array.
        # All downstream code checks "onset_val < 0" to detect non-sepsis patients.
        onset_val = -1 if series.onset_hour is None else series.onset_hour
        all_onset.append(np.array([onset_val] * len(series.labels)))

    # Stack all per-patient arrays into single large arrays.
    # np.concatenate joins them along axis 0 (rows), so the result has as many
    # rows as the total number of patient-hours across all patients.
    X           = np.concatenate(all_x,       axis=0)
    y           = np.concatenate(all_y,       axis=0)
    patient_ids = np.concatenate(all_pid,     axis=0)
    hours       = np.concatenate(all_hour,    axis=0)
    onset_hours = np.concatenate(all_onset,   axis=0)
    quality     = np.concatenate(all_quality, axis=0)
    return X, y, patient_ids, hours, onset_hours, quality, feature_names
