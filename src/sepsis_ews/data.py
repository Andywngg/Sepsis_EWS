from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class PatientSeries:
    patient_id: str
    features: np.ndarray
    labels: np.ndarray
    hours: np.ndarray
    onset_hour: int | None


def list_patient_files(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("*.psv")])


def load_patient_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|")


def compute_onset_hour(labels: np.ndarray) -> int | None:
    idx = np.where(labels == 1)[0]
    return int(idx[0]) if len(idx) else None


def _time_since_last_observed(series: pd.Series) -> np.ndarray:
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
    cols = [c for c in df.columns if c != "SepsisLabel"]
    missing_rate = df[cols].isna().mean(axis=1).values.astype(np.float32)
    return 1.0 - missing_rate


def build_features(
    df: pd.DataFrame, feature_set: str = "basic", patient_normalize: bool = False
) -> Tuple[np.ndarray, List[str]]:
    feature_cols = [c for c in df.columns if c != "SepsisLabel"]
    raw = df[feature_cols].copy()
    if patient_normalize:
        for c in raw.columns:
            mean = raw[c].mean(skipna=True)
            std = raw[c].std(skipna=True)
            if pd.isna(std) or std == 0:
                std = 1.0
            raw[c] = (raw[c] - mean) / std
    raw.insert(0, "Hour", np.arange(len(raw)))
    delta = raw.diff().fillna(0.0)
    delta.columns = [f"{c}_delta" for c in delta.columns]
    missing = raw.isna().astype(int)
    missing.columns = [f"{c}_miss" for c in missing.columns]
    feat_frames = [raw, delta, missing]

    if feature_set == "enhanced":
        raw_filled = raw.ffill().fillna(raw.median(numeric_only=True))
        for window in (3, 6):
            roll_mean = raw_filled.rolling(window=window, min_periods=1).mean()
            roll_std = raw_filled.rolling(window=window, min_periods=1).std().fillna(0.0)
            roll_mean.columns = [f"{c}_rollmean{window}" for c in roll_mean.columns]
            roll_std.columns = [f"{c}_rollstd{window}" for c in roll_std.columns]
            feat_frames.extend([roll_mean, roll_std])

        tslo = {}
        for c in raw.columns:
            tslo[f"{c}_tslo"] = _time_since_last_observed(raw[c])
        tslo_df = pd.DataFrame(tslo)
        feat_frames.append(tslo_df)

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
    all_x = []
    all_y = []
    all_pid = []
    all_hour = []
    all_onset = []
    all_quality = []
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
        all_pid.append(np.array([series.patient_id] * len(series.labels)))
        all_hour.append(series.hours)
        onset_val = -1 if series.onset_hour is None else series.onset_hour
        all_onset.append(np.array([onset_val] * len(series.labels)))

    X = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)
    patient_ids = np.concatenate(all_pid, axis=0)
    hours = np.concatenate(all_hour, axis=0)
    onset_hours = np.concatenate(all_onset, axis=0)
    quality = np.concatenate(all_quality, axis=0)
    return X, y, patient_ids, hours, onset_hours, quality, feature_names
