from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_basic_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {"auroc": float(auroc), "auprc": float(auprc)}


def apply_alert_policy(probabilities: np.ndarray, threshold: float, alert_k: int = 1) -> np.ndarray:
    raw = (probabilities >= threshold).astype(int)
    if alert_k <= 1:
        return raw
    out = np.zeros_like(raw)
    run = 0
    for i, val in enumerate(raw):
        if val:
            run += 1
        else:
            run = 0
        if run >= alert_k:
            out[i] = 1
    return out


def select_threshold_by_utility(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    onset_hours: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    utility_kind: str = "official",
    alert_k: int = 1,
) -> Tuple[float, float]:
    best_thr = 0.5
    best_util = -1e9
    for thr in thresholds:
        if utility_kind == "official":
            util = compute_official_utility(patient_ids, y_true, y_prob, float(thr), alert_k=alert_k)
        else:
            util = compute_utility(patient_ids, hours, onset_hours, y_true, y_prob, float(thr), alert_k=alert_k)
        if util > best_util:
            best_util = util
            best_thr = float(thr)
    return best_thr, float(best_util)


def compute_utility(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    onset_hours: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> float:
    # Simplified utility: reward earlier alerts, penalize false alarms.
    total = 0.0
    patients = np.unique(patient_ids)
    for pid in patients:
        mask = patient_ids == pid
        probs = y_prob[mask]
        hrs = hours[mask]
        onset = onset_hours[mask][0]
        has_sepsis = np.any(y_true[mask] == 1)

        preds = apply_alert_policy(probs, threshold, alert_k=alert_k)
        alert_idx = np.where(preds == 1)[0]
        if len(alert_idx) == 0:
            if has_sepsis:
                total -= 2.0
            continue

        first_alert_hour = hrs[alert_idx[0]]
        if has_sepsis:
            if onset < 0:
                total -= 2.0
            else:
                if first_alert_hour <= onset:
                    lead = onset - first_alert_hour
                    total += 1.0 + 0.1 * min(lead, 6)
                else:
                    delay = first_alert_hour - onset
                    total += max(0.0, 1.0 - 0.2 * delay)
        else:
            total -= 0.5
    return total / max(len(patients), 1)


def early_warning_stats(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    onset_hours: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> Dict[str, float]:
    patients = np.unique(patient_ids)
    lead_times = []
    early_hits = 0
    sepsis_count = 0
    false_alerts = 0

    for pid in patients:
        mask = patient_ids == pid
        probs = y_prob[mask]
        hrs = hours[mask]
        onset = onset_hours[mask][0]
        has_sepsis = np.any(y_true[mask] == 1)
        preds = apply_alert_policy(probs, threshold, alert_k=alert_k)
        alert_idx = np.where(preds == 1)[0]
        if len(alert_idx) == 0:
            if not has_sepsis:
                continue
            sepsis_count += 1
            continue
        first_alert_hour = hrs[alert_idx[0]]
        if has_sepsis:
            sepsis_count += 1
            if onset >= 0:
                lead = onset - first_alert_hour
                lead_times.append(float(lead))
                if lead >= 0:
                    early_hits += 1
        else:
            false_alerts += 1

    early_rate = early_hits / sepsis_count if sepsis_count else 0.0
    false_rate = false_alerts / max(len(patients) - sepsis_count, 1)
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    return {
        "early_detection_rate": float(early_rate),
        "false_alert_rate": float(false_rate),
        "median_lead_time_hours": median_lead,
    }


def compute_accuracy_f_measure(labels: np.ndarray, predictions: np.ndarray) -> Tuple[float, float]:
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(len(labels)):
        if labels[i] and predictions[i]:
            tp += 1
        elif not labels[i] and predictions[i]:
            fp += 1
        elif labels[i] and not predictions[i]:
            fn += 1
        elif not labels[i] and not predictions[i]:
            tn += 1
    if tp + fp + fn + tn:
        accuracy = float(tp + tn) / float(tp + fp + fn + tn)
    else:
        accuracy = 1.0
    if 2 * tp + fp + fn:
        f_measure = float(2 * tp) / float(2 * tp + fp + fn)
    else:
        f_measure = 1.0
    return accuracy, f_measure


def compute_prediction_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: int = -12,
    dt_optimal: int = -6,
    dt_late: int = 3,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float = -0.05,
    u_tn: float = 0.0,
) -> float:
    if np.any(labels):
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        t_sepsis = float("inf")
    n = len(labels)

    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    u = np.zeros(n)
    is_septic = bool(np.any(labels))
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                else:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif (not is_septic) and predictions[t]:
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0
                else:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            else:
                u[t] = u_tn
    return float(np.sum(u))


def compute_official_utility(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> float:
    dt_early = -12
    dt_optimal = -6
    dt_late = 3
    max_u_tp = 1
    min_u_fn = -2
    u_fp = -0.05
    u_tn = 0

    patients = np.unique(patient_ids)
    observed = 0.0
    best = 0.0
    inaction = 0.0

    for pid in patients:
        mask = patient_ids == pid
        labels = y_true[mask]
        probs = y_prob[mask]
        preds = apply_alert_policy(probs, threshold, alert_k=alert_k)

        observed += compute_prediction_utility(
            labels,
            preds,
            dt_early=dt_early,
            dt_optimal=dt_optimal,
            dt_late=dt_late,
            max_u_tp=max_u_tp,
            min_u_fn=min_u_fn,
            u_fp=u_fp,
            u_tn=u_tn,
        )

        best_preds = np.zeros_like(labels)
        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            start = max(0, t_sepsis + dt_early)
            end = min(t_sepsis + dt_late + 1, len(labels))
            best_preds[start:end] = 1
        best += compute_prediction_utility(
            labels,
            best_preds,
            dt_early=dt_early,
            dt_optimal=dt_optimal,
            dt_late=dt_late,
            max_u_tp=max_u_tp,
            min_u_fn=min_u_fn,
            u_fp=u_fp,
            u_tn=u_tn,
        )

        inaction += compute_prediction_utility(
            labels,
            np.zeros_like(labels),
            dt_early=dt_early,
            dt_optimal=dt_optimal,
            dt_late=dt_late,
            max_u_tp=max_u_tp,
            min_u_fn=min_u_fn,
            u_fp=u_fp,
            u_tn=u_tn,
        )

    denom = best - inaction
    if denom == 0:
        return 0.0
    return float((observed - inaction) / denom)


def compute_patient_level_metrics(
    patient_ids: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    patients = np.unique(patient_ids)
    labels = []
    probs = []
    for pid in patients:
        mask = patient_ids == pid
        labels.append(1 if np.any(y_true[mask] == 1) else 0)
        probs.append(float(np.max(y_prob[mask])))
    labels_arr = np.array(labels, dtype=int)
    probs_arr = np.array(probs, dtype=float)
    return compute_basic_metrics(labels_arr, probs_arr)


def lead_time_distribution(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    onset_hours: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> np.ndarray:
    patients = np.unique(patient_ids)
    lead_times = []
    for pid in patients:
        mask = patient_ids == pid
        probs = y_prob[mask]
        hrs = hours[mask]
        onset = onset_hours[mask][0]
        has_sepsis = np.any(y_true[mask] == 1)
        if not has_sepsis or onset < 0:
            continue
        preds = apply_alert_policy(probs, threshold, alert_k=alert_k)
        alert_idx = np.where(preds == 1)[0]
        if len(alert_idx) == 0:
            continue
        first_alert_hour = hrs[alert_idx[0]]
        lead_times.append(float(onset - first_alert_hour))
    return np.array(lead_times, dtype=float)


def alert_burden_stats(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> Dict[str, float]:
    patients = np.unique(patient_ids)
    total_alerts = 0.0
    total_days = 0.0
    nonsepsis_alerts = 0.0
    nonsepsis_days = 0.0
    alerts_per_patient = []

    for pid in patients:
        mask = patient_ids == pid
        probs = y_prob[mask]
        hrs = hours[mask]
        preds = apply_alert_policy(probs, threshold, alert_k=alert_k)
        alerts = float(np.sum(preds))
        duration_days = max(len(hrs), 1) / 24.0
        total_alerts += alerts
        total_days += duration_days
        alerts_per_patient.append(alerts)

        has_sepsis = np.any(y_true[mask] == 1)
        if not has_sepsis:
            nonsepsis_alerts += alerts
            nonsepsis_days += duration_days

    return {
        "alerts_per_patient_day": float(total_alerts / total_days) if total_days else 0.0,
        "alerts_per_nonsepsis_patient_day": float(nonsepsis_alerts / nonsepsis_days) if nonsepsis_days else 0.0,
        "mean_alerts_per_patient": float(np.mean(alerts_per_patient)) if alerts_per_patient else 0.0,
    }
