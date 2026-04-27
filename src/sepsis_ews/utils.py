from __future__ import annotations

# PURPOSE: All scoring, metric, and alert-policy functions used across the project.
#
# KEY FUNCTIONS (most important first):
#   compute_official_utility()   — PhysioNet normalized utility score (main metric)
#   compute_prediction_utility() — per-patient utility with time-based reward/penalty
#   apply_alert_policy()         — converts probabilities → binary alerts
#   early_warning_stats()        — early detection rate, false alert rate, lead time
#   compute_basic_metrics()      — AUROC and AUPRC
#   alert_burden_stats()         — alerts per patient-day (operational load metric)

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
    # AUROC: probability that model ranks a sepsis sample above a non-sepsis one (0.5=random, 1=perfect)
    # AUPRC: precision-recall area — better than AUROC for imbalanced data (~2% positives)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {"auroc": float(auroc), "auprc": float(auprc)}


def apply_alert_policy(probabilities: np.ndarray, threshold: float, alert_k: int = 1) -> np.ndarray:
    # Convert raw probabilities → binary alert array using the chosen threshold.
    # alert_k > 1: only fire after k consecutive hours above threshold (reduces single-spike noise).
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
    # Grid search: try every threshold, return the one that maximizes utility score.
    best_thr, best_util = 0.5, -1e9
    for thr in thresholds:
        if utility_kind == "official":
            util = compute_official_utility(patient_ids, y_true, y_prob, float(thr), alert_k=alert_k)
        else:
            util = compute_utility(patient_ids, hours, onset_hours, y_true, y_prob, float(thr), alert_k=alert_k)
        if util > best_util:
            best_util = util
            best_thr  = float(thr)
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
    # Simplified (custom) utility — rewards early alerts, penalizes false alarms.
    total = 0.0
    patients = np.unique(patient_ids)
    for pid in patients:
        mask       = patient_ids == pid
        probs      = y_prob[mask]
        hrs        = hours[mask]
        onset      = onset_hours[mask][0]
        has_sepsis = np.any(y_true[mask] == 1)

        preds     = apply_alert_policy(probs, threshold, alert_k=alert_k)
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
            total -= 0.5  # penalty for false alarm on non-sepsis patient
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
    # -----------------------------------------------------------------------
    # CLINICAL EARLY-WARNING METRICS
    #   early_detection_rate — fraction of sepsis patients alerted before/at onset
    #   false_alert_rate     — fraction of non-sepsis patients who got any alert
    #   median_lead_time     — median hours between first alert and onset
    # -----------------------------------------------------------------------
    patients = np.unique(patient_ids)
    lead_times, early_hits, sepsis_count, false_alerts = [], 0, 0, 0

    for pid in patients:
        mask       = patient_ids == pid
        probs      = y_prob[mask]
        hrs        = hours[mask]
        onset      = onset_hours[mask][0]
        has_sepsis = np.any(y_true[mask] == 1)
        preds      = apply_alert_policy(probs, threshold, alert_k=alert_k)
        alert_idx  = np.where(preds == 1)[0]
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

    early_rate  = early_hits / sepsis_count if sepsis_count else 0.0
    false_rate  = false_alerts / max(len(patients) - sepsis_count, 1)
    median_lead = float(np.median(lead_times)) if lead_times else 0.0
    return {
        "early_detection_rate":    float(early_rate),
        "false_alert_rate":        float(false_rate),
        "median_lead_time_hours":  median_lead,
    }


def compute_accuracy_f_measure(labels: np.ndarray, predictions: np.ndarray) -> Tuple[float, float]:
    tp = fp = fn = tn = 0
    for i in range(len(labels)):
        if   labels[i] and     predictions[i]: tp += 1
        elif not labels[i] and predictions[i]: fp += 1
        elif labels[i] and not predictions[i]: fn += 1
        else:                                  tn += 1
    accuracy  = float(tp + tn) / float(tp + fp + fn + tn) if (tp + fp + fn + tn) else 1.0
    f_measure = float(2 * tp) / float(2 * tp + fp + fn)   if (2 * tp + fp + fn) else 1.0
    return accuracy, f_measure


def compute_prediction_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: int   = -12,   # reward window opens 12h before onset
    dt_optimal: int = -6,    # maximum reward at 6h before onset
    dt_late: int    = 3,     # reward window closes 3h after onset
    max_u_tp: float = 1.0,   # max score for a perfect early alert
    min_u_fn: float = -2.0,  # penalty for missing sepsis in the late window
    u_fp: float     = -0.05, # penalty per hour for alerting on non-sepsis patient
    u_tn: float     = 0.0,   # no reward/penalty for correctly not alerting
) -> float:
    # -----------------------------------------------------------------------
    # PER-PATIENT UTILITY — PhysioNet scoring formula
    #
    # Scoring window relative to onset:
    #   [-12h, -6h] → linear ramp from 0 up to +1.0  (early alert, good)
    #   [-6h,  +3h] → linear decay from +1.0 to 0    (late alert, less good)
    #   outside window but alerting non-sepsis → -0.05 per hour (false alarm)
    #   missed sepsis after -6h → penalty down to -2.0
    # -----------------------------------------------------------------------
    if np.any(labels):
        t_sepsis = np.argmax(labels) - dt_optimal  # target = 6h before onset
    else:
        t_sepsis = float("inf")
    n = len(labels)

    # Slopes and intercepts for the piecewise linear reward/penalty ramps
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
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)  # early alert ramp
                else:
                    u[t] = m_2 * (t - t_sepsis) + b_2              # late alert decay
            elif (not is_septic) and predictions[t]:
                u[t] = u_fp                                          # false alarm penalty
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0                                         # no penalty for waiting
                else:
                    u[t] = m_3 * (t - t_sepsis) + b_3              # missed sepsis penalty
            else:
                u[t] = u_tn                                          # true negative, no change
    return float(np.sum(u))


def compute_official_utility(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> float:
    # -----------------------------------------------------------------------
    # OFFICIAL PHYSIONET UTILITY SCORE — main evaluation metric
    #
    # Normalized so that:
    #   0.0 = equivalent to never alerting on anyone (inaction)
    #   1.0 = perfect oracle that always alerts at exactly -6h before onset
    #
    # Formula: (observed - inaction) / (best_possible - inaction)
    # -----------------------------------------------------------------------
    dt_early, dt_optimal, dt_late = -12, -6, 3
    max_u_tp, min_u_fn, u_fp, u_tn = 1, -2, -0.05, 0

    patients = np.unique(patient_ids)
    observed = best = inaction = 0.0

    for pid in patients:
        mask   = patient_ids == pid
        labels = y_true[mask]
        probs  = y_prob[mask]
        preds  = apply_alert_policy(probs, threshold, alert_k=alert_k)

        observed += compute_prediction_utility(labels, preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

        # Perfect oracle: alert exactly in the optimal window [-12h, +3h]
        best_preds = np.zeros_like(labels)
        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_preds[max(0, t_sepsis + dt_early):min(t_sepsis + dt_late + 1, len(labels))] = 1
        best += compute_prediction_utility(labels, best_preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

        # Inaction: never alert on anyone
        inaction += compute_prediction_utility(labels, np.zeros_like(labels), dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    denom = best - inaction
    if denom == 0:
        return 0.0
    return float((observed - inaction) / denom)


def compute_patient_level_metrics(
    patient_ids: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    # Summarize each patient as their max predicted probability, then compute AUROC/AUPRC.
    # More clinically meaningful: "does the model rank sepsis patients higher overall?"
    patients = np.unique(patient_ids)
    labels, probs = [], []
    for pid in patients:
        mask = patient_ids == pid
        labels.append(1 if np.any(y_true[mask] == 1) else 0)
        probs.append(float(np.max(y_prob[mask])))
    return compute_basic_metrics(np.array(labels, dtype=int), np.array(probs, dtype=float))


def lead_time_distribution(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    onset_hours: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> np.ndarray:
    # Returns array of lead times (hours) for all sepsis patients that were caught early.
    # Lead time = onset_hour - first_alert_hour (positive = alert before onset).
    patients = np.unique(patient_ids)
    lead_times = []
    for pid in patients:
        mask       = patient_ids == pid
        probs      = y_prob[mask]
        hrs        = hours[mask]
        onset      = onset_hours[mask][0]
        has_sepsis = np.any(y_true[mask] == 1)
        if not has_sepsis or onset < 0:
            continue
        preds     = apply_alert_policy(probs, threshold, alert_k=alert_k)
        alert_idx = np.where(preds == 1)[0]
        if len(alert_idx) == 0:
            continue
        lead_times.append(float(onset - hrs[alert_idx[0]]))
    return np.array(lead_times, dtype=float)


def alert_burden_stats(
    patient_ids: np.ndarray,
    hours: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> Dict[str, float]:
    # ALERT BURDEN — operational load metric for real deployment.
    # Measures how many alerts the clinical team would receive per day.
    # High burden → alarm fatigue → nurses ignore alerts → model loses clinical value.
    patients = np.unique(patient_ids)
    total_alerts = total_days = nonsepsis_alerts = nonsepsis_days = 0.0
    alerts_per_patient = []

    for pid in patients:
        mask          = patient_ids == pid
        probs         = y_prob[mask]
        hrs           = hours[mask]
        preds         = apply_alert_policy(probs, threshold, alert_k=alert_k)
        alerts        = float(np.sum(preds))
        duration_days = max(len(hrs), 1) / 24.0  # convert hours → days
        total_alerts  += alerts
        total_days    += duration_days
        alerts_per_patient.append(alerts)

        has_sepsis = np.any(y_true[mask] == 1)
        if not has_sepsis:
            nonsepsis_alerts += alerts
            nonsepsis_days   += duration_days

    return {
        "alerts_per_patient_day":         float(total_alerts / total_days)         if total_days        else 0.0,
        "alerts_per_nonsepsis_patient_day":float(nonsepsis_alerts / nonsepsis_days) if nonsepsis_days    else 0.0,
        "mean_alerts_per_patient":         float(np.mean(alerts_per_patient))       if alerts_per_patient else 0.0,
    }
