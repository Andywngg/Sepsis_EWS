from __future__ import annotations

# Scoring, metric, and alert-policy functions used across every script in this project.
# Nothing here trains a model or loads data; it only computes numbers from predictions.

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
    # AUROC (Area Under the ROC Curve): measures how well the model ranks patients.
    # Specifically, it is the probability that a randomly picked sepsis patient
    # gets a higher risk score than a randomly picked non-sepsis patient.
    # 0.5 means the model is no better than random guessing. 1.0 is perfect.
    #
    # AUPRC (Area Under the Precision-Recall Curve): better suited to imbalanced datasets
    # like this one where only about 2% of hours are positive (sepsis).
    # AUROC can look good even if the model is bad at identifying the rare positives,
    # because it is dominated by the easy-to-classify negatives. AUPRC focuses only
    # on how well the model finds the rare positives.
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {"auroc": float(auroc), "auprc": float(auprc)}


def apply_alert_policy(probabilities: np.ndarray, threshold: float, alert_k: int = 1) -> np.ndarray:
    # Converts the model's continuous risk probabilities into a binary alert array.
    # Each position is 1 (fire alert this hour) or 0 (no alert).
    #
    # alert_k=1: fire an alert the moment the probability crosses the threshold.
    # alert_k=2: require TWO consecutive hours above the threshold before firing.
    #            This prevents a single abnormal lab value from triggering an alert.
    #            A one-hour spike is often noise; two hours in a row is a real trend.
    #            The run counter resets to zero whenever a sub-threshold hour appears.
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
    # Grid search over candidate threshold values. For each threshold, convert probabilities
    # to alerts and score those alerts with the utility function. Return the threshold
    # that gave the highest utility score.
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
    # A simplified (custom) utility score. For each patient it rewards alerting
    # before onset, penalizes alerting on patients who never got sepsis, and
    # penalizes missing sepsis patients entirely.
    # This is NOT the official PhysioNet metric; use compute_official_utility for that.
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
                # The model never fired but the patient had sepsis. Heavy penalty.
                total -= 2.0
            continue

        first_alert_hour = hrs[alert_idx[0]]
        if has_sepsis:
            if onset < 0:
                # onset < 0 means no valid onset time was recorded for this patient.
                total -= 2.0
            else:
                if first_alert_hour <= onset:
                    # Alert came before or at onset. Reward increases with earlier warning.
                    lead = onset - first_alert_hour
                    total += 1.0 + 0.1 * min(lead, 6)
                else:
                    # Alert came after onset. Reward decreases with how late it was.
                    delay = first_alert_hour - onset
                    total += max(0.0, 1.0 - 0.2 * delay)
        else:
            # The patient never had sepsis but the model still fired. False alarm penalty.
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
    # Computes three clinically meaningful performance measures:
    #   early_detection_rate: what fraction of sepsis patients got an alert before onset?
    #   false_alert_rate: what fraction of non-sepsis patients got any alert at all?
    #   median_lead_time_hours: among patients caught early, how many hours of warning did they get?
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
                continue  # no alert, no sepsis: correct and uninteresting
            sepsis_count += 1
            continue  # sepsis patient missed entirely

        first_alert_hour = hrs[alert_idx[0]]
        if has_sepsis:
            sepsis_count += 1
            if onset >= 0:
                # lead is positive when alert came before onset, negative when after
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
    # Standard classification accuracy and F1 score computed from individual hourly predictions.
    # These use a threshold to produce binary predictions; AUROC and AUPRC do not.
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
    dt_early: int   = -12,
    dt_optimal: int = -6,
    dt_late: int    = 3,
    max_u_tp: float = 1.0,
    min_u_fn: float = -2.0,
    u_fp: float     = -0.05,
    u_tn: float     = 0.0,
) -> float:
    # Scores a single patient's alert sequence using the PhysioNet piecewise reward function.
    # The scoring depends on how far each alert is from the sepsis onset time.
    #
    # Time window relative to onset (t=0 at onset):
    #   Before -12h: alerting here is treated the same as a false alarm (-0.05/hour)
    #                because there is not enough clinical evidence that far in advance.
    #   -12h to -6h: partial reward that linearly ramps up from 0 to +1.0.
    #                An alert here gives a head start but not the full reward.
    #   -6h to +3h:  reward linearly decays from +1.0 back down to 0.
    #                The best time to alert is right at -6h (maximum clinical benefit).
    #   After +3h:   no more reward for alerting; penalty grows if sepsis was missed.
    #   Not alerting on a sepsis patient in the -6h to +3h window causes a penalty
    #   that grows toward -2.0 (missed sepsis is the worst possible outcome).
    #
    # t_sepsis here is the INDEX where the label first becomes 1 minus dt_optimal.
    # So t_sepsis + dt_optimal = the actual onset index.
    if np.any(labels):
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        t_sepsis = float("inf")
    n = len(labels)

    # Pre-compute the slopes and intercepts for the three piecewise linear segments.
    # m_1, b_1: slope/intercept for the early ramp (reward rises from -12h to -6h).
    # m_2, b_2: slope/intercept for the late decay (reward falls from -6h to +3h).
    # m_3, b_3: slope/intercept for the missed-sepsis penalty (grows from -6h onward).
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
                    # Alert fired in the early ramp zone. Use the ramp formula, but
                    # cap at u_fp to avoid rewarding very early alerts like false alarms.
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                else:
                    # Alert fired in the decay zone (between -6h and +3h from onset).
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif (not is_septic) and predictions[t]:
                # Alert on a patient who never had sepsis: flat false-alarm penalty.
                u[t] = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    # No alert before -6h: no penalty yet, still time to catch it.
                    u[t] = 0
                else:
                    # No alert and we are past the optimal window: growing missed-sepsis penalty.
                    u[t] = m_3 * (t - t_sepsis) + b_3
            else:
                # True negative: correctly not alerting on a non-sepsis patient.
                u[t] = u_tn
    return float(np.sum(u))


def compute_official_utility(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    alert_k: int = 1,
) -> float:
    # The official PhysioNet competition utility score. Normalized so that:
    #   0.0 means the model is no better than never alerting on anyone.
    #   1.0 means the model is a perfect oracle that always alerts at exactly -6h.
    #
    # The normalization formula is: (observed - inaction) / (best_possible - inaction).
    # "inaction" is the score you would get from a model that never fires any alert.
    # This is NOT zero because even inaction gets penalized for missed sepsis patients.
    # Subtracting inaction from both numerator and denominator removes that baseline,
    # so a score of 0.0 means "no better than doing nothing" and anything above 0 is useful.
    dt_early, dt_optimal, dt_late = -12, -6, 3
    max_u_tp, min_u_fn, u_fp, u_tn = 1, -2, -0.05, 0

    patients = np.unique(patient_ids)
    observed = best = inaction = 0.0

    for pid in patients:
        mask   = patient_ids == pid
        labels = y_true[mask]
        probs  = y_prob[mask]
        preds  = apply_alert_policy(probs, threshold, alert_k=alert_k)

        # Score the model's actual alerts for this patient.
        observed += compute_prediction_utility(labels, preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

        # Score the hypothetical perfect oracle: alert exactly in the optimal window.
        best_preds = np.zeros_like(labels)
        if np.any(labels):
            t_sepsis = np.argmax(labels) - dt_optimal
            best_preds[max(0, t_sepsis + dt_early):min(t_sepsis + dt_late + 1, len(labels))] = 1
        best += compute_prediction_utility(labels, best_preds, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

        # Score the all-zeros policy (never alert on anyone).
        inaction += compute_prediction_utility(labels, np.zeros_like(labels), dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn)

    denom = best - inaction
    if denom == 0:
        return 0.0
    return float((observed - inaction) / denom)


def compute_patient_level_metrics(
    patient_ids: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    # Rather than scoring every individual hour, summarize each patient as a single number:
    # the highest risk score the model assigned to them at any point during their ICU stay.
    # Then compute AUROC and AUPRC on those per-patient scores.
    # This answers the question: does the model correctly flag sepsis patients as high-risk
    # overall, regardless of which specific hour triggered the high score?
    # This is more clinically relevant because a doctor cares whether a patient gets
    # flagged at all, not which exact hour the model was most confident.
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
    # For every sepsis patient who was alerted before onset, compute how many hours
    # of warning they received (onset_hour - first_alert_hour).
    # Returns an array of those lead times. Positive = alerted before onset.
    # Only includes patients where the alert actually came early (not after onset).
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
    # Measures how many alerts the clinical team would receive per patient-day.
    # This is an operational load metric: if the model fires 50 alerts per patient-day,
    # nurses will start ignoring it (alarm fatigue), which eliminates its clinical value
    # even if the model is technically accurate. We want this number to be small.
    # We separately compute the rate for non-sepsis patients because those alerts are
    # all false alarms by definition.
    patients = np.unique(patient_ids)
    total_alerts = total_days = nonsepsis_alerts = nonsepsis_days = 0.0
    alerts_per_patient = []

    for pid in patients:
        mask          = patient_ids == pid
        probs         = y_prob[mask]
        hrs           = hours[mask]
        preds         = apply_alert_policy(probs, threshold, alert_k=alert_k)
        alerts        = float(np.sum(preds))
        duration_days = max(len(hrs), 1) / 24.0  # convert hours to days
        total_alerts  += alerts
        total_days    += duration_days
        alerts_per_patient.append(alerts)

        has_sepsis = np.any(y_true[mask] == 1)
        if not has_sepsis:
            nonsepsis_alerts += alerts
            nonsepsis_days   += duration_days

    return {
        "alerts_per_patient_day":          float(total_alerts / total_days)          if total_days        else 0.0,
        "alerts_per_nonsepsis_patient_day": float(nonsepsis_alerts / nonsepsis_days)  if nonsepsis_days    else 0.0,
        "mean_alerts_per_patient":          float(np.mean(alerts_per_patient))        if alerts_per_patient else 0.0,
    }
