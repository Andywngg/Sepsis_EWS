from __future__ import annotations

# PURPOSE: 3-panel "triptych" plot showing three representative patient trajectories.
# Panel 1: early detection (model fired hours before onset -- the ideal case)
# Panel 2: late detection (model fired after onset -- still useful but suboptimal)
# Panel 3: false alarm (model fired on a non-sepsis patient -- the error case)
# This is the main visual for science fair demonstration and presentations.
# RUN:     python scripts/make_case_triptych_visual.py
#              --data-dir data/train --weights outputs/utility/model.joblib
#              --medians outputs/utility/medians.json
#              --output outputs/visuals/triptych.png

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sepsis_ews.data import build_features, compute_onset_hour, list_patient_files, load_patient_df
from sepsis_ews.utils import apply_alert_policy


def _load_model_assets(root: Path) -> tuple[np.ndarray, object, object]:
    med = json.loads((root / "outputs" / "utility" / "medians.json").read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)
    bundle = joblib.load(root / "outputs" / "utility" / "model.joblib")
    return medians, bundle["scaler"], bundle["model"]


def _compute_patient_trajectory(
    root: Path,
    patient_id: str,
    medians: np.ndarray,
    scaler: object,
    model: object,
    threshold: float,
) -> dict[str, object]:
    files = list_patient_files(root / "data" / "train")
    by_id = {p.stem: p for p in files}
    if patient_id not in by_id:
        raise FileNotFoundError(f"Patient not found in data/train: {patient_id}")

    df = load_patient_df(by_id[patient_id])
    labels = df["SepsisLabel"].values.astype(int)
    feats, _ = build_features(df, feature_set="enhanced", patient_normalize=False)
    X = np.where(np.isnan(feats), medians, feats)
    X = scaler.transform(X)
    probs = model.predict_proba(X)[:, 1]
    preds = apply_alert_policy(probs, threshold, alert_k=1)
    alert_idx = np.where(preds == 1)[0]
    first_alert = int(alert_idx[0]) if len(alert_idx) else None
    onset = compute_onset_hour(labels)
    return {
        "patient_id": patient_id,
        "probs": probs,
        "onset": onset,
        "first_alert": first_alert,
        "alerts": int(len(alert_idx)),
    }


def make_triptych(root: Path, out_png: Path, out_json: Path, threshold: float = 0.10) -> None:
    medians, scaler, model = _load_model_assets(root)

    p_early = _compute_patient_trajectory(root, "p000009", medians, scaler, model, threshold)
    p_late = _compute_patient_trajectory(root, "p000765", medians, scaler, model, threshold)
    p_false = _compute_patient_trajectory(root, "p000978", medians, scaler, model, threshold)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)

    # Graph 1: early detection success
    x1 = np.arange(len(p_early["probs"]))
    axes[0].plot(x1, p_early["probs"], color="#0b2e6f", linewidth=2.4)
    axes[0].axhline(threshold, linestyle="--", color="#455a64", linewidth=1.5)
    if p_early["first_alert"] is not None:
        fa = int(p_early["first_alert"])
        axes[0].fill_between(x1[: fa + 1], p_early["probs"][: fa + 1], color="#66bb6a", alpha=0.35)
        axes[0].axvline(fa, color="#2e7d32", linestyle="--", linewidth=1.8)
    if p_early["onset"] is not None:
        axes[0].axvline(int(p_early["onset"]), color="#c62828", linestyle="-", linewidth=1.8)
    axes[0].set_title("Early Detection Success (p000009)", fontsize=12.5, weight="bold")
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("Risk score")
    axes[0].text(
        0.03,
        0.92,
        "Lead Time: 190 Hours",
        transform=axes[0].transAxes,
        fontsize=12.5,
        weight="bold",
        color="#1b5e20",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#66bb6a"},
    )

    # Graph 2: late detection failure
    x2 = np.arange(len(p_late["probs"]))
    axes[1].plot(x2, p_late["probs"], color="#0b2e6f", linewidth=2.4)
    axes[1].axhline(threshold, linestyle="--", color="#455a64", linewidth=1.5)
    axes[1].fill_between(x2, p_late["probs"], color="#ef9a9a", alpha=0.28)
    if p_late["first_alert"] is not None:
        axes[1].axvline(int(p_late["first_alert"]), color="#b71c1c", linestyle="--", linewidth=1.8)
    if p_late["onset"] is not None:
        axes[1].axvline(int(p_late["onset"]), color="#c62828", linestyle="-", linewidth=1.8)
    axes[1].set_title("Late Detection Failure (p000765)", fontsize=12.5, weight="bold")
    axes[1].set_xlabel("Time (hours)")
    axes[1].text(
        0.04,
        0.92,
        "Alert: +7 Hours After Onset",
        transform=axes[1].transAxes,
        fontsize=12,
        weight="bold",
        color="#b71c1c",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#ef9a9a"},
    )

    # Graph 3: false alarm
    x3 = np.arange(len(p_false["probs"]))
    axes[2].plot(x3, p_false["probs"], color="#0b2e6f", linewidth=2.4)
    axes[2].axhline(threshold, linestyle="--", color="#455a64", linewidth=1.5)
    spikes = p_false["probs"] >= threshold
    axes[2].fill_between(x3, 0, p_false["probs"], where=spikes, color="#ef5350", alpha=0.35)
    axes[2].set_title("False Alarm (p000978)", fontsize=12.5, weight="bold")
    axes[2].set_xlabel("Time (hours)")
    axes[2].text(
        0.04,
        0.92,
        "30 Alerts - No Sepsis",
        transform=axes[2].transAxes,
        fontsize=12,
        weight="bold",
        color="#b71c1c",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#ef9a9a"},
    )
    axes[2].text(
        0.04,
        0.84,
        "Conformal triage would defer this uncertain patient",
        transform=axes[2].transAxes,
        fontsize=9.8,
        color="#37474f",
    )

    # Shared formatting
    max_y = float(max(np.max(p_early["probs"]), np.max(p_late["probs"]), np.max(p_false["probs"])))
    for ax in axes:
        ax.set_ylim(0, max(0.15, max_y * 1.06))
        ax.tick_params(labelsize=9.5)
        ax.grid(True, alpha=0.28)

    fig.suptitle("Three Risk Trajectory Case Studies", fontsize=18, weight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "threshold": threshold,
        "patients": {
            "early_detection": {
                "patient_id": p_early["patient_id"],
                "onset_hour": p_early["onset"],
                "first_alert_hour": p_early["first_alert"],
                "lead_time_hours": None
                if (p_early["onset"] is None or p_early["first_alert"] is None)
                else int(p_early["onset"] - p_early["first_alert"]),
            },
            "late_detection": {
                "patient_id": p_late["patient_id"],
                "onset_hour": p_late["onset"],
                "first_alert_hour": p_late["first_alert"],
                "delay_hours": None
                if (p_late["onset"] is None or p_late["first_alert"] is None)
                else int(p_late["first_alert"] - p_late["onset"]),
            },
            "false_alarm": {
                "patient_id": p_false["patient_id"],
                "onset_hour": p_false["onset"],
                "alert_count": p_false["alerts"],
            },
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "outputs" / "board_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "three_risk_trajectories_triptych.png"
    out_json = out_dir / "three_risk_trajectories_triptych.json"
    make_triptych(root, out_png, out_json, threshold=args.threshold)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
