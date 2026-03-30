from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sepsis_ews.data import list_patient_files, load_patient_df, build_features, compute_onset_hour
from sepsis_ews.utils import apply_alert_policy, save_json


def plot_case(output_dir: Path, patient_id: str, probs: np.ndarray, onset: int | None, first_alert: int | None) -> None:
    plt.figure(figsize=(6, 2.8))
    plt.plot(probs, label="Risk")
    if onset is not None:
        plt.axvline(onset, color="red", linestyle="--", label="Onset")
    if first_alert is not None:
        plt.axvline(first_alert, color="green", linestyle="--", label="First alert")
    plt.xlabel("Hour")
    plt.ylabel("Risk")
    plt.title(f"Patient {patient_id}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{patient_id}.png")
    plt.close()


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

    files = list_patient_files(data_dir)
    if args.max_patients:
        files = files[: args.max_patients]

    med = json.loads(Path(args.medians).read_text(encoding="utf-8"))
    medians = np.array(med["medians"], dtype=float)
    medians = np.where(np.isnan(medians), 0.0, medians)

    bundle = joblib.load(args.weights)
    model = bundle["model"]
    scaler = bundle["scaler"]

    candidates = {
        "early_detection": None,
        "late_detection": None,
        "false_alarm": None,
    }

    for path in files:
        df = load_patient_df(path)
        labels = df["SepsisLabel"].values.astype(int)
        feats, _ = build_features(df, feature_set=args.feature_set, patient_normalize=args.patient_normalize)
        X = np.where(np.isnan(feats), medians, feats)
        X = scaler.transform(X)
        probs = model.predict_proba(X)[:, 1]

        preds = apply_alert_policy(probs, args.threshold, alert_k=args.alert_k)
        alert_idx = np.where(preds == 1)[0]
        first_alert = int(alert_idx[0]) if len(alert_idx) else None
        onset = compute_onset_hour(labels)

        has_sepsis = onset is not None
        if has_sepsis and first_alert is not None:
            lead = onset - first_alert
            if lead >= 3:
                if candidates["early_detection"] is None or lead > candidates["early_detection"]["lead"]:
                    candidates["early_detection"] = {
                        "patient_id": path.stem,
                        "lead": lead,
                        "onset": onset,
                        "first_alert": first_alert,
                        "probs": probs,
                    }
            elif lead < 0:
                if candidates["late_detection"] is None or lead < candidates["late_detection"]["lead"]:
                    candidates["late_detection"] = {
                        "patient_id": path.stem,
                        "lead": lead,
                        "onset": onset,
                        "first_alert": first_alert,
                        "probs": probs,
                    }
        if (not has_sepsis) and first_alert is not None:
            if candidates["false_alarm"] is None or len(alert_idx) > candidates["false_alarm"]["alerts"]:
                candidates["false_alarm"] = {
                    "patient_id": path.stem,
                    "alerts": int(len(alert_idx)),
                    "onset": None,
                    "first_alert": first_alert,
                    "probs": probs,
                }

    report = {}
    for key, info in candidates.items():
        if info is None:
            continue
        report[key] = {
            "patient_id": info["patient_id"],
            "onset_hour": info["onset"],
            "first_alert_hour": info["first_alert"],
            "lead_time": info["lead"] if "lead" in info else None,
            "alerts": info.get("alerts"),
        }
        plot_case(output_dir, info["patient_id"], info["probs"], info["onset"], info["first_alert"])

    save_json(output_dir / "case_studies.json", report)
    print(f"Saved case studies to {output_dir}")


if __name__ == "__main__":
    main()
