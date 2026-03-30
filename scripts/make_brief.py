from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--baseline", required=False)
    parser.add_argument("--cross-site", required=False)
    parser.add_argument("--cross-site-label", required=False, default="Cross-site Generalization (proxy A/B split)")
    parser.add_argument("--ci", required=False)
    parser.add_argument("--policy", required=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = load_json(Path(args.metrics))
    baseline = load_json(Path(args.baseline)) if args.baseline else {}
    cross_site = load_json(Path(args.cross_site)) if args.cross_site else {}
    ci = load_json(Path(args.ci)) if args.ci else {}
    policy = load_json(Path(args.policy)) if args.policy else {}
    lines = []
    lines.append("# ISEF Brief: Sepsis Early-Warning System\n\n")
    lines.append("## Research question\n")
    lines.append(
        "Can a utility-optimized early warning system predict sepsis hours before onset "
        "using ICU vitals and labs?\n\n"
    )
    lines.append("## Methods\n")
    lines.append(
        "- Time-series features from ICU vitals/labs\n"
        "- Risk model with early-warning policy\n"
        "- Utility-based threshold selection\n\n"
    )
    lines.append("## Results (held-out patients)\n")
    if metrics.get("max_patients"):
        lines.append(f"- Subset size (max patients): {metrics.get('max_patients')}\n")
    lines.append(f"- AUROC: {metrics.get('metrics', {}).get('auroc', 0):.3f}\n")
    lines.append(f"- AUPRC: {metrics.get('metrics', {}).get('auprc', 0):.3f}\n")
    lines.append(
        f"- Patient-level AUROC: {metrics.get('patient_level_metrics', {}).get('auroc', 0):.3f}\n"
    )
    lines.append(
        f"- Patient-level AUPRC: {metrics.get('patient_level_metrics', {}).get('auprc', 0):.3f}\n"
    )
    lines.append(f"- Brier score: {metrics.get('brier_score', 0):.3f}\n")
    if "brier_score_raw" in metrics:
        lines.append(f"- Brier score (raw): {metrics.get('brier_score_raw', 0):.3f}\n")
    calib = metrics.get("calibration", {})
    if calib:
        lines.append(f"- Calibration: {calib.get('method', 'none')} ({calib.get('patients', 0)} patients)\n")
    lines.append(f"- Utility score: {metrics.get('utility_score', 0):.3f}\n")
    lines.append(f"- Official utility: {metrics.get('official_utility', 0):.3f}\n")
    lines.append(f"- Accuracy: {metrics.get('accuracy', 0):.3f}\n")
    lines.append(f"- F-measure: {metrics.get('f_measure', 0):.3f}\n")
    lines.append(
        f"- Early detection rate: {metrics.get('early_warning', {}).get('early_detection_rate', 0):.3f}\n"
    )
    lines.append(
        f"- False alert rate: {metrics.get('early_warning', {}).get('false_alert_rate', 0):.3f}\n"
    )
    lines.append(
        f"- Median lead time (hours): {metrics.get('early_warning', {}).get('median_lead_time_hours', 0):.2f}\n\n"
    )
    burden = metrics.get("alert_burden", {})
    if burden:
        lines.append(f"- Alerts per patient-day: {burden.get('alerts_per_patient_day', 0):.3f}\n")
        lines.append(
            f"- Alerts per non-sepsis patient-day: {burden.get('alerts_per_nonsepsis_patient_day', 0):.3f}\n"
        )
        lines.append(f"- Mean alerts per patient: {burden.get('mean_alerts_per_patient', 0):.3f}\n\n")
    if baseline:
        lines.append("## Baseline comparison\n")
        lines.append(
            f"- Baseline AUROC: {baseline.get('metrics', {}).get('auroc', 0):.3f} vs improved {metrics.get('metrics', {}).get('auroc', 0):.3f}\n"
        )
        lines.append(
            f"- Baseline AUPRC: {baseline.get('metrics', {}).get('auprc', 0):.3f} vs improved {metrics.get('metrics', {}).get('auprc', 0):.3f}\n"
        )
        lines.append(
            f"- Baseline utility: {baseline.get('utility_score', 0):.3f} vs improved {metrics.get('utility_score', 0):.3f}\n"
        )
        lines.append(
            f"- Baseline official utility: {baseline.get('official_utility', 0):.3f} vs improved {metrics.get('official_utility', 0):.3f}\n\n"
        )

    if ci:
        lines.append("## Statistical Confidence (bootstrap)\n")
        if ci.get("max_patients"):
            lines.append(f"- Bootstrap subset size: {ci.get('max_patients')}\n")
        for key, label in [
            ("auroc", "AUROC"),
            ("auprc", "AUPRC"),
            ("official_utility", "Official utility"),
        ]:
            if key in ci:
                lo, hi = ci[key].get("ci_low", 0), ci[key].get("ci_high", 0)
                lines.append(f"- {label} 95% CI: [{lo:.3f}, {hi:.3f}]\n")
        lines.append("\n")

    if policy:
        lines.append("## Alert Policies (clinical tradeoff)\n")
        sens = policy.get("sensitive_policy", {})
        cons = policy.get("conservative_policy", {})
        if sens:
            lines.append(
                f"- Sensitive policy: threshold {sens.get('threshold', 0):.2f}, "
                f"early detection {sens.get('early_detection_rate', 0):.3f}, "
                f"alerts/patient-day {sens.get('alerts_per_patient_day', 0):.3f}\n"
            )
        if cons:
            lines.append(
                f"- Conservative policy: threshold {cons.get('threshold', 0):.2f}, "
                f"early detection {cons.get('early_detection_rate', 0):.3f}, "
                f"alerts/patient-day {cons.get('alerts_per_patient_day', 0):.3f}\n"
            )
        lines.append("\n")
    lines.append("## Novelty\n")
    lines.append(
        "Unlike standard sepsis classifiers that optimize AUROC, this system optimizes a "
        "clinical utility score and learns an early-warning policy that balances early detection "
        "against false alarms.\n"
    )

    if cross_site:
        lines.append(f"\n## {args.cross_site_label}\n")
        a_to_b = cross_site.get("a_to_b", {})
        b_to_a = cross_site.get("b_to_a", {})
        if a_to_b:
            lines.append(
                f"- Train A -> Test B official utility: {a_to_b.get('official_utility', 0):.3f}, "
                f"AUROC: {a_to_b.get('metrics', {}).get('auroc', 0):.3f}\n"
            )
        if b_to_a:
            lines.append(
                f"- Train B -> Test A official utility: {b_to_a.get('official_utility', 0):.3f}, "
                f"AUROC: {b_to_a.get('metrics', {}).get('auroc', 0):.3f}\n"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    print(f"Saved brief to {output}")


if __name__ == "__main__":
    main()
