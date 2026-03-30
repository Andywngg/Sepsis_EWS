from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--error-analysis", required=False)
    parser.add_argument("--cross-site", required=False)
    parser.add_argument("--calibration-best", required=False)
    parser.add_argument("--ci", required=False)
    parser.add_argument("--policy", required=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = load_json(Path(args.metrics))
    errors = load_json(Path(args.error_analysis)) if args.error_analysis else {}
    cross_site = load_json(Path(args.cross_site)) if args.cross_site else {}
    calib_best = load_json(Path(args.calibration_best)) if args.calibration_best else {}
    ci = load_json(Path(args.ci)) if args.ci else {}
    policy = load_json(Path(args.policy)) if args.policy else {}

    lines = []
    lines.append("# Model Card: Sepsis Early-Warning System\n\n")
    lines.append("## Intended Use\n")
    lines.append(
        "- Research prototype for early sepsis warning using ICU vitals/labs.\n"
        "- Not approved for clinical use.\n\n"
    )
    lines.append("## Data\n")
    lines.append("- PhysioNet/CinC Challenge 2019 ICU time-series.\n")
    if metrics.get("max_patients"):
        lines.append(f"- Max patients used in evaluation: {metrics.get('max_patients')}\n")
    lines.append("\n## Model\n")
    lines.append("- Tree-based gradient boosting classifier with engineered time-series features.\n")
    lines.append("- Early-warning policy optimized by official utility.\n\n")

    lines.append("## Performance (Held-out)\n")
    lines.append(f"- AUROC: {metrics.get('metrics', {}).get('auroc', 0):.3f}\n")
    lines.append(f"- AUPRC: {metrics.get('metrics', {}).get('auprc', 0):.3f}\n")
    lines.append(
        f"- Patient-level AUROC: {metrics.get('patient_level_metrics', {}).get('auroc', 0):.3f}\n"
    )
    lines.append(
        f"- Patient-level AUPRC: {metrics.get('patient_level_metrics', {}).get('auprc', 0):.3f}\n"
    )
    lines.append(f"- Official utility: {metrics.get('official_utility', 0):.3f}\n")
    lines.append(f"- Early detection rate: {metrics.get('early_warning', {}).get('early_detection_rate', 0):.3f}\n")
    lines.append(f"- False alert rate: {metrics.get('early_warning', {}).get('false_alert_rate', 0):.3f}\n")
    lines.append(f"- Median lead time (hours): {metrics.get('early_warning', {}).get('median_lead_time_hours', 0):.2f}\n\n")

    lines.append("## Calibration\n")
    if "brier_score" in metrics:
        lines.append(f"- Brier score: {metrics.get('brier_score', 0):.3f}\n")
    if "brier_score_raw" in metrics:
        lines.append(f"- Brier score (raw): {metrics.get('brier_score_raw', 0):.3f}\n")
    calib = metrics.get("calibration", {})
    if calib:
        lines.append(
            f"- Calibration method: {calib.get('method', 'none')} "
            f"({calib.get('patients', 0)} patients)\n"
        )
    if calib_best:
        extra = ""
        if "max_patients" in calib_best:
            extra = f", max_patients {int(float(calib_best.get('max_patients', 0)))}"
        lines.append(
            f"- Best sweep policy: {calib_best.get('method', 'n/a')} "
            f"(utility {calib_best.get('official_utility', 0):.3f}, "
            f"brier {calib_best.get('brier', 0):.3f}{extra})\n"
        )
    lines.append("\n")

    if errors:
        lines.append("## Error Analysis\n")
        if errors.get("max_patients") is not None:
            lines.append(f"- Error analysis subset size: {errors.get('max_patients')}\n")
        pc = errors.get("patient_confusion", {})
        lines.append(f"- Patient-level TP/FP/FN/TN: {pc.get('tp', 0)}/{pc.get('fp', 0)}/{pc.get('fn', 0)}/{pc.get('tn', 0)}\n")
        lines.append(f"- Sensitivity: {errors.get('sensitivity', 0):.3f}\n")
        lines.append(f"- Specificity: {errors.get('specificity', 0):.3f}\n")
        lines.append(
            f"- Alert burden (alerts per patient-day): {errors.get('alert_burden', {}).get('alerts_per_patient_day', 0):.3f}\n"
        )
        lines.append("\n")

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
        lines.append("## Alert Policies\n")
        sens = policy.get("sensitive_policy", {})
        cons = policy.get("conservative_policy", {})
        if sens:
            lines.append(
                f"- Sensitive: threshold {sens.get('threshold', 0):.2f}, "
                f"early detection {sens.get('early_detection_rate', 0):.3f}, "
                f"alerts/patient-day {sens.get('alerts_per_patient_day', 0):.3f}\n"
            )
        if cons:
            lines.append(
                f"- Conservative: threshold {cons.get('threshold', 0):.2f}, "
                f"early detection {cons.get('early_detection_rate', 0):.3f}, "
                f"alerts/patient-day {cons.get('alerts_per_patient_day', 0):.3f}\n"
            )
        lines.append("\n")

    if cross_site:
        lines.append("## Cross-site Generalization\n")
        lines.append("- Cross-site results use patient-level normalization to mitigate domain shift.\n")
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
        lines.append("\n")

    lines.append("## Limitations\n")
    lines.append("- Retrospective evaluation on ICU time-series; no prospective validation.\n")
    lines.append("- Domain shift across hospitals remains a challenge.\n")
    lines.append("- Missing data patterns can bias alerts.\n\n")

    lines.append("## Ethical Considerations\n")
    lines.append("- Intended for clinical decision support only; should not replace clinician judgment.\n")
    lines.append("- Requires monitoring for bias across sites and patient populations.\n")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")
    print(f"Saved model card to {output}")


if __name__ == "__main__":
    main()
