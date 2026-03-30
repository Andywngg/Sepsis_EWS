from __future__ import annotations

import argparse
import json
from pathlib import Path

from fpdf import FPDF


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--utility-plot", required=False)
    parser.add_argument("--quality-plot", required=False)
    parser.add_argument("--calibration-plot", required=False)
    parser.add_argument("--leadtime-plot", required=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = load_json(Path(args.metrics))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "Sepsis Early-Warning System (One-Pager)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(
        0,
        6,
        "Goal: predict sepsis hours before onset using ICU vitals and labs, "
        "and optimize an early-warning policy rather than just AUROC.",
    )

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, "Key Results", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, f"AUROC: {metrics.get('metrics', {}).get('auroc', 0):.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"AUPRC: {metrics.get('metrics', {}).get('auprc', 0):.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0,
        6,
        f"Patient AUROC: {metrics.get('patient_level_metrics', {}).get('auroc', 0):.3f}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        6,
        f"Patient AUPRC: {metrics.get('patient_level_metrics', {}).get('auprc', 0):.3f}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(0, 6, f"Brier score: {metrics.get('brier_score', 0):.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Utility score: {metrics.get('utility_score', 0):.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Official utility: {metrics.get('official_utility', 0):.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Accuracy: {metrics.get('accuracy', 0):.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"F-measure: {metrics.get('f_measure', 0):.3f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        0,
        6,
        f"Early detection rate: {metrics.get('early_warning', {}).get('early_detection_rate', 0):.3f}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        6,
        f"False alert rate: {metrics.get('early_warning', {}).get('false_alert_rate', 0):.3f}",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    if args.utility_plot and Path(args.utility_plot).exists():
        pdf.ln(2)
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 7, "Utility Curve", new_x="LMARGIN", new_y="NEXT")
        pdf.image(args.utility_plot, w=150)

    if args.quality_plot and Path(args.quality_plot).exists():
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 7, "Quality Gating Tradeoff", new_x="LMARGIN", new_y="NEXT")
        pdf.image(args.quality_plot, w=150)

    if args.calibration_plot and Path(args.calibration_plot).exists():
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 7, "Calibration Curve", new_x="LMARGIN", new_y="NEXT")
        pdf.image(args.calibration_plot, w=150)

    if args.leadtime_plot and Path(args.leadtime_plot).exists():
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 7, "Lead Time Distribution", new_x="LMARGIN", new_y="NEXT")
        pdf.image(args.leadtime_plot, w=150)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output))
    print(f"Saved one-pager to {output}")


if __name__ == "__main__":
    main()
