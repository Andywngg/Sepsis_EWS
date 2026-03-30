from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--improved", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    base = load_metrics(Path(args.baseline))
    imp = load_metrics(Path(args.improved))

    rows = [
        {
            "model": "baseline",
            "auroc": base.get("metrics", {}).get("auroc", 0),
            "auprc": base.get("metrics", {}).get("auprc", 0),
            "utility": base.get("utility_score", 0),
            "official_utility": base.get("official_utility", 0),
            "accuracy": base.get("accuracy", 0),
            "f_measure": base.get("f_measure", 0),
            "early_detection_rate": base.get("early_warning", {}).get("early_detection_rate", 0),
            "false_alert_rate": base.get("early_warning", {}).get("false_alert_rate", 0),
        },
        {
            "model": "improved",
            "auroc": imp.get("metrics", {}).get("auroc", 0),
            "auprc": imp.get("metrics", {}).get("auprc", 0),
            "utility": imp.get("utility_score", 0),
            "official_utility": imp.get("official_utility", 0),
            "accuracy": imp.get("accuracy", 0),
            "f_measure": imp.get("f_measure", 0),
            "early_detection_rate": imp.get("early_warning", {}).get("early_detection_rate", 0),
            "false_alert_rate": imp.get("early_warning", {}).get("false_alert_rate", 0),
        },
    ]

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_md = Path(args.output_md)
    header = "| model | auroc | auprc | utility | official_utility | accuracy | f_measure | early_detection_rate | false_alert_rate |\n"
    sep = "|---|---|---|---|---|---|---|---|---|\n"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {model} | {auroc:.3f} | {auprc:.3f} | {utility:.3f} | {official_utility:.3f} | {accuracy:.3f} | {f_measure:.3f} | {early_detection_rate:.3f} | {false_alert_rate:.3f} |\n".format(
                **row
            )
        )
    out_md.write_text("".join(lines), encoding="utf-8")

    print(f"Saved comparison to {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
