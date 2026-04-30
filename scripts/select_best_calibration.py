from __future__ import annotations

# PURPOSE: Pick the best calibration configuration from a calibration_sweep run.
# INPUT:   calibration_sweep.csv (output of calibration_sweep.py).
# METHOD:  Sort rows by official_utility descending, then Brier score ascending.
#          The first row after sorting is the best overall calibration setting.
# OUTPUT:  best_calibration.json -- the winning row, with method, fraction, and all metrics.
# RUN:     python scripts/select_best_calibration.py
#              --sweep-csv outputs/calibration_sweep/calibration_sweep.csv
#              --output outputs/calibration_sweep/best_calibration.json

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    with Path(args.sweep_csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["auroc", "auprc", "patient_auroc", "patient_auprc", "brier", "brier_raw", "utility", "official_utility", "custom_utility", "best_threshold"]:
                if key in row:
                    row[key] = float(row[key])
            for key in ["cal_fraction", "cal_patients"]:
                if key in row:
                    row[key] = float(row[key])
            rows.append(row)

    if not rows:
        raise ValueError("No rows found in sweep CSV.")

    rows.sort(key=lambda r: (r.get("official_utility", 0.0), -r.get("brier", 0.0)), reverse=True)
    best = rows[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Saved best calibration to {output}")


if __name__ == "__main__":
    main()
