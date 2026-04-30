from __future__ import annotations

# PURPOSE: Generate synthetic patient data for demos and testing.
# WHY:     The real PhysioNet data cannot be shared publicly (it requires a data use agreement).
#          For demo purposes (science fair, code review), synthetic patients that look
#          statistically similar to real ICU data let the demo app work without the real dataset.
# OUTPUT:  A directory of .psv files in the same format as the real data.
# RUN:     python scripts/make_synthetic.py
#              --output-dir data/synthetic --n-patients 100

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patients", type=int, default=50)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--sepsis-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = ["HR", "Temp", "Resp", "SBP", "DBP", "WBC", "Lactate", "SepsisLabel"]

    for i in range(args.patients):
        has_sepsis = rng.random() < args.sepsis_rate
        onset = rng.integers(args.hours // 2, args.hours) if has_sepsis else None

        hr = rng.normal(80, 10, args.hours)
        temp = rng.normal(36.8, 0.4, args.hours)
        resp = rng.normal(16, 3, args.hours)
        sbp = rng.normal(120, 12, args.hours)
        dbp = rng.normal(75, 8, args.hours)
        wbc = rng.normal(7, 1.5, args.hours)
        lactate = rng.normal(1.2, 0.3, args.hours)

        if has_sepsis and onset is not None:
            ramp = np.linspace(0, 1, args.hours - onset)
            hr[onset:] += 20 * ramp
            temp[onset:] += 1.2 * ramp
            resp[onset:] += 6 * ramp
            sbp[onset:] -= 15 * ramp
            wbc[onset:] += 4 * ramp
            lactate[onset:] += 1.5 * ramp

        labels = np.zeros(args.hours, dtype=int)
        if has_sepsis and onset is not None:
            labels[onset:] = 1

        data = np.vstack([hr, temp, resp, sbp, dbp, wbc, lactate, labels]).T
        df = pd.DataFrame(data, columns=columns)

        # Randomly drop 10% values to simulate missingness
        mask = rng.random(df.shape) < 0.1
        df = df.mask(mask)
        df["SepsisLabel"] = labels

        df.to_csv(out_dir / f"patient_{i:04d}.psv", sep="|", index=False)

    print(f"Wrote synthetic data to {out_dir}")


if __name__ == "__main__":
    main()
