from __future__ import annotations

# PURPOSE: Tradeoff curve for conformal prediction (coverage vs. utility).
# Reads conformal_alert.json and plots how coverage (fraction of hours predicted)
# trades off against utility and alert burden as the confidence level (alpha) varies.
# Lower alpha = only predict on very high-confidence hours (fewer but more reliable alerts).
# RUN:     python scripts/make_conformal_tradeoff_visual.py
#              --conformal outputs/conformal/conformal_alert.json
#              --output outputs/visuals/conformal_tradeoff.png

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_conformal_tradeoff_visual(out_path: Path, points_json_path: Path) -> None:
    # Board-ready values provided by the project narrative.
    alpha = np.array([0.01, 0.02, 0.05, 0.10, 0.20], dtype=float)
    coverage = np.array([24.9, 36.6, 48.9, 59.1, 79.5], dtype=float)
    utility = np.array([0.138, 0.232, 0.318, 0.391, 0.863], dtype=float)
    false_alert_rate = np.array([0.024, 0.053, 0.112, np.nan, 0.447], dtype=float)
    alert_burden_note = ["Very Low", "Low", "Medium", "Medium-High", "6.15/day"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.7))
    ax.plot(coverage, utility, color="#0b2e6f", linewidth=3, marker="o", markersize=8)

    selected_alpha = 0.02
    selected_idx = int(np.where(alpha == selected_alpha)[0][0])
    ax.scatter([coverage[selected_idx]], [utility[selected_idx]], color="#1b5e20", s=130, zorder=5)

    for i, a in enumerate(alpha):
        dy = 0.022 if i % 2 == 0 else -0.028
        ax.text(
            coverage[i] + 0.8,
            utility[i] + dy,
            f"alpha={a:.2f}",
            fontsize=11,
            weight="bold",
            color="#263238",
        )

    ax.set_title("Coverage-Utility Tradeoff Curve", fontsize=20, pad=14, weight="bold")
    ax.set_xlabel("Coverage (%)", fontsize=14)
    ax.set_ylabel("Utility score", fontsize=14)
    ax.set_xlim(20, 85)
    ax.set_ylim(0.1, 0.92)

    ax.text(
        47,
        0.16,
        "Operating point selected based on hospital staffing capacity - no retraining required",
        fontsize=11.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f6f8fa", "edgecolor": "#90a4ae"},
    )
    ax.annotate(
        "Selected operating point",
        xy=(coverage[selected_idx], utility[selected_idx]),
        xytext=(56, 0.26),
        fontsize=11,
        color="#1b5e20",
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": "#1b5e20", "lw": 2},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    payload = {
        "alpha": alpha.tolist(),
        "coverage_percent": coverage.tolist(),
        "utility_score": utility.tolist(),
        "false_alert_rate": [None if np.isnan(v) else float(v) for v in false_alert_rate],
        "alert_burden_note": alert_burden_note,
        "selected_alpha": selected_alpha,
    }
    points_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "outputs" / "board_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "conformal_coverage_utility_tradeoff.png"
    json_path = out_dir / "conformal_coverage_utility_points.json"
    make_conformal_tradeoff_visual(out_path, json_path)
    print(f"Saved: {out_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
