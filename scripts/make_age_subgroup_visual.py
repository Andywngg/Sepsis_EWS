from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_age_subgroup_visual(out_png: Path, out_json: Path) -> None:
    age_groups = ["Under 40", "40-59", "60-79", "80 and above"]
    auroc = [0.853, 0.907, 0.885, 0.805]
    utility = [0.432, 0.341, 0.277, 0.214]
    n = [37, 162, 222, 79]

    x = np.arange(len(age_groups))
    colors = ["#0b2e6f"] * len(age_groups)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    bars = ax.bar(x, utility, color=colors, width=0.62)
    ax.set_title("Utility Score by Age Group", fontsize=20, weight="bold", pad=14)
    ax.set_xlabel("Age Group", fontsize=13)
    ax.set_ylabel("Utility Score", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(age_groups, fontsize=12)
    ax.set_ylim(0, 0.5)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h - 0.03,
            f"{h:.3f}",
            ha="center",
            va="top",
            fontsize=11,
            color="white",
            weight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"n={n[i]}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#37474f",
        )

    # Trend line to emphasize decline toward older ages.
    ax.plot(x, utility, color="#90a4ae", linewidth=2.0, alpha=0.75)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    payload = {
        "age_groups": age_groups,
        "auroc": auroc,
        "utility_score": utility,
        "n": n,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "outputs" / "board_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "utility_score_by_age_group.png"
    out_json = out_dir / "utility_score_by_age_group.json"
    make_age_subgroup_visual(out_png, out_json)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
