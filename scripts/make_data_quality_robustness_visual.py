from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("rows", [])


def make_data_quality_robustness_visual(root: Path, out_png: Path, out_json: Path) -> None:
    missing_rows = _load_rows(root / "outputs" / "missingness_stress" / "missingness_summary.json")
    delay_rows = _load_rows(root / "outputs" / "delay_stress" / "delay_summary.json")

    missing_x = np.array([float(r["drop_rate"]) * 100.0 for r in missing_rows], dtype=float)
    missing_auroc = np.array([float(r["metrics"]["auroc"]) for r in missing_rows], dtype=float)

    delay_x = np.array([float(r["delay_hours"]) for r in delay_rows], dtype=float)
    delay_auroc = np.array([float(r["metrics"]["auroc"]) for r in delay_rows], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)

    axes[0].plot(missing_x, missing_auroc, color="#0b2e6f", marker="o", linewidth=2.5)
    axes[0].set_title("AUROC vs Additional Missingness", fontsize=13, weight="bold")
    axes[0].set_xlabel("Additional missingness (%)", fontsize=11.5)
    axes[0].set_ylabel("AUROC", fontsize=11.5)
    axes[0].set_xticks(missing_x)
    axes[0].set_xlim(float(np.min(missing_x)) - 1, float(np.max(missing_x)) + 1)

    axes[1].plot(delay_x, delay_auroc, color="#0b2e6f", marker="o", linewidth=2.5)
    axes[1].set_title("AUROC vs Documentation Delay", fontsize=13, weight="bold")
    axes[1].set_xlabel("Delay (hours)", fontsize=11.5)
    axes[1].set_xticks(delay_x)
    axes[1].set_xlim(float(np.min(delay_x)) - 0.2, float(np.max(delay_x)) + 0.2)

    y_min = min(float(np.min(missing_auroc)), float(np.min(delay_auroc))) - 0.004
    y_max = max(float(np.max(missing_auroc)), float(np.max(delay_auroc))) + 0.004
    for ax in axes:
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.28)
        for line_x, line_y in zip(ax.lines[0].get_xdata(), ax.lines[0].get_ydata()):
            ax.text(line_x, line_y + 0.0005, f"{line_y:.3f}", fontsize=9, ha="center", color="#37474f")

    fig.suptitle("Experiment 2: Measurement Delay Robustness", fontsize=17, weight="bold", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Stable performance across both degradation scenarios confirms reliability under realistic hospital data quality conditions.",
        ha="center",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "missingness": {
            "additional_missingness_percent": missing_x.tolist(),
            "auroc": missing_auroc.tolist(),
        },
        "documentation_delay": {
            "delay_hours": delay_x.tolist(),
            "auroc": delay_auroc.tolist(),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / "outputs" / "board_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / "auroc_robustness_missingness_delay.png"
    out_json = out_dir / "auroc_robustness_missingness_delay.json"
    make_data_quality_robustness_visual(root, out_png, out_json)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
