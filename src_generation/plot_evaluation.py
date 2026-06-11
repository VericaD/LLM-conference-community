#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "faithfulness",
    "cross_section_consistency",
    "scientific_plausibility",
    "experiments_quality",
]


def load_rows(evaluations_dir: str) -> list[dict]:
    rows = []
    base = Path(evaluations_dir)

    for model_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        for path in sorted(model_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            scores = data["scores"]
            row = {
                "model": data["model"],
                "idea_id": data["idea_id"],
            }
            for metric in METRICS:
                row[metric] = scores[metric]
            rows.append(row)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot evaluation results")
    ap.add_argument("--evaluations-dir", default="./evaluations")
    ap.add_argument("--output-dir", default="./plots")
    args = ap.parse_args()

    rows = load_rows(args.evaluations_dir)
    if not rows:
        raise RuntimeError("No evaluation files found.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    # average bar plot per metric
    avg = df.groupby("model")[METRICS].mean()

    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        avg[metric].sort_values(ascending=False).plot(kind="bar")
        plt.ylabel("Average score")
        plt.title(f"Average {metric} by model")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_bar.png")
        plt.close()

    # boxplots
    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        df.boxplot(column=metric, by="model")
        plt.ylabel("Score")
        plt.title(f"{metric} distribution by model")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_box.png")
        plt.close()

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()