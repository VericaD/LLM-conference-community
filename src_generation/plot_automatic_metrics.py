#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "total_words",
    "lexical_diversity",
    "technical_density",
    "experiments_realism_score",
    "intro_method_similarity",
    "method_experiments_similarity",
    "experiments_conclusion_similarity",
    "numeric_specificity",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot automatic metrics by model")
    ap.add_argument("--metrics-csv", default="./automatic_metrics.csv")
    ap.add_argument("--output-dir", default="./plots_auto")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("model")[METRICS].mean()

    # bar plots
    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        grouped[metric].sort_values(ascending=False).plot(kind="bar")
        plt.ylabel(metric)
        plt.title(f"Average {metric} by model")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_bar.png")
        plt.close()

    # boxplots
    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        df.boxplot(column=metric, by="model")
        plt.ylabel(metric)
        plt.title(f"{metric} distribution by model")
        plt.suptitle("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_box.png")
        plt.close()

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()