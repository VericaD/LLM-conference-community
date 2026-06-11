#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TECHNICAL_TERMS = {
    "model", "models", "objective", "loss", "optimization", "optimizer",
    "embedding", "embeddings", "feature", "features", "representation",
    "representations", "training", "baseline", "baselines", "dataset",
    "datasets", "accuracy", "policy", "policies", "reward", "rewards",
    "contrastive", "metric", "metrics", "kernel", "kernels", "generalization",
    "architecture", "architectures", "gradient", "gradients", "evaluation",
    "experiment", "experiments", "classification", "clustering"
}

DATASET_TERMS = {
    "cifar", "imagenet", "mnist", "fashion-mnist", "stl", "svhn",
    "coco", "cityscapes", "kitti", "scanet", "librispeech", "squad",
    "wikitext", "mujoco", "atari"
}

METRIC_TERMS = {
    "accuracy", "f1", "precision", "recall", "auc", "bleu", "rouge",
    "ari", "nmi", "reward", "success rate", "error rate", "mse", "rmse"
}

BASELINE_TERMS = {
    "simclr", "moco", "byol", "swav", "dqn", "ppo", "sac", "td3",
    "resnet", "bert", "transformer", "vae", "gan", "deepcluster", "cpc"
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_paper_files(assembled_dir: str) -> list[Path]:
    base = Path(assembled_dir)
    files = []
    for model_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        files.extend(sorted(model_dir.glob("*.json")))
    return files


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-\_]*", text.lower())


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+\s+|\n+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)


def word_count(text: str) -> int:
    return len(tokenize_words(text))


def lexical_diversity(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def count_term_hits(text: str, terms: set[str]) -> int:
    text_l = text.lower()
    hits = 0
    for term in terms:
        # simple word/phrase count
        hits += len(re.findall(rf"\b{re.escape(term)}\b", text_l))
    return hits


def technical_density(text: str) -> float:
    words = tokenize_words(text)
    if not words:
        return 0.0
    hits = count_term_hits(text, TECHNICAL_TERMS)
    return 100.0 * hits / len(words)


def numeric_specificity(text: str) -> int:
    count_numbers = len(re.findall(r"\b\d+(\.\d+)?%?\b", text))
    count_citations = len(re.findall(r"\([A-Za-z][^)]*\d{4}[^)]*\)", text))
    count_compare = len(re.findall(r"\b(outperform|improve|improves|improved|compared to|higher than|lower than)\b", text.lower()))
    return count_numbers + count_citations + count_compare


def section_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform([a, b])
    sim = cosine_similarity(X[0:1], X[1:2])[0][0]
    return float(sim)


def experiments_realism_features(text: str) -> dict:
    return {
        "experiments_dataset_mentions": count_term_hits(text, DATASET_TERMS),
        "experiments_metric_mentions": count_term_hits(text, METRIC_TERMS),
        "experiments_baseline_mentions": count_term_hits(text, BASELINE_TERMS),
        "experiments_numeric_specificity": numeric_specificity(text),
    }


def experiments_realism_score(features: dict) -> float:
    # simple interpretable score
    score = 0.0
    score += min(features["experiments_dataset_mentions"], 3)
    score += min(features["experiments_metric_mentions"], 3)
    score += min(features["experiments_baseline_mentions"], 3)
    score += min(features["experiments_numeric_specificity"], 4)
    return score


def build_row(paper: dict) -> dict:
    sections = paper.get("sections", {})

    intro = sections.get("introduction", "")
    method = sections.get("method", "")
    experiments = sections.get("experiments", "")
    conclusion = sections.get("conclusion", "")

    full_text = "\n\n".join([intro, method, experiments, conclusion]).strip()

    exp_feats = experiments_realism_features(experiments)

    row = {
        "idea_id": paper.get("idea_id"),
        "model": paper.get("model"),

        "intro_words": word_count(intro),
        "method_words": word_count(method),
        "experiments_words": word_count(experiments),
        "conclusion_words": word_count(conclusion),
        "total_words": word_count(full_text),

        "intro_sentences": sentence_count(intro),
        "method_sentences": sentence_count(method),
        "experiments_sentences": sentence_count(experiments),
        "conclusion_sentences": sentence_count(conclusion),

        "lexical_diversity": lexical_diversity(full_text),
        "technical_density": technical_density(full_text),
        "numeric_specificity": numeric_specificity(full_text),

        "intro_method_similarity": section_similarity(intro, method),
        "method_experiments_similarity": section_similarity(method, experiments),
        "experiments_conclusion_similarity": section_similarity(experiments, conclusion),

        **exp_feats,
    }

    row["experiments_realism_score"] = experiments_realism_score(exp_feats)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute automatic metrics for assembled synthetic papers")
    ap.add_argument("--assembled-dir", default="./assembled_papers")
    ap.add_argument("--output-csv", default="./automatic_metrics.csv")
    args = ap.parse_args()

    paper_files = list_paper_files(args.assembled_dir)
    if not paper_files:
        raise RuntimeError("No assembled papers found.")

    rows = []
    for idx, path in enumerate(paper_files, start=1):
        print(f"[{idx}/{len(paper_files)}] {path}")
        paper = load_json(path)
        rows.append(build_row(paper))

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()