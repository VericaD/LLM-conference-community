#!/usr/bin/env python3
"""
Prepare the topic list for the next generation iteration.

The script reads review-pipeline JSON files from a completed iteration, keeps
the topics of accepted papers, removes duplicates, and saves them as a
`topics.txt` file for the next iteration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_paper_path(input_path: str, base_dir: Path | None) -> Path:
    p = Path(input_path)
    if base_dir is not None and not p.exists():
        p = base_dir / p.name
    return p


def extract_accepted_topics(
    review_outputs_dir: Path,
    topic_field: str,
    base_dir: Path | None,
) -> tuple[list[str], int, int, int]:
    
    files = sorted(review_outputs_dir.rglob("*.review_pipeline.json"))

    n_total = len(files)
    n_accepted = 0
    n_unresolved = 0
    topics: list[str] = []
    seen: set[str] = set()

    for path in files:
        data = load_json(path)
        decision = data.get("meta_review", {}).get("decision")
        if decision != "accept":
            continue
        n_accepted += 1

        input_path = data.get("input_path")
        if not input_path:
            n_unresolved += 1
            continue

        resolved = resolve_paper_path(input_path, base_dir)
        try:
            paper = load_json(resolved)
        except (FileNotFoundError, json.JSONDecodeError):
            n_unresolved += 1
            continue

        topic = paper.get(topic_field)
        if not topic or not isinstance(topic, str):
            n_unresolved += 1
            continue

        key = topic.strip().lower()
        if key not in seen:
            seen.add(key)
            topics.append(topic.strip())

    return topics, n_accepted, n_total, n_unresolved


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare next-iteration topics from accepted papers")

    ap.add_argument("--review-outputs-dir", type=Path, required=True, help="Directory containing review-pipeline JSON files from the completed iteration")
    ap.add_argument("--output-topics-file", type=Path, required=True, help="Path where the next iteration's topics.txt file is saved")
    ap.add_argument("--topic-field", default="topic", choices=["topic", "chosen_direction"], help="Paper field used as the next iteration's topic")
    ap.add_argument("--assembled-papers-base-dir", type=Path, default=None, help="Optional base directory used to resolve assembled paper paths")
    ap.add_argument("--max-topics", type=int, default=None, help="Optional maximum number of topics carried into the next iteration")

    args = ap.parse_args()

    topics, n_accepted, n_total, n_unresolved = extract_accepted_topics(
        args.review_outputs_dir, args.topic_field, args.assembled_papers_base_dir
    )

    print(f"Reviewed papers found: {n_total}")
    print(f"Accepted: {n_accepted}")
    if n_unresolved:
        print(f"Could not resolve original paper file for {n_unresolved} accepted paper(s) "
              f"(check --assembled-papers-base-dir)")
    print(f"Unique '{args.topic_field}' values among accepted papers: {len(topics)}")

    if args.max_topics is not None and len(topics) > args.max_topics:
        print(f"Capping at --max-topics {args.max_topics} (first-seen order)")
        topics = topics[: args.max_topics]

    if not topics:
        print("No topics extracted -- nothing written. Check decisions/paths above.")
        return

    args.output_topics_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_topics_file.write_text("\n".join(topics) + "\n", encoding="utf-8")
    print(f"Wrote {len(topics)} topics to {args.output_topics_file}")


if __name__ == "__main__":
    main()