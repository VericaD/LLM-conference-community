#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from generate import (
    DEFAULT_GEN_MODEL,
    generate_research_idea,
    generate_abstract,
    save_frozen_idea,
)


def load_topics(topics_file: str) -> list[str]:
    with open(topics_file, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics


def make_idea_id(index: int) -> str:
    return f"idea_{index:03d}"


def generate_frozen_ideas(
    topic: str,
    idea_id: str,
    model: str,
    n_titles: int,
    temperature: float,
    selected_direction: str | None = None,
) -> list[dict]:
    
    research_idea = generate_research_idea(
        topic=topic,
        model=model,
        n_titles=n_titles,
        temperature=temperature,
        selected_direction=selected_direction,
    )

    frozen_ideas = []
    
    for idx, title in enumerate(research_idea.titles, start=1):
        abstract = generate_abstract(
            topic=topic,
            title=title,
            model=model,
            temperature=temperature,
        )

        frozen_ideas.append(
            {
                "idea_id": f"{idea_id}_{idx:02d}",
                    "topic": topic,
                    "chosen_direction": research_idea.chosen_direction,
                    "refined_problem": research_idea.refined_problem,
                    "selected_title": title,
                    "abstract": abstract,
            }
        )

    return frozen_ideas

def generate_frozen_ideas_batch(
    topics: list[str],
    output_dir: str,
    model: str,
    n_titles: int,
    temperature: float,
    start_index: int = 1,
    limit: int | None = None,
    selected_direction: str | None = None,
    skip_existing: bool = True,
) -> None:
    if limit is not None:
        topics = topics[:limit]

    total = len(topics)

    for offset, topic in enumerate(topics, start=0):
        idea_num = start_index + offset
        idea_id = make_idea_id(idea_num)
        output_dir_path = Path(output_dir)
        existing_paths = sorted(output_dir_path.glob(f"{idea_id}_*.json"))

        print(f"\n[{offset + 1}/{total}] {idea_id}")
        print(f"Topic: {topic}")

        if skip_existing and existing_paths:
            print(f"[skip] already exists for prefix {idea_id}:")
            for path in existing_paths:
                print(f"  - {path}")
            continue

        frozen_ideas = generate_frozen_ideas(
            topic=topic,
            idea_id=idea_id,
            model=model,
            n_titles=n_titles,
            temperature=temperature,
            selected_direction=selected_direction,
        )

        for frozen in frozen_ideas:
            path = save_frozen_idea(
                output_dir=output_dir,
                idea_id=frozen["idea_id"],
                topic=frozen["topic"],
                chosen_direction=frozen["chosen_direction"],
                refined_problem=frozen["refined_problem"],
                selected_title=frozen["selected_title"],
                abstract=frozen["abstract"],
            )

            print(f"Chosen direction: {frozen['chosen_direction']}")
            print(f"Refined problem: {frozen['refined_problem']}")
            print(f"Selected title: {frozen['selected_title']}")
            print(f"Saved: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate frozen paper ideas from a manual topic list")
    ap.add_argument("--topics-file", required=True, help="Text file with one topic per line")
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL, help="Ollama model used for ideation")
    ap.add_argument("--output-dir", default="./frozen_ideas_v2", help="Directory for frozen idea JSON files")
    ap.add_argument("--n-titles", type=int, default=5, help="Number of candidate titles to generate")
    ap.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    ap.add_argument("--start-index", type=int, default=1, help="Starting idea index")
    ap.add_argument("--limit", type=int, default=None, help="Maximum number of topics to process")
    ap.add_argument("--selected-direction", default=None, help="Optional fixed broad direction")
    ap.add_argument("--skip-existing", action="store_true", help="Skip idea files that already exist")
    args = ap.parse_args()

    topics = load_topics(args.topics_file)
    if not topics:
        raise RuntimeError("No topics found in topics file.")

    generate_frozen_ideas_batch(
        topics=topics,
        output_dir=args.output_dir,
        model=args.model,
        n_titles=args.n_titles,
        temperature=args.temperature,
        start_index=args.start_index,
        limit=args.limit,
        selected_direction=args.selected_direction,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()