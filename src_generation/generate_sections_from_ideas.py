#!/usr/bin/env python3
"""
Generate paper sections from frozen research ideas.

Consumes frozen ideas from frozen_ideas/ and generates one JSON file per
idea, model, and section into rag_runs/
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path

from generate import (
    PAPER_SECTION_ORDER,
    ALLOWED_TARGET_SECTIONS,
    CHROMA_DIR,
    COLLECTION,
    EMBED_MODEL,
    ResearchIdea,
    get_collection,
    load_frozen_idea,
    retrieve_context_for_section,
    generate_section,
    save_run,
)


def make_safe_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", model.strip())


def list_idea_files(idea_dir: str) -> list[Path]:
    paths = sorted(Path(idea_dir).glob("*.json"))
    return [p for p in paths if p.is_file()]


def section_output_path(output_dir: str, model: str, idea_id: str, section: str) -> Path:
    safe_model = make_safe_model_name(model)
    safe_idea_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", idea_id.strip())
    safe_section = re.sub(r"[^a-zA-Z0-9_\-]+", "_", section.strip())
    return Path(output_dir) / safe_model / f"{safe_idea_id}__{safe_section}.json"


def generate_one_section_from_idea(
    idea_path: Path,
    model: str,
    target_section: str,
    collection,
    embed_model: str,
    output_dir: str,
    top_k: int,
    max_per_paper: int,
    temperature: float,
) -> Path:
    frozen = load_frozen_idea(str(idea_path))

    idea_id = frozen["idea_id"]
    topic = frozen["topic"]
    chosen_direction = frozen["chosen_direction"]
    refined_problem = frozen["refined_problem"]
    selected_title = frozen["selected_title"]
    abstract = frozen["abstract"]

    research_idea = ResearchIdea(
        chosen_direction=chosen_direction,
        refined_problem=refined_problem,
        titles=[selected_title],
    )

    context_hits = retrieve_context_for_section(
        collection=collection,
        topic=topic,
        title=selected_title,
        abstract=abstract,
        target_section=target_section,
        embed_model=embed_model,
        top_k=top_k,
        max_per_paper=max_per_paper,
    )

    generation_result = generate_section(
        topic=topic,
        chosen_direction=chosen_direction,
        refined_problem=refined_problem,
        title=selected_title,
        abstract=abstract,
        target_section=target_section,
        context_hits=context_hits,
        model=model,
        temperature=temperature,
    )

    saved_path = save_run(
        output_dir=output_dir,
        idea_id=idea_id,
        topic=topic,
        target_section=target_section,
        model=model,
        research_idea=research_idea,
        selected_title=selected_title,
        abstract=abstract,
        context_hits=context_hits,
        generation_result=generation_result,
    )

    return saved_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate sections from frozen ideas for one or more models")
    ap.add_argument("--idea-dir", default="./frozen_ideas", help="Directory containing frozen idea JSON files")
    ap.add_argument("--models", nargs="+", required=True, help="One or more Ollama/HuggingFace models")

    section_group = ap.add_mutually_exclusive_group(required=True)
    section_group.add_argument("--sections", nargs="+", choices=sorted(ALLOWED_TARGET_SECTIONS), help="Sections to generate")
    section_group.add_argument("--all-sections", action="store_true", help=f"Generate all paper sections in order: {', '.join(PAPER_SECTION_ORDER)}")

    ap.add_argument("--output-dir", default="./rag_runs", help="Directory where section JSON files are saved")
    ap.add_argument("--chroma-dir", default=CHROMA_DIR, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default=COLLECTION, help="ChromaDB collection name")

    ap.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model for retrieval")
    ap.add_argument("--top-k", type=int, default=4, help="Final number of retrieved chunks")
    ap.add_argument("--max-per-paper", type=int, default=1, help="Maximum retrieved chunks per paper")
    ap.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    ap.add_argument("--assignment-seed", type=int, default=42, help="Random seed used to assign ideas to models")
    ap.add_argument("--ideas-per-model", type=int, default=200, help="Number of shuffled ideas assigned to each model")
    ap.add_argument("--assignment-offset", type=int, default=0, help="Number of shuffled ideas to skip before assigning ideas to models")

    ap.add_argument("--skip-existing", action="store_true", help="Skip section files that already exist")

    args = ap.parse_args()

    sections = PAPER_SECTION_ORDER if args.all_sections else args.sections

    idea_files = list_idea_files(args.idea_dir)
    if not idea_files:
        raise RuntimeError(f"No frozen idea JSON files found in: {args.idea_dir}")

    collection = get_collection(args.chroma_dir, args.collection)

    rng = random.Random(args.assignment_seed)

    shuffled_ideas = idea_files[:]
    rng.shuffle(shuffled_ideas)

    n_required_ideas = len(args.models) * args.ideas_per_model

    start_rank = args.assignment_offset
    end_rank = start_rank + n_required_ideas

    if end_rank > len(shuffled_ideas):
        raise RuntimeError(
            f"Not enough ideas: need shuffled idea ranks {start_rank} to {end_rank - 1}, "
            f"but only found {len(shuffled_ideas)} ideas."
        )

    selected_ideas = shuffled_ideas[start_rank:end_rank]

    idea_assignments = {}

    for model_idx, model in enumerate(args.models):
        start = model_idx * args.ideas_per_model
        end = start + args.ideas_per_model
        model_ideas = selected_ideas[start:end]

        for idea_path in model_ideas:
            frozen = load_frozen_idea(str(idea_path))
            idea_id = frozen["idea_id"]
            idea_assignments[idea_id] = model

    assignment_counts = Counter(idea_assignments.values())

    print("\nAssignment counts:")
    for model in args.models:
        print(f"  {model}: {assignment_counts[model]}")

    idea_files = selected_ideas
    total_jobs = len(idea_files) * len(sections)
    job_idx = 0

    print("\n=== Random model assignment ===")
    print(f"Seed: {args.assignment_seed}")
    print(f"Assignment offset: {args.assignment_offset}")
    print(f"Models: {', '.join(args.models)}")

    for idea_idx, idea_path in enumerate(idea_files, start=1):
        frozen = load_frozen_idea(str(idea_path))
        idea_id = frozen["idea_id"]
        model = idea_assignments[idea_id]

        print(f"\n[idea {idea_idx}/{len(idea_files)}] {idea_id}")
        print(f"Assigned model: {model}")

        for section in sections:
            job_idx += 1
            out_path = section_output_path(args.output_dir, model, idea_id, section)

            print(
                f"[job {job_idx}/{total_jobs}] "
                f"model={model} section={section} -> {out_path.name}"
            )

            if args.skip_existing and out_path.exists():
                print(f"[skip] already exists: {out_path}")
                continue

            try:
                saved_path = generate_one_section_from_idea(
                    idea_path=idea_path,
                    model=model,
                    target_section=section,
                    collection=collection,
                    embed_model=args.embed_model,
                    output_dir=args.output_dir,
                    top_k=args.top_k,
                    max_per_paper=args.max_per_paper,
                    temperature=args.temperature,
                )

                print(f"[saved] {saved_path}")

            except Exception as e:
                print(f"[error] failed job model={model} idea={idea_id} section={section}: {e}")
                continue

if __name__ == "__main__":
    main()