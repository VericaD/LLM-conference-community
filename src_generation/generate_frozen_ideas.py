#!/usr/bin/env python3
"""
Generate and save frozen research ideas from ICLR keyword metadata.

The script extracts and normalizes frequent topics from SQLite,
refines them with an LLM, and generates titles and abstracts that are saved as
reusable JSON idea files.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

from generate import (
    DEFAULT_GEN_MODEL,
    chat_generate,
    generate_research_idea,
    generate_abstract,
    save_frozen_idea,
)

TOPICS_FILENAME = "topics.txt"

GENERIC_KEYWORDS = {
    "deep learning",
    "machine learning",
    "learning",
    "training",
    "neural networks",
    "neural network",
    "deep neural networks",
    "deep neural network",
    "deep network",
    "network",
    "theory",
    "framework",
    "dataset",
    "evaluation",
    "classification",
    "regression",
}

NORMALIZE_KEYWORDS = {
    "gan": "generative adversarial networks",
    "gans": "generative adversarial networks",
    "generative adversarial network": "generative adversarial networks",
    "generative adversarial nets": "generative adversarial networks",
    "adversarial networks": "generative adversarial networks",

    "vae": "variational autoencoders",
    "variational autoencoder": "variational autoencoders",
    "variational autoencoders": "variational autoencoders",
    "variational auto encoder": "variational autoencoders",
    "variational auto encoders": "variational autoencoders",

    "cnn": "convolutional neural networks",
    "convolutional neural network": "convolutional neural networks",
    "convolutional networks": "convolutional neural networks",

    "rnn": "recurrent neural networks",
    "recurrent neural network": "recurrent neural networks",
    "recurrent networks": "recurrent neural networks",
    "recurrent network": "recurrent neural networks",

    "lstm": "long short-term memory networks",

    "sgd": "stochastic gradient descent",

    "nlp": "natural language processing",
    "nmt": "neural machine translation",

    "rl": "reinforcement learning",
    "reinforcement learning": "reinforcement learning",
    "deep reinforcement learning": "reinforcement learning",

    "meta learning": "meta-learning",

    "automl": "automated machine learning",

    "optimisation": "optimization",
    "nonconvex optimization": "non-convex optimization",
    "non convex optimization": "non-convex optimization",

    "multi task": "multi-task learning",
    "multi task learning": "multi-task learning",
    "multitask learning": "multi-task learning",

    "model based reinforcement learning": "model-based reinforcement learning",

    "sentence embedding": "sentence embeddings",

    "question answering": "question answering",
    "sequence to sequence": "sequence to sequence",
}

def normalize_keyword(keyword: str) -> str:
    keyword = keyword.strip().lower()
    keyword = keyword.replace("-", " ")
    keyword = " ".join(keyword.split())

    return NORMALIZE_KEYWORDS.get(keyword, keyword)

def load_saved_topics(output_dir: str) -> list[str] | None:
    path = Path(output_dir) / TOPICS_FILENAME

    if not path.exists():
        return None

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def load_topics_from_keywords_db(
    db_path: str,
    sql_limit: int = 100,
    min_count: int = 1,
) -> list[str]:
    """
    Extract frequent OpenReview keywords from the SQLite database and use them
    as generation topics.

    The keywords are lightly normalized and duplicate aliases are merged.
    """

    query = """
    SELECT
      lower(trim(k.value)) AS keyword,
      COUNT(*) AS count
    FROM papers p
    JOIN json_each(json_extract(p.raw_json, '$.content.keywords')) AS k
    WHERE k.value IS NOT NULL
      AND trim(k.value) <> ''
    GROUP BY lower(trim(k.value))
    HAVING count >= ?
    ORDER BY count DESC
    LIMIT ?
    """

    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(query, (min_count, sql_limit)).fetchall()
    finally:
        con.close()

    merged_counts: dict[str, int] = {}

    for keyword, count in rows:
        topic = normalize_keyword(keyword)

        if topic in GENERIC_KEYWORDS:
            continue

        merged_counts[topic] = merged_counts.get(topic, 0) + int(count)

    sorted_topics = sorted(
        merged_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    topics = [topic for topic, count in sorted_topics]
    return topics


def parse_topic_lines(text: str) -> list[str]:
    topics = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove bullets, numbering, markdown decoration, and quotes.
        line = re.sub(r"^\s*[-*•]\s*", "", line)
        line = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", line)
        line = line.strip().strip('"').strip("'")

        if not line:
            continue

        # Avoid accidental explanatory lines.
        lower = line.lower()
        if lower.startswith("here are"):
            continue
        if lower.startswith("topics:"):
            continue

        topics.append(line)

    # Remove duplicates while preserving order.
    unique_topics = []
    seen = set()

    for topic in topics:
        key = topic.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_topics.append(topic)

    return unique_topics


def expand_topics_with_llm(
    keywords: list[str],
    model: str,
    n_topics: int,
    temperature: float = 0.2,
) -> list[str]:
    """
    Ask the LLM to turn extracted ICLR keywords into more precise research-topic
    phrases. This is only topic refinement, not full idea generation.
    """

    system_prompt = (
        "You are a careful machine learning researcher helping build a "
        "data-driven paper generation pipeline."
    )

    user_prompt = f"""
The following keywords were extracted from ICLR paper metadata.

Your task is to rewrite them into precise, diverse research topics suitable for generating ICLR-style machine learning paper ideas.

Constraints:
- Use the provided keywords as the main source of inspiration.
- Do not invent unrelated research areas.
- Prefer concrete research topics over broad keywords.
- Merge related keywords when useful.
- Avoid duplicate or near-duplicate topics.
- Each topic must be one short phrase, not a full sentence.
- Return exactly {n_topics} topics.
- Return one topic per line.
- Do not number the lines.
- Do not add explanations.

Keywords:
{chr(10).join(f"- {keyword}" for keyword in keywords)}
"""

    result = chat_generate(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt.strip(),
        temperature=temperature,
    )

    raw = result.text

    topics = parse_topic_lines(raw)

    if not topics:
        raise RuntimeError("The LLM did not return any valid expanded topics.")

    return topics[:n_topics]


def save_topics_file(topics: list[str], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(topics) + "\n", encoding="utf-8")


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

        print(f"\n[{offset + 1}/{total}] {idea_id}: {topic}")

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

            print(f"Saved {frozen['idea_id']}: {path}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate frozen paper ideas from ICLR keyword metadata")

    ap.add_argument("--db", required=True, help="Path to the SQLite database")

    ap.add_argument("--sql-limit", type=int, default=200, help="Number of raw keywords to retrieve from SQL")
    ap.add_argument("--min-keyword-count", type=int, default=1, help="Minimum keyword frequency")

    ap.add_argument("--model", default=DEFAULT_GEN_MODEL, help="Ollama model used for ideation")

    ap.add_argument("--output-dir", default="./frozen_ideas", help="Directory for frozen idea JSON files")
    ap.add_argument("--n-titles", type=int, default=5, help="Number of candidate titles to generate")

    ap.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    ap.add_argument("--start-index", type=int, default=1, help="Starting idea index")
    ap.add_argument("--limit", type=int, default=None, help="Maximum number of topics to process")
    ap.add_argument("--selected-direction", default=None, help="Optional fixed broad direction")
    ap.add_argument("--skip-existing", action="store_true", help="Skip idea files that already exist")

    args = ap.parse_args()

    saved_topics = load_saved_topics(args.output_dir)

    if saved_topics:
        topics = saved_topics
        print(f"Loaded {len(topics)} existing topics from {Path(args.output_dir) / TOPICS_FILENAME}")
    else:
        topics = load_topics_from_keywords_db(
            db_path=args.db,
            sql_limit=args.sql_limit,
            min_count=args.min_keyword_count,
        )

        print(f"Loaded {len(topics)} normalized keywords from metadata in {args.db}")

        n_requested_topics = args.limit if args.limit is not None else len(topics)

        topics = expand_topics_with_llm(
            keywords=topics,
            model=args.model,
            n_topics=n_requested_topics,
            temperature=0.2,
        )

        print(f"Generated {len(topics)} research topics with the LLM")

        if not topics:
            raise RuntimeError("No topics found.")

        topics_to_save = topics[:args.limit] if args.limit is not None else topics
        topics_path = Path(args.output_dir) / TOPICS_FILENAME
        save_topics_file(topics_to_save, str(topics_path))
        print(f"Saved final topic list to {topics_path}")

    topics_to_generate = topics[:args.limit] if args.limit is not None else topics

    generate_frozen_ideas_batch(
        topics=topics_to_generate,
        output_dir=args.output_dir,
        model=args.model,
        n_titles=args.n_titles,
        temperature=args.temperature,
        start_index=args.start_index,
        limit=None,
        selected_direction=args.selected_direction,
        skip_existing=args.skip_existing,
    )

if __name__ == "__main__":
    main()