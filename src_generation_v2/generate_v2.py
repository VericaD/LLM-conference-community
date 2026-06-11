#!/usr/bin/env python3
"""
RAG pipeline for two workflows:

1) Idea-bank generation from a file of broad research directions
   - for each broad direction, generate multiple refined problems
   - for each refined problem, generate multiple titles
   - for each title, generate one abstract
   - save every (direction, refined problem, title, abstract) as a frozen idea JSON

2) Section generation from one frozen idea
   - load one frozen idea JSON
   - retrieve section-relevant context from ChromaDB
   - generate one target section with a local Ollama model

Examples:
    # Generate an idea bank only
    python generate.py \
      --directions-file topics.txt \
      --n-refined-problems 2 \
      --n-titles-per-problem 3 \
      --generate-ideas-only

    # Generate an idea bank, then continue with one frozen idea later
    python generate.py \
      --directions-file topics.txt \
      --n-refined-problems 2 \
      --n-titles-per-problem 3

    # Generate one section from a frozen idea
    python generate.py \
      --idea-file frozen_ideas/idea_0001.json \
      --target-section introduction \
      --model qwen2.5:7b

Requirements:
    pip install chromadb ollama
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
import ollama


# ---------------------------
# Dataclasses
# ---------------------------

@dataclass
class RefinedProblemBundle:
    """One broad direction expanded into one concrete problem and several titles."""

    broad_direction: str
    refined_problem: str
    titles: list[str]


@dataclass
class FrozenIdea:
    """A single paper seed that can later be used for section generation."""

    idea_id: str
    broad_direction: str
    refined_problem: str
    title: str
    abstract: str


# ---------------------------
# Config
# ---------------------------

EMBED_MODEL = "nomic-embed-text"
DEFAULT_GEN_MODEL = "qwen2.5:7b"
CHROMA_DIR = "./chroma_iclr"
COLLECTION = "iclr_papers"

ALLOWED_TARGET_SECTIONS = {
    "introduction",
    "related_work",
    "background",
    "method",
    "experiments",
    "conclusion",
}

SECTION_RETRIEVAL_MAP = {
    "introduction": ["abstract", "introduction", "related_work", "background"],
    "related_work": ["related_work", "background", "introduction"],
    "background": ["background", "introduction", "related_work"],
    "method": ["abstract", "background", "method"],
    "experiments": ["method", "experiments", "conclusion"],
    "conclusion": ["abstract", "experiments", "conclusion"],
}


# ---------------------------
# Utility helpers
# ---------------------------


def parse_numbered_list(text: str) -> list[str]:
    """Parse a simple numbered list returned by the model.

    Expected formats include:
      1. ...
      2) ...
      3 - ...
    """
    items: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        cleaned = re.sub(r"^\d+\s*[\.\)\-:]\s*", "", line).strip().strip('"')
        if cleaned:
            items.append(cleaned)

    return items



def parse_numbered_titles(text: str) -> list[str]:
    """Backward-compatible helper kept for title parsing."""
    return parse_numbered_list(text)



def clean_title_candidates(titles: list[str]) -> list[str]:
    """Remove clearly malformed title candidates."""
    cleaned: list[str] = []

    for title in titles:
        t = title.strip().strip('"')
        lower = t.lower()

        if not t:
            continue
        if "chosen direction" in lower:
            continue
        if "refined problem" in lower:
            continue
        if len(t) < 10:
            continue

        cleaned.append(t)

    return cleaned



def load_directions(path: str) -> list[str]:
    """Load broad directions from a plain-text file, one direction per line."""
    with open(path, "r", encoding="utf-8") as f:
        directions = [line.strip() for line in f if line.strip()]

    if not directions:
        raise ValueError(f"No directions found in file: {path}")

    return directions


# ---------------------------
# Chroma helpers
# ---------------------------


def get_collection(chroma_dir: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)



def embed_query(query: str, model: str) -> list[float]:
    response = ollama.embed(model=model, input=[query])
    return response["embeddings"][0]



def build_where_clause(
    section_names: Optional[list[str]] = None,
    paper_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> Optional[dict]:
    clauses = []

    if section_names:
        if len(section_names) == 1:
            clauses.append({"section_name": section_names[0]})
        else:
            clauses.append({"$or": [{"section_name": name} for name in section_names]})

    if paper_id:
        clauses.append({"paper_id": paper_id})

    if chunk_type:
        clauses.append({"chunk_type": chunk_type})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}



def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    embed_model: str,
    n_results: int = 5,
    section_names: Optional[list[str]] = None,
    paper_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> list[dict]:
    query_vec = embed_query(query, embed_model)

    where = build_where_clause(
        section_names=section_names,
        paper_id=paper_id,
        chunk_type=chunk_type,
    )

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "score": 1 - dist,
                "document": doc,
                "metadata": meta,
            }
        )
    return hits



def deduplicate_hits_by_text(hits: list[dict], preview_chars: int = 300) -> list[dict]:
    seen = set()
    filtered = []

    for item in hits:
        key = item["document"][:preview_chars].strip()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)

    return filtered



def diversify_results(
    hits: list[dict],
    max_results: int = 5,
    max_per_paper: int = 1,
) -> list[dict]:
    selected = []
    per_paper_counts: dict[str, int] = {}

    for item in hits:
        paper_id = item["metadata"].get("paper_id")
        count = per_paper_counts.get(paper_id, 0)

        if count >= max_per_paper:
            continue

        selected.append(item)
        per_paper_counts[paper_id] = count + 1

        if len(selected) >= max_results:
            break

    return selected



def format_context(hits: list[dict]) -> str:
    parts = []
    for i, item in enumerate(hits, start=1):
        meta = item["metadata"]
        parts.append(
            f"[Chunk {i} | score={item['score']:.3f} | "
            f"paper_id={meta.get('paper_id')} | "
            f"title={meta.get('title')} | "
            f"section={meta.get('section_name')}]\n"
            f"{item['document']}"
        )
    return "\n\n" + ("\n\n" + "-" * 80 + "\n\n").join(parts) if parts else ""


# ---------------------------
# Ollama generation helpers
# ---------------------------


def chat_generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": temperature,
        },
    )
    return response["message"]["content"].strip()


# ---------------------------
# Prompt builders
# ---------------------------


def build_refined_problem_prompts(
    broad_direction: str,
    n_refined_problems: int,
) -> tuple[str, str]:
    system_prompt = (
        "You are a careful machine learning researcher designing realistic "
        "ICLR-style research problems. Stay grounded in plausible mainstream "
        "machine learning research."
    )

    user_prompt = f"""
Broad research direction:
{broad_direction}

Task:
Generate {n_refined_problems} distinct refined research problems.

Constraints:
- Each refined problem must be concrete enough for a single conference paper.
- They must be meaningfully different from each other.
- They must stay within the given broad direction.
- They must be realistic and technically plausible.
- Avoid vague, generic, or overly ambitious ideas.

Formatting rules:
- Output only a numbered list.
- One refined problem per line.
1. ...
2. ...
3. ...
"""
    return system_prompt.strip(), user_prompt.strip()



def build_title_generation_prompts(
    broad_direction: str,
    refined_problem: str,
    n_titles: int,
) -> tuple[str, str]:
    system_prompt = (
        "You are a careful machine learning researcher designing realistic "
        "ICLR-style paper titles."
    )

    user_prompt = f"""
Broad research direction:
{broad_direction}

Refined problem:
{refined_problem}

Task:
Generate {n_titles} distinct and realistic paper titles for this refined problem.

Constraints:
- Titles must be specific, technical, and plausible.
- Titles should sound like papers that could realistically appear at ICLR.
- Titles should remain faithful to the refined problem.
- Avoid vague, generic, or overly ambitious titles.
- The titles should differ meaningfully in framing, while describing the same paper family.

Formatting rules:
- Output only a numbered list.
1. ...
2. ...
3. ...
"""
    return system_prompt.strip(), user_prompt.strip()



def build_abstract_prompts(
    broad_direction: str,
    refined_problem: str,
    title: str,
) -> tuple[str, str]:
    system_prompt = (
        "You are a machine learning researcher writing a concise research abstract "
        "in the style of a strong conference paper."
    )

    user_prompt = f"""
Broad research direction:
{broad_direction}

Refined problem:
{refined_problem}

Paper title:
{title}

Task:
Write a realistic abstract for this paper.

Constraints:
- 150 to 220 words.
- Academic style.
- Clearly state the problem, the core idea, and the main result claims.
- Keep it plausible and self-consistent.
- Stay faithful to the refined problem and title.
- Do not use bullet points.
- Output only the abstract text.
"""
    return system_prompt.strip(), user_prompt.strip()



def build_section_generation_prompts(
    broad_direction: str,
    refined_problem: str,
    title: str,
    abstract: str,
    target_section: str,
    context: str,
) -> tuple[str, str]:
    section_instructions = {
        "introduction": (
            "Write an Introduction section. "
            "First present the broader problem setting and why it matters. "
            "Then explain the concrete limitation in existing approaches that motivates this work. "
            "Next introduce the core idea of the proposed approach in a way consistent with the abstract. "
            "Finally summarize the main contributions at a high level."
        ),
        "related_work": (
            "Write a Related Work section. "
            "First identify the most relevant lines of prior work related to the paper's problem. "
            "Then explain how existing approaches address the problem and where they fall short. "
            "Next position the proposed work relative to those approaches without claiming unrealistic superiority. "
            "Finally make clear what gap this paper is intended to address."
        ),
        "background": (
            "Write a Background section. "
            "First introduce the main concepts, assumptions, and terminology needed to understand the paper. "
            "Then explain the learning setting or problem formulation in a concise and technical way. "
            "Finally provide only the background needed for the method, without turning this section into a literature review."
        ),
        "method": (
            "Write a Method section. "
            "First restate the core technical goal of the approach in a way consistent with the abstract. "
            "Then describe the main components of the proposed method and how they interact. "
            "Next explain the learning objective, optimization logic, or training procedure at a high level. "
            "If the abstract does not specify equations or algorithmic details, stay appropriately high-level and do not invent unnecessary mathematical machinery. "
            "Finally clarify why this method is expected to address the identified limitation better than prior approaches."
        ),
        "experiments": (
            "Write an Experiments section. "
            "First state the specific evaluation goal of this paper in a way that is consistent with the title and abstract. "
            "Then describe a realistic experimental setup, including the type of datasets, baselines, and evaluation metrics that are appropriate for this exact proposed method. "
            "Do not replace the paper with a neighboring topic from the literature or retrieved context. "
            "Next summarize the main findings in a plausible and restrained way, without inventing overly precise or exaggerated results. "
            "If the abstract does not provide exact numbers, keep the findings qualitative or only moderately specific. "
            "Finally explain what the experiments suggest about both the strengths and limitations of the proposed approach."
        ),
        "conclusion": (
            "Write a Conclusion section. "
            "First briefly restate the problem and the proposed idea. "
            "Then summarize the main contribution and the most important experimental takeaway. "
            "Next acknowledge at least one reasonable limitation or open challenge. "
            "Finally end with a realistic statement about possible future work, without making exaggerated claims."
        ),
    }

    system_prompt = (
        "You are a careful machine learning researcher. "
        "Write one section of a research paper using retrieved context as inspiration, "
        "without copying it verbatim."
    )

    user_prompt = f"""
Broad research direction:
{broad_direction}

Refined problem:
{refined_problem}

Paper title:
{title}

Abstract:
{abstract}

Target section:
{target_section}

Task:
{section_instructions[target_section]}

Retrieved context from related papers:
{context}

Important constraints:
- The paper's identity is defined by the broad direction, refined problem, title, and abstract.
- The retrieved context is only background inspiration.
- Do not copy or adapt a retrieved paper's method as if it were the proposed method.
- Do not introduce a specific architecture, probabilistic model, or algorithm unless it is already supported by the abstract.
- If the abstract stays high-level, keep the section high-level and consistent.
- Maintain faithfulness to the proposed paper idea.
- Use academic conference-paper style.
- Do not include a section header like "Introduction" or "Method".
- Output only the section text.
- Mention one concrete challenge or limitation of existing methods.
- Clearly state what is new in the proposed approach.
- End with a brief high-level summary of the contribution.
"""

    return system_prompt.strip(), user_prompt.strip()


# ---------------------------
# Idea generation steps
# ---------------------------


def generate_refined_problems(
    broad_direction: str,
    model: str,
    n_refined_problems: int,
    temperature: float,
) -> list[str]:
    sys_p, usr_p = build_refined_problem_prompts(
        broad_direction=broad_direction,
        n_refined_problems=n_refined_problems,
    )

    raw = chat_generate(
        model=model,
        system_prompt=sys_p,
        user_prompt=usr_p,
        temperature=temperature,
    )

    problems = parse_numbered_list(raw)
    problems = [p for p in problems if len(p) >= 20]

    # Keep order, remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for problem in problems:
        key = problem.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(problem)

    deduped = deduped[:n_refined_problems]

    if not deduped:
        raise RuntimeError(f"No refined problems generated for direction: {broad_direction}")

    return deduped



def generate_titles_for_problem(
    broad_direction: str,
    refined_problem: str,
    model: str,
    n_titles: int,
    temperature: float,
) -> list[str]:
    sys_p, usr_p = build_title_generation_prompts(
        broad_direction=broad_direction,
        refined_problem=refined_problem,
        n_titles=n_titles,
    )

    raw = chat_generate(
        model=model,
        system_prompt=sys_p,
        user_prompt=usr_p,
        temperature=temperature,
    )

    titles = parse_numbered_titles(raw)
    titles = clean_title_candidates(titles)

    # Keep order, remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for title in titles:
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(title)

    deduped = deduped[:n_titles]

    if not deduped:
        raise RuntimeError(f"No titles generated for refined problem: {refined_problem}")

    return deduped



def generate_abstract(
    broad_direction: str,
    refined_problem: str,
    title: str,
    model: str,
    temperature: float,
) -> str:
    sys_p, usr_p = build_abstract_prompts(
        broad_direction=broad_direction,
        refined_problem=refined_problem,
        title=title,
    )
    return chat_generate(
        model=model,
        system_prompt=sys_p,
        user_prompt=usr_p,
        temperature=temperature,
    )


# ---------------------------
# Retrieval and section generation
# ---------------------------


def retrieve_context_for_section(
    collection: chromadb.Collection,
    broad_direction: str,
    refined_problem: str,
    title: str,
    abstract: str,
    target_section: str,
    embed_model: str,
    top_k: int,
    max_per_paper: int,
) -> list[dict]:
    retrieval_sections = SECTION_RETRIEVAL_MAP[target_section]

    retrieval_query = (
        f"Broad direction: {broad_direction}\n"
        f"Refined problem: {refined_problem}\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        f"Target section: {target_section}\n"
        f"Find relevant material for writing this section."
    )

    raw_hits = retrieve_chunks(
        collection=collection,
        query=retrieval_query,
        embed_model=embed_model,
        n_results=max(top_k * 4, 20),
        section_names=retrieval_sections,
    )

    raw_hits = deduplicate_hits_by_text(raw_hits)
    hits = diversify_results(
        raw_hits,
        max_results=top_k,
        max_per_paper=max_per_paper,
    )
    return hits



def generate_section(
    broad_direction: str,
    refined_problem: str,
    title: str,
    abstract: str,
    target_section: str,
    context_hits: list[dict],
    model: str,
    temperature: float,
) -> str:
    context = format_context(context_hits)
    sys_p, usr_p = build_section_generation_prompts(
        broad_direction=broad_direction,
        refined_problem=refined_problem,
        title=title,
        abstract=abstract,
        target_section=target_section,
        context=context,
    )
    return chat_generate(
        model=model,
        system_prompt=sys_p,
        user_prompt=usr_p,
        temperature=temperature,
    )


# ---------------------------
# Output helpers
# ---------------------------


def print_problem_bundle(bundle: RefinedProblemBundle) -> None:
    print("\nBroad direction:")
    print(bundle.broad_direction)

    print("\nRefined problem:")
    print(bundle.refined_problem)

    print("\nCandidate titles:")
    for i, title in enumerate(bundle.titles, start=1):
        print(f"{i}. {title}")



def print_context_hits(hits: list[dict]) -> None:
    print("\nRetrieved context:")
    for i, item in enumerate(hits, start=1):
        meta = item["metadata"]
        print(f"#{i} score={item['score']:.3f}")
        print(f"   paper_id: {meta.get('paper_id')}")
        print(f"   title   : {meta.get('title')}")
        print(f"   section : {meta.get('section_name')}")
        print(f"   preview : {item['document'][:220].replace(chr(10), ' ')}")
        print()



def save_frozen_idea(
    output_dir: str,
    idea_id: str,
    broad_direction: str,
    refined_problem: str,
    title: str,
    abstract: str,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_idea_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", idea_id.strip())
    path = out_dir / f"{safe_idea_id}.json"

    payload = {
        "idea_id": idea_id,
        "broad_direction": broad_direction,
        "refined_problem": refined_problem,
        "title": title,
        "abstract": abstract,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path



def load_frozen_idea(path: str) -> FrozenIdea:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = [
        "broad_direction",
        "refined_problem",
        "title",
        "abstract",
    ]
    for key in required:
        if key not in data or not str(data[key]).strip():
            raise ValueError(f"Missing required field in frozen idea file: {key}")

    idea_id = str(data.get("idea_id") or Path(path).stem)

    return FrozenIdea(
        idea_id=idea_id,
        broad_direction=str(data["broad_direction"]),
        refined_problem=str(data["refined_problem"]),
        title=str(data["title"]),
        abstract=str(data["abstract"]),
    )



def save_run(
    output_dir: str,
    idea_id: str,
    target_section: str,
    model: str,
    broad_direction: str,
    refined_problem: str,
    title: str,
    abstract: str,
    context_hits: list[dict],
    generated_section: str,
) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_\-]+", "_", model.strip())
    safe_idea_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", idea_id.strip())
    safe_section = re.sub(r"[^a-zA-Z0-9_\-]+", "_", target_section.strip())

    model_dir = Path(output_dir) / safe_model
    model_dir.mkdir(parents=True, exist_ok=True)

    run_path = model_dir / f"{safe_idea_id}__{safe_section}.json"

    payload = {
        "idea_id": idea_id,
        "target_section": target_section,
        "generation_model": model,
        "broad_direction": broad_direction,
        "refined_problem": refined_problem,
        "title": title,
        "abstract": abstract,
        "context_hits": context_hits,
        "generated_section": generated_section,
    }

    with run_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return run_path


# ---------------------------
# Orchestration
# ---------------------------


def generate_frozen_ideas_from_directions(
    directions: list[str],
    model: str,
    n_refined_problems: int,
    n_titles_per_problem: int,
    temperature: float,
    frozen_ideas_dir: str,
    start_index: int = 1,
) -> list[Path]:
    """Generate and save one frozen idea per title across all directions."""
    saved_paths: list[Path] = []
    idea_counter = start_index

    for direction in directions:
        print(f"\n=== Broad direction: {direction} ===")

        refined_problems = generate_refined_problems(
            broad_direction=direction,
            model=model,
            n_refined_problems=n_refined_problems,
            temperature=temperature,
        )

        for refined_problem in refined_problems:
            titles = generate_titles_for_problem(
                broad_direction=direction,
                refined_problem=refined_problem,
                model=model,
                n_titles=n_titles_per_problem,
                temperature=temperature,
            )

            bundle = RefinedProblemBundle(
                broad_direction=direction,
                refined_problem=refined_problem,
                titles=titles,
            )
            print_problem_bundle(bundle)

            for title in titles:
                print("\nGenerating abstract for title:")
                print(title)

                abstract = generate_abstract(
                    broad_direction=direction,
                    refined_problem=refined_problem,
                    title=title,
                    model=model,
                    temperature=temperature,
                )

                idea_id = f"idea_{idea_counter:04d}"
                path = save_frozen_idea(
                    output_dir=frozen_ideas_dir,
                    idea_id=idea_id,
                    broad_direction=direction,
                    refined_problem=refined_problem,
                    title=title,
                    abstract=abstract,
                )
                print(f"Saved frozen idea to: {path}")

                saved_paths.append(path)
                idea_counter += 1

    return saved_paths


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG pipeline for idea-bank and section generation")

    # Shared / generation config
    ap.add_argument("--model", default=DEFAULT_GEN_MODEL, help="Local Ollama generation model")
    ap.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    ap.add_argument("--frozen-ideas-dir", default="./generated_papers/frozen_ideas", help="Directory where frozen idea JSON files are stored")
    ap.add_argument("--idea-id", default=None, help="Optional stable ID for a single frozen idea run")

    # Idea-bank mode
    ap.add_argument("--directions-file", default="./topics.txt", help="Text file with one broad direction per line")
    ap.add_argument("--n-refined-problems", type=int, default=2, help="Number of refined problems to generate per direction")
    ap.add_argument("--n-titles-per-problem", type=int, default=3, help="Number of titles to generate per refined problem")
    ap.add_argument("--max-directions", type=int, default=None, help="Optional cap on how many directions to process")
    ap.add_argument("--start-idea-index", type=int, default=1, help="Starting index used to name frozen ideas")
    ap.add_argument("--generate-ideas-only", action="store_true", help="Generate frozen ideas and stop before section generation")

    # Section-generation mode
    ap.add_argument("--idea-file", default=None, help="JSON file containing one frozen idea")
    ap.add_argument("--target-section", choices=sorted(ALLOWED_TARGET_SECTIONS), help="Section to generate from a frozen idea")
    ap.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model for retrieval")
    ap.add_argument("--chroma-dir", default=CHROMA_DIR, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default=COLLECTION, help="ChromaDB collection name")
    ap.add_argument("--top-k", type=int, default=4, help="Final number of retrieved chunks")
    ap.add_argument("--max-per-paper", type=int, default=1, help="Maximum retrieved chunks per paper")
    ap.add_argument("--output-dir", default="./rag_runs", help="Directory where run JSON is saved")

    args = ap.parse_args()

    # ---------------------------
    # Mode A: idea-bank generation
    # ---------------------------
    if not args.idea_file:
        directions = load_directions(args.directions_file)

        if args.max_directions is not None:
            directions = directions[: args.max_directions]

        saved_paths = generate_frozen_ideas_from_directions(
            directions=directions,
            model=args.model,
            n_refined_problems=args.n_refined_problems,
            n_titles_per_problem=args.n_titles_per_problem,
            temperature=args.temperature,
            frozen_ideas_dir=args.frozen_ideas_dir,
            start_index=args.start_idea_index,
        )

        print("Idaes were generated! Thabk you for the wait")
        # In the new design, the default upstream behavior is to stop after
        # building the idea bank unless an explicit frozen idea is provided.
        return

    # ---------------------------
    # Mode B: section generation from one frozen idea
    # ---------------------------
    if not args.target_section:
        ap.error("--target-section is required when --idea-file is provided")

    print("[1/4] loading frozen idea ...")
    frozen = load_frozen_idea(args.idea_file)

    idea_id = args.idea_id or frozen.idea_id
    broad_direction = frozen.broad_direction
    refined_problem = frozen.refined_problem
    title = frozen.title
    abstract = frozen.abstract

    print(f"\nIdea ID: {idea_id}")
    print("\nBroad direction:")
    print(broad_direction)
    print("\nRefined problem:")
    print(refined_problem)
    print("\nTitle:")
    print(title)
    print("\nAbstract:\n")
    print(abstract)

    print("\n[2/4] loading Chroma collection ...")
    collection = get_collection(args.chroma_dir, args.collection)

    print("\n[3/4] retrieving context ...")
    context_hits = retrieve_context_for_section(
        collection=collection,
        broad_direction=broad_direction,
        refined_problem=refined_problem,
        title=title,
        abstract=abstract,
        target_section=args.target_section,
        embed_model=args.embed_model,
        top_k=args.top_k,
        max_per_paper=args.max_per_paper,
    )
    print_context_hits(context_hits)

    print("[4/4] generating target section ...")
    generated_section = generate_section(
        broad_direction=broad_direction,
        refined_problem=refined_problem,
        title=title,
        abstract=abstract,
        target_section=args.target_section,
        context_hits=context_hits,
        model=args.model,
        temperature=args.temperature,
    )

    print(f"\nGenerated {args.target_section}:\n")
    print(generated_section)

    saved_path = save_run(
        output_dir=args.output_dir,
        idea_id=idea_id,
        target_section=args.target_section,
        model=args.model,
        broad_direction=broad_direction,
        refined_problem=refined_problem,
        title=title,
        abstract=abstract,
        context_hits=context_hits,
        generated_section=generated_section,
    )

    print(f"\nSaved run to: {saved_path}")


if __name__ == "__main__":
    main()
