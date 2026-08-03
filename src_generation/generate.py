#!/usr/bin/env python3
"""
Provide reusable generation and RAG utilities for the paper generation pipeline.

The module supports research-idea and abstract generation, context retrieval
from ChromaDB, and generation of individual paper sections with Ollama or Hugging Face models. 
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import time

import chromadb
import ollama

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class ResearchIdea:
    chosen_direction: str
    refined_problem: str
    titles: list[str]

@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    total_duration_seconds: float | None = None
    prompt_eval_duration_seconds: float | None = None
    eval_duration_seconds: float | None = None

# ---------------------------
# Config
# ---------------------------

EMBED_MODEL = "nomic-embed-text"
DEFAULT_GEN_MODEL = "qwen3:14b"

CHROMA_DIR = "./chroma_iclr"
COLLECTION = "iclr_papers"
HF_MODEL_CACHE = {}

PAPER_SECTION_ORDER = [
    "introduction",
    "related_work",
    "background",
    "method",
    "experiments",
    "conclusion",
]

ALLOWED_TARGET_SECTIONS = set(PAPER_SECTION_ORDER)

SECTION_RETRIEVAL_MAP = {
    "introduction": ["abstract", "introduction", "related work", "background"],
    "related_work": ["related work", "background", "introduction"],
    "background": ["background", "introduction", "related work"],
    "method": ["abstract", "background", "method"],
    "experiments": ["method", "experiments", "conclusion"],
    "conclusion": ["abstract", "experiments", "conclusion"],
}

# Broad research directions taken from the ICLR Call for Papers
BROAD_TOPIC_DIRECTIONS = [
    "unsupervised, semi-supervised, and supervised representation learning",
    "representation learning for planning and reinforcement learning",
    "metric learning and kernel learning",
    "sparse coding and dimensionality expansion",
    "hierarchical models",
    "optimization for representation learning",
    "learning representations of outputs or states",
    "theoretical issues in deep learning",
    "visualization or interpretation of learned representations",
    "implementation issues, parallelization, software platforms, hardware",
    "applications in vision, audio, speech, natural language processing, robotics, neuroscience, computational biology, or any other field",
]

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
# Generation helpers
# Supports Ollama models and Hugging Face models with the "hf:" prefix
# ---------------------------

def ns_to_seconds(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1_000_000_000

def hf_chat_generate(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_new_tokens: int = 2500,
) -> GenerationResult:
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError(
            "Hugging Face generation requires torch and transformers. "
            "Install them before using models with the hf: prefix."
        )

    if model_name not in HF_MODEL_CACHE:
        print(f"[hf] loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        HF_MODEL_CACHE[model_name] = (tokenizer, model)

    tokenizer, model = HF_MODEL_CACHE[model_name]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            f"System:\n{system_prompt}\n\n"
            f"User:\n{user_prompt}\n\n"
            f"Assistant:\n"
        )

    input_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

    prompt_tokens = inputs["input_ids"].shape[-1]

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    total_duration_seconds = time.perf_counter() - start_time

    generated_tokens = outputs[0][prompt_tokens:]
    completion_tokens = generated_tokens.shape[-1]

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return GenerationResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        total_duration_seconds=total_duration_seconds,
        prompt_eval_duration_seconds=None,
        eval_duration_seconds=None,
    )
def chat_generate(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
) -> GenerationResult:
    if model.startswith("hf:"):
        hf_model_name = model.removeprefix("hf:")

        return hf_chat_generate(
            model_name=hf_model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

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

    text = response["message"]["content"].strip()

    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")

    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return GenerationResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        total_duration_seconds=ns_to_seconds(response.get("total_duration")),
        prompt_eval_duration_seconds=ns_to_seconds(response.get("prompt_eval_duration")),
        eval_duration_seconds=ns_to_seconds(response.get("eval_duration")),
    )

def parse_numbered_titles(text: str) -> list[str]:
    titles = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\s*[\.\)\-:]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip().strip('"')
        if line:
            titles.append(line)
    return titles

def parse_research_idea(text: str, n_titles: int) -> ResearchIdea:
    chosen_direction = ""
    refined_problem = ""
    titles: list[str] = []

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    current_field = None

    for line in lines:
        cleaned_line = line.strip()

        # Remove common markdown decoration
        normalized = cleaned_line
        normalized = re.sub(r"^[#*\-\s]+", "", normalized).strip()
        normalized_lower = normalized.lower()

        # Detect field headers
        if normalized_lower.startswith("chosen direction:"):
            value = normalized.split(":", 1)[1].strip()
            if value:
                chosen_direction = value
                current_field = None
            else:
                current_field = "chosen_direction"
            continue

        if normalized_lower.startswith("refined problem:"):
            value = normalized.split(":", 1)[1].strip()
            if value:
                refined_problem = value
                current_field = None
            else:
                current_field = "refined_problem"
            continue

        # If previous line introduced a field and value comes on next line
        if current_field == "chosen_direction" and not chosen_direction:
            chosen_direction = normalized.strip('"')
            current_field = None
            continue

        if current_field == "refined_problem" and not refined_problem:
            refined_problem = normalized.strip('"')
            current_field = None
            continue

        # Parse numbered titles
        title_candidate = re.sub(r"^[#*\-\s]*\d+\s*[\.\)\-:]\s*", "", cleaned_line).strip()
        title_candidate = title_candidate.strip('"')

        if title_candidate:
            lower_tc = title_candidate.lower()
            if not lower_tc.startswith("chosen direction") and not lower_tc.startswith("refined problem"):
                titles.append(title_candidate)

    titles = [t for t in titles if t][:n_titles]

    return ResearchIdea(
        chosen_direction=chosen_direction,
        refined_problem=refined_problem,
        titles=titles,
    )

def clean_title_candidates(titles: list[str]) -> list[str]:
    cleaned = []
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

def load_frozen_idea(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = [
        "topic",
        "chosen_direction",
        "refined_problem",
        "selected_title",
        "abstract",
    ]
    for key in required:
        if key not in data or not str(data[key]).strip():
            raise ValueError(f"Missing required field in frozen idea file: {key}")

    if "idea_id" not in data or not str(data["idea_id"]).strip():
        data["idea_id"] = Path(path).stem

    return data
# ---------------------------
# Prompt builders
# ---------------------------

def format_broad_topic_list() -> str:
    return "\n".join(f"- {item}" for item in BROAD_TOPIC_DIRECTIONS)


def build_research_idea_prompts(topic: str, n_titles: int, selected_direction: Optional[str] = None,) -> tuple[str, str]:
    system_prompt = (
        "You are a careful machine learning researcher designing realistic "
        "ICLR-style paper ideas. You must stay grounded in plausible mainstream "
        "machine learning research."
    )

    if selected_direction:
        direction_instruction = f"""
The broad research direction is fixed:
{selected_direction}

You must stay within this direction.
"""
    else:
        direction_instruction = """
Choose one broad direction from the list below, or a very small combination of compatible directions.
"""

    user_prompt = f"""
High-level theme:
{topic}

Non-exhaustive list of broad research directions:
{format_broad_topic_list()}

Task:
1. {direction_instruction.strip()}
2. Refine it into one concrete and plausible research problem suitable for a single conference paper.
3. Generate {n_titles} distinct and realistic paper titles based on that refined problem.

Constraints:
- Titles must be specific, technical, and plausible.
- Titles should sound like papers that could realistically appear at ICLR.
- Titles should reflect a clear contribution, method, analysis, or evaluation setting.
- Stay grounded in mainstream machine learning research.
- Avoid vague, generic, or overly ambitious titles.
- The titles must be meaningfully different from each other, while staying within the same refined problem area.
- Application areas are allowed, but the title must still reflect a concrete machine learning contribution.

Formatting rules:
- Do not use markdown headings.
- Do not use bold text.
- Do not use bullet points except for the numbered title list.
- Put the value for "Chosen direction:" on the same line.
- Put the value for "Refined problem:" on the same line.
- Then list the titles as:
1. ...
2. ...
3. ...
4. ...
5. ...
"""
    return system_prompt.strip(), user_prompt.strip()

def build_abstract_prompts(topic: str, title: str) -> tuple[str, str]:
    system_prompt = (
        "You are a careful machine learning researcher writing a realistic abstract "
        "for an ICLR-style conference paper. The abstract must be technically plausible, "
        "self-consistent, and appropriately restrained."
    )

    user_prompt = f"""
Topic:
{topic}

Paper title:
{title}

Task:
Write a realistic conference-paper abstract for this title.

Constraints:
- 170 to 230 words.
- Write in a formal academic style.
- Clearly state:
  1. the problem setting,
  2. the limitation in existing approaches,
  3. the proposed idea,
  4. the evaluation setting,
  5. the main takeaway.
- Keep the contribution plausible and focused.
- Do not make extreme claims.
- Do not invent very specific numerical improvements unless they are modest and realistic.
- Do not use bullet points.
- Output only the abstract text.
"""
    return system_prompt.strip(), user_prompt.strip()


def build_section_generation_prompts(
    topic: str,
    chosen_direction: str,
    refined_problem: str,
    title: str,
    abstract: str,
    target_section: str,
    context: str,
) -> tuple[str, str]:
    
    section_instructions = {
    "introduction": (
        "Write a full Introduction section. "
        "Structure the section as a coherent sequence of 7 to 9 dense academic paragraphs. "
        "Start from the broader research area and explain why the problem matters. "
        "Then narrow to the exact limitation in prior work that motivates this paper. "
        "Clearly formulate the paper's core idea in a way that is consistent with the title and abstract. "
        "Explain why this idea is promising without claiming guaranteed success. "
        "Conclude with a concise summary of the paper's main contributions and scope. "
        "The section should read like a real conference introduction, not like notes or an outline. "
        "Target length: 900 to 1200 words."
    ),
    "related_work": (
        "Write a full Related Work section. "
        "Organize the discussion into 7 to 9 coherent academic paragraphs. "
        "Identify 2 to 4 major lines of prior work that are most relevant to this paper. "
        "For each line, explain the main idea, what it achieves, and its limitations relative to the present paper. "
        "Use realistic literature framing, but do not fabricate specific citations, author names, or years. "
        "Position the proposed paper clearly and carefully, without exaggerated claims. "
        "End by clarifying the specific gap that motivates the present work. "
        "Target length: 900 to 1200 words."
    ),
    "background": (
        "Write a full Background section. "
        "Use 7 to 9 coherent academic paragraphs. "
        "Introduce the technical concepts, assumptions, notation-level ideas, and problem setting needed to understand the paper. "
        "Explain only the background necessary for the proposed method and experiments. "
        "Do not turn this section into a literature review, and do not introduce unnecessary mathematical formalism. "
        "Keep the exposition concise, technical, and faithful to the abstract. "
        "Target length: 900 to 1200 words."
    ),
    "method": (
        "Write a full Method section. "
        "Use 8 to 10 coherent academic paragraphs. "
        "Begin by restating the technical goal and the intuition behind the proposed approach. "
        "Then describe the main components of the method, how they interact, and the role each component plays. "
        "Explain the training or optimization logic at a level of detail supported by the abstract. "
        "If the abstract is high-level, keep the method high-level but still technically informative. "
        "Do not invent unnecessary equations, losses, modules, or algorithms. "
        "End by explaining why the method is expected to address the identified limitation better than prior approaches. "
        "Target length: 1000 to 1300 words."
    ),
    "experiments": (
        "Write a full Experiments section. "
        "Use 8 to 10 coherent academic paragraphs. "
        "State the evaluation goal clearly. "
        "Describe a realistic experimental design, including task setting, dataset types, baseline families, and evaluation metrics. "
        "The experiments must match the exact paper idea rather than drift toward the retrieved papers. "
        "Summarize plausible findings in a restrained and realistic way. "
        "If exact numbers are not supported by the abstract, use qualitative or moderately specific comparisons instead. "
        "Discuss both strengths and limitations of the proposed approach. "
        "Target length: 1000 to 1300 words."
    ),
    "conclusion": (
        "Write a full Conclusion section. "
        "Use 6 to 8 coherent academic paragraphs. "
        "Briefly restate the problem and the proposed idea. "
        "Summarize the main contribution and the most important experimental takeaway. "
        "Acknowledge at least one realistic limitation. "
        "End with measured future work directions that follow naturally from the paper. "
        "Avoid generic closing statements. "
        "Target length: 800 to 1000 words."
    ),
}
    system_prompt = (
        "You are a careful machine learning researcher writing a realistic, high-quality "
        "ICLR-style paper section. Your writing must be technically plausible, coherent, "
        "and faithful to the provided paper idea. Use the retrieved context only as "
        "background inspiration for terminology, structure, and literature framing. "
        "Do not copy the context, do not switch to a different paper idea, and do not "
        "invent overly specific technical details that are unsupported by the abstract."
    )
    user_prompt = f"""
Topic:
{topic}

Chosen direction:
{chosen_direction}

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
- The paper's identity is defined by the chosen direction, refined problem, title, and abstract.
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
# Pipeline steps
# ---------------------------

def generate_research_idea(topic: str, model: str, n_titles: int, temperature: float, selected_direction: Optional[str] = None,) -> ResearchIdea:
    sys_p, usr_p = build_research_idea_prompts(
        topic=topic,
        n_titles=n_titles,
        selected_direction=selected_direction,
    )

    result = chat_generate(
        model=model,
        system_prompt=sys_p,
        user_prompt=usr_p,
        temperature=temperature,
    )

    raw = result.text
    idea = parse_research_idea(raw, n_titles=n_titles)
    idea.titles = clean_title_candidates(idea.titles)

    if not idea.chosen_direction:
        idea.chosen_direction = selected_direction or "Not clearly specified"

    if not idea.refined_problem:
        idea.refined_problem = "Not clearly specified"

    if len(idea.titles) < n_titles:
        fallback_titles = parse_numbered_titles(raw)
        for title in fallback_titles:
            if title not in idea.titles:
                idea.titles.append(title)
        idea.titles = clean_title_candidates(idea.titles)
        idea.titles = idea.titles[:n_titles]
        
    if not idea.titles:
        raise RuntimeError("No candidate titles were generated.")

    return idea

def generate_abstract(
    topic: str,
    title: str,
    model: str,
    temperature: float,
) -> str:
    sys_p, usr_p = build_abstract_prompts(topic, title)
    return chat_generate(
        model=model,
        system_prompt=sys_p,
        user_prompt=usr_p,
        temperature=temperature,
    ).text

def retrieve_context_for_section(
    collection: chromadb.Collection,
    topic: str,
    title: str,
    abstract: str,
    target_section: str,
    embed_model: str,
    top_k: int,
    max_per_paper: int,
) -> list[dict]:
    if target_section not in SECTION_RETRIEVAL_MAP:
        raise ValueError(f"Unknown target section: {target_section}")

    retrieval_sections = SECTION_RETRIEVAL_MAP[target_section]

    retrieval_query = (
        f"Topic: {topic}\n"
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
    topic: str,
    chosen_direction: str,
    refined_problem: str,
    title: str,
    abstract: str,
    target_section: str,
    context_hits: list[dict],
    model: str,
    temperature: float,
) -> GenerationResult:
    context = format_context(context_hits)

    sys_p, usr_p = build_section_generation_prompts(
        topic=topic,
        chosen_direction=chosen_direction,
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

def print_context_hits(hits: list[dict]) -> None:
    print("\nRetrieved context:")
    for i, item in enumerate(hits, start=1):
        meta = item["metadata"]
        print(f"#{i} score={item['score']:.3f}")
        print(f"   paper_id: {meta.get('paper_id')}")
        print(f"   title   : {meta.get('title')}")
        print(f"   section : {meta.get('section_name')}")
        print()


def save_run(
    output_dir: str,
    idea_id: str,
    topic: str,
    target_section: str,
    model: str,
    research_idea: ResearchIdea,
    selected_title: str,
    abstract: str,
    context_hits: list[dict],
    generation_result: GenerationResult
) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_\-]+", "_", model.strip())
    safe_idea_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", idea_id.strip())
    safe_section = re.sub(r"[^a-zA-Z0-9_\-]+", "_", target_section.strip())

    model_dir = Path(output_dir) / safe_model
    model_dir.mkdir(parents=True, exist_ok=True)

    run_path = model_dir / f"{safe_idea_id}__{safe_section}.json"

    payload = {
        "idea_id": idea_id,
        "topic": topic,
        "target_section": target_section,
        "generation_model": model,
        "chosen_direction": research_idea.chosen_direction,
        "refined_problem": research_idea.refined_problem,
        "title_candidates": research_idea.titles,
        "selected_title": selected_title,
        "abstract": abstract,
        "context_hits": context_hits,
        "generated_section": generation_result.text,
        "token_usage": {
            "prompt_tokens": generation_result.prompt_tokens,
            "completion_tokens": generation_result.completion_tokens,
            "total_tokens": generation_result.total_tokens,
        },
        "timing": {
            "total_duration_seconds": generation_result.total_duration_seconds,
            "prompt_eval_duration_seconds": generation_result.prompt_eval_duration_seconds,
            "eval_duration_seconds": generation_result.eval_duration_seconds,
        }
    }

    with run_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return run_path

def save_frozen_idea(
    output_dir: str,
    idea_id: str,
    topic: str,
    chosen_direction: str,
    refined_problem: str,
    selected_title: str,
    abstract: str,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_idea_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", idea_id.strip())
    path = out_dir / f"{safe_idea_id}.json"

    payload = {
        "idea_id": idea_id,
        "topic": topic,
        "chosen_direction": chosen_direction,
        "refined_problem": refined_problem,
        "selected_title": selected_title,
        "abstract": abstract,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path

# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate one paper section from one frozen idea using RAG")

    ap.add_argument("--idea-file", required=True, help="Frozen idea JSON file to use")
    ap.add_argument("--target-section", required=True, choices=sorted(ALLOWED_TARGET_SECTIONS), help="Section to generate")

    ap.add_argument("--model", default=DEFAULT_GEN_MODEL, help="Generation moddel, use an Ollama or a HuggingFace model such as apertus")
    ap.add_argument("--output-dir", default="./rag_runs", help="Directory where run JSON is saved")

    ap.add_argument("--embed-model", default=EMBED_MODEL, help="Embedding model for retrieval")
    ap.add_argument("--chroma-dir", default=CHROMA_DIR, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default=COLLECTION, help="ChromaDB collection name")

    ap.add_argument("--top-k", type=int, default=4, help="Final number of retrieved chunks")
    ap.add_argument("--max-per-paper", type=int, default=1, help="Maximum retrieved chunks per paper")
    ap.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")

    args = ap.parse_args()

    print("[1/4] loading Chroma collection ...")
    collection = get_collection(args.chroma_dir, args.collection)

    print("[2/4] loading frozen idea ...")
    frozen = load_frozen_idea(args.idea_file)

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

    print(f"\nIdea ID: {idea_id}")
    print("\nChosen direction:")
    print(chosen_direction)
    print("\nRefined problem:")
    print(refined_problem)
    print("\nSelected title:")
    print(selected_title)
    print("\nAbstract:\n")
    print(abstract)

    print("\n[3/4] retrieving context ...")
    context_hits = retrieve_context_for_section(
        collection=collection,
        topic=topic,
        title=selected_title,
        abstract=abstract,
        target_section=args.target_section,
        embed_model=args.embed_model,
        top_k=args.top_k,
        max_per_paper=args.max_per_paper,
    )
    print_context_hits(context_hits)

    print("[4/4] generating target section ...")
    generation_result = generate_section(
        topic=topic,
        chosen_direction=chosen_direction,
        refined_problem=refined_problem,
        title=selected_title,
        abstract=abstract,
        target_section=args.target_section,
        context_hits=context_hits,
        model=args.model,
        temperature=args.temperature,
    )

    print(f"\nGenerated {args.target_section}:\n")
    print(generation_result.text)

    saved_path = save_run(
        output_dir=args.output_dir,
        idea_id=idea_id,
        topic=topic,
        target_section=args.target_section,
        model=args.model,
        research_idea=research_idea,
        selected_title=selected_title,
        abstract=abstract,
        context_hits=context_hits,
        generation_result=generation_result,
    )

    print("\nToken usage:")
    print(f"  prompt_tokens     : {generation_result.prompt_tokens}")
    print(f"  completion_tokens : {generation_result.completion_tokens}")
    print(f"  total_tokens      : {generation_result.total_tokens}")

    print(f"\nSaved run to: {saved_path}")


if __name__ == "__main__":
    main()
