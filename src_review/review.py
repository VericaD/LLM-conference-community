#!/usr/bin/env python3
"""
Generate reviews and meta-reviews for assembled research papers.

The script reads assembled paper JSON files from an input directory, retrieves
similar ICLR reviews from `chroma_iclr_reviews/`, generates independent reviews
and an Area Chair decision with Ollama, and saves one review-pipeline JSON file
per paper in the specified output directory. Failed runs are recorded in
`failures.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

import chromadb
import ollama
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_REVIEWER_MODELS = ["llama3.1:8b", "qwen3:14b", "phi4-reasoning:14b"]
DEFAULT_AREA_CHAIR_MODEL = "phi4-reasoning:14b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_REVIEW_CHROMA_DIR = "./chroma_iclr_reviews"
DEFAULT_REVIEW_COLLECTION = "iclr_reviews"
DEFAULT_TOP_K_CHUNKS = 3
MAX_REFERENCE_REVIEW_CHARS = 2000

DEFAULT_NUM_PREDICT_OFFICIAL = 800
DEFAULT_NUM_PREDICT_META = 650

SECTION_ORDER = [
    "introduction",
    "related_work",
    "background",
    "method",
    "experiments",
    "conclusion",
]

RATING_SCALE_GUIDE = """
Rating scale (assign exactly one integer from 1 to 10; use this only to
anchor what each number means, not as a target to aim for):
  10   : Exceptional, seminal paper, among the very best submissions.
  8-9  : Clear accept. Strong, well-supported contribution.
  6-7  : Accept. Solid contribution with limitations that do not undermine
         the core claims.
  5    : Marginally below the acceptance threshold; borderline case.
  3-4  : Clear rejection. Significant, unresolved concerns about
         soundness, novelty, or evaluation.
  1-2  : Trivial, incorrect, or fundamentally unsound.
""".strip()

# ---------------------------------------------------------------------------
# Official review schema 
# ---------------------------------------------------------------------------


class OfficialReview(BaseModel):

    model_config = ConfigDict(extra="allow")

    paper_id: str
    reviewer_model: str

    title: str  # one-sentence summary of this specific paper's overall verdict
    review: str  # single free-text field, internally headed: Summary / Strengths / Weaknesses / Questions / Recommendation rationale
    rating: int = Field(ge=1, le=10)
    confidence: int = Field(ge=1, le=5)

    # Analysis-only fields -- not part of the authentic ICLR schema.
    extracted_strengths: list[str] = Field(min_length=2, max_length=4)
    extracted_weaknesses: list[str] = Field(min_length=2, max_length=4)


OFFICIAL_REVIEW_SCHEMA: dict[str, Any] = OfficialReview.model_json_schema()

# ---------------------------------------------------------------------------
# Meta-review / Area Chair schema
# ---------------------------------------------------------------------------


class MetaReview(BaseModel):

    model_config = ConfigDict(extra="allow")

    paper_id: str
    area_chair_model: str

    summary: str
    consensus_strengths: list[str] = Field(min_length=1, max_length=5)
    consensus_weaknesses: list[str] = Field(min_length=1, max_length=5)
    significant_weakness_count: int = Field(ge=0, le=3)
    decision: Literal["accept", "reject"]
    justification: str


META_REVIEW_SCHEMA: dict[str, Any] = MetaReview.model_json_schema()

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_paper_id(paper: dict[str, Any], fallback_path: Path) -> str:
    return str(paper.get("idea_id") or paper.get("paper_id") or fallback_path.stem)


def build_paper_text(paper: dict[str, Any]) -> str:
    parts: list[str] = []

    for key in ("selected_title", "topic", "chosen_direction", "refined_problem", "abstract"):
        value = paper.get(key)
        if isinstance(value, str) and value.strip():
            header = key.replace("_", " ").title()
            parts.append(f"## {header}\n{value.strip()}")

    sections = paper.get("sections", {})
    if isinstance(sections, dict):
        for key in SECTION_ORDER:
            value = sections.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(f"## {key.replace('_', ' ').title()}\n{value.strip()}")

    if not parts:
        full_text = paper.get("full_text", "")
        if isinstance(full_text, str) and full_text.strip():
            parts.append(f"## Full Text\n{full_text.strip()}")

    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def get_review_collection(chroma_dir: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)


def embed_query(text: str, embed_model: str) -> list[float]:
    response = ollama.embed(model=embed_model, input=[text])
    return response["embeddings"][0]


def retrieve_review_chunks(
    collection: chromadb.Collection,
    query_text: str,
    embed_model: str,
    top_k: int,
) -> list[dict[str, Any]]:

    query_vec = embed_query(query_text, embed_model)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"score": 1 - dist, "text": doc, "metadata": meta})
    return hits


def build_retrieval_query(paper: dict[str, Any]) -> str:
    title = paper.get("selected_title", "")
    abstract = paper.get("abstract", "")
    return (
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        f"Find review text discussing a similar topic or method."
    )


def truncate_reference_review(text: str, max_chars: int = MAX_REFERENCE_REVIEW_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(".")
    if last_period > max_chars * 0.7:
        return cut[: last_period + 1]
    return cut


def format_review_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(no reference reviews retrieved)"

    parts = []
    for i, c in enumerate(chunks, start=1):
        text = truncate_reference_review(c["text"])
        metadata = c.get("metadata", {})
        rating = metadata.get("rating", "unknown")
        confidence = metadata.get("confidence", "unknown")
        parts.append(f"[Reference review {i}]\nRating: {rating} | Confidence: {confidence}\n{text}")

    return "\n\n".join(parts)

# ---------------------------------------------------------------------------
# Token usage 
# ---------------------------------------------------------------------------


def ns_to_seconds(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1_000_000_000


def extract_token_usage(response: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")

    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_duration_seconds": ns_to_seconds(response.get("total_duration")),
        "prompt_eval_duration_seconds": ns_to_seconds(response.get("prompt_eval_duration")),
        "eval_duration_seconds": ns_to_seconds(response.get("eval_duration")),
    }


def aggregate_token_usage(token_usages: list[dict[str, Any]]) -> dict[str, Any]:

    prompt_total = sum((t.get("prompt_tokens") or 0) for t in token_usages)
    completion_total = sum((t.get("completion_tokens") or 0) for t in token_usages)
    total_total = sum((t.get("total_tokens") or 0) for t in token_usages)
    return {
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "total_tokens": total_total,
    }


def generate_with_retry(
    model: str,
    prompt: str,
    schema: dict[str, Any],
    validate_fn: Any,
    temperature: float,
    think: str | bool | None,
    num_predict: int | None,
    max_retries: int = 2,
    budget_growth_factor: float = 1.6,
) -> Any:
    current_budget = num_predict
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        options: dict[str, Any] = {"temperature": temperature}
        if current_budget is not None:
            options["num_predict"] = current_budget

        generate_kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "format": schema,
            "stream": False,
            "options": options,
        }
        if think is not None:
            generate_kwargs["think"] = think

        response = ollama.generate(**generate_kwargs)
        raw = response["response"]
        token_usage = extract_token_usage(response)

        try:
            return validate_fn(raw, token_usage)
        except (ValidationError, ValueError) as e:
            last_error = e
            if attempt < max_retries:
                if current_budget is not None:
                    current_budget = int(current_budget * budget_growth_factor)
                print(
                    f"          [retry {attempt + 1}/{max_retries}] {model}: "
                    f"invalid/truncated JSON, raising num_predict to {current_budget}"
                )

    assert last_error is not None
    raise last_error

# ---------------------------------------------------------------------------
# Official review generation prompt
# ---------------------------------------------------------------------------


def build_official_review_prompt(
    paper_id: str,
    paper_text: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    instructions = f"""
    You are writing an official peer review for a top-tier ML conference,
    following the real ICLR/OpenReview official-review format: a
    one-sentence title, a single free-text review, a rating, and a
    confidence score. There are no separate strengths/weaknesses/questions
    fields -- write the review as ONE continuous text field, internally
    organized with the following prose headings, in this order:

    Summary
      What the paper claims to contribute. Read the paper and form a
      concise, neutral understanding of its problem, method, and claims.

    Strengths
      What is novel, clear, or empirically strong. Ground every point in
      specific content from the paper -- avoid generic statements that
      could apply to any paper.

    Weaknesses
      Main limitations, unclear claims, or missing experiments. Ground
      every point in specific content from the paper, the same way. When
      looking for weaknesses, actively check: are the baselines adequate
      and relevant to the method's claims? Is the evaluation limited to a
      single dataset or setting? Do ablations actually isolate the
      contribution of the proposed method? Are novelty claims supported by
      a real comparison to the closest prior work, not just an assertion?
      Are reported improvements substantial or marginal? Do not skip this
      list just because the paper reads well -- a fluent paper can still
      fail several of these checks.

    Questions
      Concrete, answerable issues for the authors to address (omit this
      heading entirely if you have none).

    Recommendation rationale
      First, explicitly state how many of the weaknesses you listed above
      you consider SIGNIFICANT -- meaning they cast real doubt on the
      paper's central claims, novelty, or the validity of its evidence, as
      opposed to minor/cosmetic issues. Then explain how that count and
      severity is reflected in the rating: zero significant weaknesses
      points toward the higher bands, one significant weakness should
      pull the rating toward the middle or lower-middle of the scale, and
      two or more significant weaknesses should pull it into the
      rejection range, regardless of how well-written the paper is. The
      rating and rationale must be consistent with the strengths/
      weaknesses you wrote -- do not raise serious weaknesses and then
      rate the paper highly, or vice versa.

    Write each heading followed by a colon, then flowing prose or short
    numbered points underneath -- the way a real reviewer would type into
    a single text box, not as separate structured data.

    {RATING_SCALE_GUIDE}

    Calibration examples (illustrative only, not real papers):

    Example A: suppose a reviewer's own Weaknesses section states that the
    method is evaluated on only one dataset and is not compared against
    the two most relevant baselines discussed in the paper's own related
    work. That is a significant gap in evidence, regardless of how
    clearly the paper is written. A rating of 8-9 would be inconsistent
    with that weakness; an appropriate rating given that gap alone would
    be closer to 4-6, before weighing anything else.

    Example B: suppose a reviewer's own Weaknesses section states that the
    paper does not engage at all with the single most closely related
    prior method (one that solves essentially the same problem with a
    similar approach), so the actual novelty of the contribution cannot
    be assessed. That is a more severe gap than Example A -- it undermines
    the paper's central claim rather than a secondary evaluation detail.
    An appropriate rating given that gap alone would be closer to 3-4,
    even if everything else about the paper is well executed.

    Let the severity of the specific weaknesses you wrote down -- not the
    paper's overall polish or confident tone -- determine how far the
    rating moves from a default mid-point, and how far down the scale it
    can go; the scale genuinely extends down to 1-2 when warranted, not
    just to the bottom of the "Accept" band. At real ML conferences, most
    submissions are rejected; do not treat acceptance as the default
    outcome.

    You are also given a small set of real reviews from past ICLR
    submissions, retrieved because they cover similar topics or methods.
    Use them ONLY as inspiration for tone, structure, and the level of
    specificity expected of a real reviewer. Do NOT copy phrases, numbers,
    or claims from them, and do NOT refer to the retrieved papers directly
    -- they are different submissions. Your review must be grounded only
    in the paper you were given.

    The `title` field is a one-sentence summary of your own verdict on
    THIS paper specifically, in your own words -- not a restatement of the
    paper's title, and not a generic template. It must reference at least
    one specific aspect of this paper (its method, its central claim, or
    its main weakness), so that the exact same title could not honestly be
    reused for a different paper.

    After writing `review`, also populate `extracted_strengths` and
    `extracted_weaknesses`: short list items restating the same points you
    already made in the Strengths and Weaknesses sections of `review`,
    concisely. Do NOT introduce any point in these lists that is not
    already present in `review`, and do NOT introduce any point in
    `review` that is missing from these lists -- the two must agree.

    State your conclusions directly and concisely. Do not think
    step-by-step inside a field or use phrases like "let me reconsider" --
    just report your conclusions.

    Output ONLY valid JSON matching the provided schema.
    """.strip()

    return f"""
{instructions}

Paper ID: {paper_id}

Retrieved reference reviews (inspiration only, not ground truth):
{format_review_chunks(retrieved_chunks)}

JSON schema:
{json.dumps(OFFICIAL_REVIEW_SCHEMA, ensure_ascii=False)}

Paper to review:
{paper_text}
""".strip()


def generate_official_review(
    paper_id: str,
    paper_text: str,
    retrieved_chunks: list[dict[str, Any]],
    reviewer_model: str,
    temperature: float,
    think: str | bool | None,
    num_predict: int | None = DEFAULT_NUM_PREDICT_OFFICIAL,
    max_retries: int = 2,
) -> OfficialReview:
    prompt = build_official_review_prompt(paper_id, paper_text, retrieved_chunks)

    def _validate(raw: str, token_usage: dict[str, Any]) -> OfficialReview:
        review = OfficialReview.model_validate_json(raw)
        parsed = review.model_dump()
        parsed["paper_id"] = paper_id
        parsed["reviewer_model"] = reviewer_model
        parsed["token_usage"] = {
            "prompt_tokens": token_usage["prompt_tokens"],
            "completion_tokens": token_usage["completion_tokens"],
            "total_tokens": token_usage["total_tokens"],
        }
        parsed["timing"] = {
            "total_duration_seconds": token_usage["total_duration_seconds"],
            "prompt_eval_duration_seconds": token_usage["prompt_eval_duration_seconds"],
            "eval_duration_seconds": token_usage["eval_duration_seconds"],
        }
        return OfficialReview.model_validate(parsed)

    return generate_with_retry(
        reviewer_model, prompt, OFFICIAL_REVIEW_SCHEMA, _validate,
        temperature, think, num_predict, max_retries,
    )

# ---------------------------------------------------------------------------
# Area Chair / meta-review prompt
# ---------------------------------------------------------------------------


def format_official_reviews(reviews: list[OfficialReview]) -> str:
    parts = []
    for i, r in enumerate(reviews, start=1):
        parts.append(
            f"### Reviewer {i} ({r.reviewer_model})\n"
            f"Title: {r.title}\n"
            f"Rating: {r.rating}/10 | Confidence: {r.confidence}/5\n"
            f"Review:\n{r.review}"
        )
    return "\n\n".join(parts)


def build_meta_review_prompt(paper_id: str, official_reviews: list[OfficialReview]) -> str:
    instructions = f"""
    You are the Area Chair for a top-tier ML conference.

    You are given the official reviews written by independent reviewers for
    the same paper. Each review has a title, a free-text review body
    (which internally covers summary, strengths, weaknesses, and
    recommendation rationale as prose, not separate fields), a rating, and
    a confidence score. There is no separate accept/reject label from the
    reviewers -- interpret their intent from the rating band and the
    review text itself, the same way a real Area Chair would.

    {RATING_SCALE_GUIDE}

    Real ML conferences reject most submissions, but this is background
    context, not a rule to apply mechanically to every paper you see. Some
    papers in this pipeline genuinely deserve acceptance; your job is to
    tell those apart from the ones that don't, case by case, rather than
    leaning toward either outcome by default.

    Before deciding, work through this explicitly:
    1. Restate the three ratings and confidences to yourself.
    2. For each review, identify any weakness you judge SIGNIFICANT --
       meaning it casts real doubt on the paper's central claims, novelty,
       or the validity of its evidence, as opposed to a minor or cosmetic
       issue -- regardless of how positive that reviewer's own rating or
       concluding language reads. `significant_weakness_count` is how many
       of the three reviews raise at least one such weakness.
    3. The raw count from step 2 does NOT by itself determine the
       decision. What matters is whether the significant weaknesses
       converge on the same underlying gap, and how central that gap is:
       - If two or more reviewers independently raise the SAME central
         gap (e.g. missing comparison to the closest prior method,
         fundamentally unsupported claims), that convergence is strong
         evidence the concern is real, and normally supports reject --
         even if the numeric ratings alone average into the accept range.
       - If each reviewer instead raises a DIFFERENT, non-overlapping
         weakness, and none of them individually undermines the core
         contribution, that is consistent with a paper that has room for
         improvement but no single disqualifying flaw -- this can still
         support accept, even when significant_weakness_count is high.
       - A single significant weakness raised by only one reviewer,
         especially when the other two rate the paper well and raise no
         comparable concern, is a genuinely borderline case -- weigh it
         against the paper's strengths rather than defaulting to either
         outcome.

    Calibration examples (illustrative only, not real papers):

    Example A (convergent, central -> reject): three reviewers rate 7, 6,
    6. Two of the three independently note that the paper does not
    compare against the most relevant prior method, despite both also
    saying the paper is otherwise well-written. Two reviewers converging
    on the same central gap outweighs the numeric average -- the correct
    decision here is reject.

    Example B (no significant weaknesses -> accept): three reviewers rate
    8, 7, 8. None of the three raises anything beyond a minor, cosmetic
    point (e.g. requesting one more ablation, or noting a typo). No
    significant weaknesses were raised -- the correct decision here is
    accept.

    Example C (high count, but scattered and non-central -> still accept):
    three reviewers rate 7, 7, 6. Each raises a different significant-
    sounding weakness -- one wants a larger-scale experiment, one wants
    more hyperparameter analysis, one wants comparison to one more
    baseline -- but none individually casts doubt on whether the core
    method works or is novel, and the three do not overlap. Even though
    significant_weakness_count would be 3 here, the correct decision is
    still accept: the concerns are real but scattered and non-central,
    not a shared, disqualifying flaw.

    You do NOT have access to the paper itself -- base your decision only
    on what the reviewers reported.

    Synthesize the reviews into:
    - summary: a brief synthesis of what the paper is about, based on the reviews
    - consensus_strengths: strengths multiple reviewers agree on (or the
      strongest points raised)
    - consensus_weaknesses: weaknesses multiple reviewers agree on (or the
      most serious points raised)
    - significant_weakness_count: your count from step 2 above (0-3) --
      report this honestly; it is a diagnostic count and does not by
      itself determine the decision (see step 3)
    - decision: "accept" or "reject"
    - justification: a concise paragraph that states the significant-
      weakness count, whether those weaknesses converge on a shared
      central gap or are scattered and non-central, and how that
      distinction -- not the count alone -- led to the decision

    State your conclusions directly and concisely. Do not think step-by-step
    in any field -- just report your conclusions.

    Output ONLY valid JSON matching the provided schema.
    """.strip()

    return f"""
{instructions}

Paper ID: {paper_id}

Official reviews:
{format_official_reviews(official_reviews)}

JSON schema:
{json.dumps(META_REVIEW_SCHEMA, ensure_ascii=False)}
""".strip()


def generate_meta_review(
    paper_id: str,
    official_reviews: list[OfficialReview],
    area_chair_model: str,
    temperature: float,
    think: str | bool | None,
    num_predict: int | None = DEFAULT_NUM_PREDICT_META,
    max_retries: int = 2,
) -> MetaReview:
    prompt = build_meta_review_prompt(paper_id, official_reviews)

    def _validate(raw: str, token_usage: dict[str, Any]) -> MetaReview:
        review = MetaReview.model_validate_json(raw)
        parsed = review.model_dump()
        parsed["paper_id"] = paper_id
        parsed["area_chair_model"] = area_chair_model
        parsed["token_usage"] = {
            "prompt_tokens": token_usage["prompt_tokens"],
            "completion_tokens": token_usage["completion_tokens"],
            "total_tokens": token_usage["total_tokens"],
        }
        parsed["timing"] = {
            "total_duration_seconds": token_usage["total_duration_seconds"],
            "prompt_eval_duration_seconds": token_usage["prompt_eval_duration_seconds"],
            "eval_duration_seconds": token_usage["eval_duration_seconds"],
        }
        return MetaReview.model_validate(parsed)

    return generate_with_retry(
        area_chair_model, prompt, META_REVIEW_SCHEMA, _validate,
        temperature, think, num_predict, max_retries,
    )

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_single_reviewer(
    paper: dict[str, Any],
    paper_id: str,
    paper_text: str,
    reviewer_model: str,
    review_collection: chromadb.Collection,
    embed_model: str,
    top_k_chunks: int,
    temperature: float,
    think: str | bool | None,
    num_predict_official: int | None = DEFAULT_NUM_PREDICT_OFFICIAL,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    query = build_retrieval_query(paper)
    chunks = retrieve_review_chunks(review_collection, query, embed_model, top_k_chunks)
    t1 = time.perf_counter()

    official_review = generate_official_review(
        paper_id, paper_text, chunks, reviewer_model, temperature, think, num_predict_official
    )
    t2 = time.perf_counter()

    return {
        "reviewer_model": reviewer_model,
        "retrieved_chunks": chunks,
        "official_review": official_review.model_dump(),
        "timings_seconds": {
            "retrieval": round(t1 - t0, 2),
            "official_review": round(t2 - t1, 2),
            "total": round(t2 - t0, 2),
        },
    }


def review_paper(
    paper_path: Path,
    reviewer_models: list[str],
    area_chair_model: str,
    review_collection: chromadb.Collection,
    embed_model: str,
    top_k_chunks: int,
    temperature: float,
    think: str | bool | None,
    num_predict_official: int | None = DEFAULT_NUM_PREDICT_OFFICIAL,
    num_predict_meta: int | None = DEFAULT_NUM_PREDICT_META,
) -> dict[str, Any]:
    paper = load_json(paper_path)
    paper_id = extract_paper_id(paper, paper_path)
    paper_text = build_paper_text(paper)

    if not paper_text.strip():
        raise ValueError("No usable paper text found")

    reviewer_runs = [
        run_single_reviewer(
            paper, paper_id, paper_text, model, review_collection,
            embed_model, top_k_chunks, temperature, think, num_predict_official,
        )
        for model in reviewer_models
    ]

    official_reviews = [OfficialReview.model_validate(r["official_review"]) for r in reviewer_runs]

    t_ac_0 = time.perf_counter()
    meta_review = generate_meta_review(
        paper_id, official_reviews, area_chair_model, temperature, think, num_predict_meta
    )
    t_ac_1 = time.perf_counter()

    meta_review_dump = meta_review.model_dump()
    paper_token_usage = aggregate_token_usage(
        [r["official_review"]["token_usage"] for r in reviewer_runs]
        + [meta_review_dump["token_usage"]]
    )

    return {
        "paper_id": paper_id,
        "input_path": str(paper_path),
        "reviewer_models": reviewer_models,
        "area_chair_model": area_chair_model,
        "reviews": reviewer_runs,
        "meta_review": meta_review_dump,
        "token_usage": paper_token_usage,
        "timings_seconds": {
            "per_reviewer_total": [r["timings_seconds"]["total"] for r in reviewer_runs],
            "area_chair": round(t_ac_1 - t_ac_0, 2),
        },
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_think(value: str | None) -> str | bool | None:
    if value is None or value.lower() == "none":
        return None
    lowered = value.lower()
    if lowered == "false":
        return False
    if lowered == "true":
        return True
    return lowered


def iter_json_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*.json") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviews and Area Chair decisions for assembled papers")

    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing assembled paper JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where review-pipeline JSON files are saved")

    parser.add_argument("--reviewer-models", nargs="+", default=DEFAULT_REVIEWER_MODELS, help="Models used as independent reviewers")
    parser.add_argument("--area-chair-model", type=str, default=DEFAULT_AREA_CHAIR_MODEL, help="Model used as the Area Chair")

    parser.add_argument("--reviews-chroma-dir", type=str, default=DEFAULT_REVIEW_CHROMA_DIR, help="Directory containing the review ChromaDB index")
    parser.add_argument("--reviews-collection", type=str, default=DEFAULT_REVIEW_COLLECTION, help="Name of the ChromaDB review collection")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Ollama model used for retrieval embeddings")
    parser.add_argument("--top-k-chunks", type=int, default=DEFAULT_TOP_K_CHUNKS, help="Number of reference-review chunks retrieved per reviewer")

    parser.add_argument("--think", type=str, default="false", help="Reasoning level: false, true, low, medium, high, or none")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature")

    parser.add_argument("--num-predict-official", type=int, default=DEFAULT_NUM_PREDICT_OFFICIAL, help="Initial output-token budget for official reviews")
    parser.add_argument("--num-predict-meta", type=int, default=DEFAULT_NUM_PREDICT_META, help="Initial output-token budget for meta-reviews")

    parser.add_argument("--limit", type=int, default=None, help="Maximum number of papers to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip papers whose output file already exists")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between papers in seconds")

    args = parser.parse_args()

    think_value = resolve_think(args.think)
    review_collection = get_review_collection(args.reviews_chroma_dir, args.reviews_collection)

    all_files = iter_json_files(args.input_dir)
    if args.limit is not None:
        all_files = all_files[: args.limit]
    if not all_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    failures_path = args.output_dir / "failures.jsonl"
    succeeded = failed = skipped = 0
    started_at = time.time()
    run_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for idx, paper_path in enumerate(all_files, start=1):
        out_path = args.output_dir / f"{paper_path.stem}.review_pipeline.json"

        if args.skip_existing and out_path.exists():
            skipped += 1
            print(f"[{idx}/{len(all_files)}] SKIP {paper_path}")
            continue

        print(f"[{idx}/{len(all_files)}] RUN  {paper_path}")
        try:
            result = review_paper(
                paper_path=paper_path,
                reviewer_models=args.reviewer_models,
                area_chair_model=args.area_chair_model,
                review_collection=review_collection,
                embed_model=args.embed_model,
                top_k_chunks=args.top_k_chunks,
                temperature=args.temperature,
                think=think_value,
                num_predict_official=args.num_predict_official,
                num_predict_meta=args.num_predict_meta,
            )
            save_json(result, out_path)
            succeeded += 1

            tu = result["token_usage"]
            run_token_usage["prompt_tokens"] += tu["prompt_tokens"]
            run_token_usage["completion_tokens"] += tu["completion_tokens"]
            run_token_usage["total_tokens"] += tu["total_tokens"]

            print(f"          OK   saved to {out_path}")
            print(
                f"          tokens: prompt={tu['prompt_tokens']} "
                f"completion={tu['completion_tokens']} total={tu['total_tokens']}"
            )
        except Exception as e:
            failed += 1
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            with failures_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "paper_path": str(paper_path),
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"          FAIL {type(e).__name__}: {e}")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    elapsed = time.time() - started_at
    print(f"\nDone in {elapsed:.1f}s. succeeded={succeeded} failed={failed} skipped={skipped}")
    print(
        f"Total tokens: prompt={run_token_usage['prompt_tokens']} "
        f"completion={run_token_usage['completion_tokens']} "
        f"total={run_token_usage['total_tokens']}"
    )


if __name__ == "__main__":
    main()