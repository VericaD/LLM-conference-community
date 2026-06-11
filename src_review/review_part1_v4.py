from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import ollama
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from pydantic import field_validator


DEFAULT_MODEL = "qwen3:8b"

'''
Prompt is improved by adding constraints 
Reviews saved in review_outputs/*_cluster_v2
'''

# =========================
# Pydantic output models
# =========================


CriterionName = Literal[
    "problem_clearly_defined",
    "method_clearly_explained",
    "structure_logical_and_coherent",
    "method_specified_in_detail",
    "assumptions_explicit_and_reasonable",
    "reasoning_logically_consistent",
    "contribution_clearly_identified",
    "comparison_to_prior_work",
    "novelty_not_just_claimed_but_justified",
    "datasets_and_setup_specified",
    "baselines_clearly_named",
    "evaluation_metrics_defined",
    "results_clearly_presented",
    "experiments_support_claims",
    "vague_method_description",
    "missing_experimental_details",
    "unsupported_claims",
    "no_comparison_to_prior_work",
    "generic_or_boilerplate_limitations",
]

CriterionVerdict = Literal["satisfied", "partially_satisfied", "not_satisfied"]


class DimensionScores(BaseModel):
    clarity: int = Field(ge=1, le=5)
    soundness: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    empirical_quality: int = Field(ge=1, le=5)


class CriterionEvaluation(BaseModel):
    name: CriterionName
    dimension: Literal["clarity", "soundness", "novelty", "empirical_quality", "penalty"]
    verdict: CriterionVerdict
    score: float 
    weight: int
    evidence: str
    explanation: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v):
        allowed = [0.0, 0.5, 1.0]
        for a in allowed:
            if abs(v - a) < 1e-3:
                return a
        raise ValueError("score must be 0.0, 0.5, or 1.0")


class AnalyticalReview(BaseModel):
    paper_id: str
    generator_model: str
    topic: str
    selected_title: str
    model_reviewing: str

    summary: str

    dimension_scores: DimensionScores

    criteria_evaluation: list[CriterionEvaluation] = Field(min_length=19, max_length=19)

    strengths: list[str] = Field(min_length=2, max_length=5)
    weaknesses: list[str] = Field(min_length=2, max_length=5)

    confidence: int = Field(ge=1, le=5)


ANALYTICAL_SCHEMA: dict[str, Any] = AnalyticalReview.model_json_schema()


SECTION_ORDER = [
    "introduction",
    "related_work",
    "background",
    "method",
    "experiments",
    "conclusion",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_paper_id(paper: dict[str, Any], fallback_path: Path) -> str:
    return str(
        paper.get("idea_id")
        or paper.get("paper_id")
        or fallback_path.stem
    )


def build_paper_text(paper: dict[str, Any]) -> str:
    parts: list[str] = []

    title = paper.get("selected_title")
    abstract = paper.get("abstract")
    topic = paper.get("topic")
    chosen_direction = paper.get("chosen_direction")
    refined_problem = paper.get("refined_problem")

    if isinstance(title, str) and title.strip():
        parts.append(f"## Title\n{title.strip()}")

    if isinstance(topic, str) and topic.strip():
        parts.append(f"## Topic\n{topic.strip()}")

    if isinstance(chosen_direction, str) and chosen_direction.strip():
        parts.append(f"## Chosen Direction\n{chosen_direction.strip()}")

    if isinstance(refined_problem, str) and refined_problem.strip():
        parts.append(f"## Refined Problem\n{refined_problem.strip()}")

    if isinstance(abstract, str) and abstract.strip():
        parts.append(f"## Abstract\n{abstract.strip()}")

    sections = paper.get("sections", {})
    if isinstance(sections, dict):
        for key in SECTION_ORDER:
            value = sections.get(key)
            if isinstance(value, str) and value.strip():
                header = key.replace("_", " ").title()
                parts.append(f"## {header}\n{value.strip()}")

    if not parts:
        full_text = paper.get("full_text", "")
        if isinstance(full_text, str) and full_text.strip():
            parts.append(f"## Full Text\n{full_text.strip()}")

    return "\n\n".join(parts)


def build_prompt(paper_id: str, paper_text: str, reviewer_model: str) -> str:
    rubric = """
    You are performing the ANALYTICAL EVALUATION stage of a scientific peer-review pipeline.

    Your task is NOT to write a final review.
    Your task is to evaluate the paper using a structured, criterion-based rubric.

    Evaluate ONLY what is explicitly present in the paper.
    Do NOT infer missing details.
    Do NOT reward fluency, confidence, or academic tone.

    ----------------------------------------
    EVALUATION DIMENSIONS
    ----------------------------------------

    You must evaluate the paper using 4 dimensions:

    1. clarity
    2. soundness
    3. novelty
    4. empirical_quality

    CLARITY:
    Measures whether the paper clearly communicates its problem, method, structure, and conclusions.

    SOUNDNESS:
    Measures whether the method and reasoning are technically coherent, plausible, justified, and sufficiently specified.

    NOVELTY:
    Measures whether the contribution is clearly distinguished from prior work and whether novelty claims are justified.

    EMPIRICAL QUALITY:
    Measures whether experiments, datasets, baselines, metrics, and results are specific enough to support the claims.

    ----------------------------------------
    CRITERIA TO EVALUATE
    ----------------------------------------

    Evaluate EXACTLY the following 19 criteria.

    For each criterion, output:
    - name
    - dimension
    - verdict: satisfied, partially_satisfied, or not_satisfied
    - score: 1.0, 0.5, or 0.0
    - weight
    - evidence: direct quote or short textual evidence from the paper
    - explanation

    CLARITY:
    1. problem_clearly_defined | dimension=clarity | weight=3
    2. method_clearly_explained | dimension=clarity | weight=3
    3. structure_logical_and_coherent | dimension=clarity | weight=2

    SOUNDNESS:
    4. method_specified_in_detail | dimension=soundness | weight=4
    5. assumptions_explicit_and_reasonable | dimension=soundness | weight=2
    6. reasoning_logically_consistent | dimension=soundness | weight=3

    NOVELTY:
    7. contribution_clearly_identified | dimension=novelty | weight=3
    8. comparison_to_prior_work | dimension=novelty | weight=4
    9. novelty_not_just_claimed_but_justified | dimension=novelty | weight=3

    EMPIRICAL QUALITY:
    10. datasets_and_setup_specified | dimension=empirical_quality | weight=4
    11. baselines_clearly_named | dimension=empirical_quality | weight=4
    12. evaluation_metrics_defined | dimension=empirical_quality | weight=3
    13. results_clearly_presented | dimension=empirical_quality | weight=3
    14. experiments_support_claims | dimension=empirical_quality | weight=4

    PENALTIES:
    15. vague_method_description | dimension=penalty | weight=-4
    16. missing_experimental_details | dimension=penalty | weight=-5
    17. unsupported_claims | dimension=penalty | weight=-4
    18. no_comparison_to_prior_work | dimension=penalty | weight=-4
    19. generic_or_boilerplate_limitations | dimension=penalty | weight=-2

    ----------------------------------------
    PENALTY RULES
    ----------------------------------------

    Penalty criteria must be evaluated critically and independently from positive criteria.

    IMPORTANT PRINCIPLE:
    The presence of content does NOT mean the absence of a flaw.

    A penalty should be triggered whenever the quality is insufficient, not only when something is completely missing.

    Penalty scoring:
    - score = 1.0:
    The flaw is clearly present and significantly affects the paper.

    - score = 0.5:
    The flaw is partially present.
    This is the DEFAULT case when content exists but is incomplete, vague, underspecified, or insufficient.

    - score = 0.0:
    The flaw is clearly absent.
    This requires strong, concrete, and detailed evidence that the issue does NOT exist.

    CRITICAL RULE:
    If a positive criterion is partially_satisfied, the corresponding penalty should usually be 0.5.

    Examples:

    - method_specified_in_detail = partially_satisfied  
    → vague_method_description = 0.5

    - datasets_and_setup_specified = partially_satisfied  
    → missing_experimental_details = 0.5

    - comparison_to_prior_work = partially_satisfied  
    → no_comparison_to_prior_work = 0.5

    - experiments_support_claims = partially_satisfied  
    → unsupported_claims = 0.5

    DO NOT assign penalty = 0.0 unless the paper clearly demonstrates strong, detailed, and complete evidence.

    If in doubt → assign 0.5.

    ----------------------------------------
    REWARD–PENALTY CONSISTENCY
    ----------------------------------------

    Positive criteria and penalties must be consistent.

    - If a positive criterion is partially_satisfied → the corresponding penalty should be 0.5.
    - If a positive criterion is not_satisfied → the corresponding penalty should be 1.0.

    It is inconsistent to assign:
    - a positive criterion = 0.5
    - and the corresponding penalty = 0.0

    Avoid such inconsistencies.

    ----------------------------------------
    STRICT EVIDENCE THRESHOLDS
    ----------------------------------------

    For ALL positive criteria:

    - satisfied (score = 1.0):
    Requires concrete, specific, detailed evidence sufficient for verification.

    - partially_satisfied (score = 0.5):
    Evidence exists but is incomplete, vague, or high-level.

    - not_satisfied (score = 0.0):
    Criterion is unsupported or absent.

    CRITICAL RULE:
    Mentioning something is NOT enough.

    When in doubt - choose partially_satisfied.


    ----------------------------------------
    CRITERION-SPECIFIC THRESHOLDS
    ----------------------------------------

    method_specified_in_detail:
    - 1.0 only if algorithm, objectives, and implementation details are clear
    - 0.5 if high-level description only

    comparison_to_prior_work:
    - 1.0 only if explicit comparison with named methods
    - 0.5 if generic discussion

    datasets_and_setup_specified:
    - 1.0 only if environments, setup, and conditions are specified
    - 0.5 if vague

    baselines_clearly_named:
    - 1.0 only if clearly named and used
    - 0.5 if partially specified

    evaluation_metrics_defined:
    - 1.0 only if metrics are defined or computable
    - 0.5 if just listed

    results_clearly_presented:
    - 1.0 only if numerical results are shown
    - 0.5 if summarized

    experiments_support_claims:
    - 1.0 only if experiments convincingly support claims
    - 0.5 if partial support

    ----------------------------------------
    SCORING RULES
    ----------------------------------------

    Assign final dimension_scores from 1 to 5.

    1 = very weak
    2 = weak
    3 = mixed / partially convincing
    4 = strong
    5 = excellent

    Constraints:

    - If method_specified_in_detail = 0.0 → soundness ≤ 2  
    - If baselines or datasets = 0.0 → empirical_quality ≤ 2  
    - If comparison_to_prior_work = 0.0 → novelty ≤ 2  
    - If vague_method_description = 1.0 → soundness ≤ 2  
    - If missing_experimental_details = 1.0 → empirical_quality ≤ 2  

    Do NOT assign 4 or 5 without strong evidence.

    ----------------------------------------
    ANTI-LENIENCY CHECK
    ----------------------------------------

    Before finalizing:

    - Are 1.0 scores truly justified by detailed evidence?
    - Are some 1.0 actually 0.5?
    - Did you confuse clarity with correctness?

    If yes → downgrade.
    ----------------------------------------
    CONFIDENCE
    ----------------------------------------

    Confidence measures how reliable your evaluation is given the completeness of the paper.

    - 5 = very high confidence
    Only if the paper provides detailed, concrete, and reproducible information across method and experiments.

    - 4 = reasonably high confidence
    Minor missing details, but overall evaluation is reliable.

    - 3 = moderate confidence
    Some important aspects are unclear or underspecified.
    
    - 2 = low confidence
    Key components (method or experiments) are vague.

    - 1 = very low confidence
    The paper is too incomplete to evaluate reliably.

    If the method or experiments are vague or incomplete, confidence should be <= 3.
    If multiple core criteria are missing, confidence should be <= 2.

    ----------------------------------------
    TASK
    ----------------------------------------

    Produce:
    1. summary
    2. dimension_scores
    3. criteria_evaluation with exactly 19 items
    4. strengths
    5. weaknesses
    6. confidence

    Output ONLY valid JSON matching the provided schema.
    """.strip()

    return f"""
{rubric}

Reviewer model: {reviewer_model}
Paper ID: {paper_id}

JSON schema:
{json.dumps(ANALYTICAL_SCHEMA, ensure_ascii=False)}

Paper to evaluate:
{paper_text}
""".strip()

def validate_and_patch_output(
    raw_json: str,
    paper: dict[str, Any],
    paper_id: str,
    model: str,
) -> dict[str, Any]:
    review = AnalyticalReview.model_validate_json(raw_json)
    parsed = review.model_dump()

    # Patch canonical metadata from source file
    parsed["paper_id"] = paper_id
    parsed["generator_model"] = str(paper.get("model", "unknown"))
    parsed["topic"] = str(paper.get("topic", "unknown"))
    parsed["selected_title"] = str(paper.get("selected_title", "unknown"))
    parsed["model_reviewing"] = model

    # Re-validate after patching
    review = AnalyticalReview.model_validate(parsed)
    return review.model_dump()


def evaluate_paper(
    paper_path: Path,
    model: str,
    think: str | bool,
    temperature: float,
    max_retries: int = 1,
) -> dict[str, Any]:
    paper = load_json(paper_path)
    paper_id = extract_paper_id(paper, paper_path)
    paper_text = build_paper_text(paper)

    if not paper_text.strip():
        raise ValueError("No usable paper text found")

    prompt = build_prompt(
        paper_id=paper_id,
        paper_text=paper_text,
        reviewer_model=model,
    )

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        retry_prompt = prompt
        if attempt > 0:
            retry_prompt = f"""
            You are revising a previous criterion-based analytical evaluation of a research paper.

            Return ONLY valid JSON matching the schema.

            Fix the review by ensuring:
            - criteria_evaluation contains exactly 19 criteria
            - every criterion has name, dimension, verdict, score, weight, evidence, and explanation
            - verdict is one of: satisfied, partially_satisfied, not_satisfied
            - score is one of: 1.0, 0.5, 0.0
            - penalty criteria use score=1.0 when the flaw is present
            - final dimension_scores respect the strict scoring rules
            - evidence is based only on the paper text
            - strengths and weaknesses are specific
            - confidence reflects how complete the paper is
            - criteria_evaluation must follow exactly the same order as the 19 criteria listed in the rubric

            Paper:
            {paper_text}

            JSON schema:
            {json.dumps(ANALYTICAL_SCHEMA, ensure_ascii=False)}
            """.strip()

        try:
            response = ollama.generate(
                model=model,
                prompt=retry_prompt,
                format=ANALYTICAL_SCHEMA,
                stream=False,
                think=think,
                options={
                    "temperature": temperature,
                },
            )

            raw = response["response"]
            return validate_and_patch_output(
                raw_json=raw,
                paper=paper,
                paper_id=paper_id,
                model=model,
            )

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e

    raise ValueError(f"Invalid analytical output after retries: {last_error}")


def resolve_think(value: str) -> str | bool:
    lowered = value.lower()
    if lowered == "false":
        return False
    if lowered == "true":
        return True
    return lowered


def get_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = input_path.relative_to(input_dir)
    return output_dir / relative.with_suffix(".analysis.json")


def iter_json_files(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.rglob("*.json")
        if p.is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing generated paper JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder where analytical outputs will be saved")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama reviewer model")
    parser.add_argument("--think", type=str, default="medium", help="false, true, low, medium, or high")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of papers to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip papers whose output file already exists")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Pause between papers")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries if validation fails")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    think_value = resolve_think(args.think)

    all_files = iter_json_files(input_dir)
    if args.limit is not None:
        all_files = all_files[:args.limit]

    if not all_files:
        print(f"No JSON files found in {input_dir}")
        return

    summary_path = output_dir / "run_summary.json"
    failures_path = output_dir / "failures.jsonl"

    processed = 0
    succeeded = 0
    skipped = 0
    failed = 0

    started_at = time.time()

    for idx, paper_path in enumerate(all_files, start=1):
        out_path = get_output_path(
            input_path=paper_path,
            input_dir=input_dir,
            output_dir=output_dir,
        )

        if args.skip_existing and out_path.exists():
            skipped += 1
            print(f"[{idx}/{len(all_files)}] SKIP  {paper_path}")
            continue

        print(f"[{idx}/{len(all_files)}] RUN   {paper_path}")

        processed += 1
        item_start = time.time()

        try:
            result = evaluate_paper(
                paper_path=paper_path,
                model=args.model,
                think=think_value,
                temperature=args.temperature,
                max_retries=args.max_retries,
            )
            save_json(result, out_path)
            succeeded += 1

            elapsed = time.time() - item_start
            print(f"           OK    saved to {out_path} ({elapsed:.1f}s)")

        except Exception as e:
            failed += 1
            elapsed = time.time() - item_start

            failure_record = {
                "paper_path": str(paper_path),
                "output_path": str(out_path),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "elapsed_seconds": round(elapsed, 3),
            }
            append_jsonl(failure_record, failures_path)

            print(f"           FAIL  {type(e).__name__}: {e}")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    total_elapsed = time.time() - started_at

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "think": args.think,
        "temperature": args.temperature,
        "total_files_found": len(iter_json_files(input_dir)),
        "total_files_considered": len(all_files),
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "elapsed_seconds": round(total_elapsed, 3),
    }
    save_json(summary, summary_path)

    print("\nRun complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()