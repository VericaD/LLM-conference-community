from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import ollama
from pydantic import BaseModel, Field, ValidationError


DEFAULT_MODEL = "qwen3:8b"


# =========================
# Pydantic output models
# =========================

class Scores(BaseModel):
    clarity: int = Field(ge=1, le=5)
    soundness: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    empirical_quality: int = Field(ge=1, le=5)


class Checklist(BaseModel):
    problem_clearly_defined: bool
    method_clearly_described: bool
    claims_supported_by_evidence: bool
    experiments_sufficient: bool
    limitations_acknowledged: bool


class EvidenceSpan(BaseModel):
    issue: str
    quote: str
    section: str


class AnalyticalReview(BaseModel):
    paper_id: str
    generator_model: str
    topic: str
    selected_title: str
    model_reviewing: str
    summary: str
    scores: Scores
    checklist: Checklist
    strengths: list[str] = Field(min_length=2, max_length=5)
    weaknesses: list[str] = Field(min_length=2, max_length=5)
    evidence_spans: list[EvidenceSpan] = Field(min_length=1, max_length=6)
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

Your task is NOT to write the final review.
Your task is to produce a structured analytical assessment of the paper.

The goal is to behave like a critical ICLR reviewer.

----------------------------------------
SCORING PRINCIPLES
----------------------------------------

Scores range from 1 to 5.

Interpretation:
- 1 = very weak
- 2 = weak
- 3 = reasonable but with clear limitations (DEFAULT)
- 4 = strong with solid support
- 5 = very strong and well-justified

Important:
- Use 3 as the default when the paper is plausible but has meaningful weaknesses
- Use 4 only when the paper provides strong and specific support
- Use 5 only for unusually strong evidence and execution
- Do NOT give high scores simply because the paper is well-written

----------------------------------------
CRITICAL EVALUATION RULES
----------------------------------------

- Do NOT treat the paper’s claims as evidence of correctness
- Statements in the abstract or conclusion do NOT count as evidence

A claim is supported only if the paper provides:
- detailed experimental setup
- clear description of baselines
- specific evaluation methodology

If such details are missing, mark:
- claims_supported_by_evidence = false
- or experiments_sufficient = false

----------------------------------------
SCORE CONSISTENCY CONSTRAINT
----------------------------------------

- If meaningful weaknesses are identified in:
  - methodology
  - experimental validation

then at least one of:
- soundness
- empirical_quality

must be <= 3

----------------------------------------
CHECKLIST CONSTRAINT
----------------------------------------

At least one of the following must be false unless the paper provides strong and detailed justification:
- claims_supported_by_evidence
- experiments_sufficient

----------------------------------------
----------------------------------------
CHECKLIST GUIDELINES
----------------------------------------

Mark a checklist item as TRUE only if the paper provides sufficient and concrete support.

Do NOT mark TRUE simply because:
- the topic is mentioned
- the claim is stated

----------------------------------------
WEAKNESSES REQUIREMENT
----------------------------------------

- At least one weakness must be genuinely critical
- Avoid generic weaknesses like:
  - "future work"
  - "could be extended"
- Prefer weaknesses that identify real issues in:
  - methodology
  - experimental validation
  - novelty claims
  - reproducibility

----------------------------------------
EVIDENCE SPANS
----------------------------------------

- Each evidence span must support a criticism or evaluation decision
- Do NOT just restate the paper’s claims
- The quote must justify:
  - a weakness
  - or a score reduction

Bad example:
- issue: "paper claims strong results"
- quote: "we outperform baselines"

Good example:
- issue: "experimental evidence is insufficiently detailed"
- quote: vague or underspecified experimental description

----------------------------------------
TASK
----------------------------------------

1. Produce a neutral summary
2. Assign rubric scores
3. Fill checklist (strictly)
4. Provide strengths (2–5)
5. Provide weaknesses (2–5, at least one critical)
6. Provide evidence spans (linked to criticisms)
7. Assign confidence (1–5)

----------------------------------------

Output ONLY valid JSON matching the schema.
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
        You are revising a previous analytical evaluation of a research paper.

        Improve the review by:
        - making strengths and weaknesses more specific
        - ensuring each weakness is justified by evidence from the paper
        - avoiding generic praise
        - checking whether the evidence spans directly support the stated issue
        - being appropriately critical about unsupported claims, weak experiments, and vague comparisons
        - keeping the summary neutral and concise
        - ensuring the review is internally consistent

        Return the improved analytical review in the same JSON format.

        Paper:
        {paper_text}
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