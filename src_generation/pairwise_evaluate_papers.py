#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import ollama


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def list_papers_by_idea(assembled_dir: str, models: list[str] | None = None, limit_per_model: int | None = None) -> dict[str, list[dict]]:
    base = Path(assembled_dir)
    if not base.exists():
        return {}

    model_dirs = [p for p in base.iterdir() if p.is_dir()]
    if models:
        wanted = set(models)
        model_dirs = [p for p in model_dirs if p.name in wanted]

    grouped: dict[str, list[dict]] = {}

    for model_dir in sorted(model_dirs):
        files = sorted(model_dir.glob("*.json"))
        if limit_per_model is not None:
            files = files[:limit_per_model]

        for path in files:
            paper = load_json(path)
            idea_id = paper["idea_id"]
            grouped.setdefault(idea_id, []).append(paper)

    return grouped


def paper_to_text(label: str, paper: dict) -> str:
    sections = paper.get("sections", {})
    return f"""
Paper {label}
Model: {paper.get("model", "")}
Title: {paper.get("selected_title", "")}
Abstract: {paper.get("abstract", "")}

Introduction:
{sections.get("introduction", "")}

Method:
{sections.get("method", "")}

Experiments:
{sections.get("experiments", "")}

Conclusion:
{sections.get("conclusion", "")}
""".strip()


def build_pairwise_prompt(paper_a: dict, paper_b: dict) -> tuple[str, str]:
    system_prompt = (
        "You are a strict reviewer for a top machine learning conference. "
        "You are comparing two synthetic research papers generated from the same idea. "
        "Be critical and discriminative. Avoid ties unless they are genuinely warranted. "
        "Return only valid JSON."
    )

    user_prompt = f"""
Compare the following two papers generated from the same frozen idea.

Evaluation dimensions:
- faithfulness: alignment with title and abstract
- cross_section_consistency: coherence across sections
- scientific_plausibility: realism and technical credibility
- experiments_quality: credibility and usefulness of the experiments section
- overall_winner: overall better paper

Important rules:
- Choose "A", "B", or "tie" for each field.
- Use "tie" only if the two papers are genuinely indistinguishable on that dimension.
- Prefer using A or B rather than tie.
- Be strict and discriminative.

{paper_to_text("A", paper_a)}

---

{paper_to_text("B", paper_b)}

Return ONLY JSON in exactly this format:
{{
  "faithfulness_winner": "A|B|tie",
  "cross_section_consistency_winner": "A|B|tie",
  "scientific_plausibility_winner": "A|B|tie",
  "experiments_quality_winner": "A|B|tie",
  "overall_winner": "A|B|tie",
  "reasoning": {{
    "faithfulness": "...",
    "cross_section_consistency": "...",
    "scientific_plausibility": "...",
    "experiments_quality": "...",
    "overall": "..."
  }}
}}
"""
    return system_prompt.strip(), user_prompt.strip()


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from judge response")


def judge_pair(paper_a: dict, paper_b: dict, judge_model: str) -> dict:
    system_prompt, user_prompt = build_pairwise_prompt(paper_a, paper_b)

    response = ollama.chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0},
    )

    raw = response["message"]["content"]
    return extract_json(raw)


def comparison_output_path(output_dir: str, idea_id: str, model_a: str, model_b: str) -> Path:
    safe_a = re.sub(r"[^a-zA-Z0-9_\-]+", "_", model_a)
    safe_b = re.sub(r"[^a-zA-Z0-9_\-]+", "_", model_b)
    return Path(output_dir) / idea_id / f"{safe_a}__vs__{safe_b}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Pairwise evaluation of assembled papers")
    ap.add_argument("--assembled-dir", default="./assembled_papers")
    ap.add_argument("--output-dir", default="./pairwise_evaluations")
    ap.add_argument("--judge-model", default="llama3.1:8b")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Max papers per model")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    grouped = list_papers_by_idea(
        args.assembled_dir,
        models=args.models,
        limit_per_model=args.limit,
    )

    if not grouped:
        raise RuntimeError("No assembled papers found.")

    idea_items = sorted(grouped.items(), key=lambda x: x[0])

    total_jobs = 0
    for _, papers in idea_items:
        total_jobs += len(list(itertools.combinations(papers, 2)))

    job_idx = 0

    for idea_idx, (idea_id, papers) in enumerate(idea_items, start=1):
        if len(papers) < 2:
            continue

        print(f"\n[idea {idea_idx}/{len(idea_items)}] {idea_id}")

        for paper_a, paper_b in itertools.combinations(papers, 2):
            job_idx += 1
            model_a = paper_a["model"]
            model_b = paper_b["model"]

            out_path = comparison_output_path(args.output_dir, idea_id, model_a, model_b)

            print(f"[job {job_idx}/{total_jobs}] {model_a} vs {model_b}")

            if args.skip_existing and out_path.exists():
                print(f"[skip] {out_path}")
                continue

            result = judge_pair(paper_a, paper_b, args.judge_model)

            payload = {
                "idea_id": idea_id,
                "judge_model": args.judge_model,
                "paper_a_model": model_a,
                "paper_b_model": model_b,
                "paper_a_title": paper_a.get("selected_title"),
                "paper_b_title": paper_b.get("selected_title"),
                "result": result,
            }

            save_json(out_path, payload)
            print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()