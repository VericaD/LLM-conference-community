#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def list_paper_files(
    assembled_dir: str,
    models: list[str] | None = None,
    limit_per_model: int | None = None,
) -> list[Path]:
    base = Path(assembled_dir)
    if not base.exists():
        return []

    model_dirs = [p for p in base.iterdir() if p.is_dir()]
    if models:
        wanted = set(models)
        model_dirs = [p for p in model_dirs if p.name in wanted]

    files = []
    for model_dir in sorted(model_dirs):
        model_files = sorted(model_dir.glob("*.json"))
        if limit_per_model is not None:
            model_files = model_files[:limit_per_model]
        files.extend(model_files)

    return files


def build_judge_prompt(paper: dict) -> tuple[str, str]:
    system_prompt = (
        "You are a strict reviewer for a top machine learning conference (ICLR). "
        "You must be critical and use the full scoring range."
    )

    user_prompt = f"""
Evaluate the following synthetic ML paper.

Title:
{paper.get("selected_title", "")}

Abstract:
{paper.get("abstract", "")}

Introduction:
{paper.get("sections", {}).get("introduction", "")}

Method:
{paper.get("sections", {}).get("method", "")}

Experiments:
{paper.get("sections", {}).get("experiments", "")}

Conclusion:
{paper.get("sections", {}).get("conclusion", "")}

Scoring rules (IMPORTANT):
- Use the full range 1–5.
- Be critical: most synthetic papers should score between 2 and 4.
- Only give 5 if the section is very strong and specific.
- Penalize vagueness and generic descriptions.

Definitions:
- faithfulness:
    5 = perfectly aligned with title and abstract
    3 = partially aligned, some drift
    1 = unrelated

- cross_section_consistency:
    5 = sections fully coherent
    3 = minor inconsistencies
    1 = contradictory

- scientific_plausibility:
    5 = realistic method with concrete details
    3 = generic or vague method
    1 = unrealistic or incorrect

- experiments_quality:
    5 = realistic setup with proper baselines and metrics
    3 = partial or weak evaluation
    1 = missing or not credible

Critical rule:
If the method lacks concrete technical details (no equations, no algorithm),
scientific_plausibility MUST be ≤ 3.

Return ONLY JSON:
{{
  "faithfulness": <int>,
  "cross_section_consistency": <int>,
  "scientific_plausibility": <int>,
  "experiments_quality": <int>
}}
"""
    return system_prompt.strip(), user_prompt.strip()

def extract_json(text: str) -> dict:
    text = text.strip()

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # fenced block or extra commentary fallback
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from judge response")


def judge_paper(paper: dict, judge_model: str) -> dict:
    system_prompt, user_prompt = build_judge_prompt(paper)

    response = ollama.chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.0},
    )

    raw = response["message"]["content"]
    scores = extract_json(raw)
    return scores


def make_eval_output_path(output_dir: str, paper_path: Path) -> Path:
    model_dir = paper_path.parent.name
    return Path(output_dir) / model_dir / paper_path.name


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate assembled synthetic papers with an LLM judge")
    ap.add_argument("--assembled-dir", default="./assembled_papers")
    ap.add_argument("--output-dir", default="./evaluations")
    ap.add_argument("--judge-model", default="llama3.1:8b")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    paper_files = list_paper_files(
        args.assembled_dir,
        models=args.models,
        limit_per_model=args.limit,
    )

    if not paper_files:
        raise RuntimeError("No assembled papers found.")

    for idx, paper_path in enumerate(paper_files, start=1):
        out_path = make_eval_output_path(args.output_dir, paper_path)

        print(f"[{idx}/{len(paper_files)}] {paper_path}")

        if args.skip_existing and out_path.exists():
            print(f"[skip] {out_path}")
            continue

        paper = load_json(paper_path)
        scores_list = [judge_paper(paper, args.judge_model) for _ in range(3)]

        avg_scores = {
            key: round(sum(s[key] for s in scores_list) / len(scores_list), 2)
            for key in scores_list[0].keys()
        }

        payload = {
            "idea_id": paper.get("idea_id"),
            "model": paper.get("model"),
            "selected_title": paper.get("selected_title"),
            "scores": avg_scores,
            "scores_list": scores_list,
        }

        save_json(out_path, payload)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()