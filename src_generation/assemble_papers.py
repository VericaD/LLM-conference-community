#!/usr/bin/env python3
"""
Assemble full paper JSON files from generated section outputs.

The script reads per-section JSON files from `rag_runs/`, groups them by model
and research idea, and saves one assembled paper JSON file per idea in
`assembled_papers/`. Missing sections are recorded in the assembled output.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from generate import PAPER_SECTION_ORDER

def list_model_dirs(rag_runs_dir: str) -> list[Path]:
    base = Path(rag_runs_dir)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def list_section_files(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("*.json"))


def parse_section_filename(path: Path) -> tuple[str, str] | None:
    """
    Expected filename format:
        idea_001__introduction.json
    Returns:
        (idea_id, section_name)
    """
    stem = path.stem
    if "__" not in stem:
        return None

    idea_id, section = stem.split("__", 1)
    if not idea_id or not section:
        return None

    return idea_id, section


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_sections_for_model(model_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Returns:
        {
          "idea_001": {
              "introduction": Path(...),
              "method": Path(...),
          },
          ...
        }
    """
    idea_map: dict[str, dict[str, Path]] = {}

    for path in list_section_files(model_dir):
        parsed = parse_section_filename(path)
        if parsed is None:
            continue

        idea_id, section = parsed
        if idea_id not in idea_map:
            idea_map[idea_id] = {}

        idea_map[idea_id][section] = path

    return idea_map


def assemble_one_paper(
    model_name: str,
    idea_id: str,
    section_paths: dict[str, Path],
    required_sections: list[str],
) -> dict:
    """
    Build one full paper JSON from multiple section JSON files.
    Assumes each section JSON has common metadata fields.
    """
    loaded_sections: dict[str, dict] = {
        section: load_json(path)
        for section, path in section_paths.items()
    }

    if not loaded_sections:
        raise RuntimeError(f"No section files found for {model_name} / {idea_id}")

    # Use the first available section file as metadata source
    first_section = next(iter(loaded_sections.values()))

    assembled_sections = {}
    available_sections = []
    missing_sections = []

    for section in required_sections:
        if section in loaded_sections:
            if "generated_section" not in loaded_sections[section]:
                raise ValueError(f"Missing generated_section in {section_paths[section]}")

            assembled_sections[section] = loaded_sections[section]["generated_section"]
            available_sections.append(section)
        else:
            missing_sections.append(section)

    assembled = {
        "idea_id": idea_id,
        "model": first_section.get("generation_model", model_name),
        "topic": first_section.get("topic"),
        "chosen_direction": first_section.get("chosen_direction"),
        "refined_problem": first_section.get("refined_problem"),
        "selected_title": first_section.get("selected_title"),
        "abstract": first_section.get("abstract"),
        "sections": assembled_sections,
        "available_sections": available_sections,
        "missing_sections": missing_sections,
    }

    return assembled


def save_assembled_paper(
    output_dir: str,
    model_dir_name: str,
    idea_id: str,
    payload: dict,
) -> Path:
    out_dir = Path(output_dir) / model_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_idea_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", idea_id.strip())
    out_path = out_dir / f"{safe_idea_id}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_path

def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble full papers from generated section files")
    ap.add_argument("--rag-runs-dir", default="./rag_runs", help="Directory containing per-model section outputs")
    ap.add_argument("--output-dir", default="./assembled_papers", help="Directory for assembled paper JSON files")
    ap.add_argument("--models", nargs="*", default=None, help="Optional list of model directory names to process, e.g. qwen2_5_7b llama3_1_8b")
    ap.add_argument("--sections", nargs="+", default=PAPER_SECTION_ORDER, help="Required sections to assemble")
    ap.add_argument("--skip-existing", action="store_true", help="Skip assembled papers that already exist")

    args = ap.parse_args()

    model_dirs = list_model_dirs(args.rag_runs_dir)
    if not model_dirs:
        raise RuntimeError(f"No model directories found in: {args.rag_runs_dir}")

    if args.models:
        wanted = set(args.models)
        model_dirs = [p for p in model_dirs if p.name in wanted]

    if not model_dirs:
        raise RuntimeError("No matching model directories found.")

    for model_idx, model_dir in enumerate(model_dirs, start=1):
        print(f"\n=== Model {model_idx}/{len(model_dirs)}: {model_dir.name} ===")

        idea_map = collect_sections_for_model(model_dir)
        if not idea_map:
            print("[skip] no section files found")
            continue

        idea_items = sorted(idea_map.items(), key=lambda x: x[0])

        for idea_idx, (idea_id, section_paths) in enumerate(idea_items, start=1):
            out_path = Path(args.output_dir) / model_dir.name / f"{idea_id}.json"

            print(f"[idea {idea_idx}/{len(idea_items)}] {idea_id}")

            if args.skip_existing and out_path.exists():
                print(f"[skip] already exists: {out_path}")
                continue

            assembled = assemble_one_paper(
                model_name=model_dir.name,
                idea_id=idea_id,
                section_paths=section_paths,
                required_sections=args.sections,
            )

            saved_path = save_assembled_paper(
                output_dir=args.output_dir,
                model_dir_name=model_dir.name,
                idea_id=idea_id,
                payload=assembled,
            )

            print(f"[saved] {saved_path}")
            if assembled["missing_sections"]:
                print(f"[warn] missing sections: {assembled['missing_sections']}")


if __name__ == "__main__":
    main()