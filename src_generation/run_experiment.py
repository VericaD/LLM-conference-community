#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 80 + "\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full synthetic paper generation experiment")
    ap.add_argument("--topics-file", required=True, help="Text file with one topic per line")
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Ollama models to use for section generation",
    )
    ap.add_argument(
        "--sections",
        nargs="+",
        default=["introduction", "method", "experiments", "conclusion"],
        help="Sections to generate",
    )
    ap.add_argument("--idea-model", default="qwen2.5:7b", help="Model used to generate frozen ideas")
    ap.add_argument("--frozen-ideas-dir", default="./frozen_ideas", help="Directory for frozen idea JSON files")
    ap.add_argument("--rag-runs-dir", default="./rag_runs", help="Directory for generated section JSON files")
    ap.add_argument("--assembled-dir", default="./assembled_papers", help="Directory for assembled full paper JSON files")
    ap.add_argument("--n-titles", type=int, default=5, help="Number of candidate titles per idea")
    ap.add_argument("--title-index", type=int, default=1, help="Selected title index for frozen idea generation")
    ap.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    ap.add_argument("--top-k", type=int, default=4, help="Final number of retrieved chunks")
    ap.add_argument("--max-per-paper", type=int, default=1, help="Maximum retrieved chunks per paper")
    ap.add_argument("--limit", type=int, default=None, help="Maximum number of topics / ideas to process")
    ap.add_argument("--start-index", type=int, default=1, help="Starting frozen idea index")
    ap.add_argument("--selected-direction", default=None, help="Optional fixed broad direction for frozen idea generation")
    ap.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent

    generate_frozen_ideas_py = str(this_dir / "generate_frozen_ideas.py")
    generate_sections_py = str(this_dir / "generate_sections_from_ideas.py")
    assemble_papers_py = str(this_dir / "assemble_papers.py")

    # Step 1: frozen ideas
    cmd1 = [
        sys.executable,
        generate_frozen_ideas_py,
        "--topics-file", args.topics_file,
        "--model", args.idea_model,
        "--output-dir", args.frozen_ideas_dir,
        "--n-titles", str(args.n_titles),
        "--title-index", str(args.title_index),
        "--temperature", str(args.temperature),
        "--start-index", str(args.start_index),
    ]
    if args.limit is not None:
        cmd1 += ["--limit", str(args.limit)]
    if args.selected_direction:
        cmd1 += ["--selected-direction", args.selected_direction]
    if args.skip_existing:
        cmd1 += ["--skip-existing"]

    run_cmd(cmd1)

    # Step 2: section generation
    cmd2 = [
        sys.executable,
        generate_sections_py,
        "--idea-dir", args.frozen_ideas_dir,
        "--models", *args.models,
        "--sections", *args.sections,
        "--output-dir", args.rag_runs_dir,
        "--top-k", str(args.top_k),
        "--max-per-paper", str(args.max_per_paper),
        "--temperature", str(args.temperature),
    ]
    if args.limit is not None:
        cmd2 += ["--limit", str(args.limit)]
    if args.skip_existing:
        cmd2 += ["--skip-existing"]

    run_cmd(cmd2)

    # Step 3: assembly
    safe_model_names = []
    for model in args.models:
        safe_model = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in model)
        safe_model_names.append(safe_model)

    cmd3 = [
        sys.executable,
        assemble_papers_py,
        "--rag-runs-dir", args.rag_runs_dir,
        "--output-dir", args.assembled_dir,
        "--models", *safe_model_names,
        "--sections", *args.sections,
    ]
    if args.skip_existing:
        cmd3 += ["--skip-existing"]

    run_cmd(cmd3)

    print("\nExperiment complete.")
    print(f"Frozen ideas:     {args.frozen_ideas_dir}")
    print(f"Section outputs:  {args.rag_runs_dir}")
    print(f"Assembled papers: {args.assembled_dir}")


if __name__ == "__main__":
    main()