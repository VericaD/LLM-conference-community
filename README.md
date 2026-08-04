# Can LLMs Dream of a Research Community ?
*Generation, Peer Review, and Iteration Compared Against ICLR 2019*

LLM agents generate ICLR-style papers, review them, and an area chair decides acceptance. Only scripts and notebooks are in this repo. All data below was generated on the university GPU cluster and the analysis notebooks were run locally.

## Files and outputs

| Script | Writes to |
|---|---|
| `src/iclr_general_ingest.py` | `out_general/iclr2019_raw/*.jsonl`, `out_general/iclr2019.sqlite` |
| `src_generation/embedding_index.py` | `chroma_iclr/` (collection `iclr_papers`) |
| `src_review/embedding_index_review.py` | `chroma_iclr_reviews/` (collection `iclr_reviews`) |
| `src_generation/generate_frozen_ideas.py` | `frozen_ideas/` (+ `topics.txt`) |
| `src_generation/generate_sections_from_ideas.py` (uses `generate.py`) | `rag_runs/<model>/idea_XXX__<section>.json` |
| `src_generation/assemble_papers.py` | `assembled_papers/<model>/<idea_id>.json` |
| `src_review/review.py` | `review_outputs/<model>/<idea_id>.review_pipeline.json` |
| `next_iteration.py` | `frozen_ideas_iter2/topics.txt` |


## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_full.txt
```

## Models
- llama3.1:8b
- qwen2.5:7b
- qwen3:8b
- qwen3:14b
- qwen3.5:9b
- phi4-reasoning:14b
- Apertus-8B-Instruct-2509

## Requirements

Python 3.10+, and Ollama running (`ollama serve`).

Locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_full.txt
```

On the cluster (conda):

```bash
conda env create -f environment.yml && conda activate thesis
```

Pull the models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b qwen2.5:7b qwen3:8b qwen3:14b qwen3.5:9b phi4-reasoning:14b
```

`hf:swiss-ai/Apertus-8B-Instruct-2509` downloads automatically via `transformers` (needs a
Hugging Face login)



## OpenReview credentials

Needed for the ingest step only:

```bash
export OPENREVIEW_USERNAME="your@email"
export OPENREVIEW_PASSWORD="..."
```

## Run in this order

```bash
# 1. Download real ICLR papers, reviews and PDF text into SQLite 
python src/iclr_general_ingest.py --year 2019 --fetch-pdf-text --stats

# 2. Index real paper sections for retrieval during generation
python src_generation/embedding_index.py --db out_general/iclr2019.sqlite --reset

# 3. Index real reviews for retrieval during reviewing
python src_review/embedding_index_review.py --db out_general/iclr2019.sqlite

# 4. Create the fixed research ideas shared by all generator models
python src_generation/generate_frozen_ideas.py --db out_general/iclr2019.sqlite --output-dir ./frozen_ideas \
--model qwen3:14b --n-titles 7 --skip-existing

# 5.a Generate all paper sections with Ollama models
python src_generation/generate_sections_from_ideas.py --idea-dir ./frozen_ideas --all-sections \
--models llama3.1:8b qwen2.5:7b qwen3:8b qwen3:14b qwen3.5:9b phi4-reasoning:14b \
--output-dir ./rag_runs --assignment-seed 42 --ideas-per-model 200 --skip-existing

# 5.b Generate all paper sections with Apertus 
python src_generation/generate_sections_from_ideas.py --idea-dir ./frozen_ideas --all-sections \
--models hf:swiss-ai/Apertus-8B-Instruct-2509 --output-dir ./rag_runs
--ideas-per-model 200 --assignment-seed 42 --assignment-offset 1200 --skip-existing

# 6. Merge sections into complete papers
python src_generation/assemble_papers.py --rag-runs-dir ./rag_runs --output-dir ./assembled_papers --skip-existing

# 7. Review one generator's papers with 3 reviewers + area chair
python src_review/review.py --input-dir assembled_papers/qwen3_14b \
  --output-dir review_outputs/qwen3_14b \
  --reviewer-models llama3.1:8b qwen3:14b phi4-reasoning:14b \
  --area-chair-model phi4-reasoning:14b --skip-existing

# 8.a Collect the topics of all accepted papers into a topic list for the next iteration
python next_iteration.py --review-outputs-dir review_outputs \
  --output-topics-file frozen_ideas_iter2/topics.txt --topic-field topic \
  --assembled-papers-base-dir ./assembled_papers

# 8.b Generate new ideas from those topics, then repeat steps 5-7 with the iter2 directories
python src_generation/generate_frozen_ideas.py --db out_general/iclr2019.sqlite \
  --output-dir ./frozen_ideas_iter2 --model qwen3:14b --n-titles 7 --skip-existing
```

Keep `--assignment-seed 42` and the same reviewer/area-chair models for every generator, otherwise
the results are not comparable.

Second iteration: `next_iteration.py` collects the topics of accepted papers into a new
`topics.txt`, which `generate_frozen_ideas.py` reuses as seeds; steps 5–7 are then repeated.

## Analysis notebooks

Edit the path variable in the first cell of each notebook before running.

| Notebook | Reads |
|---|---|
| `iclr_dataset_analysis.ipynb` | `out_general/iclr2019.sqlite` (real ICLR baseline) |
| `generation_pipeline_analysis.ipynb` | `assembled_papers/` (generated paper statistics) |
| `review_analysis_pipeline.ipynb` | `review_outputs/` (ratings, agreement, decisions) |
| `iteration_analysis.ipynb`  |	review_outputs/ (topic narrowing and idea redundancy between iterations) |

`generation_pipeline_analysis_iter2.ipynb`, `review_analysis_pipeline.ipynb_iter2` and `iteration_analysis_iter2` are analysis notebooks used for the papers generated in the second generation

