# LLM-conference-community

# ICLR OpenReview Ingestion

`ICLR_general_ingest.py` downloads and stores ICLR conference data from OpenReview.  
Given an ICLR year, it resolves the corresponding OpenReview venue, finds the main submission invitation, fetches paper submissions, retrieves all notes attached to each paper forum, and stores the data locally.

The script is used to build a structured dataset of real ICLR papers, reviews, decisions, comments, and PDF text for later analysis.

## What it fetches

For each paper, the script collects:

- paper ID and forum ID
- submission number
- title and abstract
- PDF URL
- creation and modification timestamps
- raw OpenReview JSON content

For each paper forum, it also fetches related OpenReview notes, such as:

- official reviews
- meta-reviews
- decisions
- author responses or rebuttals
- public comments and other replies

The script normalizes note types based on OpenReview invitation names.

## Outputs

By default, data is written under `out_general/`.

Example for ICLR 2019:

```text
out_general/iclr2019_raw/papers.jsonl
out_general/iclr2019_raw/notes.jsonl
out_general/iclr2019_raw/submission_invitation.txt
out_general/iclr2019.sqlite
