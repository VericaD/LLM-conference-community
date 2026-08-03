#!/usr/bin/env python3
"""
Build a ChromaDB retrieval index from ingested ICLR reviews.

The script reads official review text and metadata from an ICLR SQLite database,
filters and truncates the reviews, generates embeddings with Ollama, and stores
the resulting retrieval index in `chroma_iclr_reviews/`.
"""
from __future__ import annotations

import argparse
import sqlite3
from typing import Any

import chromadb
import ollama

# ---------------------------
# Config
# ---------------------------

EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_iclr_reviews"
COLLECTION = "iclr_reviews"

MAX_REVIEW_CHARS = 6000

# ---------------------------
# Review loading
# ---------------------------

def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    last_period = cut.rfind(".")

    if last_period > max_chars * 0.7:
        return cut[: last_period + 1]

    return cut


def fetch_reviews(db_path: str, min_chars: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                review_id,
                forum,
                paper_id,
                rating,
                confidence,
                summary,
                strengths,
                weaknesses,
                questions,
                review_text
            FROM reviews
            """
        ).fetchall()
    finally:
        conn.close()

    reviews = []
    n_truncated = 0

    for r in rows:
        (
            review_id,
            forum,
            paper_id,
            rating,
            confidence,
            summary,
            strengths,
            weaknesses,
            questions,
            review_text,
        ) = r

        parts = [
            ("Summary", summary),
            ("Strengths", strengths),
            ("Weaknesses", weaknesses),
            ("Questions", questions),
            ("Review", review_text),
        ]

        text = "\n\n".join(
            f"{label}:\n{value.strip()}"
            for label, value in parts
            if value is not None and value.strip()
        ).strip()

        if len(text) < min_chars:
            continue

        if len(text) > MAX_REVIEW_CHARS:
            n_truncated += 1
            text = truncate_text(text, MAX_REVIEW_CHARS)

        reviews.append(
            {
                "note_id": review_id,
                "forum": forum,
                "paper_id": paper_id,
                "text": text,
                "rating": rating,
                "confidence": confidence,
            }
        )

    print(f"Truncated {n_truncated} reviews longer than {MAX_REVIEW_CHARS} chars")

    return reviews

# ---------------------------
# Embedding
# ---------------------------

def embed_texts(texts: list[str], model: str, batch_size: int = 32) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = ollama.embed(model=model, input=batch)
        embeddings.extend(response["embeddings"])
    return embeddings

# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build a ChromaDB retrieval index from ICLR reviews")

    ap.add_argument("--db", required=True, help="Path to the ingested ICLR SQLite database")
    ap.add_argument("--chroma-dir", default=CHROMA_DIR, help="Directory where the ChromaDB index is stored")
    ap.add_argument("--collection", default=COLLECTION, help="Name of the ChromaDB review collection")
    ap.add_argument("--embed-model", default=EMBED_MODEL, help="Ollama model used to generate review embeddings")
    ap.add_argument("--min-chars", type=int, default=200, help="Minimum review length in characters")
    ap.add_argument("--batch-size", type=int, default=32, help="Number of reviews embedded per batch")
    args = ap.parse_args()

    reviews = fetch_reviews(args.db, args.min_chars)
    print(f"Loaded {len(reviews)} review texts from {args.db}")
    if not reviews:
        return

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_or_create_collection(args.collection)

    ids = [r["note_id"] for r in reviews]
    texts = [r["text"] for r in reviews]
    metadatas = [
        {
            "forum": r["forum"] or "",
            "paper_id": r["paper_id"] or "",
            "rating": r["rating"] if r["rating"] is not None else -1.0,
            "confidence": r["confidence"] if r["confidence"] is not None else -1.0,
        }
        for r in reviews
    ]

    embeddings = embed_texts(texts, args.embed_model, args.batch_size)

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(reviews)} reviews into collection '{args.collection}' at {args.chroma_dir}")


if __name__ == "__main__":
    main()