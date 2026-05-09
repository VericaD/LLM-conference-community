#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Optional

import chromadb
import ollama


EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_iclr"
COLLECTION = "iclr_papers"


def get_collection(chroma_dir: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)


def embed_query(query: str, model: str) -> list[float]:
    response = ollama.embed(model=model, input=[query])
    return response["embeddings"][0]


def build_where_clause(
    section_names: Optional[list[str]] = None,
    paper_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> Optional[dict]:
    clauses = []

    if section_names:
        if len(section_names) == 1:
            clauses.append({"section_name": section_names[0]})
        else:
            clauses.append({"$or": [{"section_name": name} for name in section_names]})

    if paper_id:
        clauses.append({"paper_id": paper_id})

    if chunk_type:
        clauses.append({"chunk_type": chunk_type})

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    embed_model: str,
    n_results: int = 5,
    section_names: Optional[list[str]] = None,
    paper_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
) -> list[dict]:
    query_vec = embed_query(query, embed_model)

    where = build_where_clause(
        section_names=section_names,
        paper_id=paper_id,
        chunk_type=chunk_type,
    )

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    output = []
    for doc, meta, dist in zip(docs, metas, dists):
        output.append(
            {
                "score": 1 - dist,
                "document": doc,
                "metadata": meta,
            }
        )
    return output


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, item in enumerate(chunks, start=1):
        meta = item["metadata"]
        parts.append(
            f"[Chunk {i} | score={item['score']:.3f} | "
            f"paper_id={meta.get('paper_id')} | "
            f"section={meta.get('section_name')}]\n"
            f"{item['document']}"
        )
    return "\n\n" + ("\n\n" + "-" * 80 + "\n\n").join(parts) if parts else ""

def deduplicate_hits_by_text(hits: list[dict], preview_chars: int = 300) -> list[dict]:
    seen = set()
    filtered = []

    for item in hits:
        key = item["document"][:preview_chars].strip()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)

    return filtered


def diversify_results(
    hits: list[dict],
    max_results: int = 5,
    max_per_paper: int = 1,
) -> list[dict]:
    selected = []
    per_paper_counts: dict[str, int] = {}

    for item in hits:
        paper_id = item["metadata"].get("paper_id")
        count = per_paper_counts.get(paper_id, 0)

        if count >= max_per_paper:
            continue

        selected.append(item)
        per_paper_counts[paper_id] = count + 1

        if len(selected) >= max_results:
            break

    return selected

def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve chunks from ChromaDB with optional section filters")
    ap.add_argument("--query", required=True, help="User query / generation query")
    ap.add_argument("--chroma-dir", default=CHROMA_DIR)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Optional section filter, e.g. --sections introduction method experiments",
    )
    ap.add_argument("--paper-id", default=None, help="Optional filter to a single paper")
    ap.add_argument("--chunk-type", default=None, help="Optional chunk_type filter")
    ap.add_argument("--max-per-paper", type=int, default=1,
                help="Maximum number of returned chunks per paper")
    args = ap.parse_args()

    collection = get_collection(args.chroma_dir, args.collection)

    hits = retrieve_chunks(
        collection=collection,
        query=args.query,
        embed_model=args.embed_model,
        n_results=max(args.top_k * 4, 20),
        section_names=args.sections,
        paper_id=args.paper_id,
        chunk_type=args.chunk_type,
    )


    raw_hits = deduplicate_hits_by_text(hits)

    hits = diversify_results(
        raw_hits,
        max_results=args.top_k,
        max_per_paper=args.max_per_paper,
    )


    print(f"\nQuery: {args.query}")
    if args.sections:
        print(f"Sections: {args.sections}")
    if args.paper_id:
        print(f"Paper ID: {args.paper_id}")
    if args.chunk_type:
        print(f"Chunk type: {args.chunk_type}")

    print(f"\nTop {len(hits)} results:\n")
    for i, item in enumerate(hits, start=1):
        meta = item["metadata"]
        print(f"#{i} score={item['score']:.3f}")
        print(f"   paper_id: {meta.get('paper_id')}")
        print(f"   title   : {meta.get('title')}")
        print(f"   section : {meta.get('section_name')}")
        print(f"   type    : {meta.get('chunk_type')}")
        print(f"   preview : {item['document'][:250].replace(chr(10), ' ')}")
        print()

    print("Context block:")
    print(format_context(hits))


if __name__ == "__main__":
    main()