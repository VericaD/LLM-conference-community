#!/usr/bin/env python3
"""
Build a ChromaDB retrieval index from ICLR papers stored in SQLite

The script loads paper abstracts and extracted PDF text, splits them into
section-aware chunks, generates embeddings with Ollama, stores them in
ChromaDB, and runs a retrieval sanity check
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from typing import Generator, Optional

import chromadb
import ollama

# ---------------------------
# Config
# ---------------------------

EMBED_MODEL = "nomic-embed-text"   
CHROMA_DIR = "./chroma_iclr"      
COLLECTION = "iclr_papers"

# Chunking settings      
PDF_CHUNK_SIZE = 1200         
PDF_CHUNK_OVERLAP = 200         


# ---------------------------
# Data classes
# ---------------------------

@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    forum: str
    title: str
    chunk_type: str
    index: int
    text: str
    section_name: Optional[str] = None
    section_index: Optional[int] = None

@dataclass
class Section:
    name: str
    index: int
    text: str

# ---------------------------
# SQLite helpers
# ---------------------------

def load_sample(db_path: str, n: int) -> list[dict]:
    """
    Pull a small sample of papers that have at least an abstract.
    Prefers papers that also have full PDF text extracted.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            p.paper_id,
            p.forum,
            p.title,
            p.abstract,
            t.text  AS pdf_text,
            t.status AS pdf_status
        FROM papers p
        LEFT JOIN paper_pdf_text t ON t.paper_id = p.paper_id
        WHERE p.abstract IS NOT NULL
          AND p.abstract <> ''
        ORDER BY
            CASE WHEN t.status = 'ok' THEN 0 ELSE 1 END,
            p.paper_id
        LIMIT ?
        """,
        (n,),
    )

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

# ---------------------------
# Chunking
# ---------------------------

SECTION_PATTERNS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "models",
    "experimental setup",
    "experiments",
    "results",
    "evaluation",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "appendix",
]

SECTION_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*)?\s*("
    + "|".join(re.escape(p) for p in SECTION_PATTERNS)
    + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_section_name(name: str) -> str:
    name = name.strip().lower()
    aliases = {
        "methods": "method",
        "methodology": "method",
        "approach": "method",
        "model": "method",
        "models": "method",
        "experimental setup": "experiments",
        "results": "experiments",
        "evaluation": "experiments",
        "conclusions": "conclusion",
        "future work": "conclusion",
        "preliminaries": "background",
    }
    return aliases.get(name, name)


def clean_pdf_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_into_sections(pdf_text: str) -> list[Section]:
    text = clean_pdf_text(pdf_text)
    matches = list(SECTION_RE.finditer(text))

    if not matches:
        return [Section(name="body", index=0, text=text)]

    sections: list[Section] = []

    first_start = matches[0].start()
    if first_start > 0:
        preface = text[:first_start].strip()
        if preface:
            sections.append(Section(name="front_matter", index=0, text=preface))

    for i, match in enumerate(matches):
        raw_name = match.group(1)
        section_name = normalize_section_name(raw_name)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()
        if section_text:
            sections.append(
                Section(
                    name=section_name,
                    index=len(sections),
                    text=section_text,
                )
            )

    return sections


def split_long_paragraph(para: str, max_chars: int) -> list[str]:
    para = para.strip()
    if len(para) <= max_chars:
        return [para] if para else []

    pieces = []
    start = 0
    while start < len(para):
        end = min(start + max_chars, len(para))
        if end < len(para):
            split_at = para.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        piece = para[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces


def chunk_section_by_paragraphs(
    text: str,
    max_chars: int = PDF_CHUNK_SIZE,
    overlap_paragraphs: int = 1,
) -> list[str]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    paragraphs: list[str] = []
    for para in raw_paragraphs:
        paragraphs.extend(split_long_paragraph(para, max_chars))

    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        extra = len(para) + (2 if current else 0)

        if current and current_len + extra > max_chars:
            chunks.append("\n\n".join(current))

            overlap = current[-overlap_paragraphs:] if overlap_paragraphs > 0 else []
            current = overlap[:]
            current_len = sum(len(p) for p in current) + max(0, 2 * (len(current) - 1))

        current.append(para)
        current_len += len(para) + (2 if len(current) > 1 else 0)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_text_sliding(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_paper(paper: dict) -> Generator[Chunk, None, None]:
    pid = paper["paper_id"]
    forum = paper["forum"]
    title = paper["title"] or ""

    abstract = (paper["abstract"] or "").strip()
    if abstract:
        yield Chunk(
            chunk_id=f"{pid}__abstract__0",
            paper_id=pid,
            forum=forum,
            title=title,
            chunk_type="abstract",
            index=0,
            text=f"Title: {title}\n\nAbstract: {abstract}",
            section_name="abstract",
            section_index=0,
        )

    pdf_text = (paper.get("pdf_text") or "").strip()
    if not pdf_text or paper.get("pdf_status") != "ok":
        return

    pdf_text = clean_pdf_text(pdf_text)
    sections = split_into_sections(pdf_text)

    global_idx = 1 if abstract else 0

    if len(sections) == 1 and sections[0].name == "body":
        fallback_chunks = chunk_text_sliding(
            sections[0].text,
            chunk_size=PDF_CHUNK_SIZE,
            overlap=PDF_CHUNK_OVERLAP,
        )

        for local_idx, chunk_text in enumerate(fallback_chunks):
            yield Chunk(
                chunk_id=f"{pid}__pdf_body__{local_idx}",
                paper_id=pid,
                forum=forum,
                title=title,
                chunk_type="pdf_body",
                index=global_idx,
                text=chunk_text,
                section_name="body",
                section_index=local_idx,
            )
            global_idx += 1
        return

    for section in sections:
        if section.name in {"abstract", "references", "appendix", "front_matter"}:
            continue

        section_chunks = chunk_section_by_paragraphs(
            section.text,
            max_chars=PDF_CHUNK_SIZE,
            overlap_paragraphs=1,
        )

        for local_idx, section_chunk in enumerate(section_chunks):
            chunk_text = (
                f"Title: {title}\n"
                f"Section: {section.name}\n\n"
                f"{section_chunk}"
            )

            yield Chunk(
                chunk_id=f"{pid}__sec{section.index}__{section.name}__{local_idx}",
                paper_id=pid,
                forum=forum,
                title=title,
                chunk_type="pdf_section",
                index=global_idx,
                text=chunk_text,
                section_name=section.name,
                section_index=local_idx,
            )
            global_idx += 1

# ---------------------------
# Embedding
# ---------------------------

def embed_chunks(chunks: list[Chunk], batch_size: int = 32) -> list[list[float]]:
    """
    Call the local Ollama embedding model in batches.
    Returns a list of embedding vectors (one per chunk, same order).
    """
    all_embeddings: list[list[float]] = []

    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]

        print(f"  embedding chunks {i+1}–{min(i+batch_size, total)} / {total}...")

        response = ollama.embed(model=EMBED_MODEL, input=texts)
        all_embeddings.extend(response["embeddings"])

    return all_embeddings


# ---------------------------
# ChromaDB storage
# ---------------------------

def get_or_create_collection(
    chroma_dir: str,
    collection_name: str,
    reset: bool = False,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[reset] deleted existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for retrieval
    )
    return collection


def store_chunks(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    """
    Upsert all chunks + their embeddings + metadata into ChromaDB.
    ChromaDB upsert is idempotent on chunk_id — safe to re-run.
    """
    ids        = [c.chunk_id  for c in chunks]
    documents  = [c.text      for c in chunks]
    metadatas = [
        {
            "paper_id": c.paper_id,
            "forum": c.forum,
            "title": c.title,
            "chunk_type": c.chunk_type,
            "index": c.index,
            "section_name": c.section_name,
            "section_index": c.section_index,
        }
        for c in chunks
    ]

    batch_size = 500
    total = len(chunks)
    for i in range(0, total, batch_size):
        collection.upsert(
            ids        = ids[i : i + batch_size],
            documents  = documents[i : i + batch_size],
            embeddings = embeddings[i : i + batch_size],
            metadatas  = metadatas[i : i + batch_size],
        )
        print(f"  stored {min(i + batch_size, total)} / {total} chunks in ChromaDB")


# ---------------------------
# Quick sanity check
# ---------------------------

def sanity_check(collection: chromadb.Collection, query: str = "attention mechanism") -> None:
   
    print(f"\n[sanity check] query: '{query}'")

    response  = ollama.embed(model=EMBED_MODEL, input=[query])
    query_vec = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    for rank, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        score = 1 - dist   # cosine distance -> similarity
        print(f"\n  #{rank}  score={score:.3f}  paper_id={meta['paper_id']}")
        print(f"       title: {meta['title'][:80]}")
        print(
            f"       type:  {meta['chunk_type']}  "
            f"section={meta.get('section_name')}  "
            f"index={meta['index']}")
        print(f"       text preview: {doc[:120].replace(chr(10), ' ')}...")


# ---------------------------
# Stats
# ---------------------------

def print_stats(papers: list[dict], chunks: list[Chunk]) -> None:
    has_pdf = sum(1 for p in papers if p.get("pdf_status") == "ok")
    abstract_chunks = sum(1 for c in chunks if c.chunk_type == "abstract")
    body_chunks = sum(1 for c in chunks if c.chunk_type in {"pdf_body", "pdf_section"})

    by_section: dict[str, int] = {}
    for c in chunks:
        if c.section_name:
            by_section[c.section_name] = by_section.get(c.section_name, 0) + 1

    print("\n── indexing stats ──────────────────────")
    print(f"  papers loaded       : {len(papers)}")
    print(f"  papers with PDF text: {has_pdf}")
    print(f"  abstract chunks     : {abstract_chunks}")
    print(f"  pdf chunks          : {body_chunks}")
    print(f"  total chunks        : {len(chunks)}")

    if by_section:
        print("  chunks by section   :")
        for name, count in sorted(by_section.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {name}: {count}")

    print("────────────────────────────────────────\n")

# ---------------------------
# Main
# ---------------------------

def main() -> None:

    global EMBED_MODEL

    ap = argparse.ArgumentParser(description="RAG indexing layer for ICLR papers")

    ap.add_argument("--db",     required=True, help="Path to iclr2019.sqlite")
    ap.add_argument("--sample", type=int, default=20, help="Number of papers to index")
    ap.add_argument("--reset",  action="store_true", help="Delete the existing ChromaDB collection before indexing")
    
    ap.add_argument("--chroma-dir", default=CHROMA_DIR, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default=COLLECTION, help="ChromaDB collection name")
    ap.add_argument("--embed-model", default=EMBED_MODEL, help="Ollama embedding model")

    ap.add_argument("--sanity-query", default="attention mechanism", help="Query string for the sanity check at the end")
    
    args = ap.parse_args()

    EMBED_MODEL = args.embed_model

    print(f"[1/5] loading {args.sample} papers from {args.db} ...")
    papers = load_sample(args.db, args.sample)

    print("[2/5] chunking papers ...")
    chunks = [chunk for paper in papers for chunk in chunk_paper(paper)]
    print_stats(papers, chunks)

    print("[3/5] embedding chunks with Ollama ...")
    embeddings = embed_chunks(chunks)
    print(f"      got {len(embeddings)} embeddings "
          f"(dim={len(embeddings[0]) if embeddings else 'n/a'})")

    print(f"[4/5] storing in ChromaDB at '{args.chroma_dir}' ...")
    collection = get_or_create_collection(args.chroma_dir, args.collection, reset=args.reset)
    store_chunks(collection, chunks, embeddings)
    print(f"      collection now has {collection.count()} total chunks")

    print("[5/5] running sanity check ...")
    sanity_check(collection, query=args.sanity_query)

    print("\n✓ indexing complete.")
    print(f"  ChromaDB collection '{args.collection}' is ready at '{args.chroma_dir}'")

if __name__ == "__main__":
    main()