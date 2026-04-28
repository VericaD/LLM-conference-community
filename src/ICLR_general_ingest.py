#!/usr/bin/env python3
"""
Generic ICLR OpenReview Ingest (API2) -> JSONL + SQLite

What it does:
- Resolves the correct conference venue and submission invitation
- Downloads submissions (papers)
- Downloads all notes in each paper forum (reviews/decisions/comments/etc.)
- Stores raw JSON + normalized key-value fields
- Optionally downloads PDFs and extracts plain text

Designed to work across multiple ICLR years, especially 2017/2018/2019/2024+.

Refs:
- OpenReview API2 client:
  https://docs.openreview.net/getting-started/using-the-api/installing-and-instantiating-the-python-client
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import openreview
import requests
import fitz  # PyMuPDF

DEFAULT_BASEURL = "https://api2.openreview.net"


# ---------------------------
# Config
# ---------------------------

@dataclass
class VenueConfig:
    venue_id: str
    venue_name: str
    year: int
    submission_invitation: Optional[str] = None
    submission_invitation_candidates: List[str] = field(default_factory=lambda: [
        "Post_Submission",
        "Blind_Submission",
        "Submission",
        "Paper",
    ])
    excluded_invitation_keywords: List[str] = field(default_factory=lambda: [
        "desk_reject",
        "desk_rejected",
        "withdraw",
        "withdrawn",
    ])


def build_iclr_config(year: int, venue_id_override: Optional[str] = None) -> VenueConfig:
    """
    Build a best-effort config for ICLR conference years.

    Notes:
    - 2018/2019 generally follow ICLR.cc/<year>/Conference
    - 2024 does too
    - 2017 may need manual override depending on actual OpenReview group id
    """
    if venue_id_override:
        return VenueConfig(
            venue_id=venue_id_override,
            venue_name="ICLR",
            year=year,
        )

    if year == 2017:
        # This is the most likely pattern, but if it differs in practice,
        # pass --venue-id explicitly when running.
        return VenueConfig(
            venue_id="ICLR.cc/2017/Conference",
            venue_name="ICLR",
            year=year,
            submission_invitation_candidates=[
                "Submission",
                "Blind_Submission",
                "Post_Submission",
                "Paper",
            ],
        )

    if year == 2018:
        return VenueConfig(
            venue_id="ICLR.cc/2018/Conference",
            venue_name="ICLR",
            year=year,
            submission_invitation_candidates=[
                "Blind_Submission",
                "Submission",
                "Post_Submission",
                "Paper",
            ],
        )

    if year == 2019:
        return VenueConfig(
            venue_id="ICLR.cc/2019/Conference",
            venue_name="ICLR",
            year=year,
            submission_invitation_candidates=[
                "Blind_Submission",
                "Submission",
                "Post_Submission",
                "Paper",
            ],
        )

    return VenueConfig(
        venue_id=f"ICLR.cc/{year}/Conference",
        venue_name="ICLR",
        year=year,
    )


# ---------------------------
# Utilities
# ---------------------------

def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_key(k: str) -> str:
    k = k.strip().lower()
    k = re.sub(r"[^\w\s-]", "", k)
    k = re.sub(r"[\s-]+", "_", k)
    return k


_num_prefix = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)")


def parse_numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _num_prefix.match(value)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def unwrap_api2_content_value(v: Any) -> Any:
    if isinstance(v, dict) and "value" in v and len(v) >= 1:
        return v["value"]
    return v


def detect_value_type(v: Any) -> str:
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, (dict, list)):
        return "json"
    return "text"


def infer_note_type(invitation: str, content: Any, replyto: Any) -> str:
    inv = (invitation or "").lower()

    if "official_review" in inv or inv.endswith("/-/review"):
        return "official_review"
    if "meta_review" in inv:
        return "meta_review"
    if "decision" in inv or "recommendation" in inv:
        return "decision"
    if "rebuttal" in inv or "author_response" in inv or "response" in inv:
        return "author_response"
    if "public_comment" in inv:
        return "public_comment"
    if "comment" in inv:
        return "comment"
    if "withdraw" in inv:
        return "withdrawal"
    if replyto is None:
        return "submission"
    return "reply"


def get_note_attr(note_obj: Any, key: str, default=None):
    if isinstance(note_obj, dict):
        return note_obj.get(key, default)

    if hasattr(note_obj, key):
        v = getattr(note_obj, key)
        return v if v is not None else default

    d = getattr(note_obj, "__dict__", None)
    if isinstance(d, dict) and key in d:
        return d[key] if d[key] is not None else default

    return default


def pick_best_invitation(invitations: Any) -> str:
    if not isinstance(invitations, list) or not invitations:
        return "UNKNOWN_INVITATION"

    preferred_keywords = [
        "official_review",
        "meta_review",
        "decision",
        "public_comment",
        "comment",
        "rebuttal",
        "author_response",
        "response",
        "submission",
        "paper",
    ]

    lower = [(inv, inv.lower()) for inv in invitations]

    for kw in preferred_keywords:
        for inv, inv_l in lower:
            if kw in inv_l:
                return inv

    return max(invitations, key=len)


def resolve_pdf_url(pdf_url: str, openreview_host: str = "https://openreview.net") -> str:
    if not pdf_url:
        raise ValueError("Empty pdf_url")

    if pdf_url.startswith("http://") or pdf_url.startswith("https://"):
        return pdf_url

    if pdf_url.startswith("/"):
        return openreview_host.rstrip("/") + pdf_url

    return openreview_host.rstrip("/") + "/" + pdf_url


def download_pdf_bytes(pdf_url: str, timeout_s: int = 60) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://openreview.net/",
    }

    r = requests.get(pdf_url, headers=headers, timeout=timeout_s, allow_redirects=True)
    r.raise_for_status()
    return r.content


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: List[str] = []

    try:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                parts.append(page_text)
    finally:
        doc.close()

    return "\n".join(parts).strip()


def upsert_pdf_text(
    conn: sqlite3.Connection,
    paper_id: str,
    pdf_url: str,
    status: str,
    text: Optional[str],
    error: Optional[str]
) -> None:
    n_chars = len(text) if text else 0

    conn.execute(
        """
        INSERT INTO paper_pdf_text(
          paper_id, pdf_url, retrieved_at, status, n_chars, text, error
        ) VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(paper_id) DO UPDATE SET
          pdf_url=excluded.pdf_url,
          retrieved_at=excluded.retrieved_at,
          status=excluded.status,
          n_chars=excluded.n_chars,
          text=excluded.text,
          error=excluded.error
        """,
        (paper_id, pdf_url, now_ms(), status, n_chars, text, error),
    )


# ---------------------------
# SQLite schema
# ---------------------------

DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS editions (
  edition_id TEXT PRIMARY KEY,
  venue_name TEXT NOT NULL,
  year       INTEGER NOT NULL,
  api_baseurl TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
  paper_id   TEXT PRIMARY KEY,
  edition_id TEXT NOT NULL REFERENCES editions(edition_id),
  forum      TEXT NOT NULL,
  number     INTEGER,
  title      TEXT,
  abstract   TEXT,
  pdf_url    TEXT,
  created_at INTEGER,
  modified_at INTEGER,
  raw_json   TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_forum ON papers(forum);
CREATE INDEX IF NOT EXISTS idx_papers_edition ON papers(edition_id);

CREATE TABLE IF NOT EXISTS notes (
  note_id    TEXT PRIMARY KEY,
  edition_id TEXT NOT NULL REFERENCES editions(edition_id),
  forum      TEXT NOT NULL REFERENCES papers(forum),
  replyto    TEXT,
  invitation TEXT NOT NULL,
  note_type  TEXT,
  created_at INTEGER,
  modified_at INTEGER,
  raw_json   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notes_type ON notes(note_type);
CREATE INDEX IF NOT EXISTS idx_notes_forum ON notes(forum);
CREATE INDEX IF NOT EXISTS idx_notes_invitation ON notes(invitation);
CREATE INDEX IF NOT EXISTS idx_notes_replyto ON notes(replyto);

CREATE TABLE IF NOT EXISTS note_fields (
  note_id    TEXT NOT NULL REFERENCES notes(note_id) ON DELETE CASCADE,
  field_key  TEXT NOT NULL,
  value_type TEXT NOT NULL,
  value_text TEXT,
  value_num  REAL,
  value_json TEXT,
  PRIMARY KEY (note_id, field_key)
);

CREATE INDEX IF NOT EXISTS idx_note_fields_key ON note_fields(field_key);
CREATE INDEX IF NOT EXISTS idx_note_fields_num ON note_fields(field_key, value_num);

CREATE TABLE IF NOT EXISTS invitation_schemas (
  invitation TEXT PRIMARY KEY,
  edition_id TEXT NOT NULL REFERENCES editions(edition_id),
  retrieved_at INTEGER NOT NULL,
  raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_pdf_text (
  paper_id     TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
  pdf_url      TEXT,
  retrieved_at INTEGER NOT NULL,
  status       TEXT NOT NULL,
  n_chars      INTEGER NOT NULL,
  text         TEXT,
  error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_pdf_text_status ON paper_pdf_text(status);
"""


# ---------------------------
# OpenReview fetch
# ---------------------------

def detect_api_version(config: VenueConfig) -> int:
    """
    Detect whether the venue is API2 or API1.

    Strategy:
    - try API2 first
    - if group exists and has a non-empty 'domain', treat as API2
    - otherwise fall back to API1
    """
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Missing credentials. Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD env vars."
        )

    try:
        client_v2 = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=username,
            password=password,
        )
        group = client_v2.get_group(config.venue_id)
        if getattr(group, "domain", None):
            return 2
    except Exception:
        pass

    return 1


def openreview_client(api_version: int) -> "openreview.Client":
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "Missing credentials. Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD env vars."
        )
    
    if api_version == 2:
        return openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=username,
            password=password
        )

    return openreview.Client(
        baseurl="https://api.openreview.net",
        username=username,
        password=password
    )

def get_all_notes_api1(client, **kwargs):
    notes = []
    offset = 0
    limit = 1000

    while True:
        batch = client.get_notes(limit=limit, offset=offset, **kwargs)
        if not batch:
            break
        notes.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    return notes

def get_all_invitations_api1(client, regex: str):
    invitations = []
    offset = 0
    limit = 1000

    while True:
        batch = client.get_invitations(regex=regex, limit=limit, offset=offset)
        if not batch:
            break
        invitations.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    return invitations

def debug_submission_invitations(client, venue_id: str, api_version) -> None:

    if api_version == 2:
        invs = client.get_all_invitations(prefix=f"{venue_id}/-/")
    else:
        invs = get_all_invitations_api1(client, regex=f"{venue_id}/-/.+")

    sub_like = []
    for inv in invs:
        inv_id = inv.id
        suffix = inv_id.split(f"{venue_id}/-/", 1)[-1]
        if "/" in suffix:
            continue  # skip per-paper invitations like Paper123/Meta_Review
        if "submission" in suffix.lower() or suffix.lower() == "paper":
            sub_like.append(inv_id)
    print(f"[debug] found {len(sub_like)} submission-like invitations")
    for inv_id in sorted(sub_like):
        print("  ", inv_id)


def resolve_submission_invitation(client: "openreview.Client", config: VenueConfig, api_version) -> str:

    if config.submission_invitation:
        return config.submission_invitation
    
    if api_version == 2:
        invs = client.get_all_invitations(prefix=f"{config.venue_id}/-/")
    else:
        invs = get_all_invitations_api1(client, regex=f"{config.venue_id}/-/.+")

    inv_ids = [inv.id for inv in invs]

    for suffix in config.submission_invitation_candidates:
        full_suffix = f"/-/{suffix}"
        for inv_id in inv_ids:
            if inv_id.endswith(full_suffix):
                return inv_id

    blacklist = [
        "desk_reject",
        "desk_rejected",
        "withdraw",
        "withdrawn",
        "revision",
        "camera_ready",
        "decision",
        "review",
        "comment",
        "meta_review",
        "public_comment",
        "ethics",
    ]

    candidates = []
    for inv_id in inv_ids:
        inv_l = inv_id.lower()
        if ("submission" in inv_l) or inv_l.endswith("/-/paper"):
            if not any(b in inv_l for b in blacklist):
                candidates.append(inv_id)

    if candidates:
        return min(candidates, key=len)

    raise RuntimeError(
        f"Could not find a main submission invitation under prefix {config.venue_id}/-/. "
        f"Found invitations: {inv_ids}"
    )


def fetch_invitation_schema(client: "openreview.Client", invitation_id: str) -> Dict[str, Any]:
    inv = client.get_invitation(invitation_id)
    if hasattr(inv, "to_json"):
        return inv.to_json()
    return inv.__dict__


def note_to_dict(note: Any) -> Dict[str, Any]:
    if hasattr(note, "to_json"):
        return note.to_json()
    return note.__dict__


def extract_pdf_url(submission_content: Dict[str, Any]) -> Optional[str]:
    pdf = submission_content.get("pdf")  # looks for a pdf field in the paper's content               
    if not pdf:
        return None
    pdf = unwrap_api2_content_value(pdf)
    if isinstance(pdf, str):
        return pdf
    if isinstance(pdf, dict) and "url" in pdf:
        return pdf["url"]
    return None

### -> "/pdf/abs123.pdf"


def fetch_submissions(client: "openreview.Client", submission_inv: str, api_version) -> List[Any]:

    if api_version == 2:
        return client.get_all_notes(invitation=submission_inv)
    else:
        return get_all_notes_api1(client, invitation=submission_inv)


def fetch_forum_notes(client: "openreview.Client", forum_id: str, api_version) -> List[Any]:
    root = None
    try:
        root = client.get_note(forum_id)
    except Exception:
        pass

    try:
        if api_version == 2:
            notes = client.get_all_notes(forum=forum_id)
        else:
            notes = get_all_notes_api1(client, forum=forum_id)

    except Exception:
        notes = []

    if root is not None:
        root_id = get_note_attr(root, "id")
        if all(get_note_attr(n, "id") != root_id for n in notes):
            notes.insert(0, root)

    return notes


def is_excluded_submission_for_forum(note_obj: Any, excluded_keywords: List[str]) -> bool:
    invs = get_note_attr(note_obj, "invitations")
    if isinstance(invs, list) and invs:
        best = pick_best_invitation(invs).lower()
    else:
        best = str(get_note_attr(note_obj, "invitation", "")).lower()
    return any(keyword in best for keyword in excluded_keywords)


# ---------------------------
# JSONL writers
# ---------------------------

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(jdump(r) + "\n")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------
# SQLite load + normalize
# ---------------------------

def connect_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    conn.commit()


def upsert_edition(conn: sqlite3.Connection, edition_id: str, 
                   venue_name: str, year: int, baseurl: str) -> None:
    conn.execute(
        """
        INSERT INTO editions(edition_id, venue_name, year, api_baseurl)
        VALUES(?,?,?,?)
        ON CONFLICT(edition_id) DO UPDATE SET
          venue_name=excluded.venue_name,
          year=excluded.year,
          api_baseurl=excluded.api_baseurl
        """,
        (edition_id, venue_name, year, baseurl),
    )


def insert_paper(conn: sqlite3.Connection, paper: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO papers(
          paper_id, edition_id, forum, number, title, abstract, pdf_url,
          created_at, modified_at, raw_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """,
        (
            paper["paper_id"],
            paper["edition_id"],
            paper["forum"],
            paper.get("number"),
            paper.get("title"),
            paper.get("abstract"),
            paper.get("pdf"),
            paper.get("created_at"),
            paper.get("modified_at"),
            jdump(paper),
        ),
    )


def insert_note(conn: sqlite3.Connection, note: Dict[str, Any]) -> None:
    if not note.get("invitation"):
        note["invitation"] = "UNKNOWN_INVITATION"

    conn.execute(
        """
        INSERT OR REPLACE INTO notes(
          note_id, edition_id, forum, replyto, invitation, note_type, created_at, modified_at, raw_json
        ) VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            note["note_id"],
            note["edition_id"],
            note["forum"],
            note.get("replyto"),
            note["invitation"],
            note.get("note_type"),
            note.get("created_at"),
            note.get("modified_at"),
            jdump(note),
        ),
    )


def normalize_note_fields(conn: sqlite3.Connection, note: Dict[str, Any]) -> None:
    note_id = note["note_id"]
    content = note.get("content") or {}
    if not isinstance(content, dict):
        return

    for raw_key, raw_val in content.items():
        key = normalize_key(str(raw_key))
        unwrapped = unwrap_api2_content_value(raw_val)
        vtype = detect_value_type(unwrapped)

        value_text = None
        value_num = None
        value_json = None

        if vtype == "number":
            value_num = float(unwrapped)
            value_text = str(unwrapped)
        elif vtype == "json":
            value_json = jdump(unwrapped)
        else:
            value_text = str(unwrapped)
            value_num = parse_numeric(value_text)

        conn.execute(
            """
            INSERT OR REPLACE INTO note_fields(
              note_id, field_key, value_type, value_text, value_num, value_json
            ) VALUES(?,?,?,?,?,?)
            """,
            (note_id, key, vtype, value_text, value_num, value_json),
        )


# ---------------------------
# Pipeline steps
# ---------------------------

def step_download(config: VenueConfig, api_version:int, 
                  out_dir: str, max_papers: Optional[int]) -> Tuple[str, str, str]:
    
    client = openreview_client(api_version)

    submission_inv = resolve_submission_invitation(client, config, api_version)

    print(f"[debug] venue API version: v{api_version}")
    print("[debug] venue_id:", config.venue_id)
    print("[debug] chosen submission_inv:", submission_inv)
    debug_submission_invitations(client, config.venue_id, api_version)

    submissions = fetch_submissions(client, submission_inv, api_version)

    if max_papers is not None:
        submissions = submissions[:max_papers]

    papers_rows: List[Dict[str, Any]] = []
    notes_rows: List[Dict[str, Any]] = []

    for s in submissions:
        sd = note_to_dict(s)

        content = sd.get("content") or {}
        title = unwrap_api2_content_value(content.get("title")) if isinstance(content, dict) else None
        abstract = unwrap_api2_content_value(content.get("abstract")) if isinstance(content, dict) else None
        pdf_url = extract_pdf_url(content if isinstance(content, dict) else {})

        paper_row = {
            "edition_id": config.venue_id,
            "paper_id": sd.get("id"),
            "forum": sd.get("forum") or sd.get("id"),
            "number": sd.get("number"),
            "title": title,
            "abstract": abstract,
            "pdf": pdf_url,
            "content": content,
            "created_at": sd.get("cdate") or sd.get("tcdate") or sd.get("pdate"),
            "modified_at": sd.get("mdate") or sd.get("tmdate"),
        }

        if paper_row.get("title") is None:
            continue

        forum_id = paper_row["forum"]
        forum_notes = fetch_forum_notes(client, forum_id, api_version)

        if any(is_excluded_submission_for_forum(n, config.excluded_invitation_keywords) for n in forum_notes):
            continue

        papers_rows.append(paper_row)

        for n in forum_notes:
            note_id = get_note_attr(n, "id")
            forum = get_note_attr(n, "forum")
            replyto = get_note_attr(n, "replyto")
            invs = get_note_attr(n, "invitations")
            if isinstance(invs, list) and invs:
                invitation = pick_best_invitation(invs)
            else:
                invitation = get_note_attr(n, "invitation", "UNKNOWN_INVITATION")
            content = get_note_attr(n, "content")

            note_type = infer_note_type(invitation, content, replyto)

            notes_rows.append(
                {
                    "edition_id": config.venue_id,
                    "note_id": note_id,
                    "forum": forum,
                    "replyto": replyto,
                    "invitation": invitation,
                    "note_type": note_type,
                    "signatures": get_note_attr(n, "signatures"),
                    "writers": get_note_attr(n, "writers"),
                    "readers": get_note_attr(n, "readers"),
                    "content": content,
                    "created_at": get_note_attr(n, "cdate") or get_note_attr(n, "tcdate") or get_note_attr(n, "pdate"),
                    "modified_at": get_note_attr(n, "mdate") or get_note_attr(n, "tmdate"),
                }
            )

    papers_path = os.path.join(out_dir, "papers.jsonl")
    notes_path = os.path.join(out_dir, "notes.jsonl")
    write_jsonl(papers_path, papers_rows)
    write_jsonl(notes_path, notes_rows)

    return submission_inv, papers_path, notes_path


def step_build_sqlite(
    db_path: str,
    config: VenueConfig,
    api_version: int,
    submission_inv: str,
    papers_jsonl: str,
    notes_jsonl: str
) -> None:
    
    conn = connect_db(db_path)
    init_db(conn)
    upsert_edition(
        conn=conn,
        edition_id=config.venue_id,
        venue_name=config.venue_name,
        year=config.year,
        baseurl="https://api2.openreview.net" if api_version == 2 else "https://api.openreview.net"
    )

    try:
        client = openreview_client(api_version)
        inv_schema = fetch_invitation_schema(client, submission_inv)
        conn.execute(
            """
            INSERT OR REPLACE INTO invitation_schemas(invitation, edition_id, retrieved_at, raw_json)
            VALUES(?,?,?,?)
            """,
            (submission_inv, config.venue_id, now_ms(), jdump(inv_schema)),
        )
    except Exception as e:
        print(f"[warn] Could not store invitation schema for {submission_inv}: {e}", file=sys.stderr)

    for p in read_jsonl(papers_jsonl):
        insert_paper(conn, p)

    for n in read_jsonl(notes_jsonl):
        insert_note(conn, n)
        normalize_note_fields(conn, n)

    conn.commit()
    conn.close()


def step_fetch_pdf_text(db_path: str, min_chars_ok: int = 500,
                        limit: Optional[int] = None, skip_existing: bool = True) -> None:
    
    conn = connect_db(db_path)
    cur = conn.cursor()

    query = """
      SELECT p.paper_id, p.pdf_url
      FROM papers p
      WHERE p.pdf_url IS NOT NULL
        AND p.pdf_url <> ''
    """

    if skip_existing:
        query += """
          AND NOT EXISTS (
            SELECT 1
            FROM paper_pdf_text t
            WHERE t.paper_id = p.paper_id
          )
        """

    query += " ORDER BY p.paper_id"

    if limit is not None:
        query += f" LIMIT {int(limit)}"

    cur.execute(query)
    rows = cur.fetchall()

    print(f"[pdf] papers to process: {len(rows)}")

    for i, (paper_id, pdf_url) in enumerate(rows, start=1):
        print(f"[pdf] {i}/{len(rows)} paper_id={paper_id}")

        try:
            full_pdf_url = resolve_pdf_url(pdf_url)
        except Exception as e:
            upsert_pdf_text(conn, paper_id, pdf_url, "download_error", None, str(e))
            conn.commit()
            continue

        try:
            pdf_bytes = download_pdf_bytes(full_pdf_url)
        except Exception as e:
            upsert_pdf_text(conn, paper_id, full_pdf_url, "download_error", None, str(e))
            conn.commit()
            continue

        try:
            text = extract_text_from_pdf_bytes(pdf_bytes)

            if len(text) < min_chars_ok:
                upsert_pdf_text(conn, paper_id, full_pdf_url, "no_text", text, None)
            else:
                upsert_pdf_text(conn, paper_id, full_pdf_url, "ok", text, None)

        except Exception as e:
            upsert_pdf_text(conn, paper_id, full_pdf_url, "parse_error", None, str(e))

        conn.commit()

    conn.close()


def step_quick_stats(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM papers")
    papers = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM notes")
    notes = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM note_fields")
    fields = cur.fetchone()[0]

    print(f"Papers: {papers}")
    print(f"Notes: {notes}")
    print(f"Extracted note_fields: {fields}")

    cur.execute("SELECT COUNT(*) FROM paper_pdf_text")
    pdf_rows = cur.fetchone()[0]
    print(f"PDF text rows: {pdf_rows}")

    cur.execute("""
      SELECT status, COUNT(*)
      FROM paper_pdf_text
      GROUP BY status
      ORDER BY COUNT(*) DESC
    """)
    pdf_status_rows = cur.fetchall()

    if pdf_status_rows:
        print("\nPDF extraction status:")
        for status, count in pdf_status_rows:
            print(f"  {status}: {count}")

    cur.execute("""
      SELECT field_key, COUNT(*) AS c
      FROM note_fields
      GROUP BY field_key
      ORDER BY c DESC
      LIMIT 20
    """)
    print("\nTop field keys:")
    for k, c in cur.fetchall():
        print(f"  {k}: {c}")

    conn.close()


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--year", type=int, default=2024, help="ICLR year")
    ap.add_argument("--venue-id", default=None, help="Override OpenReview venue id manually")
    # ap.add_argument("--baseurl", default=DEFAULT_BASEURL) ### UNUSED

    ap.add_argument("--out-dir", default=None, help="Output raw JSONL directory")
    ap.add_argument("--db", default=None, help="SQLite database path")

    ap.add_argument("--max-papers", type=int, default=None, help="For testing, limit number of papers")
    ap.add_argument("--skip-download", action="store_true", help="Use existing JSONL in out-dir")
    ap.add_argument("--stats", action="store_true", help="Print quick stats at the end")

    ap.add_argument("--fetch-pdf-text", action="store_true",
                    help="Download PDFs and extract plain text into SQLite")
    ap.add_argument("--pdf-limit", type=int, default=None,
                    help="For testing, limit number of PDFs to process")
    ap.add_argument("--pdf-min-chars", type=int, default=500,
                    help="Minimum extracted character count to consider PDF text extraction successful")
    ap.add_argument("--pdf-no-skip-existing", action="store_true",
                    help="Reprocess PDFs even if already stored in paper_pdf_text")

    args = ap.parse_args()

    config = build_iclr_config(year=args.year, venue_id_override=args.venue_id)

    api_version = detect_api_version(config)
    print(f"[debug] detected API version: v{api_version}")

    out_dir = args.out_dir or f"out_general/iclr{args.year}_raw"
    db_path = args.db or f"out_general/iclr{args.year}.sqlite"

    if not args.skip_download:
        submission_inv, papers_jsonl, notes_jsonl = step_download(
            config=config,
            api_version=api_version,
            out_dir=out_dir,
            max_papers=args.max_papers,
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "submission_invitation.txt"), "w", encoding="utf-8") as f:
            f.write(submission_inv + "\n")
    else:
        inv_path = os.path.join(out_dir, "submission_invitation.txt")
        if os.path.exists(inv_path):
            submission_inv = open(inv_path, "r", encoding="utf-8").read().strip()
        else:
            # Fallback only if reusing a previous download without the file
            submission_inv = config.submission_invitation or f"{config.venue_id}/-/Submission"

        papers_jsonl = os.path.join(out_dir, "papers.jsonl")
        notes_jsonl = os.path.join(out_dir, "notes.jsonl")

    step_build_sqlite(
        db_path=db_path,
        config=config,
        api_version=api_version,
        submission_inv=submission_inv,
        papers_jsonl=papers_jsonl,
        notes_jsonl=notes_jsonl,
    )

    if args.fetch_pdf_text:
        step_fetch_pdf_text(
            db_path=db_path,
            min_chars_ok=args.pdf_min_chars,
            limit=args.pdf_limit,
            skip_existing=not args.pdf_no_skip_existing,
        )

    if args.stats:
        step_quick_stats(db_path)


if __name__ == "__main__":
    main()
