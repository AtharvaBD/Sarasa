"""SQLite storage for the job-email pipeline.

Usage (from the project root):
    python -m polling.jobdb init            # create the database
    python -m polling.jobdb list            # show saved jobs, newest first
    python -m polling.jobdb export out.csv  # dump jobs to CSV
    python -m polling.jobdb json            # jobs as JSON (used by the dashboard)
"""
import csv
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import config

DB_PATH = config.path("database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_emails (
    email_id       TEXT PRIMARY KEY,
    thread_id      TEXT,
    classification TEXT NOT NULL,
    processed_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    company        TEXT NOT NULL,
    job_title      TEXT NOT NULL,
    link           TEXT,
    date_published TEXT              -- YYYY-MM-DD
    date_published TEXT,             -- YYYY-MM-DD
    applied_at     TEXT              -- date you marked it applied, or NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_unique
    ON jobs(lower(company), lower(job_title), COALESCE(link, ''));
"""

_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                    "utm_content", "gclid", "fbclid", "mc_cid", "mc_eid", "trk", "src"}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path=DB_PATH):
    with connect(path) as conn:
        # Migrate: drop the old multi-column jobs table if it's still there.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(jobs)")]
        if "dedupe_key" in cols:
            conn.execute("DROP TABLE jobs")
        elif cols and "applied_at" not in cols:      # existing table: add the new column
            conn.execute("ALTER TABLE jobs ADD COLUMN applied_at TEXT")
        conn.executescript(SCHEMA)


def canonical_url(url):
    """Drop tracking params and fragments so the same listing dedupes."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _TRACKING_PARAMS]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path.rstrip("/"), urlencode(query), ""))


def is_processed(email_id, path=DB_PATH):
    with connect(path) as conn:
        row = conn.execute("SELECT 1 FROM processed_emails WHERE email_id = ?", (email_id,)).fetchone()
        return row is not None


def mark_processed(email_id, classification, thread_id=None, path=DB_PATH):
    with connect(path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_emails (email_id, thread_id, classification, processed_at) "
            "VALUES (?, ?, ?, ?)",
            (email_id, thread_id, classification, _now()),
        )


def add_job(company, job_title, link=None, date_published=None, path=DB_PATH):
    """Insert one job. Returns True if added, False if it was a duplicate."""
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO jobs (company, job_title, link, date_published) VALUES (?, ?, ?, ?)",
            (company.strip(), job_title.strip(), canonical_url(link) or None, date_published),
        )
        return cur.rowcount == 1


def list_jobs(path=DB_PATH, with_id=False):
    cols = "rowid AS id, applied_at, " if with_id else ""
    with connect(path) as conn:
        return conn.execute(
            f"SELECT {cols}company, job_title, link, date_published FROM jobs "
            "ORDER BY date_published DESC"
        ).fetchall()

def set_applied(job_id, applied=True, path=DB_PATH):
    """Mark a job as applied (stores today's date) or clear the mark."""
    with connect(path) as conn:
        conn.execute("UPDATE jobs SET applied_at = ? WHERE rowid = ?",
                     (datetime.now().date().isoformat() if applied else None, job_id))

def get_job(job_id, path=DB_PATH):
    with connect(path) as conn:
        return conn.execute(
            "SELECT rowid AS id, company, job_title, link, date_published "
            "FROM jobs WHERE rowid = ?", (job_id,)
        ).fetchone()

def export_csv(out_path, path=DB_PATH):
    rows = list_jobs(path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["company", "job_title", "link", "date_published"])
        for r in rows:
            w.writerow(list(r))
    return len(rows)


def _print(rows):
    if not rows:
        print("(no rows)")
        return
    for r in rows:
        print(f"{r['date_published'] or '-':<10} | {r['company'][:24]:<24} | {r['job_title'][:40]}\n"
              f"           {r['link'] or '(no link)'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "init"
    if cmd == "init":
        init_db()
        print(f"Initialized {DB_PATH}")
    elif cmd == "list":
        init_db()
        _print(list_jobs())
    elif cmd == "export":
        init_db()
        n = export_csv(sys.argv[2] if len(sys.argv) > 2 else "jobs.csv")
        print(f"Exported {n} rows")
    elif cmd == "json":
        import json
        init_db()
        print(json.dumps([dict(r) for r in list_jobs(with_id=True)]))
    elif cmd == "applied":
        init_db()
        set_applied(int(sys.argv[2]), sys.argv[3] == "1")
        print("ok")
    else:
        print(__doc__)