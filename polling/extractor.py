"""Pulls (company, job_title, link, date_published) out of job-related emails and saves them."""
import json
import re

from polling import configpoll, emailclassifer as ec, jobdb, linkresolver

BODY_CHARS = configpoll.cfg["extractor"]["body_chars"]
MAX_LINKS = configpoll.cfg["extractor"]["max_links"]
ATTEMPTS = configpoll.cfg["extractor"]["attempts"]
LINK_URL_CHARS = configpoll.cfg["extractor"]["link_url_chars"]
LINK_TEXT_CHARS = configpoll.cfg["extractor"]["link_text_chars"]
RESOLVE_LINKS = configpoll.cfg["extractor"]["resolve_links"]

SCHEMA = {
    "type": "object",
    "properties": {
        "jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "job_title": {"type": "string"},
                    "link_index": {"type": "integer"},
                },
                "required": ["company", "job_title", "link_index"],
            },
        }
    },
    "required": ["jobs"],
}

SYSTEM_PROMPT = configpoll.prompt("extractor")

def _build_message(email, links):
    link_lines = "\n".join(
        f"[{i}] {l['url'][:LINK_URL_CHARS]}" + (f"  (text: {l['text'][:LINK_TEXT_CHARS]})" if l.get("text") else "")
        for i, l in enumerate(links)
    ) or "(none)"
    return (
        f"From: {email.get('from_name', '')} <{email.get('from_addr', '')}>\n"
        f"Subject: {email.get('subject', '')}\n\n"
        f"Body:\n{(email.get('body_text') or '')[:BODY_CHARS]}\n\n"
        f"Links:\n{link_lines}"
    )


def extract(email):
    """Returns a list of {company, job_title, link}. May be empty."""
    links = (email.get("urls") or [])[:MAX_LINKS]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_message(email, links)},
    ]
    items = None
    for _ in range(ATTEMPTS):  # one retry on malformed output
        try:
            raw = ec._chat_json(messages, SCHEMA)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
            items = json.loads(raw)["jobs"]
            break
        except (ValueError, KeyError, TypeError):
            continue
    if not items:
        return []

    rows = []
    for it in items:
        company = str(it.get("company", "")).strip()
        title = str(it.get("job_title", "")).strip()
        if not company or not title:
            continue
        idx = it.get("link_index", -1)
        link = links[idx]["url"] if isinstance(idx, int) and 0 <= idx < len(links) else None
        rows.append({"company": company, "job_title": title, "link": link})
    return rows


def save_jobs(email, result):
    """on_job handler: extract jobs from the email and write them to the database."""
    date_published = (email.get("received_at") or "")[:10] or None
    rows = extract(email)
    if not rows:
        print("   -> job-related, but no specific job found to save")
        return
    if RESOLVE_LINKS:
        for r in rows:
            if r["link"]:
                r["link"] = linkresolver.resolve(r["link"], hint=f"{r['company']} {r['job_title']}")
    added = sum(
        jobdb.add_job(r["company"], r["job_title"], r["link"], date_published) for r in rows
    )
    for r in rows:
        print(f"   -> {r['company']} | {r['job_title']} | {r['link'] or 'no link'}")
    print(f"   saved {added} new of {len(rows)}")