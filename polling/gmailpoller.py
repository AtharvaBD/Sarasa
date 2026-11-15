"""Gmail poller for the job-email pipeline.

Usage (from the project root):
    python -m polling.gmailpoller --once --backfill-days 14            # saves jobs to the database
    python -m polling.gmailpoller --once --backfill-days 14 --dry-run  # classify + log only
    python -m polling.gmailpoller --interval 120                       # keep polling
    
Settings live in configpoll.json. Command-line flags override them.
"""
import argparse
import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parseaddr

from bs4 import BeautifulSoup

from polling import emailclassifer as ec, extractor, jobdb, configpoll

SCOPES = configpoll.cfg["gmail"]["scopes"]
INBOX_QUERY = configpoll.cfg["gmail"]["inbox_query"]
SKIP_LABELS = set(configpoll.cfg["gmail"]["skip_labels"])
MAX_BODY_CHARS = configpoll.cfg["gmail"]["max_body_chars"]

CREDS_FILE = configpoll.cfg["paths"]["credentials"]
TOKEN_FILE = configpoll.path("token")
STATE_FILE = configpoll.path("state")

URL_RE = re.compile(configpoll.cfg["gmail"]["url_regex"])
PAGE_SIZE = configpoll.cfg["gmail"]["page_size"]
FALLBACK_DAYS = configpoll.cfg["gmail"]["fallback_days"]
HISTORY_LABEL = configpoll.cfg["gmail"]["history_label"]
HISTORY_TYPES = configpoll.cfg["gmail"]["history_types"]
LINK_TEXT_CHARS = configpoll.cfg["gmail"]["link_text_chars"]
TRAILING_PUNCT = configpoll.cfg["gmail"]["trailing_punct"]
SKIP_CLASSIFICATION = configpoll.cfg["gmail"]["skip_classification"]


# ---------- auth ----------

def get_service():
    # Imported lazily so the parsing helpers can be used/tested without Google libs.
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ---------- state/ Last place we left off ----------

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ---------- MIME parsing ----------

def _b64(data):
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _charset(part):
    for h in part.get("headers", []):
        if h["name"].lower() == "content-type":
            m = re.search(r"charset=\"?([\w-]+)", h["value"], re.I)
            if m:
                return m.group(1)
    return "utf-8"


def _decode(raw, charset):
    try:
        return raw.decode(charset, "replace")
    except LookupError:
        return raw.decode("utf-8", "replace")


def _walk(part, out):
    if part.get("parts"):
        for p in part["parts"]:
            _walk(p, out)
        return
    mime = part.get("mimeType", "")
    data = part.get("body", {}).get("data")
    if data and mime in ("text/plain", "text/html"):
        out[mime].append(_decode(_b64(data), _charset(part)))


def html_to_text_and_links(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("http://", "https://")):
            links.append({"url": href, "text": a.get_text(" ", strip=True)[:LINK_TEXT_CHARS]})
    return soup.get_text("\n", strip=True), links


def _clean(text):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_message(msg):
    """Gmail API message (format=full) -> flat dict for the pipeline."""
    payload = msg.get("payload", {})
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    bodies = {"text/plain": [], "text/html": []}
    _walk(payload, bodies)

    links, text = [], ""
    if bodies["text/html"]:
        text, links = html_to_text_and_links("\n".join(bodies["text/html"]))
    if bodies["text/plain"]:
        plain = "\n".join(bodies["text/plain"])
        if plain.strip():
            text = plain  # prefer the sender's own plain-text part; links still come from the HTML
        links += [{"url": u.rstrip(TRAILING_PUNCT), "text": ""} for u in URL_RE.findall(plain)]

    seen, uniq = set(), []
    for l in links:
        if l["url"] not in seen:
            seen.add(l["url"])
            uniq.append(l)

    name, addr = parseaddr(headers.get("from", ""))
    received = datetime.fromtimestamp(int(msg["internalDate"]) / 1000, tz=timezone.utc).isoformat() \
        if msg.get("internalDate") else None
    return {
        "id": msg["id"],
        "thread_id": msg.get("threadId"),
        "from_name": name,
        "from_addr": addr.lower(),
        "subject": headers.get("subject", ""),
        "received_at": received,
        "labels": msg.get("labelIds", []),
        "has_unsubscribe": "list-unsubscribe" in headers,
        "body_text": _clean(text)[:MAX_BODY_CHARS],
        "urls": uniq,
    }


# ---------- cheap pre-filter (no model) ----------
# Ignore Gmail Social/Forums messages and send all other emails to the classifier.

def skip_reason(email):
    """Skip only mail Gmail already labeled social/forums (configurable). The model decides the rest."""
    hit = SKIP_LABELS & set(email["labels"])
    return ", ".join(sorted(hit)) if hit else None


# ---------- fetching ----------

def fetch_email(service, message_id):
    msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    return parse_message(msg)


def _recent_ids(service, days):
    ids, token = [], None
    while True:
        r = service.users().messages().list(
            userId="me", q=f"{INBOX_QUERY} newer_than:{days}d", pageToken=token, maxResults=PAGE_SIZE
        ).execute()
        ids += [m["id"] for m in r.get("messages", [])]
        token = r.get("nextPageToken")
        if not token:
            break
    return list(reversed(ids))  # oldest first


def _history_ids(service, start_id):
    ids, token, latest = [], None, start_id
    while True:
        r = service.users().history().list(
            userId="me", startHistoryId=start_id, historyTypes=HISTORY_TYPES,
            labelId=HISTORY_LABEL, pageToken=token,
        ).execute()
        for h in r.get("history", []):
            for m in h.get("messagesAdded", []):
                ids.append(m["message"]["id"])
        latest = r.get("historyId", latest)
        token = r.get("nextPageToken")
        if not token:
            break
    return list(dict.fromkeys(ids)), latest


# ---------- polling ----------

def poll_once(service, handler, backfill_days=0, db_path=jobdb.DB_PATH):
    from googleapiclient.errors import HttpError

    state = load_state()
    current = service.users().getProfile(userId="me").execute()["historyId"]

    if "history_id" not in state:
        ids = _recent_ids(service, backfill_days) if backfill_days else []
        new_history = current
    else:
        try:
            ids, new_history = _history_ids(service, state["history_id"])
        except HttpError as e:
            if e.resp.status != 404:
                raise
            print("History id expired; falling back to a recent-mail scan.")
            ids, new_history = _recent_ids(service, max(backfill_days, FALLBACK_DAYS)), current

    handled = skipped = failed = 0
    for mid in ids:
        if jobdb.is_processed(mid, db_path):
            continue
        try:
            email = fetch_email(service, mid)
        except HttpError as e:
            if e.resp.status == 404:  # deleted before we got to it
                continue
            raise
        reason = skip_reason(email)
        if reason:
            jobdb.mark_processed(mid, SKIP_CLASSIFICATION, email["thread_id"], db_path)
            skipped += 1
            continue
        try:
            handler(email)
            handled += 1
        except ec.ModelUnavailable as e:
            # Model server is down: stop this cycle, don't advance the cursor, retry next poll.
            failed += 1
            print(f"Model unavailable, aborting this cycle: {e}")
            break
        except Exception as e:  # keep going; retry this one next cycle
            failed += 1
            print(f"Handler failed on {mid}: {e}")

    # Only advance the cursor if everything succeeded, so failures get retried.
    if not failed:
        state["history_id"] = new_history
        save_state(state)
    print(f"[{datetime.now():%H:%M:%S}] new={len(ids)} handled={handled} skipped={skipped} failed={failed}")


def main():
    poller = configpoll.cfg["poller"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="poll one time and exit")
    ap.add_argument("--interval", type=int, default=poller["interval_seconds"],
                    help="seconds between polls")
    ap.add_argument("--backfill-days", type=int, default=poller["backfill_days"],
                    help="only used on the first run")
    ap.add_argument("--dry-run", action="store_true",
                    help="classify and log only; don't extract or write to the database")
    ap.add_argument("--live", action="store_true", help=argparse.SUPPRESS)  # old flag, now the default
    args = ap.parse_args()

    jobdb.init_db()
    service = get_service()
    handler = ec.make_handler(on_job=extractor.save_jobs, dry_run=args.dry_run)
    while True:
        try:
            poll_once(service, handler, args.backfill_days)
        except Exception as e:  # network blips shouldn't kill the loop
            print(f"Poll failed: {e}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()