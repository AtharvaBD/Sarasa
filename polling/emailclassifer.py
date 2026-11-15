"""SLM email classifier for the job-email pipeline.

Labels every email as job_posting / application_status / not_job (or "unsure"), using a
small local model. Stdlib HTTP only.

Settings live in configpoll.json ("model" and "classifier" sections). Environment variables
SLM_BACKEND, SLM_MODEL and SLM_BASE_URL override the model settings for one session.

Usage (from the project root):
    python -m polling.emailclassifier --selftest
    python -m polling.emailclassifier --eval data/classifier_log.jsonl
"""
import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone

from polling import configpoll, jobdb
from system import ollama_setup

BACKEND, MODEL, BASE_URL = configpoll.model_settings()
TIMEOUT = configpoll.cfg["model"]["timeout_seconds"]

LABELS = tuple(configpoll.cfg["classifier"]["labels"])
MIN_CONFIDENCE = configpoll.cfg["classifier"]["min_confidence"]   # below this, "not_job" becomes "unsure"
INPUT_BODY_CHARS = configpoll.cfg["classifier"]["input_body_chars"]
ATTEMPTS = configpoll.cfg["classifier"]["attempts"]
LOG_MAX_URLS = configpoll.cfg["classifier"]["log_max_urls"]
LOG_FILE = configpoll.path("classifier_log")

SCHEMA = {
    "type": "object",
    "properties": {
        # "reason" comes first on purpose: the model writes a short justification
        # before committing to a label, which tends to improve small-model accuracy.
        "reason": {"type": "string"},
        "label": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "number"},
    },
    "required": ["reason", "label", "confidence"],
}

SYSTEM_PROMPT = configpoll.prompt("classifier")

class ModelUnavailable(RuntimeError):
    """Model server unreachable or returned an error. Propagates so the poller retries."""


_ready = False


def ensure_ready():
    """Run the Ollama setup once per process. No-op for non-ollama backends."""
    global _ready
    if _ready or BACKEND != "ollama":
        return
    try:
        ollama_setup.ensure_ready(MODEL, BASE_URL)
    except ollama_setup.SetupError as e:
        raise ModelUnavailable(str(e)) from e
    _ready = True


# ---------- talking to the model ----------

def _post(url, payload, timeout=TIMEOUT):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise ModelUnavailable(f"HTTP {e.code} from {url}: {body}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise ModelUnavailable(f"Can't reach model server at {url}: {e}") from e


def _chat_json(messages, schema=SCHEMA):
    """Send messages, return the model's raw text (expected to be JSON)."""
    if BACKEND == "ollama":
        payload = {
            "model": MODEL, "messages": messages, "stream": False,
            "format": schema, "think": configpoll.cfg["model"]["think"],
            "options": {"temperature": configpoll.cfg["model"]["temperature"]},
        }
        try:
            r = _post(f"{BASE_URL}/api/chat", payload)
        except ModelUnavailable as e:
            # Some models/versions reject the "think" flag; retry once without it.
            if "think" not in str(e).lower():
                raise
            payload.pop("think")
            r = _post(f"{BASE_URL}/api/chat", payload)
        return r["message"]["content"]

    # OpenAI-compatible (LM Studio, etc.)
    messages = [dict(m) for m in messages]
    if configpoll.cfg["model"]["no_think_match"] in MODEL.lower():
        messages[-1]["content"] += configpoll.cfg["model"]["no_think_suffix"]
    payload = {
        "model": MODEL, "messages": messages, "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        },
    }
    r = _post(f"{BASE_URL}/chat/completions", payload)
    return r["choices"][0]["message"]["content"]


def _parse(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in output")
        data = json.loads(raw[start:end + 1])
    if data.get("label") not in LABELS:
        raise ValueError(f"bad label: {data.get('label')!r}")
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"label": data["label"], "confidence": conf, "reason": str(data.get("reason", ""))}


# ---------- classification ----------

def build_user_message(email):
    body = (email.get("body_text") or "")[:INPUT_BODY_CHARS]
    return (
        f"From: {email.get('from_name', '')} <{email.get('from_addr', '')}>\n"
        f"Subject: {email.get('subject', '')}\n"
        f"Received: {email.get('received_at', '')}\n"
        f"Links in email: {len(email.get('urls', []))}\n\n"
        f"Body:\n{body}"
    )


def classify(email):
    """Returns {label, confidence, reason, needs_review}. label may also be "unsure".

    Recall-first policy: a low-confidence "not_job" is promoted to "unsure" so it still
    flows downstream instead of being silently dropped.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(email)},
    ]
    result, last_err = None, None
    for _ in range(ATTEMPTS):  # one retry on malformed output
        try:
            result = _parse(_chat_json(messages))
            break
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
    if result is None:
        return {"label": "unsure", "confidence": 0.0, "needs_review": True,
                "reason": f"unparseable model output ({last_err})"}

    if result["label"] == "not_job" and result["confidence"] < MIN_CONFIDENCE:
        result["label"] = "unsure"
    result["needs_review"] = result["confidence"] < MIN_CONFIDENCE
    return result


def is_job_related(result):
    return result["label"] != "not_job"


# ---------- audit log ----------

def log_result(email, result, path=LOG_FILE):
    """Append every verdict (including not_job) so you can audit misses later."""
    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "id": email.get("id"),
        "from_name": email.get("from_name"), "from_addr": email.get("from_addr"),
        "subject": email.get("subject"), "received_at": email.get("received_at"),
        "body_text": (email.get("body_text") or "")[:INPUT_BODY_CHARS],
        "urls": email.get("urls", [])[:LOG_MAX_URLS],
        "label": result["label"], "confidence": result["confidence"], "reason": result["reason"],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- poller handler ----------

def make_handler(on_job=None, dry_run=False, db_path=jobdb.DB_PATH):
    """Build the function the poller calls for each email.

    on_job(email, result): the extraction stage. Called only for job-related emails.
    dry_run: classify and log, but write nothing to the database.
    """
    ensure_ready()

    def handle(email):
        result = classify(email)
        log_result(email, result)
        flag = " REVIEW" if result["needs_review"] else ""
        print(f"[{result['label']:<18} {result['confidence']:.2f}{flag}] "
              f"{(email['subject'] or '')[:55]} | {result['reason'][:80]}")
        if dry_run:
            return result
        if is_job_related(result):
            if on_job is None:
                return result
            on_job(email, result)
        jobdb.mark_processed(email["id"], result["label"], email.get("thread_id"), db_path)
        return result
    return handle


# ---------- self-test and evaluation ----------

def _mk(subject, from_name, from_addr, body):
    return {"id": "test", "thread_id": None, "subject": subject, "from_name": from_name,
            "from_addr": from_addr, "received_at": "2026-10-08T12:00:00+00:00",
            "body_text": body, "urls": []}


SELFTEST = [
    (_mk("Your application to Boeing was received", "Boeing Careers", "no-reply@boeing.com",
         "Thanks for applying to the Analytics Intern role. We'll be in touch."), "application_status"),
    (_mk("5 new jobs match your profile", "Handshake", "no-reply@joinhandshake.com",
         "Software Engineering Intern at Acme - apply by Nov 3. Data Analyst Co-op at Initech."), "job_posting"),
    (_mk("Quick chat about our SWE intern role?", "Dana Ortiz", "dana@startup.io",
         "Saw your GitHub. We have an open internship, would you like to talk this week?"), "job_posting"),
    (_mk("Interview invitation - Automation Engineer", "Asahi Kasei Bioprocess", "hr@asahi-kasei.com",
         "Please pick a time for your interview next week."), "application_status"),
    (_mk("Your Amazon order has shipped", "Amazon", "ship@amazon.com", "Arriving Thursday."), "not_job"),
    (_mk("CS 466 Problem Set 3 due Friday", "Canvas", "notifications@instructure.com",
         "Reminder: assignment due Friday at 11:59 PM."), "not_job"),
    (_mk("Career fair next week", "Career Center", "careers@illinois.edu",
         "Come to the Illini Union to meet employers. Resume tips inside."), "not_job"),
    (_mk("Undergraduate research assistant opening", "Dr. Patel", "patel@illinois.edu",
         "My lab is hiring an undergrad RA for spring. Reply with your CV by Oct 30."), "job_posting"),
]


def selftest():
    print(f"Backend={BACKEND} model={MODEL} url={BASE_URL}")
    ok = 0
    for email, expected in SELFTEST:
        r = classify(email)
        hit = r["label"] == expected
        ok += hit
        print(f"{'OK  ' if hit else 'MISS'} expected={expected:<19} got={r['label']:<19} "
              f"conf={r['confidence']:.2f} | {email['subject'][:40]}")
    print(f"{ok}/{len(SELFTEST)} correct")


def evaluate(path):
    """Rerun the classifier on a JSONL file where each line has a 'truth' label."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("truth") in LABELS]
    if not rows:
        print("No lines with a valid 'truth' field (job_posting | application_status | not_job).")
        return
    exact = job_hits = job_total = fp = 0
    misses = []
    for row in rows:
        r = classify(row)
        truth_job = row["truth"] != "not_job"
        pred_job = is_job_related(r)           # "unsure" counts as job-related (recall-first)
        exact += r["label"] == row["truth"]
        if truth_job:
            job_total += 1
            job_hits += pred_job
        elif pred_job:
            fp += 1
        if r["label"] != row["truth"]:
            misses.append((row["truth"], r["label"], r["confidence"], row.get("subject", "")[:60], r["reason"][:70]))
    n = len(rows)
    print(f"Exact-label accuracy: {exact}/{n} = {exact / n:.1%}")
    print(f"Job recall (job emails not dropped): {job_hits}/{job_total}")
    print(f"False positives (non-job sent downstream): {fp}")
    for m in misses:
        print(f"  truth={m[0]:<19} got={m[1]:<19} conf={m[2]:.2f} | {m[3]} | {m[4]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval", metavar="FILE", help="JSONL with a 'truth' label per line")
    a = ap.parse_args()
    ensure_ready()
    if a.eval:
        evaluate(a.eval)
    else:
        selftest()