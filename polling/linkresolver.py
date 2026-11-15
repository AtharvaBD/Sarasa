"""Resolves aggregator links (Handshake, LinkedIn, ...) to the real employer posting URL.

Usage (from the project root):
    python -m polling.linkresolver --login          # one-time: sign in to Handshake, session is saved
    python -m polling.linkresolver <url> "Company Job Title"
"""
import json
import re
import sys
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

import configpoll
import emailclassifer as ec

_cfg = configpoll.cfg["resolver"]
PROFILE_DIR = configpoll.path("browser_profile")
HEADLESS = _cfg["headless"]
LOGIN_URL = _cfg["login_url"]
TIMEOUT_MS = _cfg["timeout_ms"]
POPUP_TIMEOUT_MS = _cfg["popup_timeout_ms"]
WAIT_UNTIL = _cfg["wait_until"]
PLATFORM_HOSTS = tuple(_cfg["platform_hosts"])
JUNK_HOSTS = tuple(_cfg["junk_hosts"])
TRACKING_HOSTS = tuple(_cfg["tracking_hosts"])
APPLY_RE = re.compile(_cfg["apply_regex"], re.I)
CAND_TEXT_CHARS = _cfg["candidate_text_chars"]
LLM_LINK_CHARS = _cfg["llm_link_chars"]
LLM_BODY_CHARS = _cfg["llm_body_chars"]
FAST_REDIRECT = _cfg["fast_redirect"]
FAST_TIMEOUT = _cfg["fast_timeout_seconds"]
USER_AGENT = _cfg["user_agent"]

PICK_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}},
    "required": ["choice"],
}
PICK_PROMPT = configpoll.prompt("link_picker")


def _host(url):
    return (urlparse(url).hostname or "").lower()


def _matches(url, hosts):
    h = _host(url)
    return any(h == d or h.endswith("." + d) for d in hosts)


def _is_platform(url):
    return _matches(url, PLATFORM_HOSTS)


def _is_junk(url):
    return _matches(url, JUNK_HOSTS)


def _is_unresolved(url):
    """True if the URL is still a platform or email-tracking host, not the employer."""
    return _is_platform(url) or _matches(url, TRACKING_HOSTS)


def _fast_resolve(url):
    """Cheap first pass: follow HTTP redirects with no browser. None if it didn't reach the employer."""
    try:
        r = httpx.get(url, follow_redirects=True, timeout=FAST_TIMEOUT,
                      headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError:
        return None
    final = str(r.url)
    return None if _is_unresolved(final) or _is_junk(final) else final


def _candidates(page):
    """Outbound http(s) links on the page that aren't the platform or obvious junk."""
    anchors = page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => ({text: (e.innerText || e.getAttribute('aria-label') || '').trim(), href: e.href}))",
    )
    seen, out = set(), []
    for a in anchors:
        href = a["href"]
        if not href.startswith("http") or _is_platform(href) or _is_junk(href) or href in seen:
            continue
        seen.add(href)
        out.append({"text": a["text"][:CAND_TEXT_CHARS], "url": href})
    return out


def _click_apply_popup(context, page):
    """Some 'Apply externally' buttons are <button>s that open a new tab, not <a> links."""
    try:
        btn = page.get_by_role("button", name=APPLY_RE).first
        with context.expect_page(timeout=POPUP_TIMEOUT_MS) as popup:
            btn.click()
        new = popup.value
        new.wait_for_load_state("domcontentloaded", timeout=TIMEOUT_MS)
        url = new.url
        new.close()
        return url if url.startswith("http") and not _is_unresolved(url) else None
    except Exception:
        return None


def _llm_pick(page, cands, hint):
    listing = "\n".join(
        f"[{i}] {c['url'][:LLM_LINK_CHARS]}  (text: {c['text']})" for i, c in enumerate(cands)
    )
    body = page.inner_text("body")[:LLM_BODY_CHARS]
    messages = [
        {"role": "system", "content": PICK_PROMPT},
        {"role": "user", "content": f"Job: {hint}\nPage title: {page.title()}\n\nPage text:\n{body}\n\nLinks:\n{listing}"},
    ]
    try:
        raw = ec._chat_json(messages, PICK_SCHEMA)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        idx = json.loads(raw)["choice"]
        return cands[idx]["url"] if isinstance(idx, int) and 0 <= idx < len(cands) else None
    except Exception:
        return None


def resolve(url, hint=""):
    """Return the employer's posting URL, or the original url if it can't be resolved."""
    if not url:
        return url

    # 0. No-browser pass: plain HTTP redirects.
    if FAST_REDIRECT:
        fast = _fast_resolve(url)
        if fast:
            return fast

    try:
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(PROFILE_DIR, headless=HEADLESS)
            page = context.new_page()
            page.set_default_timeout(TIMEOUT_MS)
            page.goto(url, wait_until=WAIT_UNTIL)
            page.wait_for_load_state("networkidle")

            # 1. Redirect that landed on the employer's site.
            if not _is_unresolved(page.url):
                result = page.url
            else:
                # 2. An outbound link whose text says "apply".
                cands = _candidates(page)
                apply_links = [c for c in cands if APPLY_RE.search(c["text"])]
                if len(apply_links) == 1:
                    result = apply_links[0]["url"]
                else:
                    # 3. Click the Apply button and catch the new tab.
                    result = _click_apply_popup(context, page)
                    # 4. Last resort: let the cheap model choose among candidates.
                    if not result and cands:
                        result = _llm_pick(page, apply_links or cands, hint)
            context.close()
            return result or url
    except Exception as e:
        print(f"   [resolver] failed for {url[:80]}: {e}")
        return url


def _login():
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(PROFILE_DIR, headless=False)
        page = context.new_page()
        page.goto(LOGIN_URL)
        input("Sign in in the browser window, then press Enter here to save the session...")
        context.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--login":
        _login()
    elif len(sys.argv) > 1:
        print(resolve(sys.argv[1], hint=" ".join(sys.argv[2:])))
    else:
        print(__doc__)