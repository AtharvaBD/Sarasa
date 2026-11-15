# polling: the email-to-jobs pipeline

This module reads your Gmail inbox, figures out which emails are about jobs, pulls out the company, job title and link, and saves them in a small database. Think of it as a mail sorter: it reads each new email, decides whether it matters, and files the useful ones.

## How an email travels

```
Gmail inbox
   │
   ▼
gmailpoller.py     Fetches new emails and cleans them up. Skips Gmail's Social/Forums mail.
   │
   ▼
emailclassifier.py A small local AI model (via Ollama) labels each email:
   │                 job_posting        a job you could apply for
   │                 application_status an update on an application you already sent
   │                 not_job            everything else (dropped here)
   │                 unsure             the model wasn't confident, so it's kept just in case
   ▼
extractor.py       For job emails only: the AI pulls out company, job title and link.
   │
   ▼
linkresolver.py    Turns Handshake/LinkedIn links into the employer's real posting link.
   │
   ▼
jobdb.py           Saves the job in the database. Duplicates are ignored.
```

Every email is marked "processed" once handled, so it is never read twice.

## Before you run it

```
pip install httpx playwright beautifulsoup4 google-api-python-client google-auth-oauthlib
playwright install chromium
```

You also need:
1. **Ollama running** with the model named in `config.json` (default `qwen3:4b`).
2. **Your Google OAuth file** at `systemdata/credentials.json`. The first run opens a browser to sign in to Google and saves a token so it won't ask again.
3. **A Handshake login** (one time): `python -m polling.linkresolver --login`, then sign in in the window that opens.

All commands run from the project root.

## Ways to run it

| Goal | Command |
|---|---|
| Try it safely: classify and log only, save nothing | `python -m polling.gmailpoller --once --backfill-days 14 --dry-run` |
| Real run once: classify, extract, save | `python -m polling.gmailpoller --once --backfill-days 14` |
| Keep running, checking every 2 minutes | `python -m polling.gmailpoller` |
| Change how often it checks | `python -m polling.gmailpoller --interval 300` |
| See saved jobs | `python -m polling.jobdb list` |
| Export jobs to a spreadsheet file | `python -m polling.jobdb export jobs.csv` |
| Test one link by hand | `python -m polling.linkresolver "<url>" "Company Job Title"` |
| Score the classifier against your own labels | `python -m polling.emailclassifier --eval data/classifier_log.jsonl` |
| Run the unit tests (no Gmail or Ollama needed) | `python -m unittest polling.testpolling -v` |

`--backfill-days` only matters the first time. After that the poller asks Gmail only for mail that arrived since the last run.

## What the output means

**One line per email** from the classifier:

```
[job_posting        0.92] Software Engineer Intern, Summer 2027 | Recruiter outreach about a specific internship.
[not_job            0.99] CS 466 Problem Set 3 due Friday | Course logistics, nothing to do with employment.
[unsure             0.45 REVIEW] Quick question | Unclear whether this is about a role.
```

The number is the model's confidence from 0 to 1. `REVIEW` means it is below the threshold (0.6), so check that one yourself. The text after `|` is the model's one-sentence reason.

**Extra lines under a job email** from the extractor:

```
   -> Stripe | Software Engineer Intern, Summer 2027 | https://stripe.com/jobs/123
   -> Acme | Data Analyst Co-op | no link
   saved 2 new of 2
```

`saved 1 new of 2` means one job was already in the database. The line `-> job-related, but no specific job found to save` means the email looked job-related but named no actual role.

**One summary line per check:**

```
[19:24:08] new=12 handled=9 skipped=3 failed=0
```

- `new`: emails found this round
- `handled`: emails the classifier processed
- `skipped`: Social/Forums mail ignored without using the model
- `failed`: emails that hit an error (they are retried next round)

**Messages you may see:**
- `Model unavailable, aborting this cycle`: Ollama isn't running. Start it and the next round picks up where it stopped.
- `Handler failed on <id>`: one email caused an error. The rest continue.
- `[resolver] failed for <url>`: the link couldn't be resolved, so the original link is saved instead.
- `History id expired`: Gmail forgot the saved position, so it re-scans recent mail automatically.

## Where things are saved

| What | Where |
|---|---|
| Jobs and the list of processed emails | `data/jobs.db` |
| Every classifier decision (good for checking mistakes) | `data/classifier_log.jsonl` |
| Where the poller left off | `data/poller_state.json` |
| Google sign-in token | `systemdata/token.json` |
| Saved Handshake login | `.browser_profile/` |

## Starting over

`python reset.py` deletes the token, state and database (it asks first). Add `--log` to delete the classifier log, `--browser` for the Handshake login, or `--all` for everything. Delete the state and database together, or the poller will think it is already caught up.

## Settings

Everything adjustable lives in `config.json`: the model, the confidence threshold, how many characters of each email the AI reads, which Gmail labels to skip, and the AI prompts. Change a value and re-run. No code edits are needed.

## Files in this folder

| File | Job |
|---|---|
| `gmailpoller.py` | Fetches and cleans emails, runs the loop |
| `emailclassifier.py` | Labels each email |
| `extractor.py` | Pulls out company, title and link |
| `linkresolver.py` | Finds the real job link |
| `jobdb.py` | Database storage and exports |
| `testpolling.py` | Unit tests for all of the above |