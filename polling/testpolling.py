"""Unit tests for every module in polling/ (plus config.py).

Run from the project root:
    python -m unittest polling.test_polling -v
    python -m unittest polling.test_polling.JobDbTests -v       # just one group

Live-model tests (real Ollama) are off by default. In PowerShell:
    $env:RUN_LIVE = "1"; python -m unittest polling.test_polling.LiveModelTests -v
"""
import base64
import contextlib
import csv
import importlib
import io
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import systemdata.config as config
from polling import emailclassifer as ec
from polling import jobdb


def _try_import(name):
    """Import polling.<name>; return (module, None) or (None, error) if a dependency is missing."""
    try:
        return importlib.import_module(f"polling.{name}"), None
    except ImportError as e:
        return None, e


lr, LR_ERR = _try_import("linkresolver")   # needs httpx + playwright
ex, EX_ERR = _try_import("extractor")      # imports linkresolver
gp, GP_ERR = _try_import("gmailpoller")    # needs bs4, imports extractor
try:
    import googleapiclient.errors  # noqa: F401
    HAVE_GOOGLE = True
except ImportError:
    HAVE_GOOGLE = False


@contextlib.contextmanager
def quiet():
    """Swallow the print() noise from the code under test."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def mk(subject="x", from_name="a", from_addr="a@b.com", body="body", urls=None):
    return {"id": "test", "thread_id": None, "subject": subject, "from_name": from_name,
            "from_addr": from_addr, "received_at": "2026-10-08T12:00:00+00:00",
            "body_text": body, "urls": urls or []}


def model_reply(label, confidence, reason="test"):
    return json.dumps({"reason": reason, "label": label, "confidence": confidence})


# =====================================================================
# config.py
# =====================================================================
class ConfigTests(unittest.TestCase):
    SECTIONS = ("model", "paths", "gmail", "poller", "classifier",
                "extractor", "resolver", "db", "prompts")

    def test_all_sections_present(self):
        for s in self.SECTIONS:
            self.assertIn(s, config.cfg, f"config.json is missing section '{s}'")

    def test_not_job_label_exists(self):
        # the pipeline's logic hard-codes "not_job"
        self.assertIn("not_job", config.cfg["classifier"]["labels"])

    def test_model_settings_env_override(self):
        env = {"SLM_BACKEND": "b", "SLM_MODEL": "m", "SLM_BASE_URL": "u"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(config.model_settings(), ("b", "m", "u"))

    def test_model_settings_defaults_from_json(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            m = config.cfg["model"]
            self.assertEqual(config.model_settings(), (m["backend"], m["model"], m["base_url"]))

    def test_path_returns_str_and_creates_parent_folder(self):
        p = config.path("database")
        self.assertIsInstance(p, str)
        self.assertTrue(os.path.isdir(os.path.dirname(p)))

    def test_prompt_joins_list_and_passes_string_through(self):
        with mock.patch.dict(config.cfg["prompts"], {"t1": ["a", "b"], "t2": "plain"}):
            self.assertEqual(config.prompt("t1"), "a\nb")
            self.assertEqual(config.prompt("t2"), "plain")

    def test_required_prompts_exist_and_nonempty(self):
        for name in ("classifier", "extractor", "link_picker"):
            self.assertTrue(config.prompt(name).strip(), f"prompt '{name}' is empty")

    def test_classifier_prompt_mentions_every_label(self):
        text = config.prompt("classifier")
        for label in config.cfg["classifier"]["labels"]:
            self.assertIn(label, text)


# =====================================================================
# emailclassifier.py
# =====================================================================
class ClassifierParseTests(unittest.TestCase):
    def test_parse_valid(self):
        r = ec._parse(model_reply("job_posting", 0.9))
        self.assertEqual(r["label"], "job_posting")
        self.assertAlmostEqual(r["confidence"], 0.9)

    def test_parse_strips_think_block(self):
        raw = "<think>hmm let me see</think>" + model_reply("not_job", 0.8)
        self.assertEqual(ec._parse(raw)["label"], "not_job")

    def test_parse_finds_json_inside_noise(self):
        raw = "Sure! " + model_reply("application_status", 0.7) + " hope that helps"
        self.assertEqual(ec._parse(raw)["label"], "application_status")

    def test_parse_rejects_bad_label(self):
        with self.assertRaises(ValueError):
            ec._parse(model_reply("spam", 0.9))

    def test_parse_rejects_no_json(self):
        with self.assertRaises(ValueError):
            ec._parse("I cannot classify this.")

    def test_parse_clamps_confidence(self):
        self.assertEqual(ec._parse(model_reply("job_posting", 1.7))["confidence"], 1.0)
        self.assertEqual(ec._parse(model_reply("job_posting", -3))["confidence"], 0.0)
        bad = '{"reason": "x", "label": "job_posting", "confidence": "high"}'
        self.assertEqual(ec._parse(bad)["confidence"], 0.0)


class ClassifierPolicyTests(unittest.TestCase):
    def test_low_confidence_not_job_becomes_unsure(self):
        low = ec.MIN_CONFIDENCE - 0.01
        with mock.patch.object(ec, "_chat_json", return_value=model_reply("not_job", low)):
            r = ec.classify(mk())
        self.assertEqual(r["label"], "unsure")
        self.assertTrue(r["needs_review"])

    def test_confident_not_job_stays_not_job(self):
        with mock.patch.object(ec, "_chat_json", return_value=model_reply("not_job", 0.99)):
            r = ec.classify(mk())
        self.assertEqual(r["label"], "not_job")
        self.assertFalse(r["needs_review"])

    def test_unparseable_output_returns_unsure_after_all_attempts(self):
        with mock.patch.object(ec, "_chat_json", return_value="garbage") as chat:
            r = ec.classify(mk())
        self.assertEqual(r["label"], "unsure")
        self.assertEqual(r["confidence"], 0.0)
        self.assertTrue(r["needs_review"])
        self.assertEqual(chat.call_count, ec.ATTEMPTS)

    @unittest.skipIf(ec.ATTEMPTS < 2, "needs classifier.attempts >= 2")
    def test_retries_after_malformed_output(self):
        replies = ["garbage", model_reply("job_posting", 0.9)]
        with mock.patch.object(ec, "_chat_json", side_effect=replies):
            r = ec.classify(mk())
        self.assertEqual(r["label"], "job_posting")

    def test_is_job_related(self):
        self.assertTrue(ec.is_job_related({"label": "job_posting"}))
        self.assertTrue(ec.is_job_related({"label": "application_status"}))
        self.assertTrue(ec.is_job_related({"label": "unsure"}))  # recall-first
        self.assertFalse(ec.is_job_related({"label": "not_job"}))

    def test_build_user_message_truncates_body(self):
        n = ec.INPUT_BODY_CHARS
        msg = ec.build_user_message(mk(subject="subj", body="x" * (n + 500)))
        self.assertIn("x" * n, msg)
        self.assertNotIn("x" * (n + 1), msg)
        self.assertIn("Subject: subj", msg)

    def test_log_result_writes_one_json_line(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "log.jsonl")
            result = {"label": "job_posting", "confidence": 0.9, "reason": "r"}
            ec.log_result(mk(subject="subj"), result, path=path)
            with open(path, encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(len(lines), 1)
        row = json.loads(lines[0])
        self.assertEqual(row["label"], "job_posting")
        self.assertEqual(row["subject"], "subj")


class ClassifierTransportTests(unittest.TestCase):
    """_chat_json builds the right request for each backend."""

    MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]

    def test_ollama_payload_has_required_fields(self):
        with mock.patch.object(ec, "BACKEND", "ollama"), \
             mock.patch.object(ec, "_post", return_value={"message": {"content": "{}"}}) as post:
            out = ec._chat_json(self.MSGS)
        payload = post.call_args[0][1]
        self.assertEqual(out, "{}")
        self.assertEqual(payload["model"], ec.MODEL)
        self.assertEqual(payload["messages"], self.MSGS)
        self.assertIs(payload["stream"], False)
        self.assertIn("format", payload)
        self.assertEqual(payload["options"]["temperature"], config.cfg["model"]["temperature"])
        self.assertTrue(post.call_args[0][0].endswith("/api/chat"))

    def test_ollama_retries_without_think_flag(self):
        effects = [ec.ModelUnavailable("unknown field: think"),
                   {"message": {"content": "ok"}}]
        with mock.patch.object(ec, "BACKEND", "ollama"), \
             mock.patch.object(ec, "_post", side_effect=effects) as post:
            self.assertEqual(ec._chat_json(self.MSGS), "ok")
        self.assertEqual(post.call_count, 2)
        self.assertNotIn("think", post.call_args_list[1][0][1])

    def test_ollama_other_errors_propagate(self):
        with mock.patch.object(ec, "BACKEND", "ollama"), \
             mock.patch.object(ec, "_post", side_effect=ec.ModelUnavailable("boom")):
            with self.assertRaises(ec.ModelUnavailable):
                ec._chat_json(self.MSGS)

    def test_openai_branch_adds_no_think_suffix_without_mutating_input(self):
        model = config.cfg["model"]["no_think_match"] + "-test"
        original = [dict(m) for m in self.MSGS]
        reply = {"choices": [{"message": {"content": "x"}}]}
        with mock.patch.object(ec, "BACKEND", "openai"), \
             mock.patch.object(ec, "MODEL", model), \
             mock.patch.object(ec, "_post", return_value=reply) as post:
            self.assertEqual(ec._chat_json(self.MSGS), "x")
        sent = post.call_args[0][1]["messages"]
        self.assertTrue(sent[-1]["content"].endswith(config.cfg["model"]["no_think_suffix"]))
        self.assertEqual(self.MSGS, original)
        self.assertTrue(post.call_args[0][0].endswith("/chat/completions"))


class ClassifierHandlerTests(unittest.TestCase):
    def _run(self, label, on_job=None, dry_run=False):
        result = {"label": label, "confidence": 0.9, "reason": "r", "needs_review": False}
        email = {"id": "e1", "thread_id": "t1", "subject": "s"}
        with mock.patch.object(ec, "ensure_ready"), \
             mock.patch.object(ec, "classify", return_value=result), \
             mock.patch.object(ec, "log_result") as log, \
             mock.patch.object(ec.jobdb, "mark_processed") as mark, \
             quiet():
            ec.make_handler(on_job=on_job, dry_run=dry_run)(email)
        return email, result, log, mark

    def test_job_email_calls_extractor_and_marks_processed(self):
        on_job = mock.Mock()
        email, result, log, mark = self._run("job_posting", on_job=on_job)
        on_job.assert_called_once_with(email, result)
        mark.assert_called_once()
        log.assert_called_once()

    def test_not_job_skips_extractor_but_marks_processed(self):
        on_job = mock.Mock()
        _, _, _, mark = self._run("not_job", on_job=on_job)
        on_job.assert_not_called()
        mark.assert_called_once()

    def test_dry_run_logs_but_writes_nothing(self):
        on_job = mock.Mock()
        _, _, log, mark = self._run("job_posting", on_job=on_job, dry_run=True)
        on_job.assert_not_called()
        mark.assert_not_called()
        log.assert_called_once()


class ClassifierEvaluateTests(unittest.TestCase):
    def test_evaluate_reports_accuracy_recall_and_false_positives(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "eval.jsonl")
            rows = [{"truth": "job_posting", "subject": "a"},
                    {"truth": "not_job", "subject": "b"},
                    {"truth": "not_job", "subject": "c"},
                    {"subject": "no truth label, ignored"}]
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r) for r in rows))

            def fake(row):
                label = {"a": "job_posting", "b": "not_job", "c": "job_posting"}[row["subject"]]
                return {"label": label, "confidence": 0.9, "reason": "r"}

            buf = io.StringIO()
            with mock.patch.object(ec, "classify", side_effect=fake), \
                 contextlib.redirect_stdout(buf):
                ec.evaluate(path)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        out = buf.getvalue()
        self.assertIn("2/3", out)
        self.assertIn("Job recall (job emails not dropped): 1/1", out)
        self.assertIn("False positives (non-job sent downstream): 1", out)


# =====================================================================
# jobdb.py
# =====================================================================
class JobDbTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        jobdb.init_db(self.db)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _cols(self, path, table="jobs"):
        with jobdb.connect(path) as c:
            return [r["name"] for r in c.execute(f"PRAGMA table_info({table})")]

    # ---- canonical_url ----
    def test_canonical_url_strips_tracking_fragment_slash_and_lowercases_host(self):
        got = jobdb.canonical_url("HTTPS://Example.COM/Jobs/123/?utm_source=x&id=7#frag")
        self.assertEqual(got, "https://example.com/Jobs/123?id=7")

    def test_canonical_url_empty_inputs(self):
        self.assertEqual(jobdb.canonical_url(None), "")
        self.assertEqual(jobdb.canonical_url(""), "")

    # ---- schema / migration ----
    def test_init_db_creates_tables_and_is_idempotent(self):
        jobdb.init_db(self.db)
        for col in ("company", "job_title", "link", "date_published", "applied_at"):
            self.assertIn(col, self._cols(self.db))
        self.assertIn("classification", self._cols(self.db, "processed_emails"))

    def test_migration_drops_old_dedupe_key_table(self):
        path = os.path.join(self.dir, "old1.db")
        with jobdb.connect(path) as c:
            c.execute("CREATE TABLE jobs (company TEXT, dedupe_key TEXT)")
        jobdb.init_db(path)
        cols = self._cols(path)
        self.assertNotIn("dedupe_key", cols)
        self.assertIn("applied_at", cols)

    def test_migration_adds_applied_at_to_existing_table(self):
        path = os.path.join(self.dir, "old2.db")
        with jobdb.connect(path) as c:
            c.execute("CREATE TABLE jobs (company TEXT NOT NULL, job_title TEXT NOT NULL, "
                      "link TEXT, date_published TEXT)")
        jobdb.init_db(path)
        self.assertIn("applied_at", self._cols(path))

    # ---- processed emails ----
    def test_mark_and_check_processed(self):
        self.assertFalse(jobdb.is_processed("m1", self.db))
        jobdb.mark_processed("m1", "job_posting", "t1", self.db)
        self.assertTrue(jobdb.is_processed("m1", self.db))
        jobdb.mark_processed("m1", "not_job", "t1", self.db)   # duplicate: no error
        self.assertTrue(jobdb.is_processed("m1", self.db))

    # ---- add_job / dedupe ----
    def test_add_job_then_duplicate(self):
        self.assertTrue(jobdb.add_job("Acme", "SWE Intern", "https://a.com/j", "2026-10-01", self.db))
        self.assertFalse(jobdb.add_job("Acme", "SWE Intern", "https://a.com/j", "2026-10-02", self.db))
        self.assertEqual(len(jobdb.list_jobs(self.db)), 1)

    def test_dedupe_ignores_case_and_tracking_params(self):
        self.assertTrue(jobdb.add_job("Acme", "SWE Intern", "https://a.com/j?utm_source=x", None, self.db))
        self.assertFalse(jobdb.add_job("ACME", "swe intern", "https://a.com/j/", None, self.db))

    def test_dedupe_works_when_link_is_none(self):
        self.assertTrue(jobdb.add_job("Acme", "RA", None, None, self.db))
        self.assertFalse(jobdb.add_job("Acme", "RA", None, None, self.db))

    def test_same_title_different_link_is_a_new_job(self):
        self.assertTrue(jobdb.add_job("Acme", "RA", "https://a.com/1", None, self.db))
        self.assertTrue(jobdb.add_job("Acme", "RA", "https://a.com/2", None, self.db))

    def test_add_job_strips_whitespace_and_stores_canonical_link(self):
        jobdb.add_job("  Acme ", " RA  ", "https://A.com/x/?utm_medium=m", "2026-10-01", self.db)
        row = jobdb.list_jobs(self.db)[0]
        self.assertEqual((row["company"], row["job_title"]), ("Acme", "RA"))
        self.assertEqual(row["link"], "https://a.com/x")

    # ---- listing / applied / get / export ----
    def test_list_jobs_newest_first_and_with_id(self):
        jobdb.add_job("A", "old", None, "2026-01-01", self.db)
        jobdb.add_job("B", "new", None, "2026-12-01", self.db)
        jobdb.add_job("C", "mid", None, "2026-06-01", self.db)
        self.assertEqual([r["job_title"] for r in jobdb.list_jobs(self.db)], ["new", "mid", "old"])
        row = jobdb.list_jobs(self.db, with_id=True)[0]
        self.assertIn("id", row.keys())
        self.assertIn("applied_at", row.keys())

    def test_set_applied_sets_today_and_clears(self):
        jobdb.add_job("A", "T", None, "2026-01-01", self.db)
        job_id = jobdb.list_jobs(self.db, with_id=True)[0]["id"]
        jobdb.set_applied(job_id, True, self.db)
        self.assertEqual(jobdb.list_jobs(self.db, with_id=True)[0]["applied_at"],
                         datetime.now().date().isoformat())
        jobdb.set_applied(job_id, False, self.db)
        self.assertIsNone(jobdb.list_jobs(self.db, with_id=True)[0]["applied_at"])

    def test_get_job(self):
        jobdb.add_job("A", "T", "https://a.com/1", "2026-01-01", self.db)
        job_id = jobdb.list_jobs(self.db, with_id=True)[0]["id"]
        self.assertEqual(jobdb.get_job(job_id, self.db)["company"], "A")
        self.assertIsNone(jobdb.get_job(9999, self.db))

    def test_export_csv(self):
        jobdb.add_job("A", "T1", "https://a.com/1", "2026-01-01", self.db)
        jobdb.add_job("B", "T2", None, "2026-02-01", self.db)
        out = os.path.join(self.dir, "out.csv")
        self.assertEqual(jobdb.export_csv(out, self.db), 2)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["company", "job_title", "link", "date_published"])
        self.assertEqual(len(rows), 3)


# =====================================================================
# extractor.py
# =====================================================================
def jobs_reply(*jobs):
    return json.dumps({"jobs": list(jobs)})


def links(n):
    return [{"url": f"https://x.com/{i}", "text": f"link {i}"} for i in range(n)]


@unittest.skipIf(ex is None, f"extractor not importable: {EX_ERR}")
class ExtractorTests(unittest.TestCase):
    def _extract(self, reply, urls=3):
        with mock.patch.object(ec, "_chat_json", return_value=reply):
            return ex.extract(mk(urls=links(urls)))

    def test_maps_link_index_to_url(self):
        rows = self._extract(jobs_reply({"company": "Stripe", "job_title": "SWE Intern", "link_index": 1}))
        self.assertEqual(rows, [{"company": "Stripe", "job_title": "SWE Intern",
                                 "link": "https://x.com/1"}])

    def test_bad_link_index_values_give_no_link(self):
        rows = self._extract(jobs_reply(
            {"company": "A", "job_title": "t1", "link_index": -1},
            {"company": "B", "job_title": "t2", "link_index": 99},
            {"company": "C", "job_title": "t3", "link_index": "1"},   # string, not int
        ))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["link"] is None for r in rows))

    def test_drops_rows_missing_company_or_title(self):
        rows = self._extract(jobs_reply(
            {"company": "", "job_title": "Intern", "link_index": 0},
            {"company": "Acme", "job_title": "   ", "link_index": 0},
            {"company": "Ok", "job_title": "Fine", "link_index": 0},
        ))
        self.assertEqual([r["company"] for r in rows], ["Ok"])

    def test_empty_list(self):
        self.assertEqual(self._extract(jobs_reply()), [])

    def test_strips_think_block(self):
        raw = "<think>reasoning</think>" + jobs_reply(
            {"company": "A", "job_title": "T", "link_index": -1})
        self.assertEqual(len(self._extract(raw)), 1)

    @unittest.skipIf(ex is not None and ex.ATTEMPTS < 2, "needs extractor.attempts >= 2")
    def test_retries_after_malformed_output(self):
        good = jobs_reply({"company": "A", "job_title": "T", "link_index": -1})
        with mock.patch.object(ec, "_chat_json", side_effect=["not json", good]):
            self.assertEqual(len(ex.extract(mk(urls=links(1)))), 1)

    def test_all_attempts_malformed_returns_empty(self):
        with mock.patch.object(ec, "_chat_json", return_value="nope") as chat:
            self.assertEqual(ex.extract(mk()), [])
        self.assertEqual(chat.call_count, ex.ATTEMPTS)

    def test_only_first_max_links_are_visible_to_the_model(self):
        n = ex.MAX_LINKS
        reply = jobs_reply(
            {"company": "A", "job_title": "inside", "link_index": n - 1},
            {"company": "B", "job_title": "outside", "link_index": n},
        )
        rows = self._extract(reply, urls=n + 5)
        self.assertEqual(rows[0]["link"], f"https://x.com/{n - 1}")
        self.assertIsNone(rows[1]["link"])

    def test_build_message_numbers_links_and_truncates_body(self):
        msg = ex._build_message(mk(body="y" * (ex.BODY_CHARS + 100)), links(2))
        self.assertIn("[0] https://x.com/0", msg)
        self.assertIn("(text: link 0)", msg)
        self.assertIn("[1] https://x.com/1", msg)
        self.assertNotIn("y" * (ex.BODY_CHARS + 1), msg)

    def test_build_message_without_links(self):
        self.assertIn("(none)", ex._build_message(mk(), []))

    # ---- save_jobs ----
    def test_save_jobs_writes_every_row_with_email_date(self):
        rows = [{"company": "A", "job_title": "T1", "link": "https://a.com/1"},
                {"company": "B", "job_title": "T2", "link": None}]
        with mock.patch.object(ex, "extract", return_value=rows), \
             mock.patch.object(ex, "RESOLVE_LINKS", False), \
             mock.patch.object(ex.jobdb, "add_job", return_value=True) as add, quiet():
            ex.save_jobs({"received_at": "2026-10-08T12:00:00+00:00"}, {})
        self.assertEqual(add.call_count, 2)
        add.assert_any_call("A", "T1", "https://a.com/1", "2026-10-08")
        add.assert_any_call("B", "T2", None, "2026-10-08")

    def test_save_jobs_resolves_only_rows_that_have_links(self):
        rows = [{"company": "A", "job_title": "T1", "link": "https://hs.com/1"},
                {"company": "B", "job_title": "T2", "link": None}]
        with mock.patch.object(ex, "extract", return_value=rows), \
             mock.patch.object(ex, "RESOLVE_LINKS", True), \
             mock.patch.object(ex.linkresolver, "resolve", return_value="https://real.com/job") as res, \
             mock.patch.object(ex.jobdb, "add_job", return_value=True) as add, quiet():
            ex.save_jobs({"received_at": None}, {})
        res.assert_called_once()
        add.assert_any_call("A", "T1", "https://real.com/job", None)
        add.assert_any_call("B", "T2", None, None)

    def test_save_jobs_with_no_rows_saves_nothing(self):
        with mock.patch.object(ex, "extract", return_value=[]), \
             mock.patch.object(ex.linkresolver, "resolve") as res, \
             mock.patch.object(ex.jobdb, "add_job") as add, quiet():
            ex.save_jobs({"received_at": "2026-10-08T12:00:00+00:00"}, {})
        add.assert_not_called()
        res.assert_not_called()


# =====================================================================
# linkresolver.py
# =====================================================================
def fake_playwright(page_url):
    pw = mock.MagicMock()
    ctx = pw.__enter__.return_value.chromium.launch_persistent_context.return_value
    page = ctx.new_page.return_value
    page.url = page_url
    return pw, ctx, page


@unittest.skipIf(lr is None, f"linkresolver not importable: {LR_ERR}")
class LinkResolverTests(unittest.TestCase):
    EMPLOYER = "https://boards.greenhouse.io/acme/jobs/1"

    # ---- host helpers ----
    def test_host_matching_is_by_domain_not_substring(self):
        p = lr.PLATFORM_HOSTS[0]
        self.assertTrue(lr._is_platform(f"https://{p}/jobs/1"))
        self.assertTrue(lr._is_platform(f"https://app.{p}/jobs/1"))
        self.assertFalse(lr._is_platform(f"https://not{p}/jobs/1"))
        self.assertFalse(lr._is_platform(self.EMPLOYER))

    def test_is_junk(self):
        self.assertTrue(lr._is_junk(f"https://{lr.JUNK_HOSTS[0]}/x"))
        self.assertFalse(lr._is_junk(self.EMPLOYER))

    def test_is_unresolved_covers_platform_and_tracking_hosts(self):
        self.assertTrue(lr._is_unresolved(f"https://{lr.PLATFORM_HOSTS[0]}/x"))
        self.assertTrue(lr._is_unresolved(f"https://click.{lr.TRACKING_HOSTS[0]}/x"))
        self.assertFalse(lr._is_unresolved(self.EMPLOYER))

    # ---- fast (no-browser) pass ----
    def test_fast_resolve_returns_employer_url(self):
        with mock.patch.object(lr.httpx, "get", return_value=mock.Mock(url=self.EMPLOYER)):
            self.assertEqual(lr._fast_resolve("https://tracker.example/abc"), self.EMPLOYER)

    def test_fast_resolve_rejects_platform_tracking_and_junk_results(self):
        for bad in (f"https://{lr.PLATFORM_HOSTS[0]}/login",
                    f"https://{lr.TRACKING_HOSTS[0]}/x",
                    f"https://{lr.JUNK_HOSTS[0]}/x"):
            with mock.patch.object(lr.httpx, "get", return_value=mock.Mock(url=bad)):
                self.assertIsNone(lr._fast_resolve("https://tracker.example/abc"), bad)

    def test_fast_resolve_returns_none_on_network_error(self):
        with mock.patch.object(lr.httpx, "get", side_effect=lr.httpx.ConnectError("down")):
            self.assertIsNone(lr._fast_resolve("https://tracker.example/abc"))

    # ---- candidates ----
    def test_candidates_filters_and_truncates(self):
        page = mock.Mock()
        page.eval_on_selector_all.return_value = [
            {"text": "Apply", "href": "https://jobs.acme.com/1"},
            {"text": "Dup", "href": "https://jobs.acme.com/1"},
            {"text": "plat", "href": f"https://app.{lr.PLATFORM_HOSTS[0]}/jobs/5"},
            {"text": "junk", "href": f"https://{lr.JUNK_HOSTS[0]}/acme"},
            {"text": "mail", "href": "mailto:a@b.com"},
            {"text": "A" * 500, "href": "https://other.com/2"},
        ]
        cands = lr._candidates(page)
        self.assertEqual([c["url"] for c in cands], ["https://jobs.acme.com/1", "https://other.com/2"])
        self.assertEqual(len(cands[1]["text"]), lr.CAND_TEXT_CHARS)

    # ---- LLM fallback ----
    def _llm(self, reply):
        page = mock.Mock()
        page.inner_text.return_value = "page body"
        page.title.return_value = "Title"
        cands = [{"text": "a", "url": "https://a.com/0"}, {"text": "b", "url": "https://b.com/1"}]
        with mock.patch.object(lr.ec, "_chat_json", return_value=reply):
            return lr._llm_pick(page, cands, "Acme RA")

    def test_llm_pick_valid_choice(self):
        self.assertEqual(self._llm('{"choice": 1}'), "https://b.com/1")

    def test_llm_pick_none_when_minus_one_out_of_range_or_garbage(self):
        self.assertIsNone(self._llm('{"choice": -1}'))
        self.assertIsNone(self._llm('{"choice": 7}'))
        self.assertIsNone(self._llm("not json"))

    # ---- apply-button popup ----
    def test_click_apply_popup_returns_employer_url(self):
        ctx, page, new = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        new.url = "https://jobs.acme.com/p"
        ctx.expect_page.return_value.__enter__.return_value.value = new
        self.assertEqual(lr._click_apply_popup(ctx, page), "https://jobs.acme.com/p")
        new.close.assert_called_once()

    def test_click_apply_popup_rejects_platform_url(self):
        ctx, page, new = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        new.url = f"https://{lr.PLATFORM_HOSTS[0]}/jobs/1"
        ctx.expect_page.return_value.__enter__.return_value.value = new
        self.assertIsNone(lr._click_apply_popup(ctx, page))

    def test_click_apply_popup_returns_none_on_error(self):
        page = mock.MagicMock()
        page.get_by_role.side_effect = RuntimeError("no button")
        self.assertIsNone(lr._click_apply_popup(mock.MagicMock(), page))

    # ---- resolve(): fast path ----
    def test_resolve_empty_url_passthrough(self):
        self.assertEqual(lr.resolve(""), "")
        self.assertIsNone(lr.resolve(None))

    def test_resolve_uses_fast_path_and_skips_browser(self):
        with mock.patch.object(lr, "FAST_REDIRECT", True), \
             mock.patch.object(lr, "_fast_resolve", return_value=self.EMPLOYER), \
             mock.patch.object(lr, "sync_playwright", side_effect=AssertionError("browser launched")):
            self.assertEqual(lr.resolve("https://tracker.example/abc"), self.EMPLOYER)

    def test_resolve_falls_back_to_original_url_if_browser_crashes(self):
        with mock.patch.object(lr, "FAST_REDIRECT", False), \
             mock.patch.object(lr, "sync_playwright", side_effect=RuntimeError("boom")), quiet():
            self.assertEqual(lr.resolve("https://hs.example/1"), "https://hs.example/1")

    # ---- resolve(): browser steps ----
    def _resolve(self, page_url, anchors=(), popup=None, llm=None):
        pw, ctx, page = fake_playwright(page_url)
        page.eval_on_selector_all.return_value = list(anchors)
        with mock.patch.object(lr, "FAST_REDIRECT", False), \
             mock.patch.object(lr, "sync_playwright", return_value=pw), \
             mock.patch.object(lr, "_click_apply_popup", return_value=popup), \
             mock.patch.object(lr, "_llm_pick", return_value=llm):
            result = lr.resolve("https://hs.example/1", hint="Acme RA")
        ctx.close.assert_called_once()
        return result

    def test_resolve_step1_redirect_landed_on_employer(self):
        self.assertEqual(self._resolve(self.EMPLOYER), self.EMPLOYER)

    def test_resolve_step2_single_apply_link(self):
        plat = f"https://app.{lr.PLATFORM_HOSTS[0]}/jobs/9"
        anchors = [{"text": "Apply externally", "href": "https://jobs.acme.com/9"},
                   {"text": "Share", "href": "https://other.com/s"}]
        self.assertEqual(self._resolve(plat, anchors), "https://jobs.acme.com/9")

    def test_resolve_step3_apply_button_popup(self):
        plat = f"https://app.{lr.PLATFORM_HOSTS[0]}/jobs/9"
        self.assertEqual(self._resolve(plat, popup="https://jobs.acme.com/p"),
                         "https://jobs.acme.com/p")

    def test_resolve_step4_llm_choice(self):
        plat = f"https://app.{lr.PLATFORM_HOSTS[0]}/jobs/9"
        anchors = [{"text": "Company site", "href": "https://x.com/z"},
                   {"text": "About", "href": "https://x.com/about"}]
        self.assertEqual(self._resolve(plat, anchors, llm="https://x.com/z"), "https://x.com/z")

    def test_resolve_gives_back_original_when_nothing_works(self):
        plat = f"https://app.{lr.PLATFORM_HOSTS[0]}/jobs/9"
        self.assertEqual(self._resolve(plat), "https://hs.example/1")


# =====================================================================
# gmailpoller.py
# =====================================================================
def b64(text):
    return base64.urlsafe_b64encode(text.encode()).decode()


@unittest.skipIf(gp is None, f"gmailpoller not importable: {GP_ERR}")
class GmailParsingTests(unittest.TestCase):
    def test_b64_handles_missing_padding(self):
        data = base64.urlsafe_b64encode(b"hello").decode().rstrip("=")
        self.assertEqual(gp._b64(data), b"hello")

    def test_clean_collapses_whitespace(self):
        self.assertEqual(gp._clean("a \u00a0 b\r\n\n\n\nc  "), "a b\n\nc")

    def test_html_to_text_and_links(self):
        html = ('<head><title>t</title></head><script>x=1</script>'
                '<a href="mailto:a@b.com">m</a>'
                f'<a href="https://a.com">{"z" * 500}</a>')
        text, found = gp.html_to_text_and_links(html)
        self.assertNotIn("x=1", text)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["url"], "https://a.com")
        self.assertEqual(len(found[0]["text"]), gp.LINK_TEXT_CHARS)

    def _message(self, internal="1760000000000"):
        msg = {
            "id": "m1", "threadId": "t1", "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [{"name": "From", "value": "Dana Ortiz <Dana@Startup.io>"},
                            {"name": "Subject", "value": "Hi there"},
                            {"name": "List-Unsubscribe", "value": "<x>"}],
                "parts": [
                    {"mimeType": "text/plain", "headers": [],
                     "body": {"data": b64("Apply here: https://a.com/job.")}},
                    {"mimeType": "text/html", "headers": [],
                     "body": {"data": b64('<p>Apply <a href="https://a.com/job">now</a></p>')}},
                ],
            },
        }
        if internal:
            msg["internalDate"] = internal
        return msg

    def test_parse_message_fields(self):
        e = gp.parse_message(self._message())
        self.assertEqual((e["id"], e["thread_id"]), ("m1", "t1"))
        self.assertEqual((e["from_name"], e["from_addr"]), ("Dana Ortiz", "dana@startup.io"))
        self.assertEqual(e["subject"], "Hi there")
        self.assertEqual(e["labels"], ["INBOX"])
        self.assertTrue(e["has_unsubscribe"])
        expected = datetime.fromtimestamp(1760000000, tz=timezone.utc).isoformat()
        self.assertEqual(e["received_at"], expected)

    def test_parse_message_prefers_plain_text_body(self):
        self.assertEqual(gp.parse_message(self._message())["body_text"],
                         "Apply here: https://a.com/job.")

    def test_parse_message_dedupes_links_and_keeps_html_link_text(self):
        # the plain-text copy of the URL has a trailing "." that must be stripped, then deduped
        urls = gp.parse_message(self._message())["urls"]
        self.assertEqual(urls, [{"url": "https://a.com/job", "text": "now"}])

    def test_parse_message_without_internal_date(self):
        self.assertIsNone(gp.parse_message(self._message(internal=None))["received_at"])

    def test_parse_message_html_only(self):
        msg = {"id": "m2", "payload": {"mimeType": "text/html", "headers": [],
               "body": {"data": b64('<a href="https://b.com/x">Go</a> Hello')}}}
        e = gp.parse_message(msg)
        self.assertIn("Hello", e["body_text"])
        self.assertEqual(e["urls"], [{"url": "https://b.com/x", "text": "Go"}])

    def test_skip_reason(self):
        self.assertIsNone(gp.skip_reason({"labels": ["INBOX"]}))
        if gp.SKIP_LABELS:
            label = next(iter(gp.SKIP_LABELS))
            self.assertIn(label, gp.skip_reason({"labels": ["INBOX", label]}))


@unittest.skipIf(gp is None, f"gmailpoller not importable: {GP_ERR}")
class GmailFetchTests(unittest.TestCase):
    def test_recent_ids_paginates_and_returns_oldest_first(self):
        svc = mock.MagicMock()
        execute = svc.users.return_value.messages.return_value.list.return_value.execute
        execute.side_effect = [{"messages": [{"id": "3"}, {"id": "2"}], "nextPageToken": "p2"},
                               {"messages": [{"id": "1"}]}]
        self.assertEqual(gp._recent_ids(svc, 7), ["1", "2", "3"])

    def test_history_ids_paginates_dedupes_and_tracks_latest(self):
        svc = mock.MagicMock()
        execute = svc.users.return_value.history.return_value.list.return_value.execute
        execute.side_effect = [
            {"history": [{"messagesAdded": [{"message": {"id": "a"}}, {"message": {"id": "b"}}]}],
             "historyId": "11", "nextPageToken": "n"},
            {"history": [{"messagesAdded": [{"message": {"id": "b"}}, {"message": {"id": "c"}}]}],
             "historyId": "12"},
        ]
        ids, latest = gp._history_ids(svc, "10")
        self.assertEqual(ids, ["a", "b", "c"])
        self.assertEqual(latest, "12")


@unittest.skipIf(gp is None or not HAVE_GOOGLE, f"gmailpoller/google libs missing: {GP_ERR}")
class GmailPollOnceTests(unittest.TestCase):
    def _service(self, current="150"):
        svc = mock.MagicMock()
        svc.users.return_value.getProfile.return_value.execute.return_value = {"historyId": current}
        return svc

    def _poll(self, handler, state=None, ids=("a", "b"), labels=("INBOX",), processed=(),
              new_history="200", history_error=None, recent=(), backfill_days=0):
        state = {"history_id": "100"} if state is None else state
        fetch = lambda service, mid: {"id": mid, "thread_id": "t-" + mid, "labels": list(labels)}
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(gp, "load_state", side_effect=lambda: dict(state)))
            save = st.enter_context(mock.patch.object(gp, "save_state"))
            st.enter_context(mock.patch.object(gp, "fetch_email", side_effect=fetch))
            if history_error:
                st.enter_context(mock.patch.object(gp, "_history_ids", side_effect=history_error))
            else:
                st.enter_context(mock.patch.object(gp, "_history_ids",
                                                   return_value=(list(ids), new_history)))
            recent_mock = st.enter_context(mock.patch.object(gp, "_recent_ids",
                                                             return_value=list(recent)))
            st.enter_context(mock.patch.object(gp.jobdb, "is_processed",
                                               side_effect=lambda mid, path=None: mid in processed))
            mark = st.enter_context(mock.patch.object(gp.jobdb, "mark_processed"))
            st.enter_context(quiet())
            gp.poll_once(self._service(), handler, backfill_days=backfill_days)
        return SimpleNamespace(save=save, mark=mark, recent=recent_mock)

    def test_success_calls_handler_and_advances_cursor(self):
        handler = mock.Mock()
        r = self._poll(handler)
        self.assertEqual(handler.call_count, 2)
        r.save.assert_called_once()
        self.assertEqual(r.save.call_args[0][0]["history_id"], "200")

    def test_uses_history_not_a_full_rescan_after_first_run(self):
        r = self._poll(mock.Mock())
        r.recent.assert_not_called()

    def test_already_processed_emails_are_skipped(self):
        handler = mock.Mock()
        self._poll(handler, processed=("a",))
        handler.assert_called_once()
        self.assertEqual(handler.call_args[0][0]["id"], "b")

    @unittest.skipIf(gp is not None and not gp.SKIP_LABELS, "no skip_labels configured")
    def test_skip_labelled_mail_never_reaches_handler(self):
        handler = mock.Mock()
        label = next(iter(gp.SKIP_LABELS))
        r = self._poll(handler, ids=("a",), labels=("INBOX", label))
        handler.assert_not_called()
        r.mark.assert_any_call("a", gp.SKIP_CLASSIFICATION, "t-a", gp.jobdb.DB_PATH)
        r.save.assert_called_once()   # skipping is not a failure

    def test_model_unavailable_stops_cycle_and_holds_cursor(self):
        handler = mock.Mock(side_effect=ec.ModelUnavailable("down"))
        r = self._poll(handler)
        self.assertEqual(handler.call_count, 1)
        r.save.assert_not_called()

    def test_handler_error_continues_but_holds_cursor(self):
        handler = mock.Mock(side_effect=[RuntimeError("bad email"), None])
        r = self._poll(handler)
        self.assertEqual(handler.call_count, 2)
        r.save.assert_not_called()

    def test_expired_history_id_falls_back_to_recent_scan(self):
        from googleapiclient.errors import HttpError
        err = HttpError(mock.Mock(status=404), b"expired")
        handler = mock.Mock()
        r = self._poll(handler, history_error=err, recent=("z",))
        r.recent.assert_called_once()
        self.assertEqual(handler.call_args[0][0]["id"], "z")

    def test_first_run_without_backfill_only_records_cursor(self):
        handler = mock.Mock()
        r = self._poll(handler, state={}, backfill_days=0)
        handler.assert_not_called()
        self.assertEqual(r.save.call_args[0][0]["history_id"], "150")


# =====================================================================
# Live model (off by default; needs Ollama running)
# =====================================================================
LIVE_CASES = [
    (mk("Your application to Boeing was received", "Boeing Careers", "no-reply@boeing.com",
        "Thanks for applying to the Analytics Intern role. We'll be in touch."), "application_status"),
    (mk("5 new jobs match your profile", "Handshake", "no-reply@joinhandshake.com",
        "Software Engineering Intern at Acme - apply by Nov 3. Data Analyst Co-op at Initech."), "job_posting"),
    (mk("Quick chat about our SWE intern role?", "Dana Ortiz", "dana@startup.io",
        "Saw your GitHub. We have an open internship, would you like to talk this week?"), "job_posting"),
    (mk("Interview invitation - Automation Engineer", "Asahi Kasei Bioprocess", "hr@asahi-kasei.com",
        "Please pick a time for your interview next week."), "application_status"),
    (mk("Your Amazon order has shipped", "Amazon", "ship@amazon.com", "Arriving Thursday."), "not_job"),
    (mk("CS 466 Problem Set 3 due Friday", "Canvas", "notifications@instructure.com",
        "Reminder: assignment due Friday at 11:59 PM."), "not_job"),
    (mk("Career fair next week", "Career Center", "careers@illinois.edu",
        "Come to the Illini Union to meet employers. Resume tips inside."), "not_job"),
    (mk("Undergraduate research assistant opening", "Dr. Patel", "patel@illinois.edu",
        "My lab is hiring an undergrad RA for spring. Reply with your CV by Oct 30."), "job_posting"),
]


@unittest.skipUnless(os.environ.get("RUN_LIVE") == "1", "set RUN_LIVE=1 to hit the real model")
class LiveModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ec.ensure_ready()
        except ec.ModelUnavailable as e:
            raise unittest.SkipTest(f"model server unavailable: {e}")

    def test_known_emails(self):
        for email, expected in LIVE_CASES:
            with self.subTest(subject=email["subject"]):
                r = ec.classify(email)
                self.assertEqual(r["label"], expected,
                                 f"got {r['label']} (conf={r['confidence']:.2f}): {r['reason']}")

    def test_extractor_on_a_digest(self):
        if ex is None:
            self.skipTest("extractor not importable")
        email = mk("5 new jobs match your profile", "Handshake", "no-reply@joinhandshake.com",
                   "Software Engineering Intern at Acme - apply by Nov 3.",
                   urls=[{"url": "https://app.joinhandshake.com/jobs/1", "text": "Acme SWE Intern"}])
        rows = ex.extract(email)
        self.assertTrue(rows, "extractor found nothing")
        self.assertIn("acme", rows[0]["company"].lower())


if __name__ == "__main__":
    unittest.main()