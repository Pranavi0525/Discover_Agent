"""M2.4 PII masking."""
import json
import sqlite3

import pytest

from discovery_agent import pii
from discovery_agent.storage import SqliteStore
from discovery_agent.tracking import JsonlRunTracker


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("PII_MASKING", "1")
    monkeypatch.setenv("PII_HASH_SALT", "test-salt")


@pytest.mark.parametrize("text,token", [
    ("mail me at asha.k+shop@example.co.in please", "[EMAIL]"),
    ("call 98765 43210 now", "[PHONE]"),
    ("call +91 98765-43210 now", "[PHONE]"),
    ("call 9876543210 now", "[PHONE]"),
    ("call +1 415 555 2671 now", "[PHONE]"),
    ("card 4111 1111 1111 1111 expires", "[CARD]"),
    ("aadhaar 2345 6789 0123", "[ID_NUMBER]"),
    ("deliver to pincode 600001", "[PIN]"),
    ("pin: 600001", "[PIN]"),
    ("zip 94105", "[PIN]"),
])
def test_masks_contact_pii(text, token):
    assert token in pii.mask_text(text)


@pytest.mark.parametrize("text", [
    "blue shirts under 150000",
    "find shoes under 2000 rupees",
    "budget 1,00,000 rupees",
    "order 600001 items",           # no cue word: not a PIN
    "Find blue shirts under 2000 rupees. Ignore my previous search",
])
def test_leaves_normal_text_alone(text):
    assert pii.mask_text(text) == text


def test_long_non_luhn_number_is_still_masked():
    assert "[NUMBER]" in pii.mask_text("ref 4111111111111112")


def test_tamil_digits_are_masked():
    assert "[PHONE]" in pii.mask_text("எண் \u0bef\u0bee\u0bed\u0bec\u0beb \u0bea\u0be9\u0be8\u0be7\u0be6")


def test_idempotent():
    once = pii.mask_text("a@b.com 9876543210 pin 600001")
    assert pii.mask_text(once) == once


def test_ids_urls_prices_untouched():
    decision = {
        "message": "mail a@b.com",
        "candidates": [{"product": {"id": "gid://shopify/Product/9876543210123",
                                    "url": "https://x.com/p/9876543210",
                                    "price": 1999.0}}],
        "diagnostics": {"unverified": [{"name": "delivery_city", "value": "Chennai"}]},
    }
    out = pii.mask_json_value(decision)
    assert out["message"] == "mail [EMAIL]"
    assert out["candidates"][0]["product"] == decision["candidates"][0]["product"]
    assert out["diagnostics"]["unverified"][0]["value"] == "Chennai"


def test_disabled(monkeypatch):
    monkeypatch.setenv("PII_MASKING", "0")
    assert pii.mask_text("a@b.com") == "a@b.com"
    assert pii.mask_event_for_ledger({"final_query": "a@b.com"}) == {"final_query": "a@b.com"}


def test_store_masks_but_keeps_city(tmp_path):
    store = SqliteStore(tmp_path / "t.db")
    sid = store.create_session()
    run = store.create_run(
        sid, "shirts, email me a@b.com",
        {"delivery_city": "Chennai", "preferences": "call 9876543210", "max_price": "2000"},
    )
    store.update_run(run, error="failed for a@b.com",
                     decision_json={"message": "hi 9876543210",
                                    "diagnostics": {"unverified": [{"name": "delivery_city", "value": "Chennai"}]}})
    store.log_step(run, 1, "audit", "ok", error="x a@b.com", payload={"final_query": "pin 600001"})
    row = store.get_run(run)
    assert row["query"] == "shirts, email me [EMAIL]"
    form = json.loads(row["form_inputs_json"])
    assert form == {"delivery_city": "Chennai", "preferences": "call [PHONE]", "max_price": "2000"}
    assert "[EMAIL]" in row["error"]
    assert "[PHONE]" in row["decision_json"] and "Chennai" in row["decision_json"]
    step = store.get_step_events(run)[0]
    assert "[EMAIL]" in step["error"] and "[PIN]" in step["payload_json"]


def test_ledger_hashes_city_and_masks(tmp_path):
    path = tmp_path / "l.jsonl"
    JsonlRunTracker(path).record({
        "run_id": "r1", "final_query": "shoes for Chennai, a@b.com",
        "unverified": [{"name": "delivery_city", "value": "Chennai"}],
        "attempt_log": [{"query": "shoes Chennai 9876543210"}],
    })
    text = path.read_text(encoding="utf-8")
    assert "Chennai" not in text and "a@b.com" not in text and "9876543210" not in text
    event = json.loads(text)
    assert event["unverified"][0]["value"].startswith("city#")
    assert event["run_id"] == "r1"


def test_city_hash_is_stable_and_salted(monkeypatch):
    a = pii.city_token(" chennai ")
    assert a == pii.city_token("Chennai")
    monkeypatch.setenv("PII_HASH_SALT", "other")
    assert a != pii.city_token("Chennai")


def _legacy_db(tmp_path):
    store = SqliteStore(tmp_path / "old.db")
    sid = store.create_session()
    run = store.create_run(sid, "x")
    with sqlite3.connect(store.path) as c:  # simulate rows written before masking existed
        c.execute("UPDATE runs SET query=?, form_inputs_json=?, error=? WHERE run_id=?",
                  ("mail a@b.com", json.dumps({"preferences": "9876543210", "delivery_city": "Chennai"}),
                   "boom a@b.com", run))
        c.execute("INSERT INTO step_events(run_id, attempt, step, status, error, payload_json, created_at)"
                  " VALUES (?,?,?,?,?,?,?)", (run, 1, "audit", "ok", None, '{"final_query": "pin 600001"}', "2026-01-01"))
    return store, run


def test_scrub_database(tmp_path):
    store, run = _legacy_db(tmp_path)
    assert pii.scrub_database(store.path, dry_run=True) == {"runs_changed": 1, "steps_changed": 1}
    assert store.get_run(run)["query"] == "mail a@b.com"          # dry run wrote nothing
    assert pii.scrub_database(store.path) == {"runs_changed": 1, "steps_changed": 1}
    row = store.get_run(run)
    assert row["query"] == "mail [EMAIL]" and "Chennai" in row["form_inputs_json"]
    assert pii.scrub_database(store.path) == {"runs_changed": 0, "steps_changed": 0}   # idempotent


def test_scrub_ledger(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps({"final_query": "a@b.com Chennai",
                                "unverified": [{"name": "delivery_city", "value": "Chennai"}]}) + "\n",
                    encoding="utf-8")
    assert pii.scrub_ledger(path)["changed"] == 1
    text = path.read_text(encoding="utf-8")
    assert "Chennai" not in text and "a@b.com" not in text
    assert pii.scrub_ledger(path)["changed"] == 0


def test_purge(tmp_path):
    store, run = _legacy_db(tmp_path)
    with sqlite3.connect(store.path) as c:
        c.execute("UPDATE runs SET created_at='2020-01-01T00:00:00+00:00' WHERE run_id=?", (run,))
    assert pii.purge_older_than(30, store.path, dry_run=True) == {"runs_deleted": 1}
    assert pii.purge_older_than(30, store.path) == {"runs_deleted": 1}
    assert store.get_run(run) is None and store.get_step_events(run) == []


def test_scrub_leaves_clean_rows_byte_for_byte(tmp_path):
    store = SqliteStore(tmp_path / "clean.db")
    sid = store.create_session()
    run = store.create_run(sid, "blue shirts under 2000")
    compact = '{"message":"ok","candidates":[]}'
    with sqlite3.connect(store.path) as c:
        c.execute("UPDATE runs SET decision_json=? WHERE run_id=?", (compact, run))
    assert pii.scrub_database(store.path) == {"runs_changed": 0, "steps_changed": 0}
    assert store.get_run(run)["decision_json"] == compact


def test_scrub_ledger_leaves_clean_lines_alone(tmp_path):
    path = tmp_path / "l.jsonl"
    line = json.dumps({"run_id": "r", "final_query": "blue shirts"})
    path.write_text(line + "\n", encoding="utf-8")
    assert pii.scrub_ledger(path) == {"lines": 1, "changed": 0}


def test_uuids_and_hashes_are_not_numbers():
    text = "DiscoveryDecision(run_id=UUID('dfb29d7c-f1234567890123-4a8b-9c06608af6b'), x='a1234567890123b')"
    assert pii.mask_text(text) == text
