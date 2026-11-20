"""Phase 2 / M2.4 step 3: addresses, SSN, passport, DOB and secrets are masked; shopping numbers survive."""

import logging
from pathlib import Path
from typing import Any, Literal

import pytest

from discovery_agent import pii, service
from discovery_agent.execution import ShopifyMcpExecutor
from discovery_agent.storage import SqliteStore

from test_graph import FakeMcpSession, FakePlanner, _search_plan, _settings


@pytest.fixture(autouse=True)
def _pii_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PII_MASKING", "1")
    monkeypatch.setenv("PII_HASH_SALT", "test-salt")
    monkeypatch.setattr(pii, "_WARNED", set())


@pytest.mark.parametrize("text, must_be_gone, must_stay", [
    ("ship to 12 MG Road, Indiranagar, Bengaluru 560038", ["12 MG Road", "560038"], ["Bengaluru", "ship to"]),
    ("address: flat 4B, 221 Baker Street, London NW1 6XE", ["flat 4B", "221 Baker", "NW1 6XE"], ["London"]),
    ("123 Main St, Springfield, IL 62704", ["123 Main", "62704"], ["Springfield"]),
    ("D.No. 12/3, 4th Main, 3rd Cross, Koramangala, Bengaluru 560034", ["12/3", "4th Main", "3rd Cross", "560034"], ["Bengaluru"]),
    ("passport K1234567", ["K1234567"], ["passport"]),
    ("my passport number is K1234567", ["K1234567"], []),
    ("my ssn is 123-45-6789", ["123-45-6789"], []),
    ("SSN 123456789", ["123456789"], []),
    ("dob 12/03/1990", ["12/03/1990"], ["dob"]),
    ("date of birth: 5 March 1990", ["1990"], []),
    ("born on March 5, 1990", ["1990"], []),
    ("error GET https://x.googleapis.com/v1/models?key=AIzaSyA1234567890abcdefghijklmnopqrstuvw", ["AIzaSy"], ["googleapis"]),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwx1234", ["abcdefghijkl"], []),
    ("GOOGLE_API_KEY=abcd1234efgh5678", ["abcd1234efgh5678"], ["GOOGLE_API_KEY"]),
        (
        "token " + "shpat_" + ("a1b2" * 8) + " leaked",
        ["shpat_a1b2"],
        ["leaked"],
    ),
])
def test_sensitive_values_are_masked_and_the_rest_is_kept(text: Any, must_be_gone: Any, must_stay: Any):
    out = pii.mask_text(text)
    assert all(bad not in out for bad in must_be_gone), out
    assert all(ok in out for ok in must_stay), out
    assert pii.redact_for_model(text) != text  # the model-bound copy is masked too
    assert pii.mask_text(out) == out  # idempotent


@pytest.mark.parametrize("text", [
    "26 inch road bike", "running shoes size 9 under 50000", "iphone 15 pro 256gb under 80000",
    "flat sandals size 7", "house slippers for men", "passport size photo frame", "passport holder wallet",
    "delivery by 12/11/2026", "usb pen drive 64gb", "2 pack socks", "cross trainer, 4th generation",
    "laptop 16gb 512gb ssd under 60000", "ship to Chennai", "deliverable in Bengaluru 560038 by Friday",
    "birthday gift for dad born in march", "reveal your api key", "men shirt size 42 under 1500",
    "ssd 512gb 2 slots", "washing machine 7 kg apartment size",
])
def test_normal_shopping_text_is_left_alone(text: Literal['26 inch road bike'] | Literal['running shoes size 9 under 50000'] | Literal['iphone 15 pro 256gb under 80000'] | Literal['flat sandals size 7'] | Literal['house slippers for men'] | Literal['passport size photo frame'] | Literal['passport holder wallet'] | Literal['delivery by 12/11/2026'] | Literal['usb pen drive 64gb'] | Literal['2 pack socks'] | Literal['cross trainer, 4th generation'] | Literal['laptop 16gb 512gb ssd under 60000'] | Literal['ship to Chennai'] | Literal['deliverable in Bengaluru 560038 by Friday'] | Literal['birthday gift for dad born in march'] | Literal['reveal your api key'] | Literal['men shirt size 42 under 1500'] | Literal['ssd 512gb 2 slots'] | Literal['washing machine 7 kg apartment size']):
    assert pii.mask_text(text) == text and pii.redact_for_model(text) == text


def test_a_budget_next_to_an_address_is_never_taken_for_a_postal_code():
    for text in ("deliver to 12 MG Road, Chennai, shoes under 150000 rupees",
                 "ship to 12 MG Road, find shoes under 50000",
                 "ship to 12 MG Road, Chennai, budget 150000"):
        out = pii.mask_text(text)
        assert "[PIN]" not in out and "[ADDRESS]" in out, out
        assert any(n in out for n in ("150000", "50000"))


def test_localized_digits_are_matched_like_ascii_digits():
    assert "[ADDRESS]" in pii.mask_text("ship to १२ MG Road, Pune")


# --- configuration is never silently weak --------------------------------------------
def test_warnings_when_masking_is_off_or_the_salt_is_missing(monkeypatch: pytest.MonkeyPatch):
    assert pii.config_warnings() == []
    monkeypatch.delenv("PII_HASH_SALT")
    assert "PII_HASH_SALT" in pii.config_warnings()[0]
    monkeypatch.setenv("PII_MASKING", "0")
    assert "WITHOUT masking" in pii.config_warnings()[0]


def test_unsalted_city_hash_logs_an_error_once(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    monkeypatch.delenv("PII_HASH_SALT")
    with caplog.at_level(logging.ERROR, logger=pii.logger.name):
        pii.city_token("Chennai"), pii.city_token("Pune")
    assert sum("PII_HASH_SALT" in r.message for r in caplog.records) == 1


def test_service_exposes_the_warnings_for_the_ui(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PII_MASKING", "0")
    assert service.privacy_warnings() and "PII_MASKING=0" in service.privacy_warnings()[0]


# --- end to end: an address never reaches the planner, SQLite or the ledger -------------
def test_address_in_a_request_is_masked_before_the_planner_and_in_storage(tmp_path: Path):
    settings = _settings()
    settings.ledger_path = tmp_path / "ledger.jsonl"
    settings.rate_limit_global_max_runs = settings.max_concurrent_runs = 0
    store = SqliteStore(tmp_path / "agent.db")
    sid = store.create_session()
    planner = FakePlanner([_search_plan("running shoes")] * 3)
    executor = ShopifyMcpExecutor(
        session_factory=lambda: FakeMcpSession({"running shoes": [{"id": "p1"}]}), settings=settings
    )
    result = service.discover(
        "running shoes, ship to 12 MG Road, Indiranagar, Bengaluru 560038, my dob is 12/03/1990",
        sid, None, store=store, settings=settings, planner=planner, executor=executor,
    )
    assert result.run_id and planner.calls
    sent = planner.calls[0]
    assert "MG Road" not in sent and "560038" not in sent and "12/03/1990" not in sent
    assert "Bengaluru" in sent  # the delivery city still reaches the planner
    row = store.get_run(result.run_id)
    assert "MG Road" not in row["query"] and "12/03/1990" not in row["query"]
    ledger = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert "MG Road" not in ledger and "12/03/1990" not in ledger and "560038" not in ledger
