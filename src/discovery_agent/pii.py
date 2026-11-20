"""Phase 2 / M2.4 - PII masking for everything the app writes to disk.

What is masked, and where
-------------------------
* Contact-type PII is replaced by a token everywhere it is stored: e-mail
  addresses, phone numbers (+91 and 10-digit), card numbers (13-19 digits that
  pass the Luhn check), Aadhaar-like 12-digit numbers, and PIN / ZIP codes that
  follow a cue word ("pin", "pincode", "zip").  None of these is needed to search.
* Step 3 adds: street-level addresses ("12 MG Road", "flat 4B", "12th Main") plus a
  postal code right after them; SSNs; passport numbers and dates of birth that follow
  a cue word ("passport", "dob", "date of birth", "born on"); and API keys / tokens
  that can appear in exception text.  The city and locality are kept.
  KNOWN LIMITS: a bare passport number or birth date with no cue word, and an address
  written without a number or street word ("near the temple, Green Villa"), are not
  caught.  Masking is a heuristic; it reduces exposure, it does not guarantee none.
* The delivery city is NOT masked in SQLite, because the policy check and the
  clarification re-run read it back.  In the JSONL ledger it is replaced by a
  keyed hash (set PII_HASH_SALT) and removed from free text in the same event.

Masking is structural: only fields that hold user-typed or user-derived text
are touched (see TEXT_KEYS).  Product ids, URLs and prices are never rewritten,
so a long numeric Shopify id cannot be mistaken for a phone or card number.

Switches (environment / .env):
    PII_MASKING=1        default on; 0 disables all masking
    PII_HASH_SALT=...    secret used to hash the city in the ledger
    PII_RETENTION_DAYS   used by ``--purge-days`` default (optional)

CLI (run from the repo root, venv active):
    python -m discovery_agent.pii --check "call me on 98765 43210"
    python -m discovery_agent.pii --scrub-db --scrub-ledger --dry-run
    python -m discovery_agent.pii --scrub-db --scrub-ledger
    python -m discovery_agent.pii --purge-days 30
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import unicodedata
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:  # the app loads .env through config; do the same for CLI use
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

logger = logging.getLogger(__name__)

_ON = {"1", "true", "yes", "on"}
_WARNED: set[str] = set()

# String values under these keys are scanned.  Everything else (ids, urls,
# prices, timestamps) is left exactly as it is.
TEXT_KEYS = frozenset({
    "query", "final_query", "raw_query", "message", "preferences", "note", "error",
    "reason", "detail", "value", "corrections", "delivery_city", "merchant", "size", "text",
})
CITY_KEYS = frozenset({"delivery_city"})


def enabled() -> bool:
    return os.getenv("PII_MASKING", "1").strip().lower() in _ON


# --- patterns ----------------------------------------------------------------------
_EMAIL = re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+")
_B0, _B1 = r"(?<![A-Za-z\d])", r"(?![A-Za-z\d])"  # not glued to letters or more digits
_CARD = re.compile(_B0 + r"(?:\d[ -]?){12,18}\d" + _B1)
_AADHAAR = re.compile(_B0 + r"[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}" + _B1)
_PHONE_IN = re.compile(_B0 + r"(?:\+|00)?(?:91[ -]?)?0?[6-9]\d{4}[ -]?\d{5}" + _B1)
_PHONE_INTL = re.compile(r"(?<!\d)\+\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}(?!\d)")
_PIN = re.compile(r"(?i)\b(pin(?:\s*code)?|postal\s*code|zip(?:\s*code)?)\b(\W{0,3})(\d{5,6})(?!\d)")


# --- Phase 2 step 3 patterns --------------------------------------------------------
# Deliberately cue-based: a shopping query is full of numbers that must survive
# ("size 9", "512gb", "under 150000", "iphone 15"), so a bare number or date is never
# masked as an identity document.  The cost is that an uncued passport or birth date
# is not caught; see the module notes.
_SSN = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
_SSN_CUE = re.compile(
    r"(?i)\b(ssn|social\s+security(?:\s+(?:number|no\.?|#))?)\b((?:\s+is)?\W{0,4})(\d{3}[ -]?\d{2}[ -]?\d{4})(?!\d)"
)
_PASSPORT = re.compile(
    r"(?i)\b(passport(?:\s*(?:no\.?|number|num|#))?)\b((?:\s+is)?\W{0,4})((?=[A-Z0-9]*\d)[A-Z0-9]{6,9})(?![A-Za-z0-9])"
)
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = (
    r"(?:\d{1,2}[/.\-]\d{1,2}[/.\-](?:\d{4}|\d{2})(?!\d)|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}(?!\d)"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH},?\s+\d{{4}}|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})"
)
_DOB = re.compile(r"(?i)\b(dob|d\.o\.b\.?|date\s+of\s+birth|birth\s*date|born\s+on)\b((?:\s+is)?\W{0,4})" + _DATE)

_STREET_WORDS = (
    r"(?:road|rd|street|st|lane|ln|avenue|ave|boulevard|blvd|marg|nagar|colony|layout|enclave|"
    r"residency|apartments?|society)"
)
_NOT_AN_ADDRESS = (
    r"(?!\s+(?:bike|bikes|shoe|shoes|running|trail|cycling|cycle|tyres?|tires?|racing|jacket|boots?|"
    r"sneakers?|bicycle|helmet|gloves?|glasses|bag|backpack|wear|cleats?|spikes?))"
)
# "7 kg apartment", "2 bhk apartment": a measurement or count word after the number is not a street.
_NOT_A_UNIT = (
    r"(?!(?:kg|kgs|g|gm|gb|tb|mb|ml|l|ltr|litres?|liters?|inch|inches|cm|mm|m|ton|tons|w|watts?|pack|packs|"
    r"pcs|pieces|pairs?|bhk|star|hp|mah|mp|hz|x)\b)"
)
_ADDR_STREET = re.compile(
    r"(?i)(?<![\w/-])\d{1,5}[A-Za-z]?(?:[-/]\d{1,4}[A-Za-z]?)?,?\s+" + _NOT_A_UNIT + r"(?:[A-Za-z][\w.'-]*\s+){0,3}?"
    + _STREET_WORDS + r"\b\.?" + _NOT_AN_ADDRESS
    + r"|(?<![\w/-])\d{1,3}(?:st|nd|rd|th)\s+(?:main|cross|block|phase|stage|floor)\b"
)
_ADDR_DESIGNATOR = re.compile(
    r"(?i)\b(?:flat|house|door|plot|apt|apartment|villa|d\.\s?no\.?|h\.\s?no\.?)"
    r"\s*(?:no\.?|number|#)?\s*[:\-]?\s*[\w/-]*\d[\w/-]*"
)
# A postal code is only masked when it sits right after a street-level hit, so a budget
# such as "under 150000" is never mistaken for a PIN.
_POSTAL_TAIL = re.compile(
    r"(?i)(?<![\w.])(\d{6}|\d{5}(?:-\d{4})?|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})(?![\w])"
)
_PRICE_BEFORE = re.compile(
    r"(?i)(?:under|below|within|budget|max(?:imum)?|upto|up to|less than|at most|price|rs\.?|inr|usd|₹|\$|£|€)\W{0,3}$"
)
_PRICE_AFTER = re.compile(r"(?i)^\s*(?:rupees|rs\b|inr|usd|dollars|bucks|₹|\$|€|£)")

# API keys and tokens that can leak through exception text (``planner failed: {exc}``).
_SECRET = re.compile(
    r"AIza[0-9A-Za-z_\-]{30,}|\b(?:shpat|shpca|shpss|shppa)_[0-9a-fA-F]{16,}\b|\bsk-[A-Za-z0-9_\-]{20,}"
    r"|\bhf_[A-Za-z0-9]{20,}\b|(?i:bearer)\s+[A-Za-z0-9._~+/=\-]{20,}"
)
_SECRET_PARAM = re.compile(r"(?i)([?&;]\s*(?:key|api[_-]?key|access[_-]?token|token|auth)=)[^&\s\"']+")
_SECRET_ASSIGN = re.compile(
    r"(?i)\b((?:google_|tavily_|shopify_|openai_)?(?:api[_-]?key|secret|access[_-]?token|"
    r"storefront_access_token))(\s*[=:]\s*)[^\s,;\"']{8,}"
)


def _mask_secrets(text: str) -> str:
    out = _SECRET.sub("[SECRET]", text)
    out = _SECRET_PARAM.sub(lambda m: f"{m.group(1)}[SECRET]", out)
    return _SECRET_ASSIGN.sub(lambda m: f"{m.group(1)}{m.group(2)}[SECRET]", out)


def _mask_address(text: str) -> str:
    """Mask street-level address pieces and the postal code that follows them.

    The delivery city and locality are kept: the search and the policy check need the
    city.  Only the house/street part and a postal code right after it are replaced.
    """
    norm = _ascii_digits(text)
    spans: list[tuple[int, int, str]] = []
    for pattern in (_ADDR_DESIGNATOR, _ADDR_STREET):
        spans.extend((m.start(), m.end(), "[ADDRESS]") for m in pattern.finditer(norm))
    if not spans:
        return text
    for start, end, _ in list(spans):
        tail = norm[end:end + 90]
        for m in _POSTAL_TAIL.finditer(tail):
            a, b = end + m.start(1), end + m.end(1)
            if _PRICE_BEFORE.search(norm[max(0, a - 24):a]) or _PRICE_AFTER.match(norm[b:b + 14]):
                continue
            spans.append((a, b, "[PIN]"))
    spans.sort()
    out, last = [], 0
    for start, end, token in spans:
        if start < last:  # overlaps an earlier replacement
            continue
        out.append(text[last:start])
        out.append(token)
        last = end
    out.append(text[last:])
    return "".join(out)


def _mask_identity(text: str) -> str:
    """SSN, passport and date of birth (cue-based), then street addresses."""
    out = _sub(_SSN_CUE, lambda m: f"{m.group(1)}{m.group(2)}[ID_NUMBER]", text)
    out = _sub(_SSN, lambda m: "[ID_NUMBER]", out)
    out = _sub(_PASSPORT, lambda m: f"{m.group(1)}{m.group(2)}[PASSPORT]", out)
    out = _sub(_DOB, lambda m: f"{m.group(1)}{m.group(2)}[DOB]", out)
    return _mask_address(out)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0
def _luhn_only(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[CARD]" if _luhn(digits) else match.group(0)


def redact_for_model(text: Any) -> Any:
    """Mask only what a product search never needs: e-mail, Luhn-valid cards,
    phones and Aadhaar-like numbers.  Unlike mask_text it leaves other long digit
    runs (barcodes) and PIN codes alone, so the search still works."""
    if not isinstance(text, str) or not text or not enabled():
        return text
    out = _mask_secrets(text)
    out = _sub(_EMAIL, lambda m: "[EMAIL]", out)
    out = _mask_identity(out)
    out = _sub(_CARD, _luhn_only, out)
    out = _sub(_AADHAAR, lambda m: "[ID_NUMBER]", out)
    out = _sub(_PHONE_IN, lambda m: "[PHONE]", out)
    out = _sub(_PHONE_INTL, lambda m: "[PHONE]", out)
    return out


def _card_sub(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if _luhn(digits):
        return "[CARD]"
    # Not a card.  A plain run of 13+ digits is still never a budget; separated
    # groups are left to the Aadhaar / phone rules.
    return "[NUMBER]" if raw == digits else raw


def _ascii_digits(text: str) -> str:
    """Same length as ``text``, with Tamil/Hindi/other decimal digits as 0-9 (for matching only)."""
    return "".join(
        str(unicodedata.decimal(c)) if c.isdecimal() and not c.isascii() else c for c in text
    )


def _sub(pattern: re.Pattern[str], repl, text: str) -> str:
    """``pattern.sub`` that matches on ASCII-digit text but rewrites the original text."""
    norm = _ascii_digits(text)
    out, last = [], 0
    for m in pattern.finditer(norm):
        out.append(text[last:m.start()])
        out.append(repl(m))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def mask_text(text: Any) -> Any:
    """Replace contact-type PII in one string.  Non-strings pass through."""
    if not isinstance(text, str) or not text or not enabled():
        return text
    out = _mask_secrets(text)
    out = _sub(_EMAIL, lambda m: "[EMAIL]", out)
    out = _mask_identity(out)
    out = _sub(_CARD, _card_sub, out)
    out = _sub(_AADHAAR, lambda m: "[ID_NUMBER]", out)
    out = _sub(_PHONE_IN, lambda m: "[PHONE]", out)
    out = _sub(_PHONE_INTL, lambda m: "[PHONE]", out)
    out = _sub(_PIN, lambda m: f"{m.group(1)}{m.group(2)}[PIN]", out)
    return out


def config_warnings() -> list[str]:
    """Plain sentences for every PII protection that is off or weaker than intended."""
    if not enabled():
        return ["PII_MASKING=0: personal data is stored in SQLite and the ledger, and sent to "
                "Gemini, WITHOUT masking."]
    if not os.getenv("PII_HASH_SALT", "").strip():
        return ["PII_HASH_SALT is not set: city hashes in the ledger are unsalted and reproducible."]
    return []


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.error("PII: %s", message)


def log_config_warnings() -> list[str]:
    """Log each weakened-protection warning once per process and return them."""
    warnings = config_warnings()
    for i, message in enumerate(warnings):
        _warn_once(f"cfg{i}:{message[:20]}", message)
    return warnings


def city_token(city: str) -> str:
    """Keyed hash of a city: lets you count by city without storing the name."""
    if city.startswith("city#"):
        return city  # already a token
    salt = os.getenv("PII_HASH_SALT", "").encode("utf-8")
    if not salt:
        _warn_once("salt", "PII_HASH_SALT is not set: city hashes in the ledger are unsalted, so "
                   "anyone can reproduce them from a list of city names. Set a long random secret.")
    digest = hmac.new(salt, city.strip().casefold().encode("utf-8"), hashlib.sha256).hexdigest()
    return f"city#{digest[:10]}"


# --- structured masking ------------------------------------------------------------
def mask_obj(obj: Any, keys: frozenset[str] | None = TEXT_KEYS, *, city: str = "keep",
             _under: bool = False) -> Any:
    """Mask a dict/list tree.

    ``keys=None`` masks every string (use for flat, user-typed form values).
    Otherwise only strings under ``keys`` are masked.  ``city="hash"`` replaces
    the delivery city with ``city_token``.
    """
    if not enabled():
        return obj
    return _walk(obj, keys, city, keys is None)


def _walk(obj: Any, keys: frozenset[str] | None, city: str, under: bool) -> Any:
    if isinstance(obj, str):
        return mask_text(obj) if under else obj
    if isinstance(obj, Mapping):
        city_item = obj.get("name") == "delivery_city"
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if city == "hash" and isinstance(v, str) and (ks in CITY_KEYS or (city_item and ks == "value")):
                out[k] = city_token(v)
            else:
                out[k] = _walk(v, keys, city, keys is None or ks in keys)
        return out
    if isinstance(obj, (list, tuple)):
        return [_walk(i, keys, city, under) for i in obj]
    return obj


def mask_json_value(value: Any) -> Any:
    """Prepare a value bound for a ``*_json`` column.  Returns a plain object (or None)."""
    if value is None or not enabled():
        return value
    if isinstance(value, str):
        try:
            return mask_obj(json.loads(value))
        except ValueError:
            return mask_text(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return mask_obj(value)


def _cities_in(obj: Any, found: set[str]) -> None:
    if isinstance(obj, Mapping):
        if obj.get("name") == "delivery_city" and isinstance(obj.get("value"), str):
            found.add(obj["value"].strip())
        for k, v in obj.items():
            if str(k) in CITY_KEYS and isinstance(v, str):
                found.add(v.strip())
            _cities_in(v, found)
    elif isinstance(obj, (list, tuple)):
        for i in obj:
            _cities_in(i, found)


def _replace_terms(obj: Any, pattern: re.Pattern[str]) -> Any:
    if isinstance(obj, str):
        return pattern.sub(lambda m: city_token(m.group(0)), obj)
    if isinstance(obj, Mapping):
        return {k: _replace_terms(v, pattern) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_replace_terms(i, pattern) for i in obj]
    return obj


def mask_event_for_ledger(event: Mapping[str, Any]) -> dict[str, Any]:
    """JSONL ledger rule: mask contact PII, hash the city, drop the city from free text."""
    if not enabled():
        return dict(event)
    cities: set[str] = set()
    _cities_in(event, cities)
    out = mask_obj(dict(event), city="hash")
    cities = {c for c in cities if len(c) >= 2 and not c.startswith("city#")}
    if cities:
        pattern = re.compile("|".join(re.escape(c) for c in sorted(cities, key=len, reverse=True)), re.I)
        for key in list(out):
            if key in {"final_query", "message"} or key in {"corrections", "attempt_log"}:
                out[key] = _replace_terms(out[key], pattern)
    return out


# --- clean-up of data that is already on disk --------------------------------------
_RUN_JSON = ("form_inputs_json", "intent_json", "plan_json", "decision_json")


def _db_path() -> Path:
    return Path(os.getenv("DISCOVERY_DB_PATH") or "data/agent.db")


def _ledger_path() -> Path:
    return Path(os.getenv("DISCOVERY_LEDGER_PATH") or "data/discovery-runs.jsonl")


def _scrub_json(text: str | None, *, flat: bool) -> str | None:
    if not text:
        return text
    try:
        data = json.loads(text)
    except ValueError:
        return mask_text(text)
    masked = mask_obj(data, None if flat else TEXT_KEYS)
    if masked == data:
        return text  # nothing found: keep the stored bytes (no re-formatting)
    return json.dumps(masked, ensure_ascii=False, default=str)


def _context(text: str | None) -> str:
    """A short, already-masked snippet around the first token, for dry-run output."""
    if not text:
        return ""
    m = re.search(r"\[(?:EMAIL|PHONE|CARD|NUMBER|ID_NUMBER|PIN|ADDRESS|PASSPORT|DOB|SECRET)\]", text)
    if not m:
        return text[:60]
    return text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")


def scrub_database(path: Path | None = None, *, dry_run: bool = False) -> dict[str, int]:
    """Mask rows written before masking existed.  Idempotent."""
    path = path or _db_path()
    counts = {"runs_changed": 0, "steps_changed": 0}
    if not path.exists():
        return counts
    shown = 0
    with closing(sqlite3.connect(str(path), timeout=30)) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT * FROM runs").fetchall():
            new = {"query": mask_text(row["query"]), "error": mask_text(row["error"])}
            for col in _RUN_JSON:
                new[col] = _scrub_json(row[col], flat=(col == "form_inputs_json"))
            changed = {c: v for c, v in new.items() if v != row[c]}
            if changed:
                counts["runs_changed"] += 1
                if dry_run and shown < 5:
                    shown += 1
                    col = next(iter(changed))
                    print(f"  run {row['run_id'][:8]} {col}: ...{_context(changed[col])}...")
                if not dry_run:
                    sets = ", ".join(f"{c}=?" for c in changed)
                    conn.execute(f"UPDATE runs SET {sets} WHERE run_id=?", (*changed.values(), row["run_id"]))
        for row in conn.execute("SELECT id, error, payload_json FROM step_events").fetchall():
            new_err, new_payload = mask_text(row["error"]), _scrub_json(row["payload_json"], flat=False)
            if new_err != row["error"] or new_payload != row["payload_json"]:
                counts["steps_changed"] += 1
                if not dry_run:
                    conn.execute("UPDATE step_events SET error=?, payload_json=? WHERE id=?",
                                 (new_err, new_payload, row["id"]))
        if not dry_run:
            conn.commit()
    return counts


def scrub_ledger(path: Path | None = None, *, dry_run: bool = False) -> dict[str, int]:
    """Rewrite the JSONL ledger with the ledger rule.  No backup is kept (it would hold the raw data)."""
    path = path or _ledger_path()
    counts = {"lines": 0, "changed": 0}
    if not path.exists():
        return counts
    out_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        counts["lines"] += 1
        try:
            event = json.loads(line)
        except ValueError:
            out_lines.append(mask_text(line))
            continue
        masked = mask_event_for_ledger(event)
        if masked == event:
            out_lines.append(line)  # untouched
            continue
        counts["changed"] += 1
        if dry_run and counts["changed"] <= 5:
            keys = sorted(k for k in masked if masked[k] != event.get(k))
            print(f"  ledger line {counts['lines']}: would change {keys}")
        out_lines.append(json.dumps(masked, default=str))
    if not dry_run and counts["changed"]:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return counts


def purge_older_than(days: int, path: Path | None = None, *, dry_run: bool = False) -> dict[str, int]:
    """Delete runs (and their steps and feedback) older than ``days``."""
    path = path or _db_path()
    counts = {"runs_deleted": 0}
    if not path.exists() or days <= 0:
        return counts
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with closing(sqlite3.connect(str(path), timeout=30)) as conn:
        ids = [r[0] for r in conn.execute("SELECT run_id FROM runs WHERE created_at < ?", (cutoff,))]
        counts["runs_deleted"] = len(ids)
        if ids and not dry_run:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                conn.execute(f"DELETE FROM step_events WHERE run_id IN ({marks})", chunk)
                conn.execute(f"DELETE FROM feedback WHERE run_id IN ({marks})", chunk)
                conn.execute(f"DELETE FROM runs WHERE run_id IN ({marks})", chunk)
            conn.execute("DELETE FROM request_dedup WHERE claimed_at < ?", (cutoff,))
            conn.commit()
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m discovery_agent.pii", description=__doc__.split("\n")[0])
    ap.add_argument("--check", metavar="TEXT", help="show how TEXT would be masked")
    ap.add_argument("--scrub-db", action="store_true", help="mask existing SQLite rows")
    ap.add_argument("--scrub-ledger", action="store_true", help="mask existing JSONL ledger lines")
    ap.add_argument("--purge-days", type=int, metavar="N", help="delete runs older than N days")
    ap.add_argument("--dry-run", action="store_true", help="count what would change, write nothing")
    args = ap.parse_args(argv)
    if args.check is not None:
        print(mask_text(args.check))
    if args.scrub_db:
        print("database:", scrub_database(dry_run=args.dry_run))
    if args.scrub_ledger:
        print("ledger:", scrub_ledger(dry_run=args.dry_run))
    if args.purge_days:
        print("purge:", purge_older_than(args.purge_days, dry_run=args.dry_run))
    if not (args.check is not None or args.scrub_db or args.scrub_ledger or args.purge_days):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
