"""Store-policy check for the two constraints the catalog cannot answer.

"Replaceable / returnable" and "deliverable in <city>" are not product data.
They live in a Shopify store's policy pages:

    https://<store>/policies/refund-policy
    https://<store>/policies/shipping-policy

This module reads those pages with plain keyword rules (no LLM) and reports,
PER PRODUCT, one of four states for each requested constraint:

* confirmed    - the policy text says so; the sentence is quoted as evidence
* contradicted - the policy says the opposite ("all sales are final")
* not_found    - the page was read but does not mention it
* unchecked    - the page could not be fetched (never treated as a pass)

Honesty rule (same as constraints.py): "confirmed" means "the store's own
policy text says so", not "guaranteed".  True delivery to a city needs a
postal code and a checkout call, which is out of scope for Phase 1.

Design notes
* Pure functions (``analyze_returns`` / ``analyze_shipping``) do the reading, so
  they are tested against saved policy text with no network.
* ``PolicyChecker`` does the fetching through an injectable ``fetcher``; tests
  pass a fake.  Pages are cached per store host, so ten shoes from one store
  cost two fetches, not twenty.
* Ranking is soft: confirmed products move up, contradicted ones are dropped
  (unless that would leave nothing), everything else keeps its order.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from .constraints import _CITY_ALIASES, canonical_city, city_country
from .schemas import Product, ScoredProduct

CONFIRMED = "confirmed"
CONTRADICTED = "contradicted"
NOT_FOUND = "not_found"
UNCHECKED = "unchecked"

REFUND_PATH = "/policies/refund-policy"
SHIPPING_PATH = "/policies/shipping-policy"

_MAX_TEXT_CHARS = 200_000
_EVIDENCE_CHARS = 180


# --------------------------------------------------------------------- results
@dataclass(frozen=True)
class PolicyFinding:
    status: str
    detail: str
    evidence: str | None = None
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ text tools
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAG = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|td|table|section)\b[^>]*>", re.IGNORECASE)


def html_to_text(raw: str) -> str:
    """Visible text of an HTML page, one sentence-ish line per block."""
    raw = raw[:_MAX_TEXT_CHARS * 3]
    raw = _SCRIPT_STYLE.sub(" ", raw)
    raw = _BLOCK_TAG.sub("\n", raw)
    raw = _TAG.sub(" ", raw)
    text = html.unescape(raw)
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)[:_MAX_TEXT_CHARS]


# Phrases that contain a negation word but are not a refusal.
_FALSE_NEGATIVES = re.compile(
    r"no[- ]questions?[- ]asked|no hassle|no restocking fees?|no (?:additional|extra|hidden) "
    r"(?:cost|charges?|fees?)|at no (?:extra |additional )?(?:cost|charge)|no cost|"
    r"return address|no shipping (?:cost|charges?|fees?)",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(no|not|never|cannot|can't|cant|unable|without|none|neither|nor)\b|n't\b|\bnon[- ]|"
    r"final sale|sales? (?:are|is) final",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(r"\b(except|unless|other than|apart from|only if)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*[\n\u2022|]\s*")
_CLAUSE_SPLIT = re.compile(r",|;|\s+but\s+|\s+however\s+|\s+though\s+", re.IGNORECASE)

_RETURN_TERMS: dict[str, re.Pattern[str]] = {
    "exchange": re.compile(r"\b(exchang\w*|replac\w*)", re.IGNORECASE),
    "return": re.compile(r"\breturn(?:s|ed|able)?\b", re.IGNORECASE),
    "refund": re.compile(r"\brefund\w*|\bmoney[- ]back\b", re.IGNORECASE),
}
_DAYS = re.compile(r"\b(\d{1,3})[\s-]*(?:calendar |business |working )?days?\b", re.IGNORECASE)


def _trim(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _EVIDENCE_CHARS else text[: _EVIDENCE_CHARS - 1].rstrip() + "\u2026"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(_FALSE_NEGATIVES.sub(" ", text)) if s.strip()]


def analyze_returns(
    text: str, *, url: str | None = None, prefer: tuple[str, ...] = ("exchange", "return", "refund")
) -> PolicyFinding:
    """Does this refund policy offer returns / exchange / replacement?

    Works clause by clause so "no returns, exchange only" reads as *returns
    refused, exchange offered* rather than one confusing negative.
    """
    positives: list[tuple[str, str, str]] = []  # (kind, clause, sentence)
    negatives: list[tuple[str, str]] = []
    restricted = False
    for sentence in _sentences(text):
        for clause in _CLAUSE_SPLIT.split(sentence):
            kinds = [k for k, pat in _RETURN_TERMS.items() if pat.search(clause)]
            if not kinds:
                continue
            if _NEGATION.search(clause):
                if _CONDITIONAL.search(sentence):
                    restricted = True  # "no returns except defects": not a blanket refusal
                else:
                    negatives.append((kinds[0], clause))
            else:
                positives.append((kinds[0], clause, sentence))

    if positives:
        positives.sort(key=lambda item: prefer.index(item[0]) if item[0] in prefer else len(prefer))
        kind, clause, sentence = positives[0]
        days = _DAYS.search(sentence)
        window = f" within {days.group(1)} days" if days else ""
        label = {"exchange": "Exchange/replacement", "return": "Returns", "refund": "Refunds"}[kind]
        return PolicyFinding(CONFIRMED, f"{label} mentioned{window} in the store policy.",
                             _trim(sentence), url)
    if negatives:
        return PolicyFinding(CONTRADICTED, "The store policy says returns/exchanges are not offered.",
                             _trim(negatives[0][1]), url)
    if restricted:
        return PolicyFinding(NOT_FOUND, "Policy mentions conditions or exceptions; read it before buying.",
                             None, url)
    return PolicyFinding(NOT_FOUND, "The store policy does not mention returns or exchange.", None, url)


# ------------------------------------------------------------------- shipping
_SHIP_TERM = re.compile(r"\b(ship\w*|deliver\w*|dispatch\w*)", re.IGNORECASE)
_WORLDWIDE = re.compile(
    r"\b(worldwide|world[- ]wide|globally|internationally|international (?:shipping|delivery|orders))\b",
    re.IGNORECASE,
)
_COUNTRY_TERMS: dict[str, re.Pattern[str]] = {
    "IN": re.compile(r"\b(india|pan[- ]india|nationwide|all over india|across india)\b", re.IGNORECASE),
}


def analyze_shipping(
    text: str, *, city: str, country: str | None = None, url: str | None = None
) -> PolicyFinding:
    """Does this shipping policy cover ``city`` (or its country / worldwide)?"""
    canonical = canonical_city(city) or city
    aliases = _CITY_ALIASES.get(canonical, (canonical.lower(),))
    city_pat = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)
    country_pat = _COUNTRY_TERMS.get((country or city_country(canonical) or "").upper())

    best: tuple[int, str, str] | None = None  # (strength, detail, sentence)
    refusal: str | None = None
    for sentence in _sentences(text):
        if not _SHIP_TERM.search(sentence):
            continue
        for clause in _CLAUSE_SPLIT.split(sentence):
            if not _SHIP_TERM.search(clause):
                continue
            hit_city = bool(city_pat.search(clause))
            hit_country = bool(country_pat and country_pat.search(clause))
            hit_world = bool(_WORLDWIDE.search(clause))
            if not (hit_city or hit_country or hit_world):
                continue
            if _NEGATION.search(clause):
                if (hit_city or hit_country) and not _CONDITIONAL.search(sentence):
                    refusal = refusal or clause
                continue
            if hit_city:
                cand = (3, f"The shipping policy mentions delivery to {canonical}.", sentence)
            elif hit_country:
                cand = (2, "The shipping policy says it ships across the country.", sentence)
            else:
                cand = (1, "The shipping policy says it ships worldwide.", sentence)
            if best is None or cand[0] > best[0]:
                best = cand
    if best:
        return PolicyFinding(CONFIRMED, best[1], _trim(best[2]), url)
    if refusal:
        return PolicyFinding(CONTRADICTED, f"The shipping policy says it does not ship to {canonical}.",
                             _trim(refusal), url)
    return PolicyFinding(NOT_FOUND, f"The shipping policy does not mention {canonical}.", None, url)


# -------------------------------------------------------------------- fetching
class Fetcher(Protocol):
    def fetch(self, url: str) -> str | None:  # None = could not fetch
        ...


def _is_public_https(url: str) -> bool:
    """Only fetch https pages on a real public hostname (the URL comes from catalog data)."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host or host == "localhost":
        return False
    if host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        ipaddress.ip_address(host)
        return False  # a bare IP address is never a store
    except ValueError:
        return True


def _resolves_public(host: str) -> bool:
    """False unless every address ``host`` resolves to is a public one.

    ``_is_public_https`` only looks at the spelling of the host.  A catalog entry
    can point a normal-looking name (a custom store domain) at 127.0.0.1 or a LAN
    address, so the resolved addresses are checked too.  (Known limit: the name is
    resolved again when the connection is made, so a DNS answer that changes
    between the two lookups is not caught.)
    """
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        try:
            if not ipaddress.ip_address(info[4][0]).is_global:
                return False
        except ValueError:
            return False
    return True


class HttpxFetcher:
    """Small, strict GET.

    https only; public hosts only (spelling AND resolved address, re-checked on
    every redirect hop); short per-call timeout plus a total deadline; the body is
    streamed and cut off at ``max_bytes`` instead of being downloaded in full;
    HTML/text only.
    """

    def __init__(
        self,
        timeout: float = 6.0,
        max_bytes: int = 600_000,
        *,
        total_timeout: float = 10.0,
        max_redirects: int = 3,
        transport: Any = None,  # tests pass httpx.MockTransport
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.total_timeout = total_timeout
        self.max_redirects = max_redirects
        self.transport = transport

    def _allowed(self, url: str) -> bool:
        return _is_public_https(url) and _resolves_public(urlparse(url).hostname or "")

    def fetch(self, url: str) -> str | None:
        if not self._allowed(url):
            return None
        import httpx

        try:
            deadline = time.monotonic() + self.total_timeout
            with httpx.Client(
                timeout=self.timeout,
                follow_redirects=False,  # every hop is checked here instead
                headers={"User-Agent": "DiscoveryAgent/0.1 (policy check)"},
                transport=self.transport,
            ) as client:
                for _ in range(self.max_redirects + 1):
                    if not self._allowed(url):
                        return None
                    with client.stream("GET", url) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                return None
                            url = str(response.url.join(location))
                            continue
                        if response.status_code != 200:
                            return None
                        ctype = response.headers.get("content-type", "")
                        if "html" not in ctype and "text" not in ctype:
                            return None
                        body = bytearray()
                        for chunk in response.iter_bytes():
                            body += chunk
                            if len(body) >= self.max_bytes or time.monotonic() > deadline:
                                break
                        text = bytes(body[: self.max_bytes]).decode(response.encoding or "utf-8", errors="replace")
                        return html_to_text(text)
            return None  # too many redirects
        except Exception:  # noqa: BLE001 - a failed check must never break a search
            return None


def store_host(product: Product) -> str | None:
    url = str(product.url)
    return (urlparse(url).hostname or "").lower() if _is_public_https(url) else None


# --------------------------------------------------------------------- checker
class PolicyChecker:
    def __init__(
        self,
        fetcher: Fetcher | None = None,
        *,
        ttl_seconds: float = 3600.0,
        max_hosts: int = 6,
        max_workers: int = 4,
    ) -> None:
        self.fetcher: Fetcher = fetcher or HttpxFetcher()
        self.ttl = ttl_seconds
        self.max_hosts = max_hosts
        self.max_workers = max_workers
        self._cache: dict[tuple[str, str], tuple[float, str | None]] = {}
        self._lock = threading.Lock()

    def _page(self, host: str, path: str) -> str | None:
        key = (host, path)
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < self.ttl:
                return hit[1]
        text = self.fetcher.fetch(f"https://{host}{path}")
        with self._lock:
            self._cache[key] = (now, text)
        return text

    def _check_host(self, host: str, city: str | None, return_required: bool) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if return_required:
            url = f"https://{host}{REFUND_PATH}"
            text = self._page(host, REFUND_PATH)
            out["returns"] = (
                analyze_returns(text, url=url)
                if text is not None
                else PolicyFinding(UNCHECKED, "Could not read the store's return policy page.", None, url)
            ).to_dict()
        if city:
            url = f"https://{host}{SHIPPING_PATH}"
            text = self._page(host, SHIPPING_PATH)
            out["delivery"] = (
                analyze_shipping(text, city=city, url=url)
                if text is not None
                else PolicyFinding(UNCHECKED, "Could not read the store's shipping policy page.", None, url)
            ).to_dict()
        return out

    def check_products(
        self, products: Iterable[Product], *, city: str | None, return_required: bool
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """{product_id: {"returns": finding, "delivery": finding}} (only requested keys)."""
        products = list(products)
        hosts: list[str] = []
        for product in products:
            host = store_host(product)
            if host and host not in hosts:
                hosts.append(host)
        hosts = hosts[: self.max_hosts]
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            checked = dict(zip(hosts, pool.map(lambda h: self._check_host(h, city, return_required), hosts)))
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for product in products:
            host = store_host(product)
            if host in checked:
                result[product.id] = checked[host]
            else:
                skipped = PolicyFinding(UNCHECKED, "Store policy was not checked for this product.").to_dict()
                result[product.id] = {
                    **({"returns": skipped} if return_required else {}),
                    **({"delivery": skipped} if city else {}),
                }
        return result


def summarize_policy(policy: dict[str, Any] | None, product_ids: Iterable[str]) -> dict[str, dict[str, int]]:
    """{"returns": {"confirmed": 2, "total": 3}, ...} over the products actually shown."""
    findings = (policy or {}).get("products") or {}
    ids = list(product_ids)
    out: dict[str, dict[str, int]] = {}
    for key in ("returns", "delivery"):
        rows = [findings[i][key] for i in ids if key in findings.get(i, {})]
        if rows:
            out[key] = {"confirmed": sum(r["status"] == CONFIRMED for r in rows), "total": len(ids)}
    return out


# --------------------------------------------------------------------- ranking
def apply_policy(
    ranked: list[ScoredProduct],
    *,
    city: str | None,
    return_required: bool,
    checker: PolicyChecker,
    top_n: int = 10,
) -> tuple[list[ScoredProduct], dict[str, Any]]:
    """Soft policy ranking.

    Confirmed products move up (stable, so relevance order breaks ties);
    products whose store policy contradicts the request are dropped, unless
    that would leave nothing (then all are kept and badged).  Never raises.
    """
    if not ranked or not (city or return_required):
        return ranked, {}
    head, tail = ranked[:top_n], ranked[top_n:]
    try:
        findings = checker.check_products(
            [item.product for item in head], city=city, return_required=return_required
        )
    except Exception:  # noqa: BLE001 - policy is an enrichment, never a reason to fail a search
        return ranked, {"products": {}, "summary": {"error": "policy check failed"}}

    def contradicted(item: ScoredProduct) -> bool:
        return any(f["status"] == CONTRADICTED for f in findings.get(item.product.id, {}).values())

    def confirmed_count(item: ScoredProduct) -> int:
        return sum(f["status"] == CONFIRMED for f in findings.get(item.product.id, {}).values())

    keep = [item for item in head if not contradicted(item)]
    dropped = [item.product.id for item in head if contradicted(item)]
    kept_all = not keep
    if kept_all:
        keep, dropped = list(head), []
    keep.sort(key=lambda item: -confirmed_count(item))  # stable
    summary = {
        "checked": len(head),
        "dropped_contradicted": dropped,
        "kept_all_because_all_contradicted": kept_all,
    }
    return keep + tail, {"products": findings, "summary": summary}
