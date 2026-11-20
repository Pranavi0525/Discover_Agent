"""Decide whether a product really comes from the merchant the user asked for.

A seller's displayed name is chosen by the seller, so it proves nothing. The one
identifier a seller cannot fake is the host of the URL the buyer would be sent
to, because only the real owner of nike.com can serve pages there.

Verdicts:
* verified   - a product/variant URL host belongs to the requested merchant
               (host brand label equals the requested name, or the request was
               a domain and the host is that domain or a subdomain of it).
* unverified - nothing contradicts the request but nothing proves it (only the
               seller NAME matches, or there is no usable URL). Never shown as
               a match.
* mismatch   - the evidence points to a different merchant.

Known limits: matching is by brand label, so nike.example would count as "Nike"
(same label, different top-level domain). A store whose name differs from its
domain is "unverified", never wrongly "verified". Shopify subdomains
(<name>.myshopify.com) are first-come, so they are weaker evidence than an
owned domain.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Literal
from urllib.parse import urlparse

Verdict = Literal["verified", "unverified", "mismatch"]

_GENERIC = frozenset({
    "the", "store", "stores", "shop", "shops", "official", "india", "in", "inc", "ltd",
    "limited", "pvt", "private", "llc", "co", "company", "online", "retail", "corp",
})
_SLD_SUFFIX = frozenset({"co", "com", "org", "net", "gov", "ac", "edu"})
_LABEL_OK = re.compile(r"^[a-z0-9-]+$")


def _tokens(name: str, ampersand: str) -> list[str]:
    text = unicodedata.normalize("NFKC", name).lower().replace("&", ampersand)
    words = re.findall(r"[a-z0-9]+", text)
    core = [w for w in words if w not in _GENERIC]
    return core or words


def _compacts(name: str) -> set[str]:
    """Spellings of a brand with spaces and punctuation removed. An ampersand is
    read both ways: "H&M" -> "hm"; "Marks & Spencer" -> "marksspencer" and
    "marksandspencer"."""
    return {"".join(_tokens(name, "")), "".join(_tokens(name, " and "))}


def _host(url: object) -> str | None:
    try:
        parsed = urlparse(str(url or "").strip())
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not host or "." not in host:
        return None
    return host[4:] if host.startswith("www.") else host


def brand_label(host: str) -> str | None:
    """www.nike.com -> nike, nike.co.in -> nike, nike.myshopify.com -> nike."""
    labels = host.split(".")
    if host.endswith(".myshopify.com") and len(labels) >= 3:
        label = labels[-3]
    elif len(labels) >= 3 and labels[-2] in _SLD_SUFFIX and len(labels[-1]) == 2:
        label = labels[-3]
    elif len(labels) >= 2:
        label = labels[-2]
    else:
        return None
    # Look-alike characters and punycode are never trusted as a brand.
    return label if _LABEL_OK.match(label) else None


def _requested_domain(text: str) -> str | None:
    candidate = re.sub(r"^https?://", "", text.strip().lower()).split("/", 1)[0]
    if candidate.startswith("www."):
        candidate = candidate[4:]
    if "." in candidate and re.match(r"^[a-z0-9.-]+$", candidate):
        return candidate
    return None


def merchant_match(
    requested: str | None, seller: str | None, urls: Iterable[object]
) -> Verdict:
    """How well the evidence supports "this product is sold by `requested`"."""
    if not requested or not requested.strip():
        return "verified"  # nothing was asked for, nothing to prove
    hosts = [h for h in (_host(u) for u in urls) if h]

    domain = _requested_domain(requested)
    if domain:
        if any(h == domain or h.endswith("." + domain) for h in hosts):
            return "verified"
        return "mismatch" if hosts else "unverified"

    wanted = {w for w in _compacts(requested) if len(w) >= 2}
    if not wanted:
        return "unverified"
    if any(brand_label(h) in wanted for h in hosts):
        return "verified"
    if seller and wanted & _compacts(seller):
        return "unverified"  # the name alone is self-asserted
    return "mismatch"
