"""Escape untrusted store text before it reaches Streamlit's markdown renderer.

Product titles, merchant names, size labels and policy URLs come from merchants
(via MCP or fetched pages).  ``st.markdown``/``st.caption`` interpret markdown,
so an unescaped title such as ``[Click](https://evil) **FREE**`` becomes a live
link, an image beacon, or a heading.  ``safe_text`` makes the text display
literally.  It is display-only: stored data and ranking are never altered.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote, urlsplit

MAX_DISPLAY_CHARS = 200
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_WS = re.compile(r"\s+")
# Every ASCII punctuation char that markdown, Streamlit directives (:red[...]),
# emoji shortcodes (:smile:) or LaTeX ($...$) can act on.
_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|~$:&])")


def safe_text(value: object, limit: int = MAX_DISPLAY_CHARS) -> str:
    """Return ``value`` as one line of literal, length-capped text."""
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = _CONTROL.sub("", text)
    text = _WS.sub(" ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return _SPECIAL.sub(r"\\\1", text)


def safe_http_url(value: object) -> str | None:
    """Return a clean http(s) URL, or None.  Parentheses are percent-encoded so
    the URL cannot close a markdown link early."""
    try:
        raw = str(value or "").strip()
        parts = urlsplit(raw)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc or _CONTROL.search(raw):
        return None
    if any(c.isspace() for c in raw):
        return None
    return quote(raw, safe=":/?#[]@!$&'*+,;=%~-._").replace("(", "%28").replace(")", "%29")
