
"""Deterministic reading of size, gender, age group and rating.

Same idea as ``budget.py`` and ``constraints.py``: Gemini proposes a
``UserIntent``, but the words the user actually typed ("men size 9", "4 stars
and above") are read again in plain Python, so a detail is not lost just
because the LLM skipped it.  A typed value beats the LLM's value.

No imports from ``schemas`` at runtime (``schemas`` imports the normalisers
below), so there is no import cycle.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .schemas import UserIntent

# "any" means the user was asked and said it does not matter.  It counts as an
# answer (so we stop asking) but it is never sent to the catalog or shown as
# an unverified constraint.
ANY = "any"

_GENDER_WORDS: dict[str, str] = {
    "men": "men", "man": "men", "mens": "men", "male": "men", "gents": "men",
    "boy": "men", "boys": "men",
    "women": "women", "woman": "women", "womens": "women", "female": "women",
    "ladies": "women", "lady": "women", "girl": "women", "girls": "women",
    "unisex": "unisex",
}
_AGE_WORDS: dict[str, str] = {
    "kid": "kids", "kids": "kids", "child": "kids", "children": "kids",
    "boy": "kids", "boys": "kids", "girl": "kids", "girls": "kids", "junior": "kids",
    "baby": "infant", "babies": "infant", "infant": "infant", "toddler": "infant",
    "adult": "adult", "adults": "adult",
}
_TOKEN = re.compile(r"[a-z]+(?:'s)?")

_NUMERIC_SIZE = r"\d{1,2}(?:\.5)?"
_LETTER_SIZE = r"xxs|xs|s|m|l|xl|xxl|xxxl|2xl|3xl"
# "size 9", "size uk 9", "size: M", "uk 9", "us 10.5", "eu 42", "uk9"
_SIZE_PHRASE = re.compile(
    rf"\b(?:size\s*:?\s*(?:(?:uk|us|eu|in)\s*)?(?P<a>{_NUMERIC_SIZE}|{_LETTER_SIZE})"
    rf"|(?P<unit>uk|us|eu)\s*(?P<b>{_NUMERIC_SIZE}))\b",
    re.IGNORECASE,
)
_BARE_LETTER_SIZE = re.compile(r"\b(xxs|xs|xl|xxl|xxxl|2xl|3xl)\b", re.IGNORECASE)
_ANY_SIZE = re.compile(r"\b(any size|all sizes|size (?:does ?n'?t|does not|is not|isn'?t) matter|no size)\b", re.IGNORECASE)
_ANY_GENDER = re.compile(r"\b(any gender|anyone|either gender|gender (?:does ?n'?t|does not) matter)\b", re.IGNORECASE)

_STAR = r"(?P<n>[1-5](?:\.\d)?)"
_RATING_PATTERNS = (
    re.compile(rf"\b{_STAR}\s*\+?\s*(?:stars?|\u2605)", re.IGNORECASE),
    re.compile(rf"\b(?:rating|rated|ratings?)\s*(?:of|above|over|at least|minimum|min|>=|>)?\s*{_STAR}\b", re.IGNORECASE),
    re.compile(rf"\b(?:at least|above|over|min(?:imum)?)\s*{_STAR}\s*(?:rating|rated)\b", re.IGNORECASE),
)


def is_any(value: str | None) -> bool:
    return bool(value) and value.strip().lower() == ANY


def normalize_gender(value: Any) -> Any:
    """'Men', "men's", 'male' -> 'men'.  Unknown words become None, not a guess."""
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    if not text or text in {"null", "none", "n/a", "na", "unknown", "-", "not specified"}:
        return None
    if text in {ANY, "all", "everyone", "either"}:
        return ANY
    first = _TOKEN.findall(text.replace("\u2019", "'"))
    for word in first:
        word = word.removesuffix("'s")
        if word in _GENDER_WORDS:
            return _GENDER_WORDS[word]
    return None


def normalize_age_group(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    if not text or text in {"null", "none", "n/a", "na", "unknown", "-", "not specified"}:
        return None
    if text in {ANY, "all"}:
        return ANY
    for word in _TOKEN.findall(text):
        if word in _AGE_WORDS:
            return _AGE_WORDS[word]
    return None


def normalize_size(value: Any) -> Any:
    """Keep the size as written, tidied ('UK 9' -> 'UK 9', ' xl ' -> 'XL')."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text or text.lower() in {"null", "none", "n/a", "na", "unknown", "-", "not specified"}:
        return None
    if text.lower() == ANY:
        return ANY
    return text.upper() if re.fullmatch(_LETTER_SIZE, text, re.IGNORECASE) else text[:20]


def normalize_rating(value: Any) -> Any:
    """A rating outside (0, 5] from an LLM means 'not stated', not a failed plan."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return number if 0 < number <= 5 else None


def extract_size(text: str) -> str | None:
    if _ANY_SIZE.search(text):
        return ANY
    match = _SIZE_PHRASE.search(text)
    if match:
        if match.group("b"):
            return f"{match.group('unit').upper()} {match.group('b')}"
        # keep a typed unit ("size uk 9") with the number
        unit = re.search(r"size\s*:?\s*(uk|us|eu|in)\b", match.group(0), re.IGNORECASE)
        value = match.group("a")
        value = value.upper() if re.fullmatch(_LETTER_SIZE, value, re.IGNORECASE) else value
        return f"{unit.group(1).upper()} {value}" if unit else value
    bare = _BARE_LETTER_SIZE.search(text)
    return bare.group(1).upper() if bare else None


def extract_gender(text: str) -> str | None:
    """men / women / unisex / any, or None when absent or contradictory."""
    if _ANY_GENDER.search(text):
        return ANY
    found: set[str] = set()
    for word in _TOKEN.findall(text.lower().replace("\u2019", "'")):
        word = word.removesuffix("'s")
        if word in _GENDER_WORDS:
            found.add(_GENDER_WORDS[word])
    if "unisex" in found:
        return "unisex"
    return next(iter(found)) if len(found) == 1 else None


def extract_age_group(text: str) -> str | None:
    found: set[str] = set()
    for word in _TOKEN.findall(text.lower()):
        if word in _AGE_WORDS:
            found.add(_AGE_WORDS[word])
    return next(iter(found)) if len(found) == 1 else None


def extract_min_rating(text: str) -> float | None:
    for pattern in _RATING_PATTERNS:
        match = pattern.search(text)
        if match:
            value = normalize_rating(match.group("n"))
            if value:
                return value
    return None


def strip_size_phrase(text: str) -> str:
    """Remove 'size 9' / 'uk 9' style phrases from a catalog query."""
    text = _SIZE_PHRASE.sub(" ", text)
    text = _ANY_SIZE.sub(" ", text)
    return " ".join(text.split())


def reconcile_attributes(intent: "UserIntent", text: str) -> tuple["UserIntent", list[str]]:
    """Make the intent agree with the size/gender/age/rating the user typed.

    A typed value overrides the LLM's; a value the LLM invented that the user
    never typed is left alone (it is reported as unverified, like any other
    constraint) because the user may have given it in the form instead.
    """
    updates: dict[str, object] = {}
    notes: list[str] = []
    typed = {
        "size": extract_size(text),
        "gender": extract_gender(text),
        "age_group": extract_age_group(text),
        "min_rating": extract_min_rating(text),
    }
    for field, value in typed.items():
        if value is None:
            continue
        current = getattr(intent, field)
        if current != value:
            updates[field] = value
            notes.append(f"{field} {current} -> {value}" if current else f"{field} set to {value}")
    if not updates:
        return intent, []
    return intent.model_copy(update=updates), notes
