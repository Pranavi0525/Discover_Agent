"""Currency and market helpers.

One place knows which currency belongs to which buyer country, so the config
(market default) and the Shopify request builders cannot drift apart.
"""

from __future__ import annotations

from typing import Any

# Only unambiguous single-country currencies are listed.  EUR is deliberately
# absent because it spans many countries.
CURRENCY_TO_COUNTRY = {
    "INR": "IN",
    "USD": "US",
    "GBP": "GB",
    "CAD": "CA",
    "AUD": "AU",
    "NZD": "NZ",
    "SGD": "SG",
    "JPY": "JP",
}
COUNTRY_TO_CURRENCY = {country: currency for currency, country in CURRENCY_TO_COUNTRY.items()}


# ISO 4217 minor-unit exponents that are NOT 2.  UCP amounts are integers in
# these units (USD 79.00 -> 7900, JPY 8000 -> 8000, KWD 8 -> 8000).
_ZERO_DECIMAL = frozenset({
    "BIF", "CLP", "DJF", "GNF", "ISK", "JPY", "KMF", "KRW", "PYG", "RWF",
    "UGX", "VND", "VUV", "XAF", "XOF", "XPF",
})
_THREE_DECIMAL = frozenset({"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"})


def minor_unit_exponent(currency: str | None) -> int:
    """Decimal places of a currency's minor unit (2 unless it is a known exception)."""
    code = (currency or "").strip().upper()
    if code in _ZERO_DECIMAL:
        return 0
    if code in _THREE_DECIMAL:
        return 3
    return 2


def currency_for_country(country: str | None) -> str | None:
    """Currency of a buyer country (``"IN"`` -> ``"INR"``), or None if unknown."""
    if not country:
        return None
    return COUNTRY_TO_CURRENCY.get(str(country).strip().upper())


# An LLM filling an optional string field often writes the WORD "null" (or "None",
# "N/A") instead of a JSON null.  Read literally, that becomes a merchant called
# "null" that silently filters out every product, or a delivery city "null".
NULLISH_STRINGS = frozenset(
    {"", "null", "none", "n/a", "na", "nil", "unknown", "unspecified", "not specified", "-"}
)


def is_nullish(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in NULLISH_STRINGS


def normalize_currency_code(value: Any) -> Any:
    """Strip/upper-case a currency code; blank or "null"-like means "not specified".

    Other non-strings are returned untouched so validation can reject them loudly.
    """
    if isinstance(value, str):
        if is_nullish(value):
            return None
        return value.strip().upper()
    return value


def format_money(amount: Any, currency: str | None) -> str:
    """``INR 8,000.00``; without a currency just ``8,000.00`` (never ``None 8000``)."""
    text = f"{amount:,.2f}"
    return f"{currency} {text}" if currency else text
