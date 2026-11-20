"""Spelled-out sizes (Small, Medium, Large, Extra Large...) in size matching.

Only size_status() / _size_readings() change: letter and number labels read as before, and a
size is "available" only when the SAME value is on an AVAILABLE variant.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from discovery_agent.ranking import (
    _size_readings,
    filter_and_rank_with_stats,
    size_status,
)
from discovery_agent.schemas import Product, ProductVariant, UserIntent
from discovery_agent.semantic import LexicalScorer


def _values(label):
    return {value for _, value in _size_readings(label)}


def _prod(*labels, available=True, currency="INR", price="1200.00"):
    flags = available if isinstance(available, (list, tuple)) else [available] * len(labels)
    return Product(
        id="gid://shopify/p/SIZEWORDS0001",
        title="Cotton shirt",
        url="https://example.com/p",
        source="test",
        variants=[
            ProductVariant(
                price=Decimal(price), currency=currency, available=flag,
                seller="Shop", options={"size": label},
            )
            for label, flag in zip(labels, flags)
        ],
    )


@pytest.mark.parametrize("label, expected", [
    ("Small", {"s"}), ("Medium", {"m"}), ("Large", {"l"}),
    ("small", {"s"}), ("MEDIUM", {"m"}),
    ("Extra Large", {"xl"}), ("X-Large", {"xl"}), ("XLarge", {"xl"}), ("x large", {"xl"}),
    ("Extra Small", {"xs"}), ("X-Small", {"xs"}), ("XSmall", {"xs"}),
    ("XX-Large", {"xxl"}), ("2X-Large", {"xxl"}), ("Extra Extra Large", {"xxl"}),
    ("XX-Small", {"xxs"}), ("2X-Small", {"xxs"}),
    ("Small (S)", {"s"}), ("Medium (M)", {"m"}),
])
def test_words_read_as_their_letter_code(label, expected):
    assert _values(label) == expected


@pytest.mark.parametrize("label", ["Extra Large", "X-Large", "Extra Small", "XX-Large"])
def test_a_compound_word_is_one_size_not_two(label):
    assert len(_size_readings(label)) == 1
    assert "l" not in _values(label) and "s" not in _values(label)


@pytest.mark.parametrize("label, expected", [
    ("9", [(None, "9")]), ("UK 9", [("uk", "9")]), ("9 (44)", [(None, "9"), (None, "44")]),
    ("11.5 (47)", [(None, "11.5"), (None, "47")]), ("S", [(None, "s")]), ("M", [(None, "m")]),
    ("XL", [(None, "xl")]), ("2XL", [(None, "2xl")]), ("XXS", [(None, "xxs")]),
    ("One Size", []), ("Default Title", []), ("Black", []),
])
def test_labels_without_size_words_read_exactly_as_before(label, expected):
    assert _size_readings(label) == expected


def test_medium_is_found_when_the_store_spells_sizes_out():
    assert size_status("M", _prod("Small", "Medium")) == "available"


def test_a_missing_size_is_still_unavailable():
    assert size_status("M", _prod("Small", "Large")) == "unavailable"
    assert size_status("S", _prod("Medium", "Large")) == "unavailable"


def test_a_sold_out_size_is_not_available():
    assert size_status("M", _prod("Small", "Medium", available=[True, False])) == "unavailable"
    assert size_status("M", _prod("Small", "Medium", available=[False, True])) == "available"


def test_extra_large_is_not_large():
    assert size_status("L", _prod("Small", "Extra Large")) == "unavailable"
    assert size_status("XL", _prod("Small", "Extra Large")) == "available"
    assert size_status("L", _prod("Small", "Large", "Extra Large")) == "available"


def test_a_requested_size_written_as_a_word_matches_letter_labels():
    assert size_status("Medium", _prod("S", "M")) == "available"
    assert size_status("Large", _prod("S", "M")) == "unavailable"
    assert size_status("Extra Large", _prod("S", "XL")) == "available"


def test_numeric_and_unit_behaviour_is_unchanged():
    assert size_status("9", _prod("UK 8", "UK 9")) == "available"
    assert size_status("9", _prod("UK 8", "UK 10")) == "unavailable"
    assert size_status("UK 9", _prod("US 9")) == "unknown"  # same number, conflicting system
    assert size_status("M", _prod("One Size")) == "unavailable"
    assert size_status("M", _prod()) == "unknown"
    assert size_status(None, _prod("Small")) is None


def test_ranking_reports_the_size_as_available_end_to_end():
    intent = UserIntent(query="shirt", currency="INR", max_price=Decimal(1500), size="M")
    ranked, stats = filter_and_rank_with_stats(
        intent, [_prod("Small", "Medium")], scorer=LexicalScorer()
    )
    assert [r.size_status for r in ranked] == ["available"]
    assert stats["products"][0]["size_status"] == "available"
    assert stats["dropped"]["over_budget"] == 0
