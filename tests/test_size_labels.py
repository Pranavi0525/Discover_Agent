from discovery_agent.ranking import size_status
from discovery_agent.schemas import Product, ProductVariant


def _p(*labels, available=True):
    return Product(id="x", title="shoe", url="https://example.com/p", price=1,
                   currency="INR", source="t",
                   variants=[ProductVariant(options={"size": l}, available=available)
                             for l in labels])


def test_bracketed_label_matches_bare_size():
    assert size_status("9", _p("8 (42.5)", "9 (44)")) == "available"

def test_half_size_is_not_whole_size():
    assert size_status("9", _p("9.5 (44.5)")) == "unavailable"

def test_conflicting_systems_are_unknown_not_available():
    assert size_status("UK 9", _p("US 9")) == "unknown"

def test_eu_request_matches_bracketed_eu():
    assert size_status("EU 44", _p("9 (44)")) == "available"

def test_out_of_stock_size_is_unavailable():
    assert size_status("9", _p("9 (44)", available=False)) == "unavailable"