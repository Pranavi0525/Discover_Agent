from discovery_agent.ranking import _floor_overlap
from discovery_agent.schemas import Product


def _p(title, desc=""):
    return Product(id="x", title=title, url="https://example.com/p", price=1,
                   currency="INR", source="t", description=desc)


def test_plural_matches_singular():
    assert _floor_overlap("shirts", _p("Cotton Shirt")) > 0

def test_joined_form_matches_hyphenated():
    assert _floor_overlap("tshirt", _p("Men's T-Shirt")) > 0

def test_unrelated_product_is_still_dropped():
    assert _floor_overlap("shirts", _p("Steel water bottle")) == 0