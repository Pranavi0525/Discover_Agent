from discovery_agent.execution import _normalize_product


def _payload(range_cur="INR", range_amt=5200, variants=None):
    return {"id": "gid://shopify/p/x", "title": "Nike Revolution 8",
            "url": "https://example.com/p",
            "price_range": {"min": {"amount": range_amt, "currency": range_cur},
                            "max": {"amount": 650000, "currency": "INR"}},
            "variants": variants or []}


def _v(amount, cur, avail=True):
    return {"price": {"amount": amount, "currency": cur},
            "availability": {"available": avail},
            "selected_options": [{"name": "Size", "label": "9"}]}


def test_variant_currency_beats_mislabelled_product_range():
    p = _normalize_product(_payload(variants=[_v(5200, "EUR")]), "x")
    assert (p.currency, str(p.price)) == ("EUR", "52.00")


def test_mixed_variant_currencies_give_no_price_and_a_note():
    p = _normalize_product(_payload(variants=[_v(5200, "EUR"), _v(650000, "INR")]), "x")
    assert p.price is None and p.price_note


def test_product_range_used_only_when_variants_have_no_price():
    p = _normalize_product(_payload(range_amt=429500, variants=[]), "x")
    assert (p.currency, str(p.price)) == ("INR", "4295.00")