import pytest
from discovery_agent.merchant_identity import merchant_match as m


@pytest.mark.parametrize("req,seller,urls,expected", [
    ("Nike", "Nike", ["https://www.nike.com/p/1"], "verified"),
    ("Nike India", "x", ["https://nike.co.in/p"], "verified"),
    ("nike.com", "x", ["https://store.nike.com/a"], "verified"),
    ("Nike", "Nike", ["https://nike.myshopify.com/p"], "verified"),
    ("H&M", "H&M", ["https://www2.hm.com/p"], "verified"),
    ("Marks & Spencer", "x", ["https://www.marksandspencer.com/p"], "verified"),
    ("", "anything", [], "verified"),
    # seller name alone never verifies
    ("Nike", "Nike", ["https://shop.evil.example/p/1"], "unverified"),
    ("Nike", "Nike", [], "unverified"),
    ("Nike", "NIKE, ", ["https://nike-store.com/p"], "unverified"),
    # look-alike (Cyrillic i) host is not trusted
    ("Nike", "Nike", ["https://n\u0456ke.com/p"], "unverified"),
    # clear contradictions
    ("nike.com", "Nike", ["https://nike.com.evil.example/a"], "mismatch"),
    ("Nike", "Other", ["https://adidas.com/p"], "mismatch"),
    ("Nike", "Other", [], "mismatch"),
    # non-http schemes are ignored
    ("Nike", "Nike", ["javascript:alert(1)", "ftp://nike.com/x"], "unverified"),
])
def test_merchant_match(req, seller, urls, expected):
    assert m(req, seller, urls) == expected


def test_garbage_urls_do_not_raise():
    assert m("Nike", "Nike", [None, 5, "http://[bad", ""]) == "unverified"
