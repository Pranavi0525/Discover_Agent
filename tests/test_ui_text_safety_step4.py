import pytest

from discovery_agent.text_safety import safe_http_url, safe_text


@pytest.mark.parametrize("hostile", [
    "[Click here](https://evil.example) running shoes **FREE**",
    "![x](https://evil.example/beacon.png)",
    "# Official notice: pay now",
    ":red[SYSTEM OVERRIDE] :smile:",
    "$$\\href{https://evil.example}{pay}$$",
    "<img src=x onerror=alert(1)>",
])
def test_markdown_syntax_is_neutralised(hostile):
    out = safe_text(hostile)
    for ch in "[]()<>#$":
        if ch in hostile:
            assert all(out[i - 1] == "\\" for i, c in enumerate(out) if c == ch and i), out


def test_newlines_and_invisible_characters_cannot_start_new_blocks():
    out = safe_text("Shoes\n\n# Heading\u200b\n- item")
    assert "\n" not in out and "\u200b" not in out


def test_length_is_capped():
    assert len(safe_text("a" * 5000)) <= 201


def test_none_and_numbers_are_safe():
    assert safe_text(None) == "" and safe_text(42) == "42"


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,x", "ftp://x/y", "//x", "", None,
                                 "https://a.example/p q"])
def test_non_http_urls_are_rejected(url):
    assert safe_http_url(url) is None


def test_url_cannot_close_a_markdown_link():
    out = safe_http_url("https://shop.example/p)(https://evil.example")
    assert out and "(" not in out and ")" not in out


def test_normal_url_survives():
    assert safe_http_url("https://shop.example/policies/refund-policy") == \
        "https://shop.example/policies/refund-policy"
