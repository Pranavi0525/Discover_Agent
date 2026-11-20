"""M2.3: the policy-page fetcher refuses unsafe hosts, checks every redirect and caps downloads."""
import httpx
import pytest

from discovery_agent import policy_check
from discovery_agent.policy_check import HttpxFetcher

HTML = {"content-type": "text/html; charset=utf-8"}
_REAL_RESOLVES_PUBLIC = policy_check._resolves_public


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    # Pretend every name resolves to a public address unless a test says otherwise.
    monkeypatch.setattr(policy_check, "_resolves_public", lambda host: host != "internal.example.com")


def _fetcher(handler, **kw):
    return HttpxFetcher(transport=httpx.MockTransport(handler), **kw)


def test_plain_page_is_read():
    f = _fetcher(lambda r: httpx.Response(200, headers=HTML, text="<p>30 day returns</p>"))
    assert "30 day returns" in f.fetch("https://shop.example.com/policies/refund-policy")


@pytest.mark.parametrize("url", [
    "http://shop.example.com/x", "https://localhost/x", "https://10.0.0.5/x",
    "https://store.internal/x", "https://[::1]/x", "ftp://shop.example.com/x",
])
def test_unsafe_urls_are_never_requested(url):
    calls = []
    f = _fetcher(lambda r: calls.append(r) or httpx.Response(200, headers=HTML, text="x"))
    assert f.fetch(url) is None and calls == []


def test_name_that_resolves_to_a_private_address_is_refused():
    calls = []
    f = _fetcher(lambda r: calls.append(r) or httpx.Response(200, headers=HTML, text="x"))
    assert f.fetch("https://internal.example.com/policies/refund-policy") is None and calls == []


def test_redirect_to_a_private_host_is_not_followed():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.host == "shop.example.com":
            return httpx.Response(302, headers={"location": "https://internal.example.com/secret"})
        return httpx.Response(200, headers=HTML, text="secret")

    assert _fetcher(handler).fetch("https://shop.example.com/policies/refund-policy") is None
    assert seen == ["https://shop.example.com/policies/refund-policy"]  # the second hop was never requested


def test_redirect_to_http_is_refused():
    f = _fetcher(lambda r: httpx.Response(302, headers={"location": "http://shop.example.com/x"}))
    assert f.fetch("https://shop.example.com/a") is None


def test_safe_redirect_is_followed():
    def handler(request):
        if request.url.path == "/a":
            return httpx.Response(301, headers={"location": "/b"})
        return httpx.Response(200, headers=HTML, text="<p>final</p>")

    assert "final" in _fetcher(handler).fetch("https://shop.example.com/a")


def test_redirect_loop_stops():
    f = _fetcher(lambda r: httpx.Response(302, headers={"location": "/again"}), max_redirects=3)
    assert f.fetch("https://shop.example.com/a") is None


def test_huge_body_is_cut_off_not_downloaded():
    sent = {"bytes": 0}

    def stream():
        for _ in range(10_000):          # would be ~10 GB if read in full
            sent["bytes"] += 1_000_000
            yield b"a" * 1_000_000

    f = _fetcher(lambda r: httpx.Response(200, headers=HTML, content=stream()), max_bytes=50_000)
    text = f.fetch("https://shop.example.com/x")
    assert text is not None and len(text) <= 50_000
    assert sent["bytes"] <= 5_000_000      # stopped after the first chunk or two


def test_non_text_content_is_refused():
    f = _fetcher(lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"))
    assert f.fetch("https://shop.example.com/x") is None


def test_error_status_is_none():
    assert _fetcher(lambda r: httpx.Response(404, headers=HTML, text="no")).fetch("https://shop.example.com/x") is None


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", False), ("10.1.2.3", False), ("192.168.0.9", False),
    ("169.254.169.254", False), ("::1", False), ("8.8.8.8", True),
])
def test_resolved_address_check_on_literal_addresses(monkeypatch, host, expected):
    monkeypatch.setattr(policy_check, "_resolves_public", _REAL_RESOLVES_PUBLIC)
    assert policy_check._resolves_public(host) is expected


def test_unresolvable_name_is_refused(monkeypatch):
    monkeypatch.setattr(policy_check, "_resolves_public", _REAL_RESOLVES_PUBLIC)
    monkeypatch.setattr(policy_check.socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(OSError("nx")))
    assert policy_check._resolves_public("nope.example.com") is False
