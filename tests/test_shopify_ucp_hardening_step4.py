"""Step 4: the MCP HTTP boundary must bound what it reads and reject odd shapes."""
import asyncio
import json

import httpx
import pytest

from discovery_agent.config import get_settings
from discovery_agent.providers import shopify_ucp as mod
from discovery_agent.providers.shopify_ucp import (
    ResponseTooLarge, ShopifyRateLimited, ShopifyToolError, ShopifyUcpMcpClient,
)


def _call(handler, limit=None):
    async def run():
        client = ShopifyUcpMcpClient(get_settings())
        async with client:
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                return await client.call_tool("search_catalog", {"catalog": {"query": "x"}})
            finally:
                await client._client.aclose()
    return asyncio.run(run())


def _ok(body, ctype="application/json"):
    return lambda request: httpx.Response(200, headers={"content-type": ctype}, content=body)


def test_normal_response_still_works():
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": {"products": []}}})
    assert _call(_ok(body)) == {"products": []}


def test_oversized_body_is_refused_while_reading(monkeypatch):
    monkeypatch.setattr(mod, "MAX_RESPONSE_BYTES", 1000)
    monkeypatch.setattr(mod._read_capped, "__defaults__", (1000,))
    with pytest.raises(ResponseTooLarge):
        _call(_ok(b"x" * 5000))


def test_oversized_declared_length_is_refused_before_reading(monkeypatch):
    monkeypatch.setattr(mod._read_capped, "__defaults__", (1000,))
    handler = lambda r: httpx.Response(200, headers={"content-type": "application/json", "content-length": "99999"},
                                      content=b"x" * 99999)
    with pytest.raises(ResponseTooLarge):
        _call(handler)


@pytest.mark.parametrize("body", ["[]", "\"text\"", "42", "null"])
def test_non_object_json_is_a_tool_error_not_a_retryable_crash(body):
    with pytest.raises(ShopifyToolError):
        _call(_ok(body))


def test_429_keeps_retry_after_and_never_reads_the_body():
    handler = lambda r: httpx.Response(429, headers={"retry-after": "3"}, content=b"x" * 10_000_000)
    with pytest.raises(ShopifyRateLimited) as info:
        _call(handler)
    assert info.value.retry_after == 3.0


def test_http_error_status_is_raised():
    with pytest.raises(httpx.HTTPStatusError):
        _call(lambda r: httpx.Response(500, content=b"boom"))


def test_is_error_result_is_a_tool_error():
    body = json.dumps({"result": {"isError": True, "content": [{"text": "bad request"}]}})
    with pytest.raises(ShopifyToolError):
        _call(_ok(body))


def test_event_stream_still_parsed():
    body = 'data: ' + json.dumps({"result": {"structuredContent": {"products": []}}}) + "\n\n"
    assert _call(_ok(body, "text/event-stream")) == {"products": []}
