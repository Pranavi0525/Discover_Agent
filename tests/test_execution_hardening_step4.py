"""Step 4: untrusted MCP / catalog data must never abort a run or fail open."""
import asyncio
import json
from decimal import Decimal

import pytest

from discovery_agent.config import get_settings
from discovery_agent.execution import (
    AgentState,
    ShopifyMcpExecutor,
    _extract_product_ids,
    _normalize_product,
    _unwrap_mcp_result,
)
from discovery_agent.schemas import ExecutionPlan, PlanStep, UserIntent

BASE = {"title": "Shoe", "url": "https://shop.example/p/1", "price": {"amount": 300000, "currency": "INR"}}


@pytest.mark.parametrize("extra", [
    {"variants": {"a": 1}}, {"variants": 5}, {"variants": "abc"}, {"media": {"a": 1}}, {"media": "abc"},
    {"price": True}, {"price": {"amount": float("nan"), "currency": "INR"}},
    {"price": {"amount": float("inf"), "currency": "INR"}}, {"merchant": {"a": 1}},
    {"title": {"x": 1}}, {"title": ["a"]}, {"id": ["x"]}, {"description": ["a"] * 5},
])
def test_malformed_product_never_raises(extra):
    _normalize_product({**BASE, **extra}, "gid://1")  # must not raise


@pytest.mark.parametrize("flag,expected", [("false", False), ("False", False), ("0", False), (False, False),
                                           ("true", True), (True, True)])
def test_string_availability_is_not_read_with_bool(flag, expected):
    assert _normalize_product({**BASE, "available": flag}, "x").available is expected


def test_title_and_merchant_are_length_capped():
    product = _normalize_product({**BASE, "title": "A" * 100_000, "merchant": "M" * 100_000}, "x")
    assert len(product.title) <= 300 and len(product.merchant) <= 200


def test_oversized_mcp_text_is_treated_as_no_result():
    assert _unwrap_mcp_result("x" * 3_000_000) == {"content": ""}
    assert _normalize_product("x" * 3_000_000, "x") is None


def test_product_ids_are_bounded_and_clean():
    payload = {"products": [{"id": "ok-1"}, {"id": "x" * 500}, {"id": "bad\x00id"}, {"id": True}, {"id": "ok-2"}]}
    assert _extract_product_ids(payload) == ["ok-1", "ok-2"]


def _plan():
    step = PlanStep(step_id="s1", action="search_catalog", reason="search")
    return ExecutionPlan(intent=UserIntent(query="running shoes", max_price=Decimal("5000"), currency="INR"),
                         steps=[step])


def test_one_poisoned_product_does_not_discard_the_good_one():
    good = {"title": "Good shoe", "url": "https://shop.example/p/g", "price": {"amount": 300000, "currency": "INR"}}
    bad = {**good, "title": "Poison", "variants": {"x": 1}}

    class Session:
        async def call_tool(self, name, args):
            pid = args["catalog"]["id"]
            return {"structuredContent": {"product": bad if pid == "bad" else good}}

    ex = ShopifyMcpExecutor(session_factory=lambda: Session(), settings=get_settings())
    state = AgentState(plan=_plan(), product_ids=["good", "bad"], session=Session(),
                       tool_call_limit=10, search_done=True)
    out = asyncio.run(ex._details_node(state))
    assert any(p.title == "Good shoe" for p in out["report"].products)


def test_dict_shaped_mcp_error_is_an_error():
    class Session:
        async def call_tool(self, name, args):
            return {"isError": True, "content": []}

    ex = ShopifyMcpExecutor(session_factory=lambda: Session(), settings=get_settings())
    state = AgentState(plan=_plan(), session=Session(), tool_call_limit=10)
    with pytest.raises(Exception):
        asyncio.run(ex._call_with_limits(state, Session(), "search_catalog", {"catalog": {"query": "x"}}))
