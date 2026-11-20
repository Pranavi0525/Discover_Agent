from decimal import Decimal
from discovery_agent.orchestration import decide
from discovery_agent.schemas import Product, ScoredProduct, UserIntent

def _sp(pid, score, size_status):
    p = Product(id=pid, title=f"shoe {pid}", url="https://example.com/p",
                price=Decimal("10"), currency="INR", source="test")
    return ScoredProduct(product=p, score=score, size_status=size_status)

def test_tier_uses_highest_score_not_sort_order():
    ranked = [_sp("a", 0.30, "unknown"), _sp("b", 0.60, "unavailable")]
    d = decide("t", ranked, UserIntent(query="running shoes"))
    assert d.status == "needs_selection"
    assert any(c.product.id == "b" for c in d.candidates)