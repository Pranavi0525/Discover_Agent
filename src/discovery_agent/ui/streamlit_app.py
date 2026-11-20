"""M1.6 Streamlit UI on the single ``service.discover`` path.

The page imports only ``discovery_agent.service``.  It never calls Gemini,
Shopify or SQLite itself, and ``st.session_state`` holds nothing but
``session_id`` and ``run_id`` (both mirrored into the URL so a browser refresh
or an app restart brings the history back).  Everything shown is read from
``service.load_run`` / ``service.get_steps``.
"""

from __future__ import annotations

from collections import defaultdict

import streamlit as st

from discovery_agent import service
from discovery_agent.semantic import NAME_MINILM as SEMANTIC_SCORER
from discovery_agent.text_safety import safe_http_url, safe_text

st.set_page_config(page_title="Shopping Discovery Agent", page_icon="🛍️", layout="wide")


@st.cache_resource(show_spinner="Loading the safety model…")
def _warm_guard() -> bool:
    """Runs once per server process, not once per rerun."""
    service.warm_guard()
    return True


_warm_guard()

_guard = service.guard_status()
if _guard["warning"]:
    st.warning(f"Safety check: {_guard['warning']}", icon="⚠️")
for _privacy_warning in service.privacy_warnings():
    st.warning(f"Privacy: {_privacy_warning}", icon="⚠️")

STATUS_ICON = {
    "selected": "✅", "needs_selection": "🟡", "clarify": "❓",
    "failed": "❌", "error": "⛔", "running": "⏳", "blocked": "🚫",
}
STATUS_LABEL = {
    "selected": "Found a good match",
    "needs_selection": "Several possible matches",
    "clarify": "Needs more detail",
    "failed": "Search failed",
    "error": "Something went wrong",
    "running": "Still running",
    "blocked": "Request not run",
}
CURRENCIES = ["Not sure", "INR", "USD", "EUR", "GBP", "AED", "SGD", "AUD", "CAD"]
UNVERIFIED_HELP = "Not checkable in Phase 1. Confirm on the merchant page."


# --- session bootstrap -----------------------------------------------------------
def _bootstrap() -> None:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = service.ensure_session(st.query_params.get("sid"))
    sid = st.session_state["session_id"]
    st.query_params["sid"] = sid
    if "run_id" not in st.session_state:
        wanted = st.query_params.get("run")
        known = {r["run_id"] for r in service.list_history(sid, limit=200)}
        st.session_state["run_id"] = wanted if wanted in known else None


def _select_run(run_id: str | None) -> None:
    st.session_state["run_id"] = run_id
    if run_id:
        st.query_params["run"] = run_id
    elif "run" in st.query_params:
        del st.query_params["run"]


def _when(iso: str) -> str:
    return f"{iso[5:10]} {iso[11:16]} UTC" if len(iso) >= 16 else iso


def _money(price, currency) -> str:
    return f"{currency} {price:,.2f}" if currency else f"{price:,.2f}"


_bootstrap()
session_id = st.session_state["session_id"]


# --- sidebar: history -------------------------------------------------------------
with st.sidebar:
    st.header("History")
    st.button("➕ New search", key="new_search", on_click=_select_run, args=(None,),
              use_container_width=True)
    history = service.list_history(session_id)
    if not history:
        st.caption("Your searches will appear here.")
    for row in history:
        label = f"{STATUS_ICON.get(row['status'], '•')} {row['query'][:34]} · {_when(row['created_at'])}"
        st.button(label, key=f"hist_{row['run_id']}", on_click=_select_run, args=(row["run_id"],),
                  help=f"{STATUS_LABEL.get(row['status'], row['status'])}"
                       + (" (partial results)" if row["partial"] else ""),
                  use_container_width=True)


# --- search form ------------------------------------------------------------------
st.title("🛍️ Discovery Agent")
st.caption("Describe what you want. Prices and stock come live from the store.")

configured = service.planner_configured()
if not configured:
    st.error(
        "The search assistant is not set up: no `GOOGLE_API_KEY` was found. Add it to your `.env` "
        "file and restart the app. You can still open past searches from the history."
    )

with st.form("search_form"):
    query = st.text_input("What are you looking for?", key="query",
                          placeholder="e.g. running shoes for everyday use")
    c1, c2 = st.columns(2)
    max_price = c1.number_input("Max budget (optional)", min_value=0.0, value=None, step=100.0,
                                key="max_price", placeholder="No limit")
    currency = c2.selectbox("Currency", CURRENCIES, key="currency",
                            help="Choose 'Not sure' and the assistant will resolve it or ask.")
    preferences = st.text_input("Preferences (optional)", key="preferences",
                                placeholder="e.g. lightweight, blue")
    c3, c4 = st.columns(2)
    merchant = c3.text_input("Store / merchant (optional)", key="merchant")
    delivery_city = c4.text_input("Delivery city (optional)", key="delivery_city")
    submitted = st.form_submit_button("🔎 Discover products", type="primary", key="submit",
                                      disabled=not configured, use_container_width=True)


def _form_values() -> dict:
    """Raw form values; blanks are dropped ('Not sure' currency is simply absent)."""
    values = {
        "max_price": str(max_price) if max_price else "",
        "currency": "" if currency == "Not sure" else currency,
        "preferences": preferences, "merchant": merchant, "delivery_city": delivery_city,
    }
    return {k: v for k, v in values.items() if v not in ("", None)}


def _run_discovery(text: str, form: dict) -> None:
    with st.spinner("Searching the store…"):
        result = service.discover(text, session_id, form)
    if result.run_id:
        _select_run(result.run_id)
        st.rerun()
    st.error(result.error or "The search could not be started.")


if submitted:
    _run_discovery(query, _form_values())


# --- results ----------------------------------------------------------------------
def _render_steps(run_id: str) -> None:
    events = [e for e in service.get_steps(run_id) if e["step"] != "audit"]
    if not events:
        return
    by_attempt: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_attempt[e["attempt"]].append(e)
    with st.expander(f"Steps ({len(by_attempt)} attempt{'s' if len(by_attempt) != 1 else ''})",
                     expanded=False):
        for attempt, rows in sorted(by_attempt.items()):
            st.markdown(f"**Attempt {attempt}**")
            st.table([
                {
                    "Step": r["step"],
                    "Status": r["status"],
                    "Latency (ms)": "–" if r["latency_ms"] is None else str(r["latency_ms"]),
                    "Error": r["error"] or "",
                }
                for r in rows
            ])


def _render_partial_banner(run_id: str, decision) -> None:
    events = service.get_steps(run_id)
    bad = [e for e in events if e["status"] in {"FAILED", "TIMED_OUT", "SKIPPED"}]
    counts: dict[str, int] = defaultdict(int)
    for e in bad:
        counts[e["status"]] += 1
    parts = [f"{n} {status.lower().replace('_', ' ')}" for status, n in counts.items()]
    # Count what is DISPLAYED (verified results and unverified candidates), not what was fetched.
    verified = len(decision.candidates) if decision else 0
    unverified = list(getattr(decision, "unverified_candidates", None) or []) if decision else []
    fetched_fallback = 0
    if decision:
        attempt_log = decision.diagnostics.get("attempt_log") or []
        if attempt_log:
            fetched_fallback = int(
                (attempt_log[-1].get("shopify") or {}).get("details_from_search_payload") or 0
            )
    lines = ["Some product details could not be loaded, so these results may be incomplete."]
    shown = f"Showing {verified} verified result{'s' if verified != 1 else ''}"
    if unverified:
        shown += (f" and {len(unverified)} unverified candidate"
                  f"{'s' if len(unverified) != 1 else ''} (listed separately)")
    lines.append(shown + ".")
    if parts:
        lines.append("Detail lookups: " + ", ".join(parts) + ".")
    hidden = max(0, fetched_fallback - len(unverified))
    if hidden:
        lines.append(
            f"{hidden} other product{'s' if hidden != 1 else ''} with search-result data only "
            f"{'were' if hidden != 1 else 'was'} not shown (not relevant to your search, or "
            "beyond the display limit)."
        )
    why = sorted({c.reason_text for c in unverified if c.reason_text})
    if why:
        lines.append("Why details failed: " + "; ".join(why[:3]) + ".")
    reasons = sorted({e["error"] for e in bad if e["error"]})
    if reasons:
        lines.append("Reasons: " + "; ".join(reasons[:3]))
    st.warning("\n\n".join(lines), icon="⚠️")


_POLICY_LABELS = {"returns": "Exchange / returns", "delivery": "Delivery"}
_POLICY_ICONS = {"confirmed": "\U0001F7E2", "contradicted": "\U0001F534",
                 "not_found": "\U0001F7E0", "unchecked": "\U0001F7E0"}
_POLICY_WORDS = {"confirmed": "per store policy", "contradicted": "store policy says no",
                 "not_found": "not confirmed", "unchecked": "could not check"}


def _render_policy(findings: dict | None, city: str | None) -> None:
    """One badge per requested constraint, with the store's own words as evidence."""
    for key, finding in (findings or {}).items():
        label = _POLICY_LABELS.get(key, key)
        if key == "delivery" and city:
            label = f"Delivery to {city}"
        icon = _POLICY_ICONS.get(finding["status"], "\U0001F7E0")
        st.caption(f"{icon} {label}: {_POLICY_WORDS.get(finding['status'], finding['status'])}",
                   help=safe_text(finding.get("evidence") or finding.get("detail"), 400) or None)
        _policy_url = safe_http_url(finding.get("url")) if finding["status"] == "confirmed" else None
        if _policy_url:
            st.caption(f"[Read the policy]({_policy_url})")


def _size_caption(status: str, size: str | None) -> str | None:
    """A verified size result: in stock, a real mismatch, or not confirmable from the listing."""
    if status == "available":
        return "✅ Your size is in stock on this offer"
    if status == "unavailable":
        return f"⚠️ {'Size ' + size if size else 'Your size'} is not available on this offer"
    if size:
        return f"🟠 Size {size}: could not be confirmed from the store's listing"
    return None


def _render_card(run_id: str, scored, ratings: dict[str, int],
                 policy: dict | None = None, city: str | None = None,
                 size: str | None = None, scorer: str | None = None) -> None:
    product = scored.product
    with st.container(border=True):
        if product.image_url:
            st.image(str(product.image_url))
        st.markdown(f"**{safe_text(product.title)}**")
        if product.merchant:
            st.caption(safe_text(product.merchant))
        st.write(_money(product.price, product.currency) if product.price is not None
                 else "Price unavailable")
        if product.price_note:
            st.caption(f"Note: {safe_text(product.price_note)}")
        size_line = _size_caption(scored.size_status, size)
        if size_line:
            st.caption(safe_text(size_line))
        if not product.available:
            st.caption("⚠️ Out of stock")
        # A similarity badge is only honest when the scorer is the real embedding model.
        # The lexical fallback is keyword overlap, so it is labelled as such, never as "match".
        if scorer == SEMANTIC_SCORER:
            st.badge(f"{scored.score:.0%} semantic match")
        elif scorer:
            st.caption("Ranked by keyword overlap (semantic scoring unavailable)")
        _render_policy(policy, city)
        st.link_button("View on store", str(product.url))
        up, down = st.columns(2)
        rated = ratings.get(product.id)
        up.button("👍" + (" ✓" if rated == 1 else ""), key=f"up_{run_id}_{product.id}",
                  on_click=service.record_feedback, args=(run_id, product.id, 1))
        down.button("👎" + (" ✓" if rated == -1 else ""), key=f"down_{run_id}_{product.id}",
                    on_click=service.record_feedback, args=(run_id, product.id, -1))


def _render_products(run_id: str, decision) -> None:
    if not decision or not decision.candidates:
        return
    ratings = {f["product_id"]: f["rating"] for f in service.get_feedback(run_id)}
    policy = (decision.diagnostics.get("policy") or {}).get("products") or {}
    city = next((i.get("value") for i in decision.diagnostics.get("unverified", [])
                 if i.get("name") == "delivery_city"), None)
    size = next((i.get("value") for i in decision.diagnostics.get("unverified", [])
                 if i.get("name") == "size"), None)
    scorer = (decision.diagnostics.get("relevance") or {}).get("scorer")
    st.subheader(f"Results ({len(decision.candidates)})")
    for start in range(0, len(decision.candidates), 3):
        for col, scored in zip(st.columns(3), decision.candidates[start:start + 3]):
            with col:
                _render_card(run_id, scored, ratings, policy.get(scored.product.id), city, size, scorer)


def _render_unverified_card(candidate) -> None:
    """A display-only card: no feedback buttons, no verified claims."""
    product = candidate.product
    with st.container(border=True):
        if product.image_url:
            st.image(str(product.image_url))
        st.markdown(f"**{safe_text(product.title)}**")
        st.caption(f"🟠 Not verified: product details could not be loaded ({candidate.reason_text})")
        if candidate.candidate_merchant:
            st.caption(f"{safe_text(candidate.candidate_merchant)} (merchant from search results, unverified)")
        if candidate.candidate_price is not None:
            st.write(_money(candidate.candidate_price, candidate.candidate_currency))
            st.caption("Price from search results, unverified. Not checked against your budget.")
        else:
            offers = [_money(o.price, o.currency) for o in candidate.listed_offers if o.price is not None]
            if offers:
                st.write(safe_text(" · ".join(offers[:3])))
                st.caption("Offers as listed in the search results, unverified. "
                           "Not checked against your budget.")
            else:
                st.write("Price not listed in the search results")
        if candidate.payload_lists_requested_currency is False:
            st.caption("The search results list other currencies only; whether an offer in "
                       "your currency exists was not checked.")
        st.caption("Size: not checked (details unavailable)")
        st.link_button("View on store", str(product.url))


def _render_unverified(decision) -> None:
    items = list(getattr(decision, "unverified_candidates", None) or []) if decision else []
    if not items:
        return
    st.subheader(f"Unverified candidates ({len(items)})")
    st.info("These products matched your search, but their details could not be loaded, so the "
            "final offer, price and budget compliance were **not verified**. Prices and "
            "merchants come from the search results only. Confirm on the store page before "
            "relying on them.")
    for start in range(0, len(items), 3):
        for col, candidate in zip(st.columns(3), items[start:start + 3]):
            with col:
                _render_unverified_card(candidate)


def _render_clarification(run_id: str, decision, meta: dict) -> None:
    form = dict(meta["form_inputs"])
    rounds = int(form.get("clarify_round", 0) or 0)
    if rounds >= service.max_clarifications():
        st.info("That's the limit for follow-up questions on one search. "
                "Start a new search with more detail.")
        return
    with st.form(f"clarify_{run_id}"):
        answer = st.text_input("Your answer", key=f"clarify_answer_{run_id}")
        go = st.form_submit_button("Search again", key=f"clarify_submit_{run_id}")
    if go and answer.strip():
        answer = answer.strip()
        # A bare currency code answers a "which currency?" question for a form budget.
        if len(answer) == 3 and answer.isalpha() and form.get("max_price") and not form.get("currency"):
            form["currency"] = answer.upper()
        form["clarify_round"] = rounds + 1
        _run_discovery(f"{meta['query']} {answer}"[:500], form)
    elif go:
        st.warning("Type an answer first.")


run_id = st.session_state.get("run_id")
result = service.load_run(run_id) if run_id else None
meta = service.load_run_meta(run_id) if run_id else None

if result is None or meta is None:
    if run_id:  # stale ID (e.g. database replaced)
        _select_run(None)
    if not history:
        st.info("Nothing here yet. Describe a product above to start your first search.")
else:
    decision = result.decision
    st.divider()
    st.markdown(f"### {STATUS_ICON.get(result.status, '')} {STATUS_LABEL.get(result.status, result.status)}")
    st.caption(f"Run ID `{result.run_id}` · Status `{result.status}`"
               + (" · partial results" if result.partial else "") + f" · “{meta['query']}”")

    if result.status == "blocked":
        st.warning(result.error or "That request was not run. Please describe the product you want.")
    elif result.status == "error":
        st.error(result.error or "The search could not be completed. Please try again.")
    elif result.status == "running":
        st.info("This search did not finish (the app may have been closed mid-run).")
    elif decision is not None:
        {"selected": st.success, "needs_selection": st.info,
         "clarify": st.warning, "failed": st.error}.get(result.status, st.info)(decision.message)

    if result.partial:
        _render_partial_banner(run_id, decision)

    if decision is not None:
        corrections = decision.diagnostics.get("corrections") or []
        if corrections:
            st.markdown("**Assumptions made**")
            for note in corrections:
                st.caption(f"• {note}")
        for item in result.unverified:
            text = (f"delivery to {item['value']}" if item.get("name") == "delivery_city"
                    else item.get("value", ""))
            st.caption(f"🟠 Unverified: {text}", help=UNVERIFIED_HELP)

    _render_products(run_id, decision)
    _render_unverified(decision)
    if result.status == "clarify":
        _render_clarification(run_id, decision, meta)
    _render_steps(run_id)

    if decision is not None and (decision.diagnostics.get("attempt_log")):
        with st.expander("Technical details"):
            st.json(decision.diagnostics["attempt_log"])