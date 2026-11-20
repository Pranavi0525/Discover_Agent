"""Phase 3 (M3.1 + M3.2 steps 1-3): signed mandates and reconciliation.

* ECDSA over NIST P-256 (secp256r1) with SHA-256, signatures as base64url DER ("ES256").
* User Intent Mandate (UIM): what the user allowed (item, budget, currency, merchants).
* Merchant Cart Mandate (MCM): what the merchant is actually offering, signed by the merchant.
* reconcile(): the tallying engine. Returns a hash that binds BOTH signed payloads; the
  bank/payment step (M3.2 step 4, not here) only needs that hash, the merchant and the total.

Nothing here talks to a network or stores a key.  Private keys never enter a mandate.
"""
from __future__ import annotations

import base64
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .merchant_identity import merchant_match
from .schemas import UserIntent

ALG = "ES256"
DEFAULT_UIM_TTL = timedelta(hours=1)
MAX_CLOCK_SKEW = timedelta(seconds=30)


class MandateError(Exception):
    """A mandate is malformed, forged, expired or violates the user's intent."""


# --------------------------------------------------------------------- keys
def generate_private_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def private_key_to_pem(key: ec.EllipticCurvePrivateKey, password: bytes | None = None) -> bytes:
    enc = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc
    )


def public_key_to_pem(key: ec.EllipticCurvePublicKey) -> bytes:
    return key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


def load_public_key(pem: bytes) -> ec.EllipticCurvePublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, ec.SECP256R1):
        raise MandateError("public key must be an ECDSA P-256 key")
    return key


def key_id(public_key: ec.EllipticCurvePublicKey) -> str:
    """Stable fingerprint: first 16 hex chars of SHA-256 over the DER public key."""
    der = public_key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return hashlib.sha256(der).hexdigest()[:16]


# ----------------------------------------------------------------- payloads
def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


class _Mandate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 1
    mandate_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")
    issued_at: datetime
    expires_at: datetime

    @field_validator("issued_at", "expires_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        return _utc(v)


class UserIntentMandate(_Mandate):
    kind: str = "user_intent"
    item_summary: str = Field(min_length=2, max_length=300)
    max_price: Decimal = Field(ge=0)
    allowed_merchants: list[str] = Field(min_length=1, max_length=10)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    product_id: str = Field(min_length=1, max_length=200)
    title: str = Field(max_length=300)
    quantity: int = Field(ge=1, le=100)
    unit_price: Decimal = Field(ge=0)


class MerchantCartMandate(_Mandate):
    kind: str = "merchant_cart"
    cart_id: str = Field(min_length=1, max_length=200)
    merchant_name: str = Field(min_length=1, max_length=200)
    merchant_url: str = Field(min_length=8, max_length=500)  # host is what proves identity
    line_items: list[LineItem] = Field(min_length=1, max_length=50)
    subtotal: Decimal = Field(ge=0)
    shipping: Decimal = Field(ge=0, default=Decimal("0"))

    @property
    def total(self) -> Decimal:
        return self.subtotal + self.shipping


class SignedMandate(BaseModel):
    """Payload (as canonical JSON text) + signature + the signer's key id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    alg: str = ALG
    key_id: str
    payload: str
    signature: str


def canonical_json(mandate: BaseModel) -> str:
    """Deterministic text: sorted keys, no spaces, Decimals as strings, UTC ISO times."""
    data = mandate.model_dump(mode="json")
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_mandate(mandate: BaseModel, private_key: ec.EllipticCurvePrivateKey) -> SignedMandate:
    payload = canonical_json(mandate)
    sig = private_key.sign(payload.encode(), ec.ECDSA(hashes.SHA256()))
    return SignedMandate(
        key_id=key_id(private_key.public_key()), payload=payload, signature=_b64(sig)
    )


def _verify(
    signed: SignedMandate, public_key: ec.EllipticCurvePublicKey, model: type[BaseModel]
):
    if signed.alg != ALG:
        raise MandateError(f"unsupported algorithm {signed.alg!r}")
    if signed.key_id != key_id(public_key):
        raise MandateError("signed by a different key than the one trusted")
    try:
        public_key.verify(_unb64(signed.signature), signed.payload.encode(), ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError) as exc:
        raise MandateError("signature does not match the payload") from exc
    try:
        mandate = model.model_validate_json(signed.payload)
    except ValidationError as exc:
        raise MandateError("payload is not a valid mandate") from exc
    # Re-serialising must reproduce the signed text, so nothing unsigned can ride along.
    if canonical_json(mandate) != signed.payload:
        raise MandateError("payload is not in canonical form")
    return mandate


def _check_times(mandate: _Mandate, now: datetime) -> None:
    now = _utc(now)
    if mandate.issued_at > now + MAX_CLOCK_SKEW:
        raise MandateError("mandate is issued in the future")
    if now >= mandate.expires_at:
        raise MandateError("mandate has expired")
    if mandate.expires_at <= mandate.issued_at:
        raise MandateError("mandate expires before it is issued")


def verify_user_intent(
    signed: SignedMandate, user_public_key: ec.EllipticCurvePublicKey, now: datetime
) -> UserIntentMandate:
    mandate = _verify(signed, user_public_key, UserIntentMandate)
    _check_times(mandate, now)
    return mandate


def verify_cart(
    signed: SignedMandate, merchant_public_key: ec.EllipticCurvePublicKey, now: datetime
) -> MerchantCartMandate:
    mandate = _verify(signed, merchant_public_key, MerchantCartMandate)
    _check_times(mandate, now)
    return mandate


# ------------------------------------------------------------------ builders
def user_intent_from(
    intent: UserIntent,
    *,
    now: datetime,
    ttl: timedelta = DEFAULT_UIM_TTL,
    allowed_merchants: list[str] | None = None,
) -> UserIntentMandate:
    """A mandate needs hard bounds: budget, currency and at least one merchant."""
    merchants = allowed_merchants or ([intent.merchant] if intent.merchant else [])
    if intent.max_price is None or intent.currency is None:
        raise MandateError("a mandate needs both a budget and a currency")
    if not merchants:
        raise MandateError("a mandate needs at least one allowed merchant")
    return UserIntentMandate(
        item_summary=intent.query,
        max_price=intent.max_price,
        currency=intent.currency.upper(),
        allowed_merchants=merchants,
        issued_at=now,
        expires_at=now + ttl,
    )


# ------------------------------------------------------------ reconciliation
@dataclass(frozen=True)
class Reconciliation:
    mandate_hash: str  # SHA-256 over both signed payloads: what the bank verifies
    merchant: str
    total: Decimal
    currency: str
    cart_id: str


def reconcile(
    signed_uim: SignedMandate,
    signed_mcm: SignedMandate,
    *,
    user_public_key: ec.EllipticCurvePublicKey,
    merchant_public_key: ec.EllipticCurvePublicKey,
    now: datetime,
) -> Reconciliation:
    """Tallying engine.  Raises MandateError on the first violated rule."""
    uim = verify_user_intent(signed_uim, user_public_key, now)
    mcm = verify_cart(signed_mcm, merchant_public_key, now)

    if mcm.currency != uim.currency:
        raise MandateError(f"cart currency {mcm.currency} != intent currency {uim.currency}")

    computed = sum((li.unit_price * li.quantity for li in mcm.line_items), Decimal("0"))
    if computed != mcm.subtotal:
        raise MandateError("cart subtotal does not equal its line items")

    if mcm.total > uim.max_price:
        raise MandateError(f"cart total {mcm.total} exceeds the budget {uim.max_price}")

    # Same evidence rule as search: only the URL host proves the merchant.  A matching
    # seller NAME alone ("unverified") is not enough to spend money.
    if not any(
        merchant_match(allowed, mcm.merchant_name, [mcm.merchant_url]) == "verified"
        for allowed in uim.allowed_merchants
    ):
        raise MandateError("cart merchant is not one of the allowed merchants")

    digest = hashlib.sha256(
        (signed_uim.payload + "\n" + signed_mcm.payload).encode()
    ).hexdigest()
    return Reconciliation(
        mandate_hash=digest,
        merchant=mcm.merchant_name,
        total=mcm.total,
        currency=mcm.currency,
        cart_id=mcm.cart_id,
    )
