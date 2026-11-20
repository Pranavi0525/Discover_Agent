import sys
from discovery_agent.policy_check import HttpxFetcher, analyze_returns, analyze_shipping

host = sys.argv[1]  # e.g. some-store.myshopify.com or www.store.in
f = HttpxFetcher()

refund = f.fetch(f"https://{host}/policies/refund-policy")
print("REFUND PAGE:", "COULD NOT FETCH" if refund is None else refund[:600])
if refund:
    print(analyze_returns(refund))

ship = f.fetch(f"https://{host}/policies/shipping-policy")
print("\nSHIPPING PAGE:", "COULD NOT FETCH" if ship is None else ship[:600])
if ship:
    print(analyze_shipping(ship, city="Chennai"))