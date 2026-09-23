"""Ray Serve's request router overflows its own backoff after ~510 s of waiting.

`RequestRouter._compute_backoff_s` (ray/serve/_private/request_router/request_router.py)
clamps *after* evaluating the power:

    return min(
        self.initial_backoff_s * (self.backoff_multiplier**attempt),
        self.max_backoff_s,
    )

`backoff_multiplier` defaults to the int 2, so `2**attempt` is an exact
arbitrary-precision int that never overflows on its own; the float multiply
then raises `OverflowError` instead of saturating, and `min` never gets to
clamp it. With the defaults (0.025 s initial, x2, 0.5 s cap) the backoff is
already pinned at the 0.5 s cap from attempt ~5, so a request reaches the
overflowing attempt 1024 after roughly 510 s of waiting to be routed.

`_fulfill_pending_requests` logs it as "Unexpected error in
_fulfill_pending_requests" and its `finally` drops the routing task; the
pending request is left unfulfilled. A new request or a replica-set update
restarts a routing task and picks it up, so a busy deployment recovers -- but
nothing restarts one periodically, so a request stranded at the tail of a run
has nothing to rescue it.

Any deployment whose work items outlast the router's patience hits this: it
needs only that some request waits ~510 s for a free replica. `pfb degrid-msv4`
did, with multi-minute items and an in-flight queue deeper than the replicas
could accept (fixed by tying the two together -- see `core/degrid_msv4.py`).

Run: uv run python scripts/ray_issues/serve_router_backoff_overflow.py
No cluster, no MS, no deployment -- it only does the router's arithmetic.
"""

from ray.serve._private.common import DeploymentHandleSource, DeploymentID
from ray.serve._private.request_router.pow_2_router import PowerOfTwoChoicesRequestRouter

router = PowerOfTwoChoicesRequestRouter(
    deployment_id=DeploymentID(name="d", app_name="a"),
    handle_source=DeploymentHandleSource.PROXY,
)

print(
    f"initial_backoff_s={router.initial_backoff_s} "
    f"backoff_multiplier={router.backoff_multiplier!r} "
    f"max_backoff_s={router.max_backoff_s}"
)

waited = 0.0
for attempt in range(2000):
    try:
        waited += router._compute_backoff_s(attempt)
    except OverflowError as e:
        print(f"OverflowError at attempt={attempt} after ~{waited:.0f}s of routing: {e}")
        break
else:
    print("no overflow -- defaults must have changed")
