# Error Code Reference

This reference lists the standard error codes surfaced by platform services, their meaning, and the recommended remediation. Support engineers should consult this table before escalating a ticket.

## E104 — Rate limit exceeded

The service rejected a request because the caller exceeded its allocated request quota within the sliding window. This is most common on the `payment-gateway` and `search-index` services during traffic spikes.

**Recommended action:** Verify the client is honouring the `Retry-After` header. If the spike is legitimate, request a temporary quota increase through the capacity team. Do not retry aggressively; exponential backoff is required. A sustained E104 across many clients usually indicates a misconfigured upstream retry loop.

## E205 — Database connection failure

The service could not obtain a connection from the database pool. Typical causes are pool exhaustion, a failover in progress, or a network partition between the service and its primary replica.

**Recommended action:** Check the connection pool saturation metric first. If the pool is exhausted, look for a leaked transaction or a slow query holding connections. If a failover is in progress, the error is transient and should clear within 60 seconds.

## E301 — Authentication token expired

The `auth-service` rejected a token whose expiry timestamp is in the past. This is expected behaviour when a session outlives its token lifetime, but a sudden surge indicates a clock-skew problem between issuer and validator.

**Recommended action:** Confirm the client is refreshing tokens before expiry. If many clients report E301 simultaneously, check NTP synchronisation on the auth nodes. See the auth troubleshooting guide for the full escalation path.

## E402 — Downstream dependency unavailable

The service returned a degraded response because a required downstream dependency did not respond within the timeout budget. The service applied a circuit breaker to protect itself.

**Recommended action:** Identify which downstream is open on the circuit-breaker dashboard. E402 is a symptom, not a root cause; the owning team of the failing downstream must be engaged.
