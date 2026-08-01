# Auth Service Troubleshooting Guide

The `auth-service` issues and validates session tokens for all platform services. This guide covers the most frequent incident classes and the recommended maintenance procedure.

## Symptom: widespread E301 (token expired)

When many clients report E301 at the same time, the cause is almost never individual sessions expiring. Investigate in this order:

1. Check NTP synchronisation across all auth nodes. Clock skew greater than 30 seconds between the issuing node and the validating node causes valid tokens to appear expired.
2. Confirm the token lifetime configuration was not shortened by a recent deploy.
3. Inspect the key-rotation schedule. A key rotated without an overlap window invalidates tokens issued under the previous key.

## Symptom: intermittent login failures

Intermittent failures usually trace back to a single unhealthy node behind the load balancer. Drain the suspect node and observe whether the error rate drops.

## Scheduled maintenance procedure

The auth-service requires a key rotation every 90 days. The rotation must use a dual-key overlap window of at least 24 hours so that tokens issued under the old key remain valid until they naturally expire. Skipping the overlap window is the single most common cause of a self-inflicted E301 incident.

Before any maintenance, drain traffic gradually and confirm the standby region is healthy. Never rotate keys in more than one region simultaneously.
