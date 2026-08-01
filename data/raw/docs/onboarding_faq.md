# Support Engineer Onboarding FAQ

This FAQ answers the questions new support engineers ask most often in their first week.

## How do I find the current status of a service?

Every platform service publishes its health on the status dashboard. A service is one of `healthy`, `degraded`, or `down`. A `degraded` service is still serving traffic but has breached at least one service-level objective.

## What is a ticket priority and how is it set?

Ticket priority is one of `critical`, `high`, `medium`, or `low`. Priority is derived from customer impact, not from how loud the reporter is. A single customer blocked on a workaround is `medium`; a whole region unable to authenticate is `critical`.

## When should I escalate?

Escalate when an incident has an active `critical` ticket that has not moved toward resolution within its response target, or when an error code points to a downstream you do not own (for example E402). Escalation means engaging the owning team, not reassigning the ticket and walking away.

## Which services exist?

The core services you will see on tickets are `auth-service`, `payment-gateway`, and `search-index`. Each has an owning team and an on-call rotation. When in doubt about ownership, check the service catalogue.

## How do I close a ticket correctly?

A ticket may only be closed once its root cause is recorded and the customer has confirmed resolution. Closing a ticket without a recorded root cause is the most common audit finding.
