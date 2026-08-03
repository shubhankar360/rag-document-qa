# Orbit Analytics — API guide

## Authentication

Create an API key under Settings, Developers, API Keys. Keys are shown once at
creation time and cannot be retrieved afterwards, so store the value in your own
secret manager.

Rotating a key issues a new secret and keeps the old one valid for twenty four
hours, which lets you deploy the replacement without downtime. Revoking a key
takes effect immediately with no grace period.

## Rate limits

The ingest endpoint accepts sixty requests per second per API key. Requests
beyond that limit receive HTTP 429 with a Retry-After header indicating how long
to wait before retrying.

To stay under the limit, batch up to five hundred events into a single request.
Batched events count individually against your monthly event quota but as one
request against the rate limit.

## Errors

A 401 response means the API key is missing or revoked. A 403 means the key
lacks the required scope. A 422 means the event payload failed schema
validation; the response body names the offending field.
