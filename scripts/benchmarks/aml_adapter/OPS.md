# AML adapter stability and operations note

## Authentication

- Require `AML_ADAPTER_API_KEY` in production; bind the same secret and one
  supported scheme (`Bearer`, `Token`, or `X-Api-Key`) in AML's evaluation
  configuration. Do not put secrets in endpoint URLs, logs, commits, or the
  public repository.
- Keep `AML_ADAPTER_MCP_TOKEN` separate from the AML-facing key. It is the
  adapter-to-AutoMem credential only.
- `/health` is intentionally unauthenticated as required by AML. It performs a
  read-only `check_database_health` MCP tool call and returns 503 if AutoMem is
  not ready.

## Capacity and rate limits

- AML can configure Add concurrency of 16–64 and Search concurrency of
  16–256. Deploy the adapter behind a reverse proxy with an explicit worker and
  connection budget, start conservatively (16 Add / 32 Search), then raise only
  after load testing the full AutoMem + FalkorDB + Qdrant path.
- Return 429 with `Retry-After` at the gateway when the known capacity is
  exceeded; AML retries 429 with bounded backoff. Do not queue Add behind the
  HTTP response: AML requires it to be searchable before 200.
- Set reverse-proxy and MCP upstream timeouts above the observed p99 but below
  the platform request timeout. Alert on sustained 5xx, 429, and p95 latency.

## 30-day availability plan

1. Deploy two adapter instances in separate failure domains behind HTTPS and a
   health-checked load balancer; pin the container/image and AutoMem version
   used for the submitted system version.
2. Keep managed FalkorDB/Qdrant backups, monitoring, and an on-call owner for
   the required 30 days. Test a restore before submitting.
3. Run synthetic authenticated Add → Search probes every minute against a
   dedicated non-evaluation `user_id`; redact request content from logs.
4. Retain only the data needed for the active AML job. Delete evaluation data,
   including derived copies and logs, within 30 days after completion. Use a
   dedicated AutoMem deployment/namespace for each AML run so deletion is
   auditable and cannot affect normal user memories.

## Submission gate

Do not submit a formal run until the official AML compatibility smoke passes
against the deployed endpoint, authenticated capacity testing is complete, the
submitted AutoMem configuration is pinned and recorded, and the 30-day
on-call/backup plan has an assigned owner. Resolve AutoMem MCP's current
50-result single-call ceiling if competitive `top_k=100` retrieval is required;
this adapter remains schema-compliant by returning fewer results.
