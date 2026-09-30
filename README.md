# Northstar Agentic Trading Platform

A safe-first research prototype for the trading-system roadmap. It now includes historical OHLCV replay, provider-neutral candle normalization, quality gates, deterministic indicators and regime snapshots, structured news classification, evidence-linked hypotheses, portfolio sizing, configurable compliance policies, a paper-only idempotent OMS contract, LLM output guardrails, an audit trail, and a local dashboard.

## Safety and scope

- Market data is deterministic synthetic sample data. It is not a quote for any real instrument or market.
- Backtest and paper replay use the same strategy and simulation engine. Paper replay is historical replay, not a live feed.
- Broker order execution is not implemented or enabled. Angel One integration is read-only market data, and no real order can be sent from this app.
- The risk gate sizes long-only entries and blocks entries when the kill switch is active. It is a development control, not a production trading risk system.
- The JSON candle adapter and keyword-based news classifier are provider-neutral examples. They do not connect to licensed market/news feeds or infer market facts independently. Live-mode candle ingestion rejects data older than 30 seconds; historical imports remain allowed.
- Angel One SmartAPI is the selected market-data provider. The adapter builds a NIFTY 50/NIFTY BANK plus nearest-expiry NFO options watchlist around ATM ±5 strikes, stores normalized ticks separately, and aggregates provisional one-minute bars. Cash equities are not subscribed until a symbol list is provided; futures are deferred.
- Angel One advertises SmartAPI at no API subscription charge, subject to its current account eligibility and terms; normal account, brokerage, taxes, and other charges may still apply. An active Angel One account, API key, and daily client-code/PIN/TOTP authentication are required. Authentication is initiated only from the loopback-only local dialog; PIN/TOTP are not stored, and session tokens are memory-only until logout, shutdown, or midnight expiry.
- Angel One data requires `ANGELONE_MARKET_DATA_ENABLED=true` and `ANGELONE_API_KEY`. The flag is off by default. It enables market data only; broker order execution remains unavailable. Angel One's 2026 notice describes a static-IP requirement for API order execution; this build has no order API. Confirm provider terms before using real data. No credentials are needed for tests.
- News keeps publication and ingestion timestamps, deduplicates normalized content across sources, and marks items older than 24 hours stale. The classifier is keyword-based, not an LLM or investment-research service.
- The OMS has a paper adapter only. Compliance policies must be explicitly configured with an authoritative exchange calendar; no country policy is enabled by default.
- Copilot and OpenAI are optional LLM providers, but inference stays disabled unless `LLM_ENABLED=true`. Every output passes the evidence and quantity guardrails; no model tools, MCP servers, or broker access are enabled.
- Results are illustrative only. Backtests do not model authoritative exchange calendars, corporate actions, partial fills, market impact, taxes, borrow, or survivorship/look-ahead-safe point-in-time universes.

Do not connect broker credentials or treat sample results as evidence of a profitable strategy.

## Run locally

From the repository root, with the repository's `uv` environment installed:

```powershell
uv run uvicorn --app-dir agentic-trading-platform api:app --reload
```

Open <http://127.0.0.1:8000>. The first launch creates `agentic-trading-platform/data/platform.sqlite3` and seeds 240 deterministic sample candles. The database is local runtime state and should not be committed.

For optional Angel One development, install `uv pip install -r agentic-trading-platform/requirements-angelone.txt`, copy `.env.example` to `agentic-trading-platform/.env`, set `ANGELONE_API_KEY`, and set `ANGELONE_MARKET_DATA_ENABLED=true`. Restart the service, then use **Login** in the dashboard and **Start feed** after successful authentication. Do not paste API keys, PINs, TOTP seeds/codes, or session tokens into chat. PIN/TOTP are not written to `.env` or the database.

The LLM gateway is separately opt-in and remains disabled by default. For Copilot, install `uv pip install -r agentic-trading-platform/requirements-copilot.txt`, sign in through the GitHub Copilot CLI using your GitHub account, then set `LLM_ENABLED=true`, `LLM_PROVIDER=copilot`, and optionally `COPILOT_MODEL=auto`. Requests consume the AI credits included with your Copilot plan; Copilot Pro does not include OpenAI API credits. The SDK session exposes no tools or MCP servers, rejects tool requests, and is deleted after each request. Alternatively, set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and `OPENAI_MODEL` for separately billed OpenAI API use. LLM requests are only available with a connected, fresh Angel One feed and never create orders.

If the SmartAPI app portal rejects loopback redirects, the optional `redirect_stub.py` serves a fixed page that discards query parameters. Run it on `127.0.0.1:8011`, then create a temporary Cloudflare Quick Tunnel to that port only:

```powershell
uv run --project .. python agentic-trading-platform/redirect_stub.py --port 8011
cloudflared tunnel --url http://127.0.0.1:8011
```

Register the printed `https://<random>.trycloudflare.com/angelone/callback` URL in SmartAPI. Quick Tunnel URLs are temporary and change when restarted. This callback does not capture SmartAPI publisher tokens or complete publisher-login; this app authenticates using the separate local TOTP form. The tunnel must never target the trading dashboard/API port.

## Tests

```powershell
Set-Location agentic-trading-platform
uv run --project .. python -m unittest discover -s tests -v
```

## API surface

- `GET /api/health`, `/api/instruments`, `/api/market/{symbol}/candles`, `/api/market/{symbol}/features`, `/api/signals/{symbol}`
- `POST /api/market/ingest` accepts normalized provider candle messages in `historical` or `live` mode and reports rejected duplicates, stale data, and ordering violations
- `GET /api/angelone/status`, `POST /api/angelone/login`, `POST /api/angelone/logout`, `POST /api/angelone/feed/start`, and `POST /api/angelone/history` control local market-data access; credential actions reject non-loopback callers
- `GET /api/market/{symbol}/ticks` and `WS /ws/market/ticks` read/publish normalized tick messages
- `POST /api/news`, `GET /api/news` for caller-supplied articles and deterministic event extraction
- `POST /api/hypotheses/{symbol}` returns a deterministic candidate and guardrail decision; stale demo data is rejected
- `POST /api/llm/hypotheses/{symbol}` is disabled unless a provider is explicitly configured and fresh feed data exists
- `POST /api/backtests`, `POST /api/paper/replay`
- `GET /api/risk`, `POST /api/risk/kill-switch`
- `GET /api/events`

The read-only provider adapter is in `angelone_adapter.py`; it calls only authentication, market-data, historical-data and WebSocket functionality. The pure service contracts for portfolio snapshots, stop-distance sizing, market-specific compliance, paper OMS, and model-output validation live in `portfolio.py`, `compliance.py`, `execution.py`, and `llm_gateway.py`. OMS remains paper-only; Copilot and OpenAI providers are separately opt-in.

## Architecture boundary

`core.py` owns the broker-independent candle, strategy, risk, and replay contracts. Signals are computed only from bars already observed and fill at the following bar's open. `store.py` keeps numerical OHLCV separate from versioned audit events in SQLite. `api.py` composes them; the browser UI never has access to broker credentials or execution APIs.

The appropriate next increments are licensed provider adapters, point-in-time historical data and a validation-grade backtester, followed by durable event streaming. Production news/RAG/LLM proposal services should only be introduced after those inputs and interfaces are reliable. Any eventual broker adapter must remain behind independent deterministic risk and market-specific compliance gates, with live execution disabled by default.

Cloud deployment remains deferred per the near-zero initial budget. ECS Fargate, RDS, S3, and EventBridge/SQS are not assumed free; no AWS resources were created. Production work still required includes real-account feed verification, cash watchlist symbols, feed failover, durable streaming, PostgreSQL/S3 migration, exchange calendars and legal review, stronger authentication/RBAC and secret management, persistent order/portfolio state, real model evaluation, broker OMS integration, tracing/metrics/alerts, and deployment/restore procedures.