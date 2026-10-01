# Northstar Agentic Trading Platform

A safe-first research prototype for the trading-system roadmap. It now includes historical OHLCV replay, provider-neutral candle normalization, quality gates, deterministic indicators and regime snapshots, structured news classification, evidence-linked hypotheses, portfolio sizing, configurable compliance policies, a paper-only idempotent OMS contract, LLM output guardrails, an audit trail, and a local dashboard.

The India-first reference-data foundation is in `india_market.py`. It defines explicit exchange, instrument, derivative contract, session, holiday, expiry-calendar, lot-size, tick-size, and corporate-action records. It contains no assumed NSE/BSE sessions, holidays, lot sizes, tick sizes, expiries, or transaction charges; authoritative provider/reference data must be supplied before those rules can be used. It is a configuration/domain layer, not a production compliance certification.

The local real-time pipeline normalizes accepted Angel One ticks, aggregates provisional `1m` bars, and emits deterministic feature snapshots only when a later tick closes the prior minute. Snapshots are persisted as audit events and published on `WS /ws/market/features`; this is an in-process Python stream, not Kafka, and does not authorize orders.

News Phase 6 is available through `POST /api/news` for one article and `POST /api/news/batch` for up to 100 provider-normalized articles. Ingestion validates ISO timestamps and source URLs, records publication and ingestion times, normalizes instruments, deduplicates normalized content across sources, and persists articles plus intelligence in SQLite. Deterministic extraction preserves a primary event (`event_type`) and all matched event types in `extracted_events`, with sentiment, severity, horizon, confidence, source reliability, evidence, and a 24-hour freshness flag. `GET /api/news` returns the stored records. No external news subscription or licensed feed is connected by default; callers must supply provider-normalized article payloads.

RAG Phase 7 is local and deterministic. `POST /api/rag/memory` stores historical trading context with a stable document id, kind, text, and metadata; `POST /api/rag/search` embeds a query and performs cosine retrieval with exact metadata filters such as `instrument`, `timeframe`, `source`, and `event_type`. Embeddings use a reproducible hashing model and vectors are persisted in the SQLite `rag_documents` table. Search returns provenance and similarity scores, never raw vectors, and has no execution authority. This is a research-grade local vector store, not a substitute for a managed embedding model or production ANN index.

AI Phase 8 uses `LLMGateway` as the provider boundary for OpenAI or Copilot structured JSON responses. `StrategyAgent` delegates evidence-linked hypothesis generation to that gateway, while `AdversarialAgent` independently checks freshness, evidence coverage, contradictory evidence, direction constraints, and reward/risk levels. `POST /api/llm/hypotheses/{symbol}` returns both the structured strategy decision and adversarial review, and records both in the audit trail. Tool use, quantity selection, and broker execution remain disabled.

Phase 9 is composed by `DecisionEngine` in `decision_engine.py`. It requires a calibrated probability, calculates explicit India round-trip costs and net expected value, sizes the candidate from deterministic equity/stop/exposure limits, evaluates portfolio concentration and correlation limits, and applies the configured market compliance policy. It returns `eligible_for_risk_review` only; zero calibration samples, stale/missing compliance state, negative net EV, or breached limits fail closed. The engine has no broker or order authority.

Phase 10 is represented by `LivePaperTradingSession` in `paper_trading.py`. It accepts normalized live ticks, rejects stale or future data before decision-making, consumes a full decision result, and submits only to the in-memory `PaperBrokerAdapter`. Paper mode and human approval are explicit gates; disabled paper mode, rejected decisions, missing approval, stale data, risk failures, and compliance failures produce no simulated fill. The session has no live broker path.

Phase 11 adds broker and OMS contracts in `broker_adapter.py` and `oms.py`. `AngelOneReadOnlyBrokerAdapter` exposes portfolio reconciliation only, `DisabledLiveOrderAdapter` rejects live submissions, and `OrderManagementSystem` provides idempotent paper-order state. `OrderReconciler` compares local OMS orders with broker snapshots and reports status, fill-quantity, price, missing-order, and unknown-order discrepancies without silently mutating either side. Compliance and execution remain deterministic and paper-only.

Phase 12 adds the controlled-live gate in `controlled_live.py`. Live trading defaults to disabled, starts with an active kill switch, requires expiring human approval, enforces strict capital allocation, per-order value, freshness, and daily-loss limits, and writes audit events for approvals, gate decisions, and submission outcomes. The gate is separate from broker connectivity; the current Angel One adapter remains read-only, so no live order can be submitted until a separately authorized live-order adapter is implemented and supplied.

The connected runtime path is available through `RuntimeDecisionPipeline` in `runtime_pipeline.py`. When explicitly installed in `angel_runtime["decision_pipeline"]`, an accepted live tick flows through multiple deterministic alphas and aggregation, the structured strategy agent, adversarial approval, historical calibration, `DecisionEngine`, paper OMS execution, the dedicated trade journal, and RAG trading-memory write-back. The callback also supplies historical similarity, India regime, current portfolio status, and an explicit breadth-unavailable result when no point-in-time universe is configured. The pipeline is unset by default; live broker orders remain disabled.

Every accepted paper trade also receives a validated `TradeTrace` from `trade_governance.py`. The trace requires measurable alpha/probability/expected-value values, evidence-linked explanation, reproducibility and context hashes, a historical-validation/backtest reference, portfolio and risk controls, compliance output, and audit identifiers. It is stored with the trade journal and written back to trading memory, making the trade measurable, explainable, reproducible, backtestable, auditable, and risk-controlled.

Candles are stored independently by `(symbol, timeframe, timestamp)`. Supported intervals are `1m`, `3m`, `5m`, `10m`, `15m`, `30m`, `1h`, `1d`, and `1w`; existing rows from before timeframe tracking are preserved as `legacy` rather than assigned a guessed interval. Candle, feature, signal, hypothesis, LLM, backtest, and paper-replay endpoints default to `1d`; pass `?timeframe=1m` (or another supported interval) to select a different series. Angel One history intervals map to the corresponding canonical timeframe, and live ticks aggregate as `1m`.

`GET /api/market/regime?timeframe=1d` classifies the broad India market using aligned NIFTY 50 and NIFTY BANK feature snapshots. It returns `INSUFFICIENT_DATA` if either series or sufficient feature history is unavailable. Bull/bear strength uses the configurable `INDIA_STRONG_MOMENTUM_THRESHOLD_PCT` (initial research default `2.0`); this is a model threshold, not an exchange rule. Regime output has no execution authority.

`POST /api/market/breadth` accepts an explicit symbol universe, timeframe, and optional sector mapping. It reports advances/declines, SMA participation, volume breadth, 252-bar highs/lows when history permits, sector breadth, index participation, and missing/misaligned coverage. It does not assume current or historical NIFTY constituent membership; supply a point-in-time universe before interpreting the result as index breadth.

`GET /api/market/{symbol}/similarity?timeframe=1d&horizon_bars=5` finds nearest historical quantitative feature states within the same instrument/timeframe, then measures outcomes only in candles strictly after each historical state. It returns return quantiles, MAE/MFE, a fixed forward horizon, and the selected analogues. The sample win rate is descriptive, not a calibrated probability or execution signal.

`POST /api/india/expected-value` calculates configurable round-trip Indian costs and net EV using `Decimal`; every product rate, GST base, brokerage cap choice, slippage, and impact assumption must be supplied. Its probability is labeled caller-supplied/unvalidated. The calibration utility withholds calibrated probabilities below its configured minimum sample count; durable prediction/outcome history and a production calibration pipeline are not yet implemented.

`PortfolioOptimizer` in `portfolio.py` is a deterministic candidate screen for explicit symbol/sector/asset limits, correlation inputs, and positive net EV. Missing correlations fail closed. It currently returns `eligible_for_risk_review` only; it is not wired into OMS submission or a live portfolio reconciliation service.

`POST /api/backtests/walk-forward` runs expanding-window out-of-sample folds for the existing deterministic strategy. Warmup bars are excluded from scoring and test windows cannot overlap. Backtest and walk-forward requests can supply an explicit per-product India charge schedule (`india_product`, `india_charge_rates`); set `fee_bps=0` with that schedule to avoid double counting. Generic bps fees remain available for demo runs. Market impact and volume participation are configurable; partial BUY remainders are cancelled, partial SELLs leave residual holdings visible, and a fold-end liquidation can remain incomplete if volume is insufficient. `execution_delay_bars` is a bar-level stale-decision proxy, not measured wall-clock latency. This remains a simulation, not a broker-fill or production execution model.

Phase 4 is still incomplete: replay does not yet apply corporate actions, derivatives expiry/contract changes, historical news, or historical portfolio snapshots. Participation-limited fills are a simple bar-volume proxy; execution delay is measured in bars, and market impact is a configured bps assumption rather than a market-depth model. Production-quality validation also requires authoritative point-in-time datasets and settlement rules.
## Safety and scope
- Market data is deterministic synthetic sample data. It is not a quote for any real instrument or market.
- Backtest and paper replay use the same strategy and simulation engine. Paper replay is historical replay, not a live feed.
- Broker order execution is not implemented or enabled. Angel One integration is read-only market data, and no real order can be sent from this app.
- The risk gate sizes long-only entries and blocks entries when the kill switch is active. It is a development control, not a production trading risk system.
- The JSON candle adapter and keyword-based news classifier are provider-neutral examples. They do not connect to licensed market/news feeds or infer market facts independently. Live-mode candle ingestion rejects data older than 30 seconds; historical imports remain allowed.
- Angel One SmartAPI is the selected market-data provider. The adapter builds a NIFTY 50/NIFTY BANK plus nearest-expiry NFO options watchlist around ATM ±5 strikes, stores normalized ticks separately, and aggregates provisional one-minute bars. Cash equities are not subscribed until a symbol list is provided; futures are deferred.
- India reference refresh imports the public Angel One instrument master into versioned, content-addressed SQLite snapshots. It does not log in, open a market feed, or enable orders. Unsupported or incomplete master rows are counted and skipped; sessions, holidays, and other exchange rules remain unconfigured until supplied by their authoritative sources.
- Angel One advertises SmartAPI at no API subscription charge, subject to its current account eligibility and terms; normal account, brokerage, taxes, and other charges may still apply. An active Angel One account, API key, and daily client-code/PIN/TOTP authentication are required. Authentication is initiated only from the loopback-only local dialog; PIN/TOTP are not stored, and session tokens are memory-only until logout, shutdown, or midnight expiry.
- Angel One data requires `ANGELONE_MARKET_DATA_ENABLED=true` and `ANGELONE_API_KEY`. The flag is off by default. It enables market data only; broker order execution remains unavailable. Angel One's 2026 notice describes a static-IP requirement for API order execution; this build has no order API. Confirm provider terms before using real data. No credentials are needed for tests.
- News keeps publication and ingestion timestamps, deduplicates normalized content across sources, and marks items older than 24 hours stale. The classifier is keyword-based, not an LLM or investment-research service.
- The OMS has a paper adapter only. Compliance policies must be explicitly configured with an authoritative exchange calendar; no country policy is enabled by default.
- Copilot and OpenAI are optional LLM providers, but inference stays disabled unless `LLM_ENABLED=true`. Every output passes the evidence and quantity guardrails; no model tools, MCP servers, or broker access are enabled.
- Results are illustrative only. Configured costs, impact, latency, and partial fills are simulation assumptions, not broker-accurate execution. Backtests still do not model authoritative exchange calendars, corporate actions, derivative expiry/contract changes, historical news/portfolio replay, borrow, or survivorship-safe point-in-time universes.

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

From the trading-platform directory, install the lightweight test tools and run the suite. Pytest also runs the existing `unittest` cases and reports line and branch coverage:

```powershell
uv pip install --python ..\.venv\Scripts\python.exe -r requirements-test.txt
..\.venv\Scripts\python.exe -m pytest
```

Run one test module or the feed-to-strategy-to-LLM API integration check:

```powershell
..\.venv\Scripts\python.exe -m pytest tests/test_runtime_pipeline.py
..\.venv\Scripts\python.exe -m pytest tests/test_api_provider_status.py
```

The suite uses `unit` as the default marker for untagged tests. Boundary suites carry `component`, `integration`, `event_driven`, `e2e`, `replay`, and `chaos` markers. Select a pyramid layer with `-m`, and run bounded load checks separately:

```powershell
..\.venv\Scripts\python.exe -m pytest -m unit
..\.venv\Scripts\python.exe -m pytest -m component
..\.venv\Scripts\python.exe -m pytest -m integration
..\.venv\Scripts\python.exe -m pytest -m event_driven
..\.venv\Scripts\python.exe -m pytest -m e2e
..\.venv\Scripts\python.exe -m pytest -m replay
..\.venv\Scripts\python.exe -m pytest -m chaos
..\.venv\Scripts\python.exe -m pytest -m performance
```

Property-based financial invariants use deterministic Hypothesis examples. Tests use temporary SQLite databases, fake HTTP/provider clients, and paper or disabled broker adapters; they do not read production broker credentials, connect to Angel One/Copilot/OpenAI, or submit real orders. The suite currently includes failing regression tests for conflicting OMS idempotency keys and non-finite risk/compliance inputs; those failures identify production defects and should not be skipped or weakened.

## API surface

- `GET /api/health`, `/api/instruments`, `/api/market/{symbol}/candles?timeframe=1d`, `/api/market/{symbol}/features?timeframe=1d`, `/api/market/{symbol}/features/multi-timeframe`, `/api/market/regime?timeframe=1d`, `/api/signals/{symbol}?timeframe=1d`
- `POST /api/market/breadth` calculates deterministic cross-sectional breadth for the supplied symbol list; it does not load or assume an index constituent universe
- `GET /api/market/{symbol}/similarity` searches historical states and scores strictly forward outcomes; outputs are not calibrated probabilities
- `POST /api/india/expected-value` computes net expected value from caller-supplied probability, gross win/loss, explicit cost schedule, slippage, and impact; it never submits an order
- `POST /api/market/ingest` accepts normalized provider candle messages in `historical` or `live` mode and reports rejected duplicates, stale data, and ordering violations
- `GET /api/angelone/status`, `POST /api/angelone/login`, `POST /api/angelone/logout`, `POST /api/angelone/feed/start`, and `POST /api/angelone/history` control local market-data access; credential actions reject non-loopback callers
- `GET /api/india/reference-data/status` and loopback-only `POST /api/india/reference-data/refresh` inspect/import canonical instrument reference data; refresh requires Angel One opt-in and `ANGELONE_API_KEY`, but does not start a feed
- `GET /api/market/{symbol}/ticks`, `WS /ws/market/ticks`, and `WS /ws/market/features` read/publish normalized ticks and closed-minute feature snapshots
- `POST /api/news`, `GET /api/news` for caller-supplied articles and deterministic event extraction
- `POST /api/hypotheses/{symbol}` returns a deterministic candidate and guardrail decision; stale demo data is rejected
- `POST /api/llm/hypotheses/{symbol}` is disabled unless a provider is explicitly configured and fresh feed data exists
- `POST /api/backtests`, `POST /api/backtests/walk-forward`, `POST /api/paper/replay`; simulation requests may provide explicit India charge rates, market impact, bar delay, and volume participation assumptions
- `GET /api/risk`, `POST /api/risk/kill-switch`
- `GET /api/events`

The read-only provider adapter is in `angelone_adapter.py`; it calls only authentication, market-data, historical-data and WebSocket functionality. The pure service contracts for portfolio snapshots, stop-distance sizing, market-specific compliance, paper OMS, and model-output validation live in `portfolio.py`, `compliance.py`, `execution.py`, and `llm_gateway.py`. OMS remains paper-only; Copilot and OpenAI providers are separately opt-in.

## Architecture boundary

`core.py` owns the broker-independent candle, strategy, risk, and replay contracts. Signals are computed only from bars already observed and fill at the following bar's open. `store.py` keeps numerical OHLCV separate from versioned audit events in SQLite. `api.py` composes them; the browser UI never has access to broker credentials or execution APIs.

The appropriate next increments are licensed provider adapters, point-in-time historical data and a validation-grade backtester, followed by durable event streaming. Production news/RAG/LLM proposal services should only be introduced after those inputs and interfaces are reliable. Any eventual broker adapter must remain behind independent deterministic risk and market-specific compliance gates, with live execution disabled by default.

Cloud deployment remains deferred per the near-zero initial budget. ECS Fargate, RDS, S3, and EventBridge/SQS are not assumed free; no AWS resources were created. Production work still required includes real-account feed verification, cash watchlist symbols, feed failover, durable streaming, PostgreSQL/S3 migration, exchange calendars and legal review, stronger authentication/RBAC and secret management, persistent order/portfolio state, real model evaluation, broker OMS integration, tracing/metrics/alerts, and deployment/restore procedures.