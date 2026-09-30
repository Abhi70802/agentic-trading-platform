const byId = (id) => document.getElementById(id);
const money = (value) => new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(value);
let toastTimer;
let marketSocket;
let marketHeartbeat;
let marketReconnect;
let marketWatchlist = [];
const latestTicks = new Map();

function notify(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3000);
}

async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Request failed");
  return payload;
}

function renderChart(points) {
  const svg = byId("equity-chart");
  if (!points.length) return;
  const width = 900;
  const height = 250;
  const left = 55;
  const right = 12;
  const top = 12;
  const bottom = 23;
  const values = points.map((point) => point.equity);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const spread = Math.max(high - low, high * 0.005, 1);
  const x = (index) => left + index * (width - left - right) / Math.max(points.length - 1, 1);
  const y = (value) => top + (high + spread * 0.08 - value) * (height - top - bottom) / (spread * 1.16);
  const line = points.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(point.equity).toFixed(1)}`).join(" ");
  const area = `${line} L${x(points.length - 1).toFixed(1)},${height - bottom} L${left},${height - bottom} Z`;
  const grid = [0, 1, 2, 3].map((step) => {
    const value = high - spread * step / 3;
    const rowY = y(value);
    return `<line class="chart-grid" x1="${left}" y1="${rowY}" x2="${width - right}" y2="${rowY}"/><text class="chart-label" x="0" y="${rowY + 3}">${Math.round(value).toLocaleString("en-IN")}</text>`;
  }).join("");
  svg.innerHTML = `<defs><linearGradient id="area-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#3b9a70" stop-opacity=".2"/><stop offset="100%" stop-color="#3b9a70" stop-opacity="0"/></linearGradient></defs>${grid}<path class="chart-area" d="${area}"/><path class="chart-line" d="${line}"/>`;
}

function renderTrades(trades) {
  const table = byId("trades-table");
  byId("trade-count").textContent = `${trades.length} record${trades.length === 1 ? "" : "s"}`;
  table.replaceChildren();
  if (!trades.length) {
    const row = table.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 5;
    cell.className = "empty-cell";
    cell.textContent = "No closed trades in this replay.";
    return;
  }
  for (const trade of [...trades].reverse().slice(0, 8)) {
    const row = table.insertRow();
    const values = [trade.symbol, formatDate(trade.entry_time), formatDate(trade.exit_time), trade.quantity, money(trade.pnl)];
    values.forEach((value, index) => {
      const cell = row.insertCell();
      cell.textContent = value;
      if (index === 4) cell.className = `pnl ${trade.pnl >= 0 ? "positive" : "negative"}`;
    });
  }
}

function formatDate(value) {
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "2-digit", timeZone: "UTC" });
}

function renderEvents(events) {
  const container = byId("event-list");
  container.replaceChildren();
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "empty-cell";
    empty.textContent = "No simulation events recorded.";
    container.append(empty);
    return;
  }
  for (const event of events.slice(0, 9)) {
    const item = document.createElement("div");
    item.className = "event-item";
    const mark = document.createElement("span");
    mark.className = "event-mark";
    const copy = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = event.event_type.replaceAll(".", " · ");
    const time = document.createElement("small");
    time.textContent = `${formatDate(event.timestamp)} · ${event.correlation_id.slice(0, 8)}`;
    const payload = document.createElement("code");
    payload.textContent = JSON.stringify(event.payload).slice(0, 150);
    copy.append(title, time, payload);
    item.append(mark, copy);
    container.append(item);
  }
}

function renderFeatures(snapshot) {
  const features = snapshot.features;
  byId("feature-rsi").textContent = features.rsi_14 == null ? "—" : features.rsi_14.toFixed(1);
  byId("feature-atr").textContent = features.atr_14 == null ? "—" : features.atr_14.toFixed(2);
  byId("feature-volume").textContent = features.volume_zscore_20 == null ? "—" : `${features.volume_zscore_20.toFixed(1)}σ`;
  byId("feature-regime").textContent = snapshot.regime.trend.replaceAll("_", " ");
}

function renderHypothesis(result) {
  const { hypothesis, validation } = result;
  const actionable = hypothesis.direction === "LONG";
  byId("hypothesis-status").textContent = validation.approved ? "VALIDATED" : actionable ? "REJECTED" : "NO SIGNAL";
  byId("hypothesis-status").classList.toggle("rejected", !validation.approved && actionable);
  byId("hypothesis-thesis").textContent = hypothesis.thesis;
  const evidence = hypothesis.supporting_evidence.map((item) => item.source_id).join(", ");
  const reason = validation.reasons[0] || "Risk and data guardrails passed; execution remains disabled.";
  byId("hypothesis-evidence").textContent = `${reason} · Evidence: ${evidence}`;
}

function renderNews(items) {
  const list = byId("news-list");
  byId("news-count").textContent = `${items.length} item${items.length === 1 ? "" : "s"}`;
  list.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-cell";
    empty.textContent = "No news has been ingested. External feeds are not connected.";
    list.append(empty);
    return;
  }
  for (const item of items.slice(0, 5)) {
    const article = document.createElement("div");
    article.className = "news-item";
    const title = document.createElement("strong");
    title.textContent = item.title;
    const meta = document.createElement("small");
    meta.textContent = `${item.source} · Published ${formatDate(item.published_at)} · Ingested ${formatDate(item.ingested_at)}`;
    if (item.original_url) {
      const sourceLink = document.createElement("a");
      sourceLink.className = "news-source-link";
      sourceLink.href = item.original_url;
      sourceLink.target = "_blank";
      sourceLink.rel = "noopener noreferrer";
      sourceLink.textContent = "Open source ↗";
      meta.append(" · ", sourceLink);
    }
    const tag = document.createElement("span");
    tag.className = "news-tag";
    if (item.intelligence.is_stale) tag.classList.add("stale");
    tag.textContent = `${item.intelligence.is_stale ? "STALE · " : ""}${item.intelligence.event_type} · ${item.intelligence.sentiment} · ${Math.round(item.intelligence.confidence * 100)}%`;
    article.append(title, meta, tag);
    list.append(article);
  }
}

function renderProviderStatus(angel, llm) {
  const angelDot = byId("angelone-dot");
  angelDot.classList.toggle("online", angel.connected);
  const angelStatus = angel.connected
    ? angel.spot_reference_stale
      ? `Connected · ${angel.watchlist_count} subscriptions · waiting for fresh index ticks`
      : `Connected · ${angel.watchlist_count} subscriptions`
    : angel.session_active
      ? "Session active · feed not started"
      : !angel.enabled
        ? "Disabled · local opt-in required"
        : !angel.credentials_configured
          ? "Set ANGELONE_API_KEY in local .env"
          : !angel.sdk_installed
            ? "Install optional SmartAPI SDK"
            : angel.last_error || "Ready · log in for today";
  byId("angelone-provider-status").textContent = angelStatus;
  byId("angelone-login-open").disabled = !angel.ready_to_login || angel.session_active;
  byId("angelone-feed-start").disabled = !angel.session_active || angel.connected;
  byId("angelone-logout").hidden = !angel.session_active;
  const providerLabel = llm.provider === "openai" ? "OpenAI" : "Copilot";
  byId("llm-provider-name").textContent = llm.provider === "openai" ? "OPENAI GATEWAY" : "GITHUB COPILOT SDK";
  byId("llm-dot").classList.toggle("online", llm.enabled);
  byId("llm-provider-status").textContent = llm.enabled
    ? `Enabled · ${llm.model} · tools off`
    : llm.provider === "copilot" && llm.sdk_installed === false
      ? "Install optional Copilot SDK"
      : "Disabled · set LLM_ENABLED=true to opt in";
  const llmButton = byId("llm-review-button");
  llmButton.disabled = !llm.enabled || !angel.connected || angel.spot_reference_stale;
  llmButton.title = llmButton.disabled
    ? `Enable ${providerLabel} and wait for fresh Angel One index ticks first`
    : "Generate an evidence-constrained research hypothesis";
  llmButton.textContent = `Ask ${providerLabel} ↗`;
  byId("provider-summary").textContent = angel.connected
    ? angel.spot_reference_stale
      ? "Angel One feed connected · option strikes provisional until fresh index ticks · orders disabled"
      : "Angel One live market data · broker orders disabled · LLM research opt-in"
    : "Synthetic sample data · Angel One feed disconnected · broker orders disabled";
  const feedPill = byId("market-feed-pill");
  const marketLabel = byId("market-mode-label");
  const marketDot = byId("market-mode-dot");
  if (!angel.connected) {
    feedPill.textContent = "DISCONNECTED";
    feedPill.className = "status-pill disconnected";
    marketLabel.textContent = "DEMO DATA";
    marketDot.classList.remove("online");
    byId("market-feed-note").textContent = angel.session_active
      ? "Session active. Start the feed to subscribe to NIFTY and BANKNIFTY options."
      : "Synthetic research data is active. Authenticate and start the Angel One feed to receive live NSE/NFO ticks.";
  } else if (angel.spot_reference_stale) {
    feedPill.textContent = "PROVISIONAL";
    feedPill.className = "status-pill provisional";
    marketLabel.textContent = "WAITING FOR FRESH INDEX TICKS";
    marketDot.classList.remove("online");
    byId("market-feed-note").textContent = "The WebSocket is connected, but the index quotes used to choose ATM strikes were stale. No stale ticks are used for signals; waiting for fresh NIFTY and BANKNIFTY updates.";
  } else {
    feedPill.textContent = "LIVE";
    feedPill.className = "status-pill live";
    marketLabel.textContent = "LIVE FEED";
    marketDot.classList.add("online");
    byId("market-feed-note").textContent = "Fresh Angel One ticks are flowing. Quotes are for research only; broker order execution is disabled.";
  }
  marketWatchlist = angel.watchlist || [];
  renderLiveWatchlist();
}

function renderLiveWatchlist() {
  const tableBody = byId("live-ticks");
  tableBody.replaceChildren();
  if (!marketWatchlist.length) {
    const row = tableBody.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 8;
    cell.className = "empty-cell";
    cell.textContent = "No market-feed subscriptions yet.";
    return;
  }
  for (const instrument of marketWatchlist) {
    const symbol = instrument.symbol;
    const tick = latestTicks.get(symbol);
    const row = tableBody.insertRow();
    const segment = instrument.instrument_type === "INDEX"
      ? "INDEX"
      : `NFO ${instrument.instrument_type || "OPTION"}`;
    const fields = [
      symbol,
      segment,
      instrument.expiry || "—",
      instrument.strike == null ? "—" : Number(instrument.strike).toLocaleString("en-IN"),
      tick ? Number(tick.last_price).toLocaleString("en-IN", { maximumFractionDigits: 2 }) : "—",
      tick?.open_interest == null ? "—" : Number(tick.open_interest).toLocaleString("en-IN"),
      tick ? `${formatDepth(tick.bids)} / ${formatDepth(tick.asks)}` : "— / —",
      tick ? new Date(tick.timestamp).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—",
    ];
    fields.forEach((value, index) => {
      const cell = row.insertCell();
      cell.textContent = value;
      if (index === 4 && tick) cell.className = "live-price";
    });
  }
}

function formatDepth(levels) {
  const best = levels?.[0];
  return best ? `${Number(best.price).toLocaleString("en-IN", { maximumFractionDigits: 2 })} × ${Number(best.quantity).toLocaleString("en-IN")}` : "—";
}

function connectMarketSocket() {
  if (marketSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(marketSocket.readyState)) return;
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  marketSocket = new WebSocket(`${scheme}//${location.host}/ws/market/ticks`);
  marketSocket.addEventListener("open", () => {
    marketSocket.send("ping");
    marketHeartbeat = window.setInterval(() => {
      if (marketSocket?.readyState === WebSocket.OPEN) marketSocket.send("ping");
    }, 20_000);
  });
  marketSocket.addEventListener("message", (event) => {
    try {
      const tick = JSON.parse(event.data);
      if (!tick.symbol) return;
      latestTicks.set(tick.symbol, tick);
      renderLiveWatchlist();
    } catch {
      return;
    }
  });
  marketSocket.addEventListener("close", () => {
    clearInterval(marketHeartbeat);
    marketReconnect = window.setTimeout(connectMarketSocket, 3000);
  });
  marketSocket.addEventListener("error", () => marketSocket?.close());
}

async function refreshProviderStatus() {
  const [angel, llm] = await Promise.all([
    request("/api/angelone/status"),
    request("/api/llm/status"),
  ]);
  renderProviderStatus(angel, llm);
}

async function askModel() {
  const button = byId("llm-review-button");
  const providerLabel = byId("llm-provider-name").textContent.includes("OPENAI") ? "OpenAI" : "Copilot";
  button.disabled = true;
  button.textContent = "Reviewing…";
  try {
    const result = await request(`/api/llm/hypotheses/${encodeURIComponent(byId("instrument").value)}`, { method: "POST" });
    const decision = result.decision;
    const validation = result.validation;
    byId("hypothesis-status").textContent = validation.approved ? "VALIDATED" : "REJECTED";
    byId("hypothesis-status").classList.toggle("rejected", !validation.approved);
    byId("hypothesis-thesis").textContent = decision.thesis;
    byId("hypothesis-evidence").textContent = `${validation.reasons[0] || "Review only · no execution authority"} · Model ${result.reproducibility.model} · Context ${result.reproducibility.context_hash.slice(0, 12)}`;
    notify(`${providerLabel} hypothesis received. No order was created.`);
  } catch (error) {
    notify(error.message);
  } finally {
    await refreshProviderStatus();
  }
}

async function submitAngelLogin(event) {
  event.preventDefault();
  const form = byId("angelone-login-form");
  const values = Object.fromEntries(new FormData(form));
  const submit = byId("angelone-login-submit");
  submit.disabled = true;
  submit.textContent = "Authenticating…";
  try {
    await request("/api/angelone/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    byId("angelone-login-dialog").close();
    notify("Angel One session started. Choose Start feed to receive market data.");
  } catch (error) {
    notify(error.message);
  } finally {
    form.reset();
    values.pin = "";
    values.totp = "";
    submit.disabled = false;
    submit.textContent = "Authenticate";
    await refreshProviderStatus();
  }
}

async function startAngelFeed() {
  const button = byId("angelone-feed-start");
  button.disabled = true;
  try {
    const result = await request("/api/angelone/feed/start", { method: "POST" });
    notify(`Market feed requested for ${result.watchlist_count} instruments. Orders remain disabled.`);
    await refreshProviderStatus();
  } catch (error) {
    notify(error.message);
    await refreshProviderStatus();
  }
}

async function logoutAngelOne() {
  try {
    await request("/api/angelone/logout", { method: "POST" });
    notify("Angel One session ended and in-memory tokens cleared.");
  } catch (error) {
    notify(error.message);
  } finally {
    await refreshProviderStatus();
  }
}

function renderResult(result) {
  byId("ending-equity").textContent = money(result.ending_equity);
  byId("equity-subtitle").textContent = `From ${money(result.starting_cash)} units · ${result.bar_count} synthetic bars`;
  byId("return-value").textContent = `${result.return_pct >= 0 ? "+" : ""}${result.return_pct.toFixed(2)}%`;
  byId("return-value").className = `metric-value ${result.return_pct >= 0 ? "positive" : "negative"}`;
  byId("drawdown-value").textContent = `${result.max_drawdown_pct.toFixed(2)}%`;
  byId("trades-value").textContent = result.closed_trades;
  byId("win-rate").textContent = `Win rate ${result.win_rate_pct.toFixed(1)}%`;
  byId("chart-caption").textContent = `${result.mode === "paper" ? "Paper replay" : "Backtest"} · ${result.symbol}`;
  byId("chart-start").innerHTML = `STARTING EQUITY<strong>${money(result.starting_cash)} units</strong>`;
  byId("chart-end").innerHTML = `LATEST EQUITY<strong>${money(result.ending_equity)} units</strong>`;
  renderChart(result.equity_curve);
  renderTrades(result.trades);
}

async function run(mode) {
  const button = byId(mode === "paper" ? "paper-button" : "backtest-button");
  const original = button.innerHTML;
  button.disabled = true;
  button.textContent = "Running…";
  try {
    const result = await request(mode === "paper" ? "/api/paper/replay" : "/api/backtests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: byId("instrument").value }),
    });
    renderResult(result);
    await refreshEvents();
    notify(`${mode === "paper" ? "Paper replay" : "Backtest"} complete. No live orders were sent.`);
  } catch (error) {
    notify(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

async function refreshEvents() {
  renderEvents(await request("/api/events?limit=30"));
}

async function refreshNews() {
  renderNews(await request("/api/news?limit=20"));
}

async function submitNews(event) {
  event.preventDefault();
  const form = byId("news-form");
  const values = Object.fromEntries(new FormData(form));
  const submit = byId("news-submit");
  const publicationDate = new Date(values.published_at);
  if (Number.isNaN(publicationDate.getTime())) {
    notify("Enter a valid publication time.");
    return;
  }
  submit.disabled = true;
  submit.textContent = "Ingesting…";
  try {
    const result = await request("/api/news", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: values.source.trim(),
        title: values.title.trim(),
        original_url: values.original_url.trim() || null,
        published_at: publicationDate.toISOString(),
        instruments: values.instruments.split(",").map((symbol) => symbol.trim().toUpperCase()).filter(Boolean),
        sector: values.sector.trim() || null,
        country: values.country.trim() || null,
        source_reliability: Number(values.source_reliability),
        content: values.content.trim(),
      }),
    });
    byId("news-dialog").close();
    form.reset();
    if (result.duplicate) notify("This article was already ingested from another source.");
    else notify("Article ingested and classified.");
    await Promise.all([refreshNews(), refreshEvents()]);
    const proposal = await request(`/api/hypotheses/${encodeURIComponent(byId("instrument").value)}`, { method: "POST" });
    renderHypothesis(proposal);
  } catch (error) {
    notify(error.message);
  } finally {
    submit.disabled = false;
    submit.textContent = "Ingest article";
  }
}

async function refreshRisk() {
  const risk = await request("/api/risk");
  byId("kill-switch").checked = risk.kill_switch;
  byId("risk-status").textContent = risk.kill_switch ? "HALTED" : "READY";
  byId("risk-status").classList.toggle("halted", risk.kill_switch);
}

async function initialize() {
  try {
    const instruments = await request("/api/instruments");
    const selector = byId("instrument");
    const selected = selector.value;
    selector.replaceChildren();
    for (const instrument of instruments) {
      const option = document.createElement("option");
      option.value = instrument.symbol;
      option.textContent = `${instrument.symbol} · ${instrument.symbol === "DEMO" ? "synthetic" : "ingested"}`;
      selector.append(option);
    }
    if (instruments.some((item) => item.symbol === selected)) selector.value = selected;
    const symbol = selector.value;
    const encodedSymbol = encodeURIComponent(symbol);
    const [candles, events, health, features, news, proposal, angel, llm] = await Promise.all([
      request(`/api/market/${encodedSymbol}/candles?limit=1`),
      request("/api/events?limit=30"),
      request("/api/health"),
      request(`/api/market/${encodedSymbol}/features`),
      request("/api/news?limit=20"),
      request(`/api/hypotheses/${encodedSymbol}`, { method: "POST" }),
      request("/api/angelone/status"),
      request("/api/llm/status"),
    ]);
    renderEvents(events);
    renderFeatures(features);
    renderNews(news);
    renderHypothesis(proposal);
    renderProviderStatus(angel, llm);
    if (health.broker_execution_enabled) notify("Unexpected execution state: check server configuration.");
    await refreshRisk();
    if (!candles.length) throw new Error("No sample candles are available.");
    await run("backtest");
  } catch (error) {
    notify(`Could not connect to the local API: ${error.message}`);
  }
}

byId("backtest-button").addEventListener("click", () => run("backtest"));
byId("paper-button").addEventListener("click", () => run("paper"));
byId("llm-review-button").addEventListener("click", askModel);
byId("angelone-login-open").addEventListener("click", () => byId("angelone-login-dialog").showModal());
byId("angelone-login-close").addEventListener("click", () => byId("angelone-login-dialog").close());
byId("angelone-login-cancel").addEventListener("click", () => byId("angelone-login-dialog").close());
byId("angelone-login-form").addEventListener("submit", submitAngelLogin);
byId("angelone-login-dialog").addEventListener("close", () => byId("angelone-login-form").reset());
byId("angelone-feed-start").addEventListener("click", startAngelFeed);
byId("angelone-logout").addEventListener("click", logoutAngelOne);
byId("news-add-open").addEventListener("click", () => byId("news-dialog").showModal());
byId("news-dialog-close").addEventListener("click", () => byId("news-dialog").close());
byId("news-cancel").addEventListener("click", () => byId("news-dialog").close());
byId("news-form").addEventListener("submit", submitNews);
byId("news-dialog").addEventListener("close", () => byId("news-form").reset());
byId("refresh-button").addEventListener("click", initialize);
byId("instrument").addEventListener("change", initialize);
byId("events-refresh").addEventListener("click", async (event) => {
  event.preventDefault();
  try { await refreshEvents(); } catch (error) { notify(error.message); }
});
byId("kill-switch").addEventListener("change", async (event) => {
  event.target.disabled = true;
  try {
    const risk = await request("/api/risk/kill-switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: event.target.checked }),
    });
    byId("risk-status").textContent = risk.kill_switch ? "HALTED" : "READY";
    byId("risk-status").classList.toggle("halted", risk.kill_switch);
    notify(risk.kill_switch ? "Kill switch active. New entries will be rejected." : "Kill switch cleared.");
  } catch (error) {
    event.target.checked = !event.target.checked;
    notify(error.message);
  } finally {
    event.target.disabled = false;
  }
});

window.setInterval(() => {
  refreshProviderStatus().catch(() => {});
}, 5000);

initialize();
connectMarketSocket();