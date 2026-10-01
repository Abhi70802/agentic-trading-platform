const byId = (id) => document.getElementById(id);
const money = (value) => new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(value);
const moneyPrecise = (value) => new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
const api = window.tradingApi;
const alertPreferences = loadAlertPreferences();
let toastTimer;
let marketSocket;
let marketHeartbeat;
let marketReconnect;
let reconnectDelay = 1000;
let streamConnected = false;
let marketFeedConnected = false;
let marketWatchlist = [];
let instruments = [];
let watchlistCandles = new Map();
let watchlistRequests = new Set();
let pinnedSymbols = new Set();
const latestTicks = new Map();
const seenTickKeys = new Set();
const seenTickQueue = [];
let latestRisk;
let latestHypothesis;
let latestSimilarity;
const chartTimeframes = ["1d", "1h", "15m", "5m", "1m"];
let instrumentDetailGeneration = 0;
let instrumentChartGeneration = 0;
let selectedChartTimeframe = "1d";
let instrumentFeatureSnapshots = {};
let instrumentChartCandles = [];
let candidateScanGeneration = 0;
let candidateScanAt = 0;
let candidateScanSnapshot = null;
let candidateScanKey = "";
let recentAuditEvents = [];

function notify(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3000);
}

async function optional(fetcher) {
  try {
    return await fetcher();
  } catch {
    return null;
  }
}

async function getHistoricalSimilarity(symbol, analysis = null) {
  const snapshot = analysis || await optional(() => api.getMultiTimeframeFeatures(symbol));
  const availableBars = Number(snapshot?.timeframes?.["1d"]?.bar_count || 0);
  if (availableBars < 50) return { status: "INSUFFICIENT_HISTORY", available_bars: availableBars };
  return optional(() => api.getSimilarity(symbol));
}

function candidateUniverse(rows) {
  const aliases = new Map([["NIFTY 50", "NIFTY"], ["NIFTY BANK", "BANKNIFTY"], ["INDIA VIX", "INDIAVIX"]]);
  const instrumentsBySymbol = new Map(rows.map((item) => [item.symbol.toUpperCase(), item]));
  const selected = byId("instrument").value.toUpperCase();
  const ordered = [selected, ...pinnedSymbols];
  const seen = new Set();
  const candidates = [];
  const symbols = [];
  let skipped = 0;
  for (const requested of ordered) {
    const symbol = aliases.get(requested.toUpperCase()) || requested.toUpperCase();
    const instrument = instrumentsBySymbol.get(symbol);
    if (!instrument || seen.has(symbol)) continue;
    if (instrumentGroup(symbol) === "DERIVATIVE" && requested !== selected && !pinnedSymbols.has(requested)) continue;
    seen.add(symbol);
    symbols.push(symbol);
    if (Number(instrument.bars) < 21) {
      skipped++;
    } else {
      candidates.push(instrument);
    }
    if (symbols.length >= 12) break;
  }
  return { candidates, skipped, symbols };
}

function renderCandidateQueue(snapshot) {
  const list = byId("candidate-list");
  const items = snapshot.items || [];
  byId("candidate-queue-count").textContent = `${items.length} signal${items.length === 1 ? "" : "s"} · ${snapshot.scanned} scanned`;
  byId("candidate-screen-note").textContent = `${snapshot.skipped} skipped for insufficient or unavailable daily history. SMA signals are deterministic research, not AI advice or execution approval.`;
  list.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-cell";
    empty.textContent = "No pinned instrument has enough stored daily history for screening.";
    list.append(empty);
    return;
  }
  for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-row";
    button.dataset.candidateSymbol = item.symbol;
    button.setAttribute("aria-pressed", String(item.symbol === byId("instrument").value));
    button.title = `Review ${item.symbol} in the decision workspace`;
    const details = document.createElement("span");
    details.className = "candidate-row-details";
    const symbol = document.createElement("strong");
    symbol.textContent = item.symbol;
    const meta = document.createElement("small");
    meta.textContent = `${item.instrument.data_source === "synthetic_demo_ohlcv" ? "SYNTHETIC SAMPLE" : "STORED OHLCV"} · ${item.bar_count} daily bars · ${formatDate(item.as_of)}`;
    details.append(symbol, meta);
    const action = document.createElement("span");
    action.className = `candidate-action ${item.action.toLowerCase()}`;
    action.textContent = item.action;
    button.append(details, action);
    list.append(button);
  }
}

async function refreshCandidateQueue(rows = instruments, force = false) {
  const universe = candidateUniverse(rows);
  const scanKey = universe.symbols.join("|");
  if (!force && candidateScanSnapshot && candidateScanKey === scanKey && Date.now() - candidateScanAt < 30000) {
    renderCandidateQueue(candidateScanSnapshot);
    return;
  }
  const scanGeneration = ++candidateScanGeneration;
  const button = byId("candidate-refresh");
  button.disabled = true;
  byId("candidate-queue-count").textContent = "Checking daily history…";
  byId("candidate-screen-note").textContent = "Reading stored signals only; no orders or AI decisions are created by this scan.";
  const results = await Promise.all(universe.candidates.map(async (instrument) => {
    const analysis = await optional(() => api.getMultiTimeframeFeatures(instrument.symbol));
    const daily = analysis?.timeframes?.["1d"];
    if (!daily || daily.bar_count < 21) return null;
    const signal = await optional(() => api.getSignal(instrument.symbol));
    if (!signal) return null;
    return { ...signal, instrument, bar_count: daily.bar_count };
  }));
  if (scanGeneration !== candidateScanGeneration) return;
  const items = results.filter(Boolean).sort((first, second) => {
    const priority = { BUY: 0, HOLD: 1, SELL: 2 };
    return (priority[first.action] ?? 3) - (priority[second.action] ?? 3) || first.symbol.localeCompare(second.symbol);
  });
  candidateScanSnapshot = {
    items,
    scanned: universe.symbols.length,
    skipped: universe.skipped + universe.candidates.length - items.length,
  };
  candidateScanKey = scanKey;
  candidateScanAt = Date.now();
  renderCandidateQueue(candidateScanSnapshot);
  button.disabled = false;
}

function safePinnedSymbols() {
  try {
    const saved = JSON.parse(localStorage.getItem("northstar.watchlist.pins") || "[]");
    return new Set(Array.isArray(saved) ? saved.filter((symbol) => typeof symbol === "string") : []);
  } catch {
    return new Set();
  }
}

function updateGlobalClock() {
  byId("ist-clock").textContent = `${new Intl.DateTimeFormat("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date())} IST`;
}

function tickTime(tick) {
  const timestamp = tick?.market_timestamp ?? tick?.timestamp;
  const parsed = timestamp ? new Date(timestamp) : null;
  return parsed && Number.isFinite(parsed.getTime()) ? parsed : null;
}

function tickIsFresh(tick, now = Date.now()) {
  const observed = tickTime(tick)?.getTime();
  const age = observed == null ? Number.POSITIVE_INFINITY : now - observed;
  return age >= 0 && age <= 30_000;
}

function updateMarketDataStatus() {
  const strip = byId("market-data-strip");
  const title = byId("market-data-title");
  const detail = byId("market-data-detail");
  const newest = [...latestTicks.values()]
    .map((tick) => tickTime(tick))
    .filter(Boolean)
    .sort((left, right) => right - left)[0];
  const ageSeconds = newest ? Math.max(0, Math.floor((Date.now() - newest.getTime()) / 1000)) : null;

  strip.classList.remove("offline", "stale", "fresh", "waiting");
  if (!marketFeedConnected) {
    strip.classList.add("offline");
    title.textContent = "DATA DISCONNECTED";
    detail.textContent = "Showing stored or synthetic data. Live market data is unavailable.";
  } else if (!newest) {
    strip.classList.add("waiting");
    title.textContent = "WAITING FOR LIVE TICKS";
    detail.textContent = "The feed is connected, but no live quote has been received in this session.";
  } else if (ageSeconds > 30 || Date.now() < newest.getTime()) {
    strip.classList.add("stale");
    title.textContent = "MARKET DATA STALE";
    detail.textContent = `Last accepted tick ${ageSeconds}s ago. Stored prices are not live.`;
  } else {
    strip.classList.add("fresh");
    title.textContent = "LIVE MARKET DATA";
    detail.textContent = `Latest accepted tick ${ageSeconds}s ago · verify instrument time before acting.`;
  }

  const stream = byId("stream-status");
  stream.textContent = streamConnected ? "TICK SOCKET CONNECTED" : "TICK SOCKET DISCONNECTED";
  stream.classList.toggle("connected", streamConnected);
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
    cell.colSpan = 8;
    cell.className = "empty-cell";
    cell.textContent = "No closed trades in this replay.";
    return;
  }
  for (const trade of [...trades].reverse().slice(0, 8)) {
    const row = table.insertRow();
    const entryTimestamp = Date.parse(trade.entry_time);
    const exitTimestamp = Date.parse(trade.exit_time);
    const holdDays = Number.isFinite(entryTimestamp) && Number.isFinite(exitTimestamp)
      ? Math.max(0, Math.round((exitTimestamp - entryTimestamp) / 86400000))
      : null;
    const entryCost = Number(trade.entry_cost);
    const tradeReturn = entryCost > 0 ? trade.pnl / entryCost * 100 : null;
    const exitReason = trade.exit_reason === "evaluation_window_end" ? "Window end" : "Strategy signal";
    const values = [
      trade.symbol,
      formatDate(trade.entry_time),
      formatDate(trade.exit_time),
      trade.quantity,
      holdDays === null ? "—" : `${holdDays}d`,
      moneyPrecise(trade.pnl),
      tradeReturn === null ? "—" : `${tradeReturn >= 0 ? "+" : ""}${tradeReturn.toFixed(2)}%`,
      exitReason,
    ];
    values.forEach((value, index) => {
      const cell = row.insertCell();
      cell.textContent = value;
      if (index === 5 || index === 6) cell.className = `pnl ${trade.pnl >= 0 ? "positive" : "negative"}`;
    });
  }
}

function renderPaperTimeline(events) {
  const container = byId("paper-order-timeline");
  const completedRuns = events.filter((event) => event.event_type === "simulation.completed" && event.payload?.mode === "paper");
  const runIds = new Set(completedRuns.map((event) => event.correlation_id));
  const paperEvents = events.filter((event) => runIds.has(event.correlation_id) && (
    event.event_type === "signal.generated"
    || event.event_type.startsWith("order.")
    || event.event_type === "risk.rejected"
    || event.event_type === "trade.closed"
    || event.event_type === "simulation.completed"
  ));
  const latestRun = completedRuns[0];
  byId("paper-oms-state").textContent = latestRun ? "PAPER REPLAY" : "NO PAPER RUNS";
  byId("paper-oms-summary").textContent = latestRun
    ? `${latestRun.payload.symbol} · ${latestRun.payload.bar_count} bars · ${Number(latestRun.payload.return_pct).toFixed(2)}% replay return · ${latestRun.payload.strategy?.name || "strategy unavailable"}`
    : "No simulated order lifecycle has been recorded.";
  byId("paper-replay-mode").textContent = latestRun ? `${latestRun.payload.symbol} · ${formatDate(latestRun.timestamp)}` : "Not run";
  byId("paper-order-count").textContent = `${paperEvents.length} record${paperEvents.length === 1 ? "" : "s"}`;
  container.replaceChildren();
  if (!paperEvents.length) {
    const empty = document.createElement("div");
    empty.className = "empty-cell";
    empty.textContent = "No paper replay events recorded. Live broker orders are disabled.";
    container.append(empty);
    return;
  }
  for (const event of paperEvents.slice(0, 40)) {
    const item = document.createElement("div");
    item.className = "paper-timeline-item";
    const marker = document.createElement("span");
    marker.className = `paper-timeline-mark ${event.event_type.replaceAll(".", "-")}`;
    const content = document.createElement("span");
    const title = document.createElement("strong");
    const payload = event.payload || {};
    if (event.event_type === "signal.generated") title.textContent = `${payload.symbol} signal · ${payload.action}`;
    else if (event.event_type === "order.filled") title.textContent = `${payload.symbol} filled · ${payload.side} ${payload.quantity}`;
    else if (event.event_type === "order.partially_filled") title.textContent = `${payload.symbol} partial fill · ${payload.filled_quantity ?? payload.quantity}`;
    else if (event.event_type === "trade.closed") title.textContent = `${payload.symbol} trade closed · ${formatRupees(Number(payload.pnl))}`;
    else if (event.event_type === "simulation.completed") title.textContent = `${payload.symbol} replay completed · ${Number(payload.return_pct).toFixed(2)}%`;
    else title.textContent = `${payload.symbol || "Paper run"} · ${event.event_type.replaceAll(".", " ")}`;
    const meta = document.createElement("small");
    const timestamp = new Intl.DateTimeFormat("en-IN", {
      timeZone: "Asia/Kolkata", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
    }).format(new Date(event.timestamp));
    meta.textContent = `${timestamp} IST · ${event.correlation_id.slice(0, 8)}`;
    content.append(title, meta);
    item.append(marker, content);
    container.append(item);
  }
}

function formatDate(value) {
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "2-digit", timeZone: "UTC" });
}

function formatMarketDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString("en-IN", {
    timeZone: "Asia/Kolkata",
    day: "numeric",
    month: "short",
    year: "2-digit",
  });
}

function formatMarketTime(tick) {
  const timestamp = tick.market_timestamp ?? tick.timestamp;
  if (!timestamp) return "—";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function loadAlertPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem("northstar.alert-preferences") || "{}");
    return { feed: saved.feed !== false, simulation: saved.simulation !== false };
  } catch {
    return { feed: true, simulation: true };
  }
}

function renderOperationalAlerts(events) {
  recentAuditEvents = events;
  const feedTypes = new Set([
    "market.connection_error",
    "market.connection_closed",
    "market.reconnect_exhausted",
    "market.tick_rejected",
  ]);
  const simulationTypes = new Set(["risk.rejected", "order.liquidity_rejected"]);
  const alerts = events.filter((event) => (
    (alertPreferences.feed && feedTypes.has(event.event_type))
    || (alertPreferences.simulation && event.source === "simulation_engine" && simulationTypes.has(event.event_type))
  ));
  const count = byId("alert-count");
  const list = byId("alert-list");
  count.textContent = `${alerts.length} MATCH${alerts.length === 1 ? "" : "ES"}`;
  list.replaceChildren();
  if (!alerts.length) {
    const empty = document.createElement("div");
    empty.className = "empty-cell";
    empty.textContent = "No matching audit alerts.";
    list.append(empty);
    return;
  }
  for (const event of alerts.slice(0, 6)) {
    const item = document.createElement("div");
    item.className = "event-item";
    const mark = document.createElement("span");
    mark.className = "event-mark";
    mark.classList.toggle("alert-mark-simulation", event.source === "simulation_engine");
    const copy = document.createElement("div");
    const title = document.createElement("strong");
    const simulated = event.source === "simulation_engine";
    title.textContent = `${simulated ? "SIMULATION" : "FEED"} · ${event.event_type.replaceAll(".", " · ")}`;
    const time = document.createElement("small");
    time.textContent = `${formatDate(event.timestamp)} · ${event.correlation_id.slice(0, 8)}`;
    const detail = document.createElement("code");
    detail.textContent = event.payload?.reason || event.payload?.code || event.payload?.symbol || event.source;
    copy.append(title, time, detail);
    item.append(mark, copy);
    list.append(item);
  }
}

function renderEvents(events) {
  renderOperationalAlerts(events);
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

function renderEvidenceList(id, evidence, emptyMessage) {
  const list = byId(id);
  list.replaceChildren();
  if (!evidence?.length) {
    const empty = document.createElement("span");
    empty.textContent = emptyMessage;
    list.append(empty);
    return;
  }
  for (const item of evidence) {
    const row = document.createElement("div");
    row.className = "evidence-row";
    row.textContent = `${item.claim || "Evidence"} · ${item.source_id || "source unavailable"}`;
    list.append(row);
  }
}

function renderHypothesis(result, similarity = latestSimilarity) {
  if (!result) return;
  const hypothesis = result.hypothesis || result.decision || result;
  const validation = result.validation || { approved: false, reasons: ["Validation unavailable"] };
  latestHypothesis = { hypothesis, validation };
  const actionable = hypothesis.direction === "LONG";
  const decision = latestRisk?.kill_switch || !actionable || !validation.approved ? "NO TRADE" : "HOLD";
  byId("signal-instrument").textContent = `${hypothesis.instrument || byId("instrument").value} · ${hypothesis.strategy || "strategy unavailable"}`;
  byId("hypothesis-status").textContent = decision;
  byId("hypothesis-status").classList.toggle("rejected", decision === "NO TRADE");
  byId("hypothesis-decision").textContent = decision;
  byId("hypothesis-thesis").textContent = hypothesis.thesis || "No trade thesis was returned.";
  renderEvidenceList("hypothesis-why", hypothesis.supporting_evidence, "No supporting evidence supplied.");
  renderEvidenceList("hypothesis-counter", hypothesis.contradictory_evidence, "No disconfirming evidence supplied; absence is not confirmation.");
  const reasons = validation.reasons || [];
  byId("hypothesis-evidence").textContent = reasons.length ? reasons.join(" · ") : "Guardrails passed for review only; this does not authorize an order.";
  byId("hypothesis-probability").textContent = "Not calibrated · raw confidence is not a probability.";
  byId("hypothesis-expected-value").textContent = "Not calculated · calibrated probability and an India charge schedule are required.";
  const entry = hypothesis.entry_range;
  const stop = Number(hypothesis.stop_loss);
  const entryPrice = entry && Number(entry.high);
  byId("hypothesis-risk").textContent = Number.isFinite(stop) && Number.isFinite(entryPrice) && entryPrice > stop
    ? `Stop distance ₹${(entryPrice - stop).toFixed(2)} per unit; maximum loss needs a risk-sized quantity.`
    : "Not calculated · no valid entry and stop-loss range.";
  byId("hypothesis-size").textContent = validation.approved
    ? "Not returned by the active decision pipeline; no order quantity is authorized."
    : "0 units eligible · no validated setup and funded equity snapshot.";
  if (!similarity || !similarity.sample_count) {
    byId("hypothesis-historical").textContent = similarity?.status === "INSUFFICIENT_HISTORY"
      ? `Insufficient stored history for analogues (${similarity.available_bars} bars; at least 50 required).`
      : "No comparable historical setups returned for this instrument.";
    byId("historical-analogue-list").replaceChildren();
  } else {
    const hitRate = similarity.win_rate_pct == null ? "—" : `${similarity.win_rate_pct.toFixed(1)}%`;
    const average = similarity.average_return_pct == null ? "—" : `${similarity.average_return_pct.toFixed(2)}%`;
    byId("hypothesis-historical").textContent = `${similarity.sample_count} nearest setups · ${similarity.horizon_bars}-bar horizon · descriptive positive-return frequency ${hitRate} · mean forward return ${average}. Not calibrated.`;
    const matches = byId("historical-analogue-list");
    matches.replaceChildren();
    for (const match of (similarity.matches || []).slice(0, 3)) {
      const row = document.createElement("div");
      row.className = "analogue-row";
      const date = document.createElement("span");
      date.textContent = formatDate(match.reference_timestamp);
      const outcome = document.createElement("strong");
      outcome.textContent = `${match.forward_return_pct >= 0 ? "+" : ""}${match.forward_return_pct.toFixed(2)}%`;
      row.append(date, outcome);
      matches.append(row);
    }
  }
}

function renderNews(items) {
  const list = byId("news-list");
  const symbol = byId("instrument").value.toUpperCase();
  const relevantItems = items.filter((item) => Array.isArray(item.instruments)
    && item.instruments.some((instrument) => typeof instrument === "string" && instrument.toUpperCase() === symbol));
  byId("news-title").textContent = `${symbol} news`;
  byId("news-count").textContent = `${relevantItems.length} item${relevantItems.length === 1 ? "" : "s"}`;
  list.replaceChildren();
  if (!relevantItems.length) {
    const empty = document.createElement("div");
    empty.className = "empty-cell";
    empty.textContent = `No articles tagged ${symbol}. External news feeds are not connected.`;
    list.append(empty);
    return;
  }
  for (const item of relevantItems.slice(0, 5)) {
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

function svgElement(name, attributes = {}, text = null) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text !== null) element.textContent = text;
  return element;
}

function showCandleChartMessage(message) {
  const svg = byId("instrument-candles");
  svg.replaceChildren(svgElement("text", {
    x: "50%", y: "50%", "text-anchor": "middle", class: "chart-placeholder",
  }, message));
  byId("instrument-ohlc").textContent = "OHLC unavailable";
  byId("instrument-bar-count").textContent = "No historical bars";
}

function formatInstrumentPrice(value) {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(value);
}

function renderCandleChart(candles) {
  const svg = byId("instrument-candles");
  const width = 900;
  const height = 350;
  const left = 68;
  const right = 18;
  const top = 14;
  const priceBottom = 250;
  const volumeTop = 276;
  const volumeBottom = 323;
  const data = [...candles]
    .filter((candle) => [candle.open, candle.high, candle.low, candle.close, candle.volume].every(Number.isFinite))
    .sort((first, second) => Date.parse(first.timestamp) - Date.parse(second.timestamp))
    .slice(-250);
  if (!data.length) {
    showCandleChartMessage("No valid stored candles for this timeframe");
    return;
  }

  const highest = Math.max(...data.map((candle) => candle.high));
  const lowest = Math.min(...data.map((candle) => candle.low));
  const priceRange = Math.max(highest - lowest, Math.abs(highest) * 0.005, 0.01);
  const priceTop = highest + priceRange * 0.08;
  const priceFloor = lowest - priceRange * 0.08;
  const plotWidth = width - left - right;
  const slotWidth = plotWidth / data.length;
  const x = (index) => left + slotWidth * (index + 0.5);
  const y = (price) => top + (priceTop - price) * (priceBottom - top) / (priceTop - priceFloor);
  const maxVolume = Math.max(1, ...data.map((candle) => candle.volume));
  const nodes = [];

  for (let step = 0; step <= 4; step++) {
    const price = priceTop - (priceTop - priceFloor) * step / 4;
    const rowY = y(price);
    nodes.push(svgElement("line", { x1: left, y1: rowY, x2: width - right, y2: rowY, class: "candle-grid" }));
    nodes.push(svgElement("text", { x: left - 8, y: rowY + 3, "text-anchor": "end", class: "candle-axis-label" }, formatInstrumentPrice(price)));
  }

  data.forEach((candle, index) => {
    const rising = candle.close >= candle.open;
    const candleGroup = svgElement("g", { class: rising ? "candle-up" : "candle-down" });
    const title = svgElement("title", {}, `${formatDate(candle.timestamp)} · O ${formatInstrumentPrice(candle.open)} · H ${formatInstrumentPrice(candle.high)} · L ${formatInstrumentPrice(candle.low)} · C ${formatInstrumentPrice(candle.close)} · V ${Math.round(candle.volume).toLocaleString("en-IN")}`);
    candleGroup.append(title, svgElement("line", { x1: x(index), y1: y(candle.high), x2: x(index), y2: y(candle.low), class: "candle-wick" }));
    const bodyTop = Math.min(y(candle.open), y(candle.close));
    const bodyHeight = Math.max(1, Math.abs(y(candle.open) - y(candle.close)));
    candleGroup.append(svgElement("rect", {
      x: x(index) - Math.max(1, Math.min(12, slotWidth * 0.62)) / 2,
      y: bodyTop,
      width: Math.max(1, Math.min(12, slotWidth * 0.62)),
      height: bodyHeight,
      class: "candle-body",
    }));
    nodes.push(candleGroup);
    const volumeHeight = Math.max(1, candle.volume / maxVolume * (volumeBottom - volumeTop));
    nodes.push(svgElement("rect", {
      x: x(index) - Math.max(1, Math.min(12, slotWidth * 0.62)) / 2,
      y: volumeBottom - volumeHeight,
      width: Math.max(1, Math.min(12, slotWidth * 0.62)),
      height: volumeHeight,
      class: rising ? "volume-up" : "volume-down",
    }));
  });

  const overlays = [
    { name: "sma20", period: 20, className: "sma20-line" },
    { name: "sma50", period: 50, className: "sma50-line" },
  ];
  for (const overlay of overlays) {
    if (!byId("instrument-overlays").querySelector(`[data-chart-overlay="${overlay.name}"]`).checked || data.length < overlay.period) continue;
    let sum = 0;
    const pathPoints = [];
    for (let index = 0; index < data.length; index++) {
      sum += data[index].close;
      if (index >= overlay.period) sum -= data[index - overlay.period].close;
      if (index >= overlay.period - 1) pathPoints.push(`${pathPoints.length ? "L" : "M"}${x(index).toFixed(1)},${y(sum / overlay.period).toFixed(1)}`);
    }
    nodes.push(svgElement("path", { d: pathPoints.join(" "), class: overlay.className }));
  }

  const dateIndexes = [...new Set([0, Math.floor((data.length - 1) / 2), data.length - 1])];
  for (const index of dateIndexes) {
    nodes.push(svgElement("text", {
      x: x(index), y: height - 7, "text-anchor": index === 0 ? "start" : index === data.length - 1 ? "end" : "middle", class: "candle-axis-label",
    }, formatDate(data[index].timestamp)));
  }
  nodes.push(svgElement("text", { x: left, y: volumeTop - 6, class: "candle-axis-label" }, "VOLUME"));
  svg.replaceChildren(...nodes);
  svg.setAttribute("aria-label", `${byId("instrument").value} ${selectedChartTimeframe} candlestick chart with volume`);

  const latest = data[data.length - 1];
  byId("instrument-ohlc").textContent = `O ${formatInstrumentPrice(latest.open)} · H ${formatInstrumentPrice(latest.high)} · L ${formatInstrumentPrice(latest.low)} · C ${formatInstrumentPrice(latest.close)}`;
  byId("instrument-bar-count").textContent = `${data.length} bars · ${selectedChartTimeframe.toUpperCase()}`;
  byId("instrument-chart-asof").textContent = `HISTORICAL · ${formatDate(latest.timestamp)}`;
  const missingOverlays = overlays.filter((overlay) => data.length < overlay.period).map((overlay) => `SMA ${overlay.period}`);
  byId("instrument-chart-note").textContent = `Stored OHLCV · not a live quote${missingOverlays.length ? ` · ${missingOverlays.join(" and ")} need more bars` : ""}.`;
}

function renderInstrumentIndicators(snapshot, timeframe) {
  const features = snapshot?.features || {};
  const setValue = (id, value, digits = 2, suffix = "") => {
    byId(id).textContent = Number.isFinite(value) ? `${Number(value).toFixed(digits)}${suffix}` : "Unavailable";
  };
  byId("instrument-indicator-timeframe").textContent = timeframe ? timeframe.toUpperCase() : "NO HISTORY";
  setValue("detail-rsi", features.rsi_14, 1);
  setValue("detail-macd", features.macd);
  setValue("detail-atr", features.atr_14);
  setValue("detail-atr-pct", features.atr_pct, 2, "%");
  setValue("detail-sma20", features.sma_20);
  setValue("detail-sma50", features.sma_50);
  byId("detail-ema").textContent = Number.isFinite(features.ema_12) && Number.isFinite(features.ema_26)
    ? `${formatInstrumentPrice(features.ema_12)} / ${formatInstrumentPrice(features.ema_26)}` : "Unavailable";
  byId("detail-bollinger").textContent = Number.isFinite(features.bollinger_lower) && Number.isFinite(features.bollinger_upper)
    ? `${formatInstrumentPrice(features.bollinger_lower)} to ${formatInstrumentPrice(features.bollinger_upper)}` : "Unavailable";
  setValue("detail-volume", features.volume_zscore_20, 1, "σ");
  setValue("detail-momentum", features.momentum_20_pct, 2, "%");
  byId("instrument-indicator-note").textContent = snapshot
    ? `${snapshot.bar_count} stored bars · as of ${formatDate(snapshot.timestamp)}. Unavailable values need more history.`
    : "No indicator snapshot for this timeframe; values are not estimated.";
}

function renderInstrumentTimeframes(availableTimeframes) {
  const container = byId("instrument-timeframes");
  container.replaceChildren();
  const available = chartTimeframes.filter((timeframe) => availableTimeframes.includes(timeframe));
  selectedChartTimeframe = available.includes(selectedChartTimeframe) ? selectedChartTimeframe : available.includes("1d") ? "1d" : available[0] || "1d";
  for (const timeframe of chartTimeframes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "timeframe-button";
    button.dataset.timeframe = timeframe;
    button.textContent = timeframe.toUpperCase();
    button.disabled = !available.includes(timeframe);
    button.setAttribute("aria-pressed", String(timeframe === selectedChartTimeframe));
    if (button.disabled) button.title = `No stored ${timeframe} candles for this instrument`;
    container.append(button);
  }
}

async function loadInstrumentCandles(symbol, timeframe, detailGeneration) {
  const chartGeneration = ++instrumentChartGeneration;
  const candles = await optional(() => api.getCandles(symbol, { limit: 250, timeframe }));
  if (chartGeneration !== instrumentChartGeneration || detailGeneration !== instrumentDetailGeneration) return;
  instrumentChartCandles = candles || [];
  if (!candles?.length) {
    showCandleChartMessage("No stored candles for this timeframe");
    byId("instrument-chart-asof").textContent = "HISTORY UNAVAILABLE";
    byId("instrument-chart-note").textContent = "No candles are stored for this instrument and timeframe.";
    byId("instrument-detail-meta").textContent = "Stored history unavailable";
    return;
  }
  renderCandleChart(candles);
}

async function renderInstrumentDetail(symbol, analysis, news) {
  const detailGeneration = ++instrumentDetailGeneration;
  instrumentFeatureSnapshots = analysis?.timeframes || {};
  const instrument = instruments.find((item) => item.symbol.toUpperCase() === symbol.toUpperCase());
  const source = instrument?.data_source === "synthetic_demo_ohlcv" ? "SYNTHETIC SAMPLE" : "STORED OHLCV";
  byId("instrument-chart-symbol").textContent = symbol;
  byId("instrument-detail-meta").textContent = `${source} · ${instrument?.bars ?? 0} bars in store`;
  byId("instrument-chart-note").textContent = "Loading stored candles…";
  renderInstrumentTimeframes(Object.keys(instrumentFeatureSnapshots));
  const selectedSnapshot = instrumentFeatureSnapshots[selectedChartTimeframe];
  renderInstrumentIndicators(selectedSnapshot, selectedSnapshot ? selectedChartTimeframe : null);
  renderNews(news);
  const available = chartTimeframes.filter((timeframe) => Object.hasOwn(instrumentFeatureSnapshots, timeframe));
  if (!available.includes(selectedChartTimeframe)) {
    instrumentChartGeneration++;
    instrumentChartCandles = [];
    showCandleChartMessage("No stored candles for this instrument");
    byId("instrument-chart-asof").textContent = "HISTORY UNAVAILABLE";
    byId("instrument-chart-note").textContent = "No stored timeframe is available for this instrument.";
    byId("instrument-detail-meta").textContent = "Stored history unavailable";
    return;
  }
  await loadInstrumentCandles(symbol, selectedChartTimeframe, detailGeneration);
}

function renderProviderStatus(angel, llm) {
  marketFeedConnected = angel.feed_running === true;
  const angelDot = byId("angelone-dot");
  angelDot.classList.toggle("online", angel.connected);
  const feedRunning = angel.feed_running === true;
  const angelStatus = angel.connected
    ? angel.spot_reference_stale
      ? `Connected · ${angel.watchlist_count} subscriptions · waiting for fresh index ticks`
      : `Connected · ${angel.watchlist_count} subscriptions`
    : feedRunning
      ? `REST fallback · ${angel.watchlist_count} subscriptions · WebSocket reconnecting`
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
  byId("angelone-feed-start").disabled = !angel.session_active || angel.connected || feedRunning;
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
  byId("portfolio-refresh").disabled = !angel.session_active;
  byId("portfolio-status").textContent = angel.session_active
    ? "Session active · refresh is a read-only positions and holdings request."
    : "Unavailable · authenticate with Angel One to request a read-only positions snapshot.";
  byId("health-angel").textContent = angel.connected
    ? "Feed connected"
    : feedRunning
      ? "REST fallback · WebSocket reconnecting"
      : angel.session_active
        ? "Session active · feed stopped"
        : angel.enabled ? "Enabled · disconnected" : "Disabled";
  byId("health-llm").textContent = llm.enabled ? `Enabled · ${providerLabel}` : "Disabled";
  byId("model-provider").textContent = providerLabel;
  byId("model-name").textContent = llm.model || "Not configured";
  byId("model-status").textContent = llm.enabled ? "NO EVALUATION" : "DISABLED";
  byId("provider-summary").textContent = angel.connected
    ? angel.spot_reference_stale
      ? "Angel One feed connected · option strikes provisional until fresh index ticks · orders disabled"
      : "Angel One live market data · broker orders disabled · LLM research opt-in"
    : feedRunning
      ? "Angel One REST quote fallback · WebSocket reconnecting · broker orders disabled"
      : "Synthetic sample data · Angel One feed disconnected · broker orders disabled";
  const feedPill = byId("market-feed-pill");
  const marketLabel = byId("market-mode-label");
  const marketDot = byId("market-mode-dot");
  if (!angel.connected) {
    feedPill.textContent = feedRunning ? "REST FALLBACK" : "DISCONNECTED";
    feedPill.className = feedRunning ? "status-pill provisional" : "status-pill disconnected";
    marketLabel.textContent = feedRunning ? "REST FALLBACK" : "DEMO DATA";
    marketDot.classList.remove("online");
    byId("market-feed-note").textContent = feedRunning
      ? "The WebSocket is reconnecting while the REST quote fallback runs. Check each instrument's market time; broker order execution is disabled."
      : angel.session_active
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
  updateMarketDataStatus();
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
      tick ? formatMarketTime(tick) : "—",
    ];
    fields.forEach((value, index) => {
      const cell = row.insertCell();
      cell.textContent = value;
      if (index === 4 && tick) cell.className = "live-price";
    });
  }
}

function instrumentGroup(symbol) {
  const normalized = symbol.toUpperCase();
  if (normalized === "DEMO") return "DEMO";
  if (/(?:CE|PE|FUT)$/.test(normalized) && /\d/.test(normalized)) return "DERIVATIVE";
  if (["NIFTY", "NIFTY 50", "BANKNIFTY", "NIFTY BANK", "SENSEX", "INDIA VIX", "INDIAVIX"].includes(normalized)) return "INDEX";
  return "EQUITY";
}

function defaultPinnedSymbols(rows) {
  const available = new Set(rows.map((item) => item.symbol.toUpperCase()));
  const preferred = ["NIFTY 50", "NIFTY", "BANKNIFTY", "NIFTY BANK", "SENSEX", "DEMO"]
    .filter((symbol) => available.has(symbol));
  const core = rows.filter((item) => instrumentGroup(item.symbol) !== "DERIVATIVE").map((item) => item.symbol);
  return new Set([...preferred, ...core].slice(0, 8));
}

function initializeWatchlist(rows) {
  instruments = rows;
  pinnedSymbols = safePinnedSymbols();
  const validSymbols = new Set(rows.map((item) => item.symbol));
  pinnedSymbols = new Set([...pinnedSymbols].filter((symbol) => validSymbols.has(symbol)));
  if (!pinnedSymbols.size) pinnedSymbols = defaultPinnedSymbols(rows);

  const datalist = byId("instrument-suggestions");
  datalist.replaceChildren();
  for (const item of rows) {
    const option = document.createElement("option");
    option.value = item.symbol;
    datalist.append(option);
  }
  renderWatchlist();
}

function watchlistChange(symbol) {
  const candles = watchlistCandles.get(symbol)?.candles || [];
  if (candles.length < 2 || !Number(candles[0].close)) return null;
  return (Number(candles[candles.length - 1].close) / Number(candles[candles.length - 2].close) - 1) * 100;
}

function renderWatchlist(loadMissing = true) {
  const query = byId("watchlist-search").value.trim().toUpperCase();
  const group = byId("watchlist-group").value;
  const sort = byId("watchlist-sort").value;
  let matches = instruments.filter((item) => {
    const symbol = item.symbol.toUpperCase();
    return (!query || symbol.includes(query)) && (group === "ALL" || instrumentGroup(symbol) === group);
  });
  matches.sort((left, right) => {
    const leftPinned = pinnedSymbols.has(left.symbol);
    const rightPinned = pinnedSymbols.has(right.symbol);
    if (sort === "PINNED" && leftPinned !== rightPinned) return leftPinned ? -1 : 1;
    if (sort === "CHANGE") return (watchlistChange(right.symbol) ?? -Infinity) - (watchlistChange(left.symbol) ?? -Infinity);
    return left.symbol.localeCompare(right.symbol, "en-IN");
  });
  const visible = matches.slice(0, 20);
  byId("watchlist-count").textContent = `${visible.length} shown · ${matches.length} match${matches.length === 1 ? "" : "es"}`;

  const body = byId("watchlist-rows");
  body.replaceChildren();
  if (!visible.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 6;
    cell.className = "empty-cell";
    cell.textContent = query ? "No configured instrument matches this search." : "No configured instruments in this group.";
  }

  for (const item of visible) {
    const row = body.insertRow();
    const snapshot = watchlistCandles.get(item.symbol);
    const candles = snapshot?.candles || [];
    const latest = candles[candles.length - 1];
    const tick = latestTicks.get(item.symbol);
    const live = marketFeedConnected && tickIsFresh(tick);
    const last = live ? Number(tick.last_price) : latest?.close;
    const change = watchlistChange(item.symbol);

    const pinCell = row.insertCell();
    const pin = document.createElement("button");
    pin.type = "button";
    pin.className = `pin-toggle${pinnedSymbols.has(item.symbol) ? " pinned" : ""}`;
    pin.dataset.pinSymbol = item.symbol;
    pin.setAttribute("aria-pressed", String(pinnedSymbols.has(item.symbol)));
    pin.setAttribute("aria-label", `${pinnedSymbols.has(item.symbol) ? "Unpin" : "Pin"} ${item.symbol}`);
    pin.title = pin.getAttribute("aria-label");
    pin.textContent = pinnedSymbols.has(item.symbol) ? "★" : "☆";
    pinCell.append(pin);

    const symbolCell = row.insertCell();
    const symbolButton = document.createElement("button");
    symbolButton.type = "button";
    symbolButton.className = "watchlist-symbol";
    symbolButton.dataset.selectSymbol = item.symbol;
    symbolButton.textContent = item.symbol;
    symbolCell.append(symbolButton);

    const lastCell = row.insertCell();
    lastCell.className = live ? "live-price" : "stored-price";
    const priceValue = document.createElement("strong");
    priceValue.textContent = Number.isFinite(Number(last))
      ? Number(last).toLocaleString("en-IN", { maximumFractionDigits: 2 })
      : snapshot?.unavailable || snapshot?.error ? "Unavailable" : "Loading…";
    const provenance = document.createElement("small");
    provenance.className = "price-source";
    provenance.textContent = live
      ? "LIVE TICK"
      : snapshot?.unavailable ? "NO BAR HISTORY" : snapshot?.error ? "UNAVAILABLE" : item.data_source === "synthetic_demo_ohlcv" ? "SYNTHETIC" : "STORED";
    lastCell.replaceChildren(priceValue, provenance);

    const changeCell = row.insertCell();
    changeCell.textContent = change == null ? "—" : `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`;
    if (change != null) changeCell.className = change >= 0 ? "positive" : "negative";

    const volumeCell = row.insertCell();
    const volume = live ? tick.volume : latest?.volume;
    volumeCell.textContent = volume == null ? "—" : Number(volume).toLocaleString("en-IN");

    const asOf = live ? tickTime(tick) : latest ? new Date(latest.timestamp) : null;
    row.insertCell().textContent = asOf
      ? asOf.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false })
      : snapshot?.unavailable || snapshot?.error ? "—" : "Loading…";
  }

  if (loadMissing) {
    for (const item of visible) {
      if (watchlistCandles.has(item.symbol) || watchlistRequests.has(item.symbol)) continue;
      if (instrumentGroup(item.symbol) === "DERIVATIVE") {
        watchlistCandles.set(item.symbol, { candles: [], unavailable: true });
        continue;
      }
      watchlistRequests.add(item.symbol);
      api.getCandles(item.symbol, { limit: 2 }).then((candles) => {
        watchlistCandles.set(item.symbol, { candles });
      }).catch((error) => {
        watchlistCandles.set(item.symbol, { candles: [], error: error.message });
      }).finally(() => {
        watchlistRequests.delete(item.symbol);
        renderWatchlist(false);
      });
    }
  }
}

function togglePinnedSymbol(symbol) {
  if (pinnedSymbols.has(symbol)) pinnedSymbols.delete(symbol);
  else pinnedSymbols.add(symbol);
  try {
    localStorage.setItem("northstar.watchlist.pins", JSON.stringify([...pinnedSymbols]));
  } catch {
    notify("Pin changed for this session; browser storage is unavailable.");
  }
  renderWatchlist(false);
  void refreshCandidateQueue(instruments, true);
}

function formatRupees(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? `₹${new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(amount)}` : "—";
}

function formatMarketTile(prefix, candles, instrument) {
  const value = byId(`${prefix}-value`);
  const change = byId(`${prefix}-change`);
  const asOf = byId(`${prefix}-asof`);
  if (!candles?.length) {
    value.textContent = "—";
    change.textContent = "Index history unavailable";
    asOf.textContent = "No market history loaded";
    return null;
  }
  const latest = candles[candles.length - 1];
  const previous = candles.length > 1 ? candles[candles.length - 2] : null;
  value.textContent = Number(latest.close).toLocaleString("en-IN", { maximumFractionDigits: 2 });
  if (previous && Number(previous.close)) {
    const percent = (Number(latest.close) / Number(previous.close) - 1) * 100;
    change.textContent = `${percent >= 0 ? "+" : ""}${percent.toFixed(2)}% · previous stored bar`;
    change.className = percent >= 0 ? "positive" : "negative";
  } else {
    change.textContent = "Change unavailable · one bar stored";
    change.className = "";
  }
  const source = instrument?.data_source === "synthetic_demo_ohlcv" ? "Synthetic demo" : "Stored OHLCV";
  asOf.textContent = `${source} · ${formatMarketDate(latest.timestamp)}`;
  return latest.timestamp;
}

function renderMarketOverview(data) {
  const dates = [
    formatMarketTile("nifty", data.nifty, data.niftyInstrument),
    formatMarketTile("banknifty", data.banknifty, data.bankInstrument),
    formatMarketTile("sensex", data.sensex, data.sensexInstrument),
    formatMarketTile("india-vix", data.vix, data.vixInstrument),
  ].filter(Boolean).sort();
  byId("market-asof").textContent = dates.length ? `Latest stored bar · ${formatMarketDate(dates[dates.length - 1])}` : "No Indian index history loaded";
  const regime = data.regime || {};
  byId("market-regime").textContent = (regime.regime || "INSUFFICIENT_DATA").replaceAll("_", " ");
  byId("market-regime-posture").textContent = `Risk posture · ${(regime.risk_posture || "UNKNOWN").replaceAll("_", " ")}`;
  byId("market-regime-reason").textContent = regime.reason || "Index history is not configured.";

  const breadth = data.breadth;
  byId("sector-list").replaceChildren();
  const sectors = Object.entries(breadth?.sector_breadth || {});
  byId("sector-count").textContent = sectors.length ? `${sectors.length} sectors` : "No sector map";
  if (!sectors.length) {
    const empty = document.createElement("p");
    empty.className = "empty-cell";
    empty.textContent = "Sector constituents and mappings are not configured.";
    byId("sector-list").append(empty);
  } else {
    for (const [name, stats] of sectors) {
      const row = document.createElement("div");
      row.className = "sector-row";
      const label = document.createElement("span");
      label.textContent = name;
      const participation = document.createElement("span");
      participation.textContent = `${stats.advances} up · ${stats.declines} down`;
      const ratio = document.createElement("strong");
      ratio.textContent = stats.advance_decline_ratio == null ? "—" : stats.advance_decline_ratio.toFixed(2);
      row.append(label, participation, ratio);
      byId("sector-list").append(row);
    }
  }
  byId("breadth-advances").textContent = breadth?.advances ?? "—";
  byId("breadth-declines").textContent = breadth?.declines ?? "—";
  byId("breadth-unchanged").textContent = breadth?.unchanged ?? "—";
  byId("breadth-ratio").textContent = breadth?.advance_decline_ratio == null ? "—" : breadth.advance_decline_ratio.toFixed(2);
  byId("breadth-asof").textContent = breadth?.as_of ? formatDate(breadth.as_of) : "No aligned universe";
  byId("breadth-note").textContent = breadth
    ? `${breadth.sample_count} of ${breadth.universe_size} symbols aligned · ${breadth.timeframe} bars. ${breadth.missing_symbols.length} missing.`
    : "No imported NSE equity universe is available for breadth.";
}

function renderPortfolio(snapshot) {
  const positions = snapshot.positions || [];
  const holdings = snapshot.holdings || [];
  const rows = [...positions.map((item) => ({ ...item, _kind: "POSITION" })), ...holdings.map((item) => ({ ...item, _kind: "HOLDING" }))];
  const body = byId("portfolio-rows");
  body.replaceChildren();
  byId("portfolio-position-count").textContent = `${rows.length}`;
  byId("portfolio-captured").textContent = snapshot.captured_at ? `As of ${formatDate(snapshot.captured_at)}` : "Snapshot time unavailable";
  if (!rows.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 7;
    cell.className = "empty-cell";
    cell.textContent = "No non-zero broker positions or holdings in this snapshot.";
  }
  let totalPnl = 0;
  let pnlCount = 0;
  let totalExposure = 0;
  let exposureCount = 0;
  for (const item of rows) {
    const quantity = Number(item.quantity ?? item.net_quantity);
    const average = Number(item.average_price ?? item.average_entry ?? item.avg_price);
    const last = Number(item.last_price ?? item.market_price ?? item.ltp);
    const marketValue = Number(item.market_value ?? (Number.isFinite(quantity) && Number.isFinite(last) ? quantity * last : NaN));
    const pnl = Number(item.unrealized_pnl ?? item.pnl ?? (Number.isFinite(quantity) && Number.isFinite(average) && Number.isFinite(last) ? quantity * (last - average) : NaN));
    const row = body.insertRow();
    const values = [item._kind, item.symbol || item.tradingsymbol || "—", Number.isFinite(quantity) ? quantity : "—", formatRupees(average), formatRupees(last), formatRupees(marketValue), formatRupees(pnl)];
    values.forEach((value, index) => {
      const cell = row.insertCell();
      cell.textContent = value;
      if (index === 6 && Number.isFinite(pnl)) cell.className = pnl >= 0 ? "positive" : "negative";
    });
    if (Number.isFinite(pnl)) { totalPnl += pnl; pnlCount += 1; }
    if (Number.isFinite(marketValue)) { totalExposure += Math.abs(marketValue); exposureCount += 1; }
  }
  byId("portfolio-pnl").textContent = pnlCount ? formatRupees(totalPnl) : "Unavailable";
  byId("portfolio-exposure").textContent = exposureCount ? formatRupees(totalExposure) : "Unavailable";
  byId("portfolio-margin").textContent = "Not supplied by read-only adapter";
  const reconciliation = snapshot.reconciliation || {};
  byId("portfolio-status").textContent = `Read-only broker snapshot · ${reconciliation.state || "received"} · internal fill ledger unavailable · live orders disabled.`;
}

function renderRisk(risk) {
  latestRisk = risk;
  byId("kill-switch").checked = risk.kill_switch;
  byId("risk-status").textContent = risk.kill_switch ? "HALTED" : "READY";
  byId("risk-status").classList.toggle("halted", risk.kill_switch);
  byId("risk-max-position").textContent = `${(risk.max_position_fraction * 100).toFixed(0)}% of equity`;
  byId("risk-max-exposure").textContent = `${(risk.max_exposure_fraction * 100).toFixed(0)}% of equity`;
  byId("risk-position-utilization").textContent = "Unavailable · equity not supplied";
  byId("risk-exposure-utilization").textContent = "Unavailable · equity not supplied";
  if (latestHypothesis) renderHypothesis(latestHypothesis, latestSimilarity);
}

function renderMarketDataHealth(angel, fallbackQuality, fallbackMarketData) {
  const tickQuality = angel.quality || fallbackQuality || {};
  const feedRunning = angel.connected || angel.feed_running;
  byId("health-market").textContent = angel.connected
    ? "Angel One feed"
    : angel.feed_running
      ? "Angel One REST fallback"
      : (fallbackMarketData || "Unavailable").replaceAll("_", " ");
  byId("quality-accepted").textContent = tickQuality.accepted ?? "—";
  byId("quality-duplicates").textContent = tickQuality.duplicates ?? "—";
  byId("quality-invalid").textContent = (tickQuality.invalid || 0) + (tickQuality.out_of_order || 0);
  byId("quality-stale").textContent = tickQuality.stale ?? "—";
  byId("market-mode-label").textContent = angel.connected
    ? feedRunning ? "LIVE FEED" : "SESSION ACTIVE · FEED OFF"
    : feedRunning ? "REST FALLBACK" : "DEMO / STORED DATA";
  byId("market-mode-dot").classList.toggle("online", feedRunning);
}

function renderSystemHealth(health, angel, llm, reference) {
  byId("health-api").textContent = health.status === "ok" ? "Online · local" : "Unavailable";
  byId("health-execution").textContent = health.broker_execution_enabled ? "Enabled" : "Disabled · safe mode";
  byId("health-reference").textContent = reference.configured ? `Loaded · ${reference.provider}` : "Not configured";
  renderMarketDataHealth(angel, health.data_quality, health.market_data);
}

async function refreshPortfolioSnapshot() {
  const button = byId("portfolio-refresh");
  button.disabled = true;
  button.textContent = "Refreshing…";
  try {
    const snapshot = await api.getPortfolioSnapshot();
    renderPortfolio(snapshot);
    notify("Read-only portfolio snapshot received. No orders were changed.");
  } catch (error) {
    byId("portfolio-status").textContent = `Snapshot unavailable · ${error.message}`;
    notify(error.message);
  } finally {
    button.textContent = "Refresh snapshot";
    await refreshProviderStatus();
  }
}

function formatDepth(levels) {
  const best = levels?.[0];
  return best ? `${Number(best.price).toLocaleString("en-IN", { maximumFractionDigits: 2 })} × ${Number(best.quantity).toLocaleString("en-IN")}` : "—";
}

function connectMarketSocket() {
  if (marketSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(marketSocket.readyState)) return;
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${scheme}//${location.host}/ws/market/ticks`);
  marketSocket = socket;
  socket.addEventListener("open", () => {
    streamConnected = true;
    reconnectDelay = 1000;
    updateMarketDataStatus();
    socket.send("ping");
    marketHeartbeat = window.setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) socket.send("ping");
    }, 20_000);
  });
  socket.addEventListener("message", (event) => {
    try {
      const tick = JSON.parse(event.data);
      if (!tick.symbol || !Number.isFinite(Number(tick.last_price)) || Number(tick.last_price) <= 0) return;
      const timestamp = tickTime(tick);
      if (!timestamp) return;
      const prior = latestTicks.get(tick.symbol);
      if (prior && tickTime(prior) > timestamp) return;
      const tickKey = [tick.symbol, timestamp.toISOString(), tick.last_price, tick.volume ?? ""].join("|");
      if (seenTickKeys.has(tickKey)) return;
      seenTickKeys.add(tickKey);
      seenTickQueue.push(tickKey);
      if (seenTickQueue.length > 2000) seenTickKeys.delete(seenTickQueue.shift());
      latestTicks.set(tick.symbol, tick);
      renderLiveWatchlist();
      renderWatchlist(false);
      updateMarketDataStatus();
    } catch {
      return;
    }
  });
  socket.addEventListener("close", () => {
    clearInterval(marketHeartbeat);
    streamConnected = false;
    updateMarketDataStatus();
    const delay = reconnectDelay;
    reconnectDelay = Math.min(30_000, reconnectDelay * 2);
    marketReconnect = window.setTimeout(connectMarketSocket, delay);
  });
  socket.addEventListener("error", () => socket.close());
}

async function refreshProviderStatus() {
  const [angel, llm] = await Promise.all([
    api.getAngelOneStatus(),
    api.getLlmStatus(),
  ]);
  renderProviderStatus(angel, llm);
  renderMarketDataHealth(angel);
}

async function askModel() {
  const button = byId("llm-review-button");
  const providerLabel = byId("llm-provider-name").textContent.includes("OPENAI") ? "OpenAI" : "Copilot";
  button.disabled = true;
  button.textContent = "Reviewing…";
  try {
    const result = await api.askLlmHypothesis(byId("instrument").value);
    renderHypothesis({ hypothesis: result.decision, validation: result.validation }, latestSimilarity);
    byId("hypothesis-evidence").textContent = `${result.validation.reasons[0] || "Review only · no execution authority"} · Model ${result.reproducibility.model} · Context ${result.reproducibility.context_hash.slice(0, 12)}`;
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
    await api.loginAngelOne(values);
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
    const result = await api.startAngelOneFeed();
    notify(`Market feed requested for ${result.watchlist_count} instruments. Orders remain disabled.`);
    await refreshProviderStatus();
  } catch (error) {
    notify(error.message);
    await refreshProviderStatus();
  }
}

async function logoutAngelOne() {
  try {
    await api.logoutAngelOne();
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

function renderWalkForward(result) {
  const summary = byId("walk-forward-summary");
  summary.textContent = `${result.symbol} · ${result.fold_count} expanding folds · ${result.total_test_bars} out-of-sample bars · mean fold return ${result.mean_fold_return_pct >= 0 ? "+" : ""}${result.mean_fold_return_pct.toFixed(2)}% · worst fold drawdown ${result.worst_fold_drawdown_pct.toFixed(2)}% · returns not compounded`;
  const table = byId("walk-forward-folds");
  table.replaceChildren();
  for (const fold of result.folds) {
    const row = table.insertRow();
    const returnPct = `${fold.return_pct >= 0 ? "+" : ""}${fold.return_pct.toFixed(2)}%`;
    const drawdownPct = `${fold.max_drawdown_pct.toFixed(2)}%`;
    const values = [
      fold.fold,
      fold.training_bars,
      `${formatDate(fold.test_start)} - ${formatDate(fold.test_end)}`,
      fold.test_bars,
      returnPct,
      drawdownPct,
      fold.closed_trades,
      fold.risk_rejections,
    ];
    values.forEach((value, index) => {
      const cell = row.insertCell();
      cell.textContent = value;
      if (index === 4) cell.className = `pnl ${fold.return_pct >= 0 ? "positive" : "negative"}`;
    });
  }
}

async function runWalkForward() {
  const button = byId("walk-forward-button");
  const original = button.innerHTML;
  button.disabled = true;
  button.textContent = "Evaluating…";
  try {
    const result = await api.runWalkForward({ symbol: byId("instrument").value });
    renderWalkForward(result);
    notify("Walk-forward evaluation complete. Fold returns are not compounded.");
  } catch (error) {
    byId("walk-forward-summary").textContent = `Walk-forward unavailable · ${error.message}`;
    notify(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

async function run(mode) {
  const button = byId(mode === "paper" ? "paper-button" : "backtest-button");
  const original = button.innerHTML;
  button.disabled = true;
  button.textContent = "Running…";
  try {
    const result = await api.runSimulation(mode, { symbol: byId("instrument").value });
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
  const [events, paperEvents] = await Promise.all([api.getEvents(100), api.getPaperEvents(40)]);
  renderEvents(events);
  renderPaperTimeline(paperEvents);
}

async function refreshNews() {
  renderNews(await api.getNews(20));
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
    const result = await api.createNews({
        source: values.source.trim(),
        title: values.title.trim(),
        original_url: values.original_url.trim() || null,
        published_at: publicationDate.toISOString(),
        instruments: values.instruments.split(",").map((symbol) => symbol.trim().toUpperCase()).filter(Boolean),
        sector: values.sector.trim() || null,
        country: values.country.trim() || null,
        source_reliability: Number(values.source_reliability),
        content: values.content.trim(),
    });
    byId("news-dialog").close();
    form.reset();
    if (result.duplicate) notify("This article was already ingested from another source.");
    else notify("Article ingested and classified.");
    await Promise.all([refreshNews(), refreshEvents()]);
    const proposal = await api.getHypothesis(byId("instrument").value);
    latestSimilarity = await getHistoricalSimilarity(byId("instrument").value);
    renderHypothesis(proposal, latestSimilarity);
  } catch (error) {
    notify(error.message);
  } finally {
    submit.disabled = false;
    submit.textContent = "Ingest article";
  }
}

async function refreshRisk() {
  renderRisk(await api.getRisk());
}

function selectInstrument(value, targetSection = "instrument-detail") {
  const entered = value.trim();
  if (!entered) return;
  const match = instruments.find((item) => item.symbol.toUpperCase() === entered.toUpperCase());
  if (!match) {
    byId("watchlist-search").value = entered;
    renderWatchlist();
    byId("watchlist").scrollIntoView({ behavior: "smooth", block: "start" });
    notify("Instrument is not configured; showing matching watchlist results.");
    return;
  }
  const selector = byId("instrument");
  const alreadySelected = selector.value === match.symbol;
  selector.value = match.symbol;
  byId("global-search").value = "";
  byId("watchlist-search").value = "";
  renderWatchlist(false);
  if (!alreadySelected) initialize();
  if (candidateScanSnapshot) renderCandidateQueue(candidateScanSnapshot);
  byId(targetSection).scrollIntoView({ behavior: "smooth", block: "start" });
}

function toggleSidebar() {
  const mobile = window.matchMedia("(max-width: 760px)").matches;
  if (mobile) {
    document.body.classList.toggle("sidebar-open");
    byId("rail-backdrop").hidden = !document.body.classList.contains("sidebar-open");
    byId("sidebar-toggle").setAttribute("aria-expanded", String(document.body.classList.contains("sidebar-open")));
    return;
  }
  document.body.classList.toggle("sidebar-collapsed");
  byId("sidebar-toggle").setAttribute("aria-expanded", String(!document.body.classList.contains("sidebar-collapsed")));
}

function closeMobileSidebar() {
  if (!window.matchMedia("(max-width: 760px)").matches) return;
  document.body.classList.remove("sidebar-open");
  byId("rail-backdrop").hidden = true;
  byId("sidebar-toggle").setAttribute("aria-expanded", "false");
}

async function initialize() {
  try {
    const availableInstruments = await api.getInstruments();
    initializeWatchlist(availableInstruments);
    const selector = byId("instrument");
    const selected = selector.value;
    selector.replaceChildren();
    for (const instrument of availableInstruments) {
      const option = document.createElement("option");
      option.value = instrument.symbol;
      option.textContent = `${instrument.symbol} · ${instrument.data_source === "synthetic_demo_ohlcv" ? "synthetic" : "stored"}`;
      selector.append(option);
    }
    if (availableInstruments.some((item) => item.symbol === selected)) selector.value = selected;
    const symbol = selector.value;
    void refreshCandidateQueue(availableInstruments);
    latestHypothesis = null;
    latestSimilarity = null;
    const [candles, events, paperEvents, health, features, news, proposal, angel, llm, risk, regime, reference, multiTimeframe] = await Promise.all([
      api.getCandles(symbol, { limit: 1 }),
      api.getEvents(100),
      api.getPaperEvents(40),
      api.getHealth(),
      api.getFeatures(symbol),
      api.getNews(20),
      api.getHypothesis(symbol),
      api.getAngelOneStatus(),
      api.getLlmStatus(),
      api.getRisk(),
      api.getRegime(),
      api.getReferenceStatus(),
      optional(() => api.getMultiTimeframeFeatures(symbol)),
    ]);
    const findInstrument = (names) => {
      const normalized = new Map(availableInstruments.map((item) => [item.symbol.toUpperCase(), item]));
      return names.map((name) => normalized.get(name)).find(Boolean) || null;
    };
    const readIndexHistory = async (names) => {
      for (const name of names) {
        const instrument = findInstrument([name]);
        if (!instrument) continue;
        const candles = await optional(() => api.getCandles(instrument.symbol, { limit: 2 }));
        if (candles?.length) return { instrument, candles };
      }
      return { instrument: null, candles: null };
    };
    const [niftyHistory, bankHistory, sensexHistory, vixHistory] = await Promise.all([
      readIndexHistory(["NIFTY", "NIFTY 50"]),
      readIndexHistory(["BANKNIFTY", "NIFTY BANK"]),
      readIndexHistory(["SENSEX"]),
      readIndexHistory(["INDIA VIX", "INDIAVIX"]),
    ]);
    const { instrument: niftyInstrument, candles: niftyCandles } = niftyHistory;
    const { instrument: bankInstrument, candles: bankCandles } = bankHistory;
    const { instrument: sensexInstrument, candles: sensexCandles } = sensexHistory;
    const { instrument: vixInstrument, candles: vixCandles } = vixHistory;
    const equitySymbols = availableInstruments
      .filter((item) => instrumentGroup(item.symbol) === "EQUITY")
      .map((item) => item.symbol.toUpperCase());
    const [breadth, similarity] = await Promise.all([
      equitySymbols.length ? optional(() => api.getBreadth({ symbols: equitySymbols, timeframe: "1d" })) : Promise.resolve(null),
      getHistoricalSimilarity(symbol, multiTimeframe),
    ]);
    renderEvents(events);
    renderPaperTimeline(paperEvents);
    renderFeatures(features);
    await renderInstrumentDetail(symbol, multiTimeframe, news);
    latestSimilarity = similarity;
    renderRisk(risk);
    renderHypothesis(proposal, similarity);
    renderProviderStatus(angel, llm);
    renderSystemHealth(health, angel, llm, reference);
    renderMarketOverview({ nifty: niftyCandles, banknifty: bankCandles, sensex: sensexCandles, vix: vixCandles, niftyInstrument, bankInstrument, sensexInstrument, vixInstrument, breadth, regime });
    if (health.broker_execution_enabled) notify("Unexpected execution state: check server configuration.");
    if (!candles.length) throw new Error("No sample candles are available.");
  } catch (error) {
    notify(`Could not connect to the local API: ${error.message}`);
  }
}

byId("backtest-button").addEventListener("click", () => run("backtest"));
byId("walk-forward-button").addEventListener("click", runWalkForward);
byId("paper-button").addEventListener("click", () => run("paper"));
byId("llm-review-button").addEventListener("click", askModel);
byId("angelone-login-open").addEventListener("click", () => byId("angelone-login-dialog").showModal());
byId("angelone-login-close").addEventListener("click", () => byId("angelone-login-dialog").close());
byId("angelone-login-cancel").addEventListener("click", () => byId("angelone-login-dialog").close());
byId("angelone-login-form").addEventListener("submit", submitAngelLogin);
byId("angelone-login-dialog").addEventListener("close", () => byId("angelone-login-form").reset());
byId("angelone-feed-start").addEventListener("click", startAngelFeed);
byId("angelone-logout").addEventListener("click", logoutAngelOne);
byId("news-add-open").addEventListener("click", () => {
  byId("news-form").elements.namedItem("instruments").value = byId("instrument").value;
  byId("news-dialog").showModal();
});
byId("candidate-refresh").addEventListener("click", () => refreshCandidateQueue(instruments, true));
byId("candidate-list").addEventListener("click", (event) => {
  const symbol = event.target.closest("[data-candidate-symbol]")?.dataset.candidateSymbol;
  if (symbol) selectInstrument(symbol, "signals");
});
byId("news-dialog-close").addEventListener("click", () => byId("news-dialog").close());
byId("news-cancel").addEventListener("click", () => byId("news-dialog").close());
byId("news-form").addEventListener("submit", submitNews);
byId("news-dialog").addEventListener("close", () => byId("news-form").reset());
byId("refresh-button").addEventListener("click", initialize);
byId("instrument").addEventListener("change", initialize);
byId("instrument-timeframes").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-timeframe]");
  if (!button || button.disabled || button.dataset.timeframe === selectedChartTimeframe) return;
  selectedChartTimeframe = button.dataset.timeframe;
  for (const option of byId("instrument-timeframes").querySelectorAll("[data-timeframe]")) {
    option.setAttribute("aria-pressed", String(option === button));
  }
  renderInstrumentIndicators(instrumentFeatureSnapshots[selectedChartTimeframe], selectedChartTimeframe);
  byId("instrument-chart-note").textContent = "Loading stored candles…";
  await loadInstrumentCandles(byId("instrument").value, selectedChartTimeframe, instrumentDetailGeneration);
});
byId("instrument-overlays").addEventListener("change", () => {
  if (instrumentChartCandles.length) renderCandleChart(instrumentChartCandles);
});
byId("watchlist-search").addEventListener("input", () => renderWatchlist());
byId("watchlist-group").addEventListener("change", () => renderWatchlist());
byId("watchlist-sort").addEventListener("change", () => renderWatchlist(false));
byId("watchlist-rows").addEventListener("click", (event) => {
  const pin = event.target.closest("[data-pin-symbol]");
  if (pin) return togglePinnedSymbol(pin.dataset.pinSymbol);
  const symbol = event.target.closest("[data-select-symbol]")?.dataset.selectSymbol;
  if (symbol) selectInstrument(symbol);
});
byId("global-search").addEventListener("change", (event) => selectInstrument(event.target.value));
byId("global-search").addEventListener("keydown", (event) => {
  if (event.key === "Enter") selectInstrument(event.currentTarget.value);
});
byId("sidebar-toggle").addEventListener("click", toggleSidebar);
byId("rail-backdrop").addEventListener("click", closeMobileSidebar);
document.querySelectorAll(".nav-item").forEach((link) => link.addEventListener("click", () => {
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item === link));
  closeMobileSidebar();
}));
byId("portfolio-refresh").addEventListener("click", refreshPortfolioSnapshot);
byId("events-refresh").addEventListener("click", async (event) => {
  event.preventDefault();
  try { await refreshEvents(); } catch (error) { notify(error.message); }
});
["alerts-feed-toggle", "alerts-simulation-toggle"].forEach((id) => byId(id).addEventListener("change", () => {
  alertPreferences.feed = byId("alerts-feed-toggle").checked;
  alertPreferences.simulation = byId("alerts-simulation-toggle").checked;
  try {
    localStorage.setItem("northstar.alert-preferences", JSON.stringify(alertPreferences));
  } catch {}
  renderOperationalAlerts(recentAuditEvents);
}));
byId("kill-switch").addEventListener("change", async (event) => {
  event.target.disabled = true;
  try {
    const risk = await api.setKillSwitch(event.target.checked);
    renderRisk(risk);
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
window.setInterval(() => {
  updateGlobalClock();
  updateMarketDataStatus();
}, 1000);
window.setInterval(() => {
  refreshEvents().catch(() => {});
}, 15000);
window.setInterval(() => renderWatchlist(false), 5000);

byId("alerts-feed-toggle").checked = alertPreferences.feed;
byId("alerts-simulation-toggle").checked = alertPreferences.simulation;
updateGlobalClock();
initialize();
connectMarketSocket();