(() => {
  /** @typedef {{symbol: string, bars: number, latest_timestamp: string, data_source?: string}} InstrumentDto */
  /** @typedef {{symbol: string, timeframe: string, timestamp: string, open: number, high: number, low: number, close: number, volume: number}} CandleDto */
  /** @typedef {{symbol: string, market_timestamp?: string, timestamp?: string, last_price: number, volume?: number, open_interest?: number}} TickDto */

  async function request(path, options = {}) {
    const response = await fetch(`/api${path}`, options);
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      throw new Error(payload?.detail || `Request failed (${response.status})`);
    }
    return payload;
  }

  function post(path, body) {
    return request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  const encode = encodeURIComponent;
  const api = Object.freeze({
    /** @returns {Promise<InstrumentDto[]>} */
    getInstruments: () => request("/instruments"),
    /** @returns {Promise<CandleDto[]>} */
    getCandles: (symbol, { limit = 250, timeframe = "1d" } = {}) =>
      request(`/market/${encode(symbol)}/candles?limit=${limit}&timeframe=${encode(timeframe)}`),
    getFeatures: (symbol, timeframe = "1d") =>
      request(`/market/${encode(symbol)}/features?timeframe=${encode(timeframe)}`),
    getMultiTimeframeFeatures: (symbol) =>
      request(`/market/${encode(symbol)}/features/multi-timeframe`),
    getEvents: (limit = 50) => request(`/events?limit=${limit}`),
    getPaperEvents: (limit = 40) => request(`/paper/events?limit=${limit}`),
    runWalkForward: (body) => post("/backtests/walk-forward", body),
    getHealth: () => request("/health"),
    getSignal: (symbol, timeframe = "1d") =>
      request(`/signals/${encode(symbol)}?timeframe=${encode(timeframe)}`),
    getNews: (limit = 50) => request(`/news?limit=${limit}`),
    getHypothesis: (symbol) => post(`/hypotheses/${encode(symbol)}`, {}),
    getAngelOneStatus: () => request("/angelone/status"),
    getLlmStatus: () => request("/llm/status"),
    getRisk: () => request("/risk"),
    getRegime: (timeframe = "1d") => request(`/market/regime?timeframe=${encode(timeframe)}`),
    getReferenceStatus: () => request("/india/reference-data/status"),
    getBreadth: (body) => post("/market/breadth", body),
    getSimilarity: (symbol, { timeframe = "1d", horizonBars = 5, topK = 5 } = {}) =>
      request(`/market/${encode(symbol)}/similarity?timeframe=${encode(timeframe)}&horizon_bars=${horizonBars}&top_k=${topK}`),
    getPortfolioSnapshot: () => request("/angelone/portfolio/reconcile"),
    askLlmHypothesis: (symbol) => post(`/llm/hypotheses/${encode(symbol)}`, {}),
    createNews: (body) => post("/news", body),
    loginAngelOne: (body) => post("/angelone/login", body),
    startAngelOneFeed: () => post("/angelone/feed/start", {}),
    logoutAngelOne: () => post("/angelone/logout", {}),
    setKillSwitch: (enabled) => post("/risk/kill-switch", { enabled }),
    runSimulation: (mode, body) => post(mode === "paper" ? "/paper/replay" : "/backtests", body),
  });

  Object.defineProperty(window, "tradingApi", { value: api, configurable: false, writable: false });
})();