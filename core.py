"""Deterministic strategy, risk, and simulation primitives for the safe-first MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import floor, sqrt
from typing import Literal, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class Candle:
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol or self.open <= 0 or self.close <= 0:
            raise ValueError("Candles require a symbol and positive prices")
        if self.low <= 0 or self.high < max(self.open, self.close, self.low):
            raise ValueError("Invalid candle price range")
        if self.low > min(self.open, self.close):
            raise ValueError("Candle low cannot exceed open or close")
        datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))


Action = Literal["BUY", "SELL", "HOLD"]


class Strategy(Protocol):
    def action(self, history: list[Candle]) -> Action: ...


@dataclass(frozen=True)
class SmaCrossStrategy:
    fast_window: int = 8
    slow_window: int = 21

    def __post_init__(self) -> None:
        if self.fast_window < 2 or self.slow_window <= self.fast_window:
            raise ValueError("Require 2 <= fast_window < slow_window")

    def action(self, history: list[Candle]) -> Action:
        if len(history) < self.slow_window + 1:
            return "HOLD"
        previous_fast = _sma(history[:-1], self.fast_window)
        previous_slow = _sma(history[:-1], self.slow_window)
        current_fast = _sma(history, self.fast_window)
        current_slow = _sma(history, self.slow_window)
        if previous_fast <= previous_slow and current_fast > current_slow:
            return "BUY"
        if previous_fast >= previous_slow and current_fast < current_slow:
            return "SELL"
        return "HOLD"


def _sma(candles: list[Candle], window: int) -> float:
    return sum(candle.close for candle in candles[-window:]) / window


@dataclass(frozen=True)
class RiskConfig:
    max_position_fraction: float = 0.10
    max_exposure_fraction: float = 0.50
    max_slippage_bps: float = 50.0
    kill_switch: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction must be in (0, 1]")
        if not 0 < self.max_exposure_fraction <= 1:
            raise ValueError("max_exposure_fraction must be in (0, 1]")
        if self.max_slippage_bps < 0:
            raise ValueError("max_slippage_bps cannot be negative")


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    quantity: int
    reason: str


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def size_entry(
        self,
        *,
        price: float,
        equity: float,
        current_exposure: float,
        data_is_fresh: bool = True,
    ) -> RiskDecision:
        if self.config.kill_switch:
            return RiskDecision(False, 0, "Emergency kill switch is active")
        if not data_is_fresh:
            return RiskDecision(False, 0, "Market data is stale")
        if price <= 0 or equity <= 0:
            return RiskDecision(False, 0, "Price and equity must be positive")
        remaining_exposure = equity * self.config.max_exposure_fraction - current_exposure
        max_order_value = equity * self.config.max_position_fraction
        quantity = floor(max(0.0, min(remaining_exposure, max_order_value)) / price)
        if quantity < 1:
            return RiskDecision(False, 0, "Risk limits leave no permissible quantity")
        return RiskDecision(True, quantity, "Approved within configured exposure limits")


@dataclass(frozen=True)
class SimulationConfig:
    starting_cash: float = 100_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.starting_cash <= 0 or self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Cash must be positive and cost assumptions non-negative")


def run_simulation(
    candles: list[Candle],
    strategy: Strategy,
    *,
    config: SimulationConfig | None = None,
    risk_engine: RiskEngine | None = None,
    mode: Literal["backtest", "paper"] = "backtest",
) -> dict:
    """Replay bars without look-ahead; close-derived signals fill at the next bar open."""
    if len(candles) < 2:
        raise ValueError("At least two candles are required")
    if any(candles[index].timestamp >= candles[index + 1].timestamp for index in range(len(candles) - 1)):
        raise ValueError("Candles must be strictly chronological")
    if len({candle.symbol for candle in candles}) != 1:
        raise ValueError("A simulation accepts candles for exactly one instrument")

    config = config or SimulationConfig()
    risk_engine = risk_engine or RiskEngine()
    if config.slippage_bps > risk_engine.config.max_slippage_bps:
        raise ValueError("Configured slippage exceeds the risk engine limit")
    cash = config.starting_cash
    quantity = 0
    entry_cost = 0.0
    entry_time = ""
    trades: list[dict] = []
    events: list[dict] = []
    equity_curve: list[dict] = []
    risk_rejections = 0
    correlation_id = str(uuid4())

    for index, candle in enumerate(candles):
        if index > 0:
            action = strategy.action(candles[:index])
            if action != "HOLD":
                events.append(_event("signal.generated", candle.timestamp, correlation_id, {
                    "symbol": candle.symbol,
                    "action": action,
                    "based_on": candles[index - 1].timestamp,
                    "fill_at": candle.timestamp,
                    "mode": mode,
                }))
            if action == "BUY" and quantity == 0:
                estimated_price = candle.open * (1 + config.slippage_bps / 10_000)
                decision = risk_engine.size_entry(
                    price=estimated_price,
                    equity=cash,
                    current_exposure=0.0,
                    data_is_fresh=True,
                )
                if not decision.approved:
                    risk_rejections += 1
                    events.append(_event("risk.rejected", candle.timestamp, correlation_id, {
                        "symbol": candle.symbol,
                        "reason": decision.reason,
                    }))
                else:
                    quantity = decision.quantity
                    fill_price = estimated_price
                    fee = fill_price * quantity * config.fee_bps / 10_000
                    cash -= fill_price * quantity + fee
                    entry_cost = fill_price * quantity + fee
                    entry_time = candle.timestamp
                    events.append(_event("order.filled", candle.timestamp, correlation_id, {
                        "symbol": candle.symbol,
                        "side": "BUY",
                        "quantity": quantity,
                        "price": fill_price,
                        "mode": mode,
                    }))
            elif action == "SELL" and quantity > 0:
                fill_price = candle.open * (1 - config.slippage_bps / 10_000)
                fee = fill_price * quantity * config.fee_bps / 10_000
                proceeds = fill_price * quantity - fee
                cash += proceeds
                trade = {
                    "symbol": candle.symbol,
                    "entry_time": entry_time,
                    "exit_time": candle.timestamp,
                    "quantity": quantity,
                    "entry_cost": round(entry_cost, 2),
                    "exit_proceeds": round(proceeds, 2),
                    "pnl": round(proceeds - entry_cost, 2),
                    "exit_price": round(fill_price, 4),
                }
                trades.append(trade)
                events.append(_event("order.filled", candle.timestamp, correlation_id, {
                    "symbol": candle.symbol,
                    "side": "SELL",
                    "quantity": quantity,
                    "price": fill_price,
                    "mode": mode,
                }))
                events.append(_event("trade.closed", candle.timestamp, correlation_id, trade))
                quantity = 0
                entry_cost = 0.0
                entry_time = ""

        equity = cash + quantity * candle.close
        equity_curve.append({"timestamp": candle.timestamp, "equity": round(equity, 2)})

    ending_equity = cash + quantity * candles[-1].close
    returns = [
        equity_curve[index]["equity"] / equity_curve[index - 1]["equity"] - 1
        for index in range(1, len(equity_curve))
        if equity_curve[index - 1]["equity"] > 0
    ]
    peak = equity_curve[0]["equity"]
    max_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point["equity"])
        if peak:
            max_drawdown = max(max_drawdown, (peak - point["equity"]) / peak)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns) if returns else 0.0
    wins = sum(trade["pnl"] > 0 for trade in trades)
    result = {
        "mode": mode,
        "symbol": candles[0].symbol,
        "strategy": {"name": "sma_cross", "fast_window": getattr(strategy, "fast_window", None), "slow_window": getattr(strategy, "slow_window", None)},
        "data_source": "deterministic_demo_ohlcv",
        "bar_count": len(candles),
        "starting_cash": round(config.starting_cash, 2),
        "ending_equity": round(ending_equity, 2),
        "return_pct": round((ending_equity / config.starting_cash - 1) * 100, 3),
        "max_drawdown_pct": round(max_drawdown * 100, 3),
        "sharpe_annualized": round(mean_return / sqrt(variance) * sqrt(252), 3) if variance > 0 else 0.0,
        "win_rate_pct": round(wins / len(trades) * 100, 2) if trades else 0.0,
        "closed_trades": len(trades),
        "open_quantity": quantity,
        "risk_rejections": risk_rejections,
        "equity_curve": equity_curve,
        "trades": trades,
        "events": events,
    }
    result["events"].append(_event("simulation.completed", candles[-1].timestamp, correlation_id, {
        key: result[key] for key in ("mode", "symbol", "strategy", "bar_count", "starting_cash", "ending_equity", "return_pct", "max_drawdown_pct")
    }))
    return result


def demo_candles(symbol: str = "DEMO") -> list[Candle]:
    """Reproducible synthetic data for UI and test-driving; not market data."""
    candles: list[Candle] = []
    price = 100.0
    start = date(2025, 1, 1)
    for index in range(240):
        drift = 0.13 if (index // 32) % 2 == 0 else -0.08
        change = drift + ((index * 17) % 13 - 6) * 0.11
        open_price = price
        close = max(20.0, price + change)
        high = max(open_price, close) + 0.3 + (index % 4) * 0.04
        low = min(open_price, close) - 0.3 - (index % 3) * 0.05
        timestamp = datetime.combine(start + timedelta(days=index), datetime.min.time(), tzinfo=timezone.utc).isoformat()
        candles.append(Candle(symbol, timestamp, round(open_price, 4), round(high, 4), round(low, 4), round(close, 4), 100_000 + (index * 7919) % 80_000))
        price = close
    return candles


def _event(event_type: str, timestamp: str, correlation_id: str, payload: dict) -> dict:
    return {
        "event_id": str(uuid4()),
        "event_type": event_type,
        "timestamp": timestamp,
        "correlation_id": correlation_id,
        "source": "simulation_engine",
        "version": 1,
        "schema_version": "1.0",
        "payload": payload,
    }