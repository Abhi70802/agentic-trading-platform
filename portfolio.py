"""Portfolio snapshots and deterministic risk-based position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: int
    average_entry: float
    market_price: float
    sector: str = "UNKNOWN"
    asset_class: str = "EQUITY"

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.market_price - self.average_entry)


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: float
    positions: tuple[Position, ...]
    realized_pnl: float = 0.0
    high_water_mark: float | None = None

    def as_dict(self) -> dict:
        exposure = sum(position.market_value for position in self.positions)
        equity = self.cash + exposure
        sectors: dict[str, float] = {}
        assets: dict[str, float] = {}
        for position in self.positions:
            sectors[position.sector] = sectors.get(position.sector, 0.0) + position.market_value
            assets[position.asset_class] = assets.get(position.asset_class, 0.0) + position.market_value
        return {
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "gross_exposure": round(exposure, 2),
            "leverage": round(exposure / equity, 4) if equity > 0 else 0.0,
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(sum(position.unrealized_pnl for position in self.positions), 2),
            "drawdown_pct": round((1 - equity / self.high_water_mark) * 100, 3)
            if self.high_water_mark and equity < self.high_water_mark else 0.0,
            "sector_exposure": {key: round(value, 2) for key, value in sectors.items()},
            "asset_class_exposure": {key: round(value, 2) for key, value in assets.items()},
            "positions": [
                {
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "average_entry": position.average_entry,
                    "market_price": position.market_price,
                    "market_value": round(position.market_value, 2),
                    "unrealized_pnl": round(position.unrealized_pnl, 2),
                    "sector": position.sector,
                    "asset_class": position.asset_class,
                }
                for position in self.positions
            ],
        }


@dataclass(frozen=True)
class SizingConfig:
    max_loss_fraction: float = 0.005
    max_position_fraction: float = 0.10
    max_exposure_fraction: float = 0.50

    def __post_init__(self) -> None:
        if not 0 < self.max_loss_fraction <= 1:
            raise ValueError("max_loss_fraction must be in (0, 1]")
        if not 0 < self.max_position_fraction <= 1 or not 0 < self.max_exposure_fraction <= 1:
            raise ValueError("Exposure fractions must be in (0, 1]")


class PositionSizer:
    def __init__(self, config: SizingConfig | None = None):
        self.config = config or SizingConfig()

    def size_long(
        self,
        *,
        equity: float,
        entry_price: float,
        stop_price: float,
        existing_exposure: float,
    ) -> int:
        if equity <= 0 or entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
            return 0
        risk_per_share = entry_price - stop_price
        loss_budget = equity * self.config.max_loss_fraction
        exposure_budget = max(0.0, equity * self.config.max_exposure_fraction - existing_exposure)
        notional_budget = min(equity * self.config.max_position_fraction, exposure_budget)
        return max(0, floor(min(loss_budget / risk_per_share, notional_budget / entry_price)))