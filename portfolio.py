"""Portfolio snapshots and deterministic risk-based position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: int
    average_entry: float
    market_price: float
    sector: str = "UNKNOWN"
    asset_class: str = "EQUITY"

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool) or self.quantity < 1:
            raise ValueError("Position quantity must be a positive integer")
        if any(not _finite_number(value) or value <= 0 for value in (self.average_entry, self.market_price)):
            raise ValueError("Position prices must be finite and positive")
        if not _finite_number(self.market_value) or not _finite_number(self.unrealized_pnl):
            raise ValueError("Position market value and P&L must be finite")

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

    def __post_init__(self) -> None:
        if not _finite_number(self.cash) or not _finite_number(self.realized_pnl):
            raise ValueError("Portfolio cash and realized P&L must be finite numbers")
        if self.high_water_mark is not None and (
            not _finite_number(self.high_water_mark) or self.high_water_mark <= 0
        ):
            raise ValueError("Portfolio high-water mark must be finite and positive")
        exposure = sum(position.market_value for position in self.positions)
        equity = self.cash + exposure
        unrealized_pnl = sum(position.unrealized_pnl for position in self.positions)
        if not all(_finite_number(value) for value in (exposure, equity, unrealized_pnl)):
            raise ValueError("Portfolio exposure, equity, and P&L must be finite")

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
        if any(not _finite_number(value) for value in (
            self.max_loss_fraction, self.max_position_fraction, self.max_exposure_fraction,
        )):
            raise ValueError("Sizing fractions must be finite numbers")
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
        if any(not _finite_number(value) for value in (equity, entry_price, stop_price, existing_exposure)):
            raise ValueError("Sizing inputs must be finite numbers")
        if equity <= 0 or entry_price <= 0 or stop_price <= 0 or stop_price >= entry_price:
            return 0
        equity_amount = Decimal(str(equity))
        entry_amount = Decimal(str(entry_price))
        risk_per_share = entry_amount - Decimal(str(stop_price))
        loss_budget = equity_amount * Decimal(str(self.config.max_loss_fraction))
        exposure_budget = max(
            Decimal("0"),
            equity_amount * Decimal(str(self.config.max_exposure_fraction)) - Decimal(str(existing_exposure)),
        )
        notional_budget = min(
            equity_amount * Decimal(str(self.config.max_position_fraction)),
            exposure_budget,
        )
        return max(0, int(min(loss_budget / risk_per_share, notional_budget / entry_amount)))


@dataclass(frozen=True)
class PortfolioOptimizationLimits:
    max_gross_exposure_fraction: Decimal
    max_symbol_exposure_fraction: Decimal
    max_sector_exposure_fraction: Decimal
    max_asset_class_exposure_fraction: Decimal
    max_correlation_weighted_exposure_fraction: Decimal

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, Decimal) or not value.is_finite() or not Decimal("0") < value <= Decimal("1"):
                raise ValueError(f"{name} must be a finite decimal fraction in (0, 1]")


class PortfolioOptimizer:
    """Screen a candidate against portfolio constraints; never creates or submits an order."""

    def __init__(self, limits: PortfolioOptimizationLimits):
        self.limits = limits

    def evaluate_addition(
        self,
        portfolio: PortfolioSnapshot,
        *,
        symbol: str,
        sector: str,
        asset_class: str,
        notional: Decimal,
        expected_net_value: Decimal,
        correlations_to_positions: dict[str, Decimal],
    ) -> dict:
        if not symbol.strip() or not sector.strip() or not asset_class.strip():
            raise ValueError("Candidate symbol, sector, and asset class are required")
        if not isinstance(notional, Decimal) or not notional.is_finite() or notional <= 0:
            raise ValueError("Candidate notional must be a finite positive decimal")
        if not isinstance(expected_net_value, Decimal) or not expected_net_value.is_finite():
            raise ValueError("Expected net value must be a finite decimal")

        equity = Decimal(str(portfolio.cash)) + sum(
            (Decimal(str(position.quantity)) * Decimal(str(position.market_price)) for position in portfolio.positions),
            Decimal("0"),
        )
        if equity <= 0:
            return self._rejected("Portfolio equity must be positive")

        candidate_symbol = symbol.strip().upper()
        candidate_sector = sector.strip().upper()
        candidate_asset = asset_class.strip().upper()
        symbol_exposure = sum((self._position_notional(p) for p in portfolio.positions if p.symbol.upper() == candidate_symbol), Decimal("0"))
        sector_exposure = sum((self._position_notional(p) for p in portfolio.positions if p.sector.upper() == candidate_sector), Decimal("0"))
        asset_exposure = sum((self._position_notional(p) for p in portfolio.positions if p.asset_class.upper() == candidate_asset), Decimal("0"))
        gross_exposure = sum((self._position_notional(p) for p in portfolio.positions), Decimal("0"))

        reasons: list[str] = []
        correlated_exposure = notional
        for held_symbol in sorted({p.symbol.upper() for p in portfolio.positions if p.symbol.upper() != candidate_symbol}):
            correlation = correlations_to_positions.get(held_symbol)
            if correlation is None:
                reasons.append(f"Correlation data is missing for {held_symbol}")
                continue
            if not isinstance(correlation, Decimal) or not correlation.is_finite() or not Decimal("-1") <= correlation <= Decimal("1"):
                reasons.append(f"Correlation for {held_symbol} must be a finite decimal in [-1, 1]")
                continue
            held_notional = sum(
                (self._position_notional(p) for p in portfolio.positions if p.symbol.upper() == held_symbol),
                Decimal("0"),
            )
            correlated_exposure += abs(correlation) * held_notional

        projected = {
            "gross_exposure_fraction": (gross_exposure + notional) / equity,
            "symbol_exposure_fraction": (symbol_exposure + notional) / equity,
            "sector_exposure_fraction": (sector_exposure + notional) / equity,
            "asset_class_exposure_fraction": (asset_exposure + notional) / equity,
            "correlation_weighted_exposure_fraction": correlated_exposure / equity,
        }
        checks = (
            ("gross_exposure_fraction", self.limits.max_gross_exposure_fraction, "Maximum gross exposure would be exceeded"),
            ("symbol_exposure_fraction", self.limits.max_symbol_exposure_fraction, "Maximum symbol exposure would be exceeded"),
            ("sector_exposure_fraction", self.limits.max_sector_exposure_fraction, "Maximum sector exposure would be exceeded"),
            ("asset_class_exposure_fraction", self.limits.max_asset_class_exposure_fraction, "Maximum asset-class exposure would be exceeded"),
            ("correlation_weighted_exposure_fraction", self.limits.max_correlation_weighted_exposure_fraction, "Maximum correlation-weighted exposure would be exceeded"),
        )
        for metric, limit, reason in checks:
            if projected[metric] > limit:
                reasons.append(reason)
        if expected_net_value <= 0:
            reasons.append("Expected net value is not positive")
        return {
            "eligible_for_risk_review": not reasons,
            "reasons": reasons,
            "equity": str(equity),
            "candidate_notional": str(notional),
            "expected_net_value": str(expected_net_value),
            "projected_exposure_fractions": {key: str(value) for key, value in projected.items()},
            "execution_authority": False,
        }

    @staticmethod
    def _position_notional(position: Position) -> Decimal:
        return abs(Decimal(str(position.quantity)) * Decimal(str(position.market_price)))

    @staticmethod
    def _rejected(reason: str) -> dict:
        return {
            "eligible_for_risk_review": False,
            "reasons": [reason],
            "execution_authority": False,
        }


def _finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False