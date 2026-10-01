"""Deterministic Phase 9 decision orchestration; never submits orders."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from compliance import ComplianceRegistry
from india_economics import IndiaExpectedValueEngine, IndiaProductType
from portfolio import PortfolioOptimizer, PortfolioSnapshot, PositionSizer


class DecisionEngine:
    def __init__(
        self,
        *,
        expected_value_engine: IndiaExpectedValueEngine,
        portfolio_optimizer: PortfolioOptimizer,
        position_sizer: PositionSizer,
        compliance_registry: ComplianceRegistry,
    ):
        self.expected_value_engine = expected_value_engine
        self.portfolio_optimizer = portfolio_optimizer
        self.position_sizer = position_sizer
        self.compliance_registry = compliance_registry

    def evaluate_long_candidate(
        self,
        *,
        portfolio: PortfolioSnapshot,
        market: str,
        product: IndiaProductType,
        symbol: str,
        sector: str,
        asset_class: str,
        entry_price: Decimal,
        stop_price: Decimal,
        average_win: Decimal,
        average_loss: Decimal,
        calibration: dict,
        buy_turnover: Decimal,
        sell_turnover: Decimal,
        slippage_bps_per_side: Decimal,
        market_impact_bps_per_side: Decimal,
        correlations_to_positions: dict[str, Decimal],
        now: datetime,
        order_type: str,
        calendar_session: bool | None,
    ) -> dict:
        calibrated_probability = calibration.get("calibrated_probability")
        if (
            calibration.get("status") != "CALIBRATED"
            or not isinstance(calibrated_probability, Decimal)
            or not calibrated_probability.is_finite()
            or not Decimal("0") <= calibrated_probability <= Decimal("1")
        ):
            return self._rejected("A finite calibrated probability is required before decision evaluation")
        finite_decimal_inputs = (
            entry_price, stop_price, average_win, average_loss, buy_turnover, sell_turnover,
            slippage_bps_per_side, market_impact_bps_per_side,
        )
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in finite_decimal_inputs):
            return self._rejected("Decision prices, risk, turnover, and costs must be finite decimals")
        cost = self.expected_value_engine.round_trip_cost(
            product=product,
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            slippage_bps_per_side=slippage_bps_per_side,
            market_impact_bps_per_side=market_impact_bps_per_side,
        )
        expected_value = self.expected_value_engine.expected_value(
            win_probability=calibrated_probability,
            average_win=average_win,
            average_loss=average_loss,
            round_trip_cost=cost,
        )
        existing_exposure = sum(position.market_value for position in portfolio.positions)
        equity = portfolio.cash + existing_exposure
        sized_quantity = self.position_sizer.size_long(
            equity=equity,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            existing_exposure=existing_exposure,
        )
        candidate_notional = Decimal(sized_quantity) * entry_price
        if candidate_notional <= 0:
            optimizer = self._rejected("Risk-based position sizing returned zero quantity")
        else:
            optimizer = self.portfolio_optimizer.evaluate_addition(
                portfolio,
                symbol=symbol,
                sector=sector,
                asset_class=asset_class,
                notional=candidate_notional,
                expected_net_value=expected_value.expected_net_value,
                correlations_to_positions=correlations_to_positions,
            )
        policy = self.compliance_registry.get(market)
        compliance = policy.evaluate(
            now=now,
            order_type=order_type,
            quantity=sized_quantity,
            notional=float(candidate_notional),
            calendar_session=calendar_session,
        )
        reasons = list(optimizer.get("reasons", [])) + list(compliance.reasons)
        if not expected_value.edge_positive:
            reasons.append("Expected net value is not positive")
        return {
            "eligible_for_risk_review": not reasons,
            "reasons": reasons,
            "probability": {key: _stringify(value) for key, value in calibration.items()},
            "expected_value": expected_value.as_dict(),
            "round_trip_cost": cost.as_dict(),
            "position_sizing": {
                "quantity": sized_quantity,
                "entry_price": str(entry_price),
                "stop_price": str(stop_price),
                "candidate_notional": str(candidate_notional),
            },
            "portfolio_optimization": optimizer,
            "compliance": {"allowed": compliance.allowed, "reasons": list(compliance.reasons)},
            "execution_authority": False,
        }

    @staticmethod
    def _rejected(reason: str) -> dict:
        return {
            "eligible_for_risk_review": False,
            "reasons": [reason],
            "execution_authority": False,
        }


def _stringify(value):
    return str(value) if isinstance(value, Decimal) else value
