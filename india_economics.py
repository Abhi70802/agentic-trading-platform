"""Configurable Indian transaction costs and deterministic net expected value."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class IndiaProductType(StrEnum):
    EQUITY_DELIVERY = "EQUITY_DELIVERY"
    EQUITY_INTRADAY = "EQUITY_INTRADAY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


@dataclass(frozen=True)
class IndiaProductChargeRates:
    brokerage_rate: Decimal
    stt_buy_rate: Decimal
    stt_sell_rate: Decimal
    exchange_transaction_rate: Decimal
    sebi_turnover_rate: Decimal
    gst_rate: Decimal
    stamp_duty_buy_rate: Decimal
    other_turnover_rate: Decimal
    gst_base_components: tuple[str, ...]
    brokerage_cap_per_order: Decimal | None

    def __post_init__(self) -> None:
        for name in (
            "brokerage_rate", "stt_buy_rate", "stt_sell_rate",
            "exchange_transaction_rate", "sebi_turnover_rate", "gst_rate",
            "stamp_duty_buy_rate", "other_turnover_rate",
        ):
            _validate_rate(name, getattr(self, name))
        allowed_gst_components = {"brokerage", "exchange_charges", "sebi_charges", "other_charges"}
        if not self.gst_base_components or not set(self.gst_base_components) <= allowed_gst_components:
            raise ValueError("GST base components must be explicitly selected from supported charge components")
        if len(self.gst_base_components) != len(set(self.gst_base_components)):
            raise ValueError("GST base components must be unique")
        if self.brokerage_cap_per_order is not None:
            _validate_amount("brokerage cap", self.brokerage_cap_per_order, allow_zero=True)


@dataclass(frozen=True)
class IndiaTransactionCostSchedule:
    rates_by_product: dict[IndiaProductType, IndiaProductChargeRates]

    def for_product(self, product: IndiaProductType) -> IndiaProductChargeRates:
        try:
            return self.rates_by_product[product]
        except KeyError as error:
            raise ValueError(f"No transaction-charge schedule configured for {product.value}") from error


@dataclass(frozen=True)
class RoundTripCost:
    product: IndiaProductType
    buy_turnover: Decimal
    sell_turnover: Decimal
    brokerage: Decimal
    stt: Decimal
    exchange_charges: Decimal
    sebi_charges: Decimal
    other_charges: Decimal
    gst: Decimal
    stamp_duty: Decimal
    slippage: Decimal
    market_impact: Decimal
    total_cost: Decimal
    currency: str = "INR"

    def __post_init__(self) -> None:
        monetary_values = (
            self.buy_turnover, self.sell_turnover, self.brokerage, self.stt,
            self.exchange_charges, self.sebi_charges, self.other_charges,
            self.gst, self.stamp_duty, self.slippage, self.market_impact, self.total_cost,
        )
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in monetary_values):
            raise ValueError("Round-trip turnover and costs must be finite decimals")
        if self.buy_turnover <= 0 or self.sell_turnover <= 0 or any(value < 0 for value in monetary_values[2:]):
            raise ValueError("Round-trip turnover must be positive and costs non-negative")

    def as_dict(self) -> dict:
        return {
            "product": self.product.value,
            "currency": self.currency,
            **{
                key: str(value)
                for key, value in self.__dict__.items()
                if isinstance(value, Decimal)
            },
        }


@dataclass(frozen=True)
class ExpectedValueResult:
    win_probability: Decimal
    average_win: Decimal
    average_loss: Decimal
    expected_gross_value: Decimal
    transaction_cost: Decimal
    expected_net_value: Decimal
    risk_reward_ratio: Decimal | None
    edge_positive: bool
    currency: str = "INR"

    def as_dict(self) -> dict:
        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in self.__dict__.items()
        }


class IndiaExpectedValueEngine:
    def __init__(self, schedule: IndiaTransactionCostSchedule):
        self.schedule = schedule

    def order_cost(self, *, product: IndiaProductType, side: str, turnover: Decimal) -> Decimal:
        _validate_amount("order turnover", turnover)
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Order side must be BUY or SELL")
        rates = self.schedule.for_product(product)
        cap = rates.brokerage_cap_per_order
        brokerage = turnover * rates.brokerage_rate
        if cap is not None:
            brokerage = min(brokerage, cap)
        exchange_charges = turnover * rates.exchange_transaction_rate
        sebi_charges = turnover * rates.sebi_turnover_rate
        other_charges = turnover * rates.other_turnover_rate
        stt = turnover * (rates.stt_buy_rate if side == "BUY" else rates.stt_sell_rate)
        stamp_duty = turnover * rates.stamp_duty_buy_rate if side == "BUY" else Decimal("0")
        taxable_components = {
            "brokerage": brokerage,
            "exchange_charges": exchange_charges,
            "sebi_charges": sebi_charges,
            "other_charges": other_charges,
        }
        gst = sum((taxable_components[name] for name in rates.gst_base_components), Decimal("0")) * rates.gst_rate
        return sum((brokerage, stt, exchange_charges, sebi_charges, other_charges, gst, stamp_duty), Decimal("0"))

    def round_trip_cost(
        self,
        *,
        product: IndiaProductType,
        buy_turnover: Decimal,
        sell_turnover: Decimal,
        slippage_bps_per_side: Decimal = Decimal("0"),
        market_impact_bps_per_side: Decimal = Decimal("0"),
    ) -> RoundTripCost:
        _validate_amount("buy turnover", buy_turnover)
        _validate_amount("sell turnover", sell_turnover)
        _validate_amount("slippage basis points", slippage_bps_per_side, allow_zero=True)
        _validate_amount("market impact basis points", market_impact_bps_per_side, allow_zero=True)
        rates = self.schedule.for_product(product)
        cap = rates.brokerage_cap_per_order

        def brokerage_for(turnover: Decimal) -> Decimal:
            charge = turnover * rates.brokerage_rate
            return min(charge, cap) if cap is not None else charge

        turnover = buy_turnover + sell_turnover
        brokerage = brokerage_for(buy_turnover) + brokerage_for(sell_turnover)
        stt = buy_turnover * rates.stt_buy_rate + sell_turnover * rates.stt_sell_rate
        exchange_charges = turnover * rates.exchange_transaction_rate
        sebi_charges = turnover * rates.sebi_turnover_rate
        other_charges = turnover * rates.other_turnover_rate
        gst_base = {
            "brokerage": brokerage,
            "exchange_charges": exchange_charges,
            "sebi_charges": sebi_charges,
            "other_charges": other_charges,
        }
        gst = sum((gst_base[name] for name in rates.gst_base_components), Decimal("0")) * rates.gst_rate
        stamp_duty = buy_turnover * rates.stamp_duty_buy_rate
        slippage = turnover * slippage_bps_per_side / Decimal("10000")
        market_impact = turnover * market_impact_bps_per_side / Decimal("10000")
        total_cost = sum((
            brokerage, stt, exchange_charges, sebi_charges, other_charges, gst,
            stamp_duty, slippage, market_impact,
        ), Decimal("0"))
        return RoundTripCost(
            product, buy_turnover, sell_turnover, brokerage, stt,
            exchange_charges, sebi_charges, other_charges, gst, stamp_duty,
            slippage, market_impact, total_cost,
        )

    @staticmethod
    def expected_value(
        *,
        win_probability: Decimal,
        average_win: Decimal,
        average_loss: Decimal,
        round_trip_cost: RoundTripCost,
    ) -> ExpectedValueResult:
        if (
            not isinstance(win_probability, Decimal)
            or not win_probability.is_finite()
            or not Decimal("0") <= win_probability <= Decimal("1")
        ):
            raise ValueError("Win probability must be between zero and one")
        if not isinstance(round_trip_cost, RoundTripCost) or not round_trip_cost.total_cost.is_finite():
            raise ValueError("Round-trip cost must be finite")
        _validate_amount("average win", average_win, allow_zero=True)
        _validate_amount("average loss", average_loss, allow_zero=True)
        expected_gross = win_probability * average_win - (Decimal("1") - win_probability) * average_loss
        expected_net = expected_gross - round_trip_cost.total_cost
        reward_ratio = average_win / average_loss if average_loss > 0 else None
        return ExpectedValueResult(
            win_probability=win_probability,
            average_win=average_win,
            average_loss=average_loss,
            expected_gross_value=expected_gross,
            transaction_cost=round_trip_cost.total_cost,
            expected_net_value=expected_net,
            risk_reward_ratio=reward_ratio,
            edge_positive=expected_net > 0,
        )


def _validate_rate(name: str, value: Decimal) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0 or value > 1:
        raise ValueError(f"{name} must be an explicit finite decimal rate between zero and one")


def _validate_amount(name: str, value: Decimal, *, allow_zero: bool = False) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} decimal")
    minimum_ok = value >= 0 if allow_zero else value > 0
    if not minimum_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} decimal")