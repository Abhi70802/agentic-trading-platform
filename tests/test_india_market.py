import unittest
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
import tempfile

from india_market import (
    AssetType,
    Contract,
    Exchange,
    IndiaMarketPolicy,
    Instrument,
    LotSize,
    OptionType,
    TickSize,
    TradingSession,
)
from store import EventStore


class IndiaMarketPolicyTests(unittest.TestCase):
    def test_empty_policy_has_no_implicit_market_rules(self):
        policy = IndiaMarketPolicy()
        self.assertEqual(policy.exchanges, ())
        self.assertEqual(policy.sessions, ())
        with self.assertRaisesRegex(ValueError, "No Indian exchange reference"):
            policy.exchange("NSE")

    def test_reference_data_validates_exchange_instrument_and_contract(self):
        exchange = Exchange("NSE-TEST", "Synthetic NSE fixture", "IN", "Asia/Kolkata")
        underlying = Instrument("NSE-TEST", "INDEX-TEST", AssetType.INDEX, "Synthetic index")
        option = Instrument("NSE-TEST", "OPTION-TEST", AssetType.OPTION, "Synthetic option")
        contract = Contract(
            instrument=option,
            underlying=underlying,
            expiry=date(2027, 1, 1),
            lot_size=LotSize(7, date(2026, 1, 1)),
            tick_size=TickSize(Decimal("0.05"), date(2026, 1, 1)),
            strike=Decimal("100.00"),
            option_type=OptionType.CALL,
            product_type="SYNTHETIC-TEST",
        )
        policy = IndiaMarketPolicy(
            exchanges=(exchange,),
            instruments=(underlying,),
            contracts=(contract,),
            sessions=(TradingSession(
                exchange_code="NSE-TEST",
                segment="SYNTHETIC-TEST",
                opens_at=time(9, 0),
                closes_at=time(16, 0),
                weekdays=frozenset({0, 1, 2, 3, 4}),
            ),),
        )

        self.assertEqual(policy.exchange("nse-test"), exchange)
        self.assertEqual(policy.instrument("nse-test", "index-test"), underlying)
        self.assertTrue(contract.lot_size.accepts(14))
        self.assertFalse(contract.lot_size.accepts(15))
        self.assertTrue(contract.tick_size.accepts(Decimal("100.15")))
        self.assertFalse(contract.tick_size.accepts(Decimal("100.13")))

    def test_futures_calls_and_puts_preserve_expiry_strike_and_lot_metadata(self):
        exchange = Exchange("NSE-TEST", "Synthetic NSE fixture", "IN", "Asia/Kolkata")
        underlying = Instrument("NSE-TEST", "INDEX-TEST", AssetType.INDEX, "Synthetic index")
        future_instrument = Instrument("NSE-TEST", "FUT-TEST", AssetType.FUTURE, "Synthetic future")
        call_instrument = Instrument("NSE-TEST", "CALL-TEST", AssetType.OPTION, "Synthetic call")
        put_instrument = Instrument("NSE-TEST", "PUT-TEST", AssetType.OPTION, "Synthetic put")
        effective_from = date(2026, 1, 1)
        future = Contract(
            future_instrument, underlying, date(2026, 10, 29), LotSize(25, effective_from),
            TickSize(Decimal("0.05"), effective_from),
        )
        call = Contract(
            call_instrument, underlying, date(2026, 10, 1), LotSize(25, effective_from),
            TickSize(Decimal("0.05"), effective_from), Decimal("22000"), OptionType.CALL,
        )
        put = Contract(
            put_instrument, underlying, date(2026, 10, 8), LotSize(50, effective_from),
            TickSize(Decimal("0.05"), effective_from), Decimal("22100"), OptionType.PUT,
        )

        policy = IndiaMarketPolicy(
            exchanges=(exchange,),
            instruments=(underlying,),
            contracts=(future, call, put),
        )

        self.assertEqual(policy.contracts, (future, call, put))
        self.assertIsNone(future.strike)
        self.assertIsNone(future.option_type)
        self.assertEqual(call.option_type, OptionType.CALL)
        self.assertEqual(put.option_type, OptionType.PUT)
        self.assertNotEqual(call.expiry, put.expiry)
        self.assertNotEqual(call.strike, put.strike)
        self.assertNotEqual(call.lot_size.quantity, put.lot_size.quantity)
        self.assertTrue(put.lot_size.accepts(100))
        self.assertFalse(put.lot_size.accepts(75))

    def test_invalid_contract_metadata_is_rejected_before_policy_registration(self):
        underlying = Instrument("NSE-TEST", "INDEX-TEST", AssetType.INDEX, "Synthetic index")
        option = Instrument("NSE-TEST", "OPTION-TEST", AssetType.OPTION, "Synthetic option")
        effective_from = date(2026, 1, 1)

        def make_contract(**overrides):
            values = {
                "instrument": option,
                "underlying": underlying,
                "expiry": date(2026, 10, 1),
                "lot_size": LotSize(25, effective_from),
                "tick_size": TickSize(Decimal("0.05"), effective_from),
                "strike": Decimal("22000"),
                "option_type": OptionType.CALL,
            }
            values.update(overrides)
            return Contract(**values)

        invalid_metadata = (
            ("boolean lot size", lambda: LotSize(True, effective_from), "Lot size"),
            ("fractional lot size", lambda: LotSize(25.5, effective_from), "Lot size"),
            ("string expiry", lambda: make_contract(expiry="2026-10-01"), "expiry"),
            (
                "datetime expiry",
                lambda: make_contract(expiry=datetime(2026, 10, 1)),
                "expiry",
            ),
            ("float strike", lambda: make_contract(strike=22000.0), "strike"),
            ("string option side", lambda: make_contract(option_type="CALL"), "call or put"),
        )
        for name, create_invalid, message in invalid_metadata:
            with self.subTest(metadata=name):
                with self.assertRaisesRegex(ValueError, message):
                    create_invalid()

        lot_size = LotSize(25, effective_from)
        self.assertFalse(lot_size.accepts(True))
        self.assertFalse(lot_size.accepts(25.0))

    def test_policy_rejects_references_without_configured_exchange(self):
        with self.assertRaisesRegex(ValueError, "Exchange reference is missing"):
            IndiaMarketPolicy(instruments=(
                Instrument("NSE-TEST", "INDEX-TEST", AssetType.INDEX, "Synthetic index"),
            ))

    def test_option_contract_requires_strike_and_option_type(self):
        underlying = Instrument("NSE-TEST", "INDEX-TEST", AssetType.INDEX, "Synthetic index")
        option = Instrument("NSE-TEST", "OPTION-TEST", AssetType.OPTION, "Synthetic option")
        with self.assertRaisesRegex(ValueError, "positive strike"):
            Contract(
                instrument=option,
                underlying=underlying,
                expiry=date(2027, 1, 1),
                lot_size=LotSize(1, date(2026, 1, 1)),
                tick_size=TickSize(Decimal("0.01"), date(2026, 1, 1)),
                product_type="SYNTHETIC-TEST",
            )

    def test_reference_snapshot_hash_is_stable_and_content_addressed(self):
        exchange = Exchange("NSE-TEST", "Synthetic NSE fixture", "IN", "Asia/Kolkata")
        empty_policy = IndiaMarketPolicy()
        populated_policy = IndiaMarketPolicy(exchanges=(exchange,))
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "reference.db")
            first = store.save_india_reference_snapshot(
                provider="fixture",
                fetched_at="2026-10-01T00:00:00+00:00",
                policy=empty_policy,
            )
            repeated = store.save_india_reference_snapshot(
                provider="fixture",
                fetched_at="2026-10-01T00:01:00+00:00",
                policy=empty_policy,
            )
            changed = store.save_india_reference_snapshot(
                provider="fixture",
                fetched_at="2026-10-01T00:02:00+00:00",
                policy=populated_policy,
            )
            loaded = store.read_india_reference_snapshot("fixture", changed["snapshot_hash"])

        self.assertEqual(first["snapshot_hash"], repeated["snapshot_hash"])
        self.assertNotEqual(first["snapshot_hash"], changed["snapshot_hash"])
        self.assertEqual(loaded["payload"]["exchanges"][0]["code"], "NSE-TEST")


if __name__ == "__main__":
    unittest.main()