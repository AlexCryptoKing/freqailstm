"""Gate.io exchange subclass"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, PriceType, TradingMode
from freqtrade.exchange import Exchange
from freqtrade.misc import safe_value_fallback2


logger = logging.getLogger(__name__)


class Gate(Exchange):
    """
    Gate.io exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: Dict = {
        "ohlcv_candle_limit": 1000,
        "order_time_in_force": ["GTC", "IOC"],
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit"},
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "marketOrderRequiresPrice": True,
        "trades_has_history": False,  # Endpoint would support this - but ccxt doesn't.
    }

    _ft_has_futures: Dict = {
        "needs_trading_fees": True,
        "marketOrderRequiresPrice": False,
        "stop_price_type_field": "price_type",
        "stop_price_type_value_mapping": {
            PriceType.LAST: 0,
            PriceType.MARK: 1,
            PriceType.INDEX: 2,
        },
    }

    _supported_trading_mode_margin_pairs: List[Tuple[TradingMode, MarginMode]] = [
        # TradingMode.SPOT always supported and not required in this list
        # (TradingMode.MARGIN, MarginMode.CROSS),
        # (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED)
    ]

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> Dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if ordertype == "market" and self.trading_mode == TradingMode.FUTURES:
            params["type"] = "market"
            params.update({"timeInForce": "IOC"})
        return params

    def get_trades_for_order(
        self, order_id: str, pair: str, since: datetime, params: Optional[Dict] = None
    ) -> List:
        trades = super().get_trades_for_order(order_id, pair, since, params)

        if self.trading_mode == TradingMode.FUTURES:
            # Futures usually don't contain fees in the response.
            # As such, futures orders on gate will not contain a fee, which causes
            # a repeated "update fee" cycle and wrong calculations.
            # Therefore we patch the response with fees if it's not available.
            # An alternative also containing fees would be
            # privateFuturesGetSettleAccountBook({"settle": "usdt"})
            pair_fees = self._trading_fees.get(pair, {})
            if pair_fees:
                for idx, trade in enumerate(trades):
                    fee = trade.get("fee", {})
                    if fee and fee.get("cost") is None:
                        takerOrMaker = trade.get("takerOrMaker", "taker")
                        if pair_fees.get(takerOrMaker) is not None:
                            trades[idx]["fee"] = {
                                "currency": self.get_pair_quote_currency(pair),
                                "cost": trade["cost"] * pair_fees[takerOrMaker],
                                "rate": pair_fees[takerOrMaker],
                            }
        return trades

    def get_order_id_conditional(self, order: Dict[str, Any]) -> str:
        return safe_value_fallback2(order, order, "id_stop", "id")

    def fetch_stoploss_order(self, order_id: str, pair: str, params: Optional[Dict] = None) -> Dict:
        order = self.fetch_order(order_id=order_id, pair=pair, params={"stop": True})
        if order.get("status", "open") == "closed":
            # Places a real order - which we need to fetch explicitly.
            val = "trade_id" if self.trading_mode == TradingMode.FUTURES else "fired_order_id"

            if new_orderid := order.get("info", {}).get(val):
                order1 = self.fetch_order(order_id=new_orderid, pair=pair, params=params)
                order1["id_stop"] = order1["id"]
                order1["id"] = order_id
                order1["type"] = "stoploss"
                order1["stopPrice"] = order.get("stopPrice")
                order1["status_stop"] = "triggered"

                return order1
        return order

    def cancel_stoploss_order(
        self, order_id: str, pair: str, params: Optional[Dict] = None
    ) -> Dict:
        return self.cancel_order(order_id=order_id, pair=pair, params={"stop": True})
