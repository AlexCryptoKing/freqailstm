import numpy as np
import pandas_ta as pta
from typing import Optional
import math
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
from functools import reduce
from typing import Dict
from datetime import timedelta, datetime, timezone
from freqtrade.persistence import Trade
from pandas import DataFrame, Series
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy,BooleanParameter
from technical.pivots_points import pivots_points

from scipy.signal import argrelextrema


import logging
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange.exchange_utils import *
from freqtrade.strategy import IStrategy, RealParameter
from technical.pivots_points import pivots_points

logger = logging.getLogger(__name__)

"""

ifconfig
sudo ifconfig wlan1 down
source ./.venv/bin/activate



freqtrade download-data -c config-torch.json --timerange 20240101-20240909 --timeframes 5m
freqtrade download-data -c config-torch.json --timerange 20220101-20240909 --timeframes 1h 2h 4h
freqtrade backtesting --cache none -c config-torch.json --breakdown day week month  --timeframe-detail 5m --timerange 20240201-20240301

freqtrade trade --strategy AlexStrategyFinalV85Hyper --config config-torch.json --freqaimodel PyTorchLSTMRegressor
freqtrade backtesting --breakdown day week --cache none -c config-torch.json --timerange 20240209-20240220


freqtrade backtesting --strategy AlexStrategyFinalV85 -c config-torch.json --freqaimodel PyTorchLSTMRegressor --timerange 20240801-20240820
freqtrade backtesting --strategy AlexStrategyFinalV85 -c config-torch.json --freqaimodel XGBoostRegressor --breakdown day week month   --timerange 20240801-20240810

freqtrade lookahead-analysis --strategy AlexStrategyFinalV85 -c config-torch.json --timerange 20240101-20240820

CHANGELOG:
V83 - Corrected Market Remige Filter R, and Dynamic Weights Adjustment
V85 - added Trailing Dynamic, Leverage Dynamic and Stop Loss


"""


class AlexStrategyFinalV9(IStrategy):
    """
    This is an example strategy that uses the LSTMRegressor model to predict the target score.
    Use at your own risk.
    This is a simple example strategy and should be used for educational purposes only.
    """
    # Hyperspace parameters:
    # Buy hyperspace params:
    # Hyperspace parameters:
    buy_params = {
        "threshold_buy": 0.59453,
        "w0": 0.54347,
        "w1": 0.82226,
        "w2": 0.56675,
        "w3": 0.77918,
        "w4": 0.98488,
        "w5": 0.31368,
        "w6": 0.75916,
        "w7": 0.09226,
        "w8": 0.85667,
    }

    sell_params = {
        "threshold_sell": 0.80573,
    
    }

    # ROI table:
    minimal_roi = {
         "0": 0.239,
      "79": 0.058,
      "231": 0.029,
      "543": 0
    }

    # Stoploss:
    stoploss = -0.305  # Were letting the model decide when to sell

    # Trailing stop:
    trailing_stop = False
    trailing_only_offset_is_reached = False

    # Define ATR-based dynamic trailing stop and offset
    min_trailing_stop = DecimalParameter(0.05, 0.25, default=0.10, space="buy")
    max_trailing_stop = DecimalParameter(0.10, 0.50, default=0.30, space="buy")
    min_trailing_offset = DecimalParameter(0.05, 0.30, default=0.10, space="buy")
    max_trailing_offset = DecimalParameter(0.10, 0.50, default=0.30, space="buy")
    
    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    startup_candle_count = 100
    leverage_value = 10.0


    threshold_buy = RealParameter(-1, 1, default=0, space='buy')
    threshold_sell = RealParameter(-1, 1, default=0, space='sell')

    # Weights for calculating the aggregate score - normalized to sum to 1
    w0 = RealParameter(0, 1, default=0.10, space='buy')  # moving average (normalized_ma)
    w1 = RealParameter(0, 1, default=0.15, space='buy')  # MACD (normalized_macd)
    w2 = RealParameter(0, 1, default=0.10, space='buy')  # Rate of Change (ROC)
    w3 = RealParameter(0, 1, default=0.15, space='buy')  # RSI (normalized_rsi)
    w4 = RealParameter(0, 1, default=0.10, space='buy')  # Bollinger Band width
    w5 = RealParameter(0, 1, default=0.10, space='buy')  # CCI (normalized_cci)
    w6 = RealParameter(0, 1, default=0.10, space='buy')  # OBV (normalized_obv)
    w7 = RealParameter(0, 1, default=0.05, space='buy')  # ATR (normalized_atr)
    w8 = RealParameter(0, 1, default=0.10, space='buy')  # Stochastic Oscillator (normalized_stoch)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):
# #----------------------------------
#         dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
#         dataframe["%-adx-period"] = ta.ADX(dataframe, window=period)
#         dataframe["%-er-period"] = pta.er(dataframe['close'], length=period)
#         dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
#         dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
#         dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
#         dataframe["%-cti-period"] = pta.cti(dataframe['close'], length=period)
#         dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
#         dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(dataframe['close'], timeperiod=period)
#         dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
#         dataframe["%-atr-periodp"] = dataframe["%-atr-period"] / dataframe['close'] * 1000
# #----------------------------------

        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(
            dataframe['close'], slowperiod=12,
            fastperiod=26)
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=2)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], slowperiod=12,
                                                                                    fastperiod=26)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)

        # Step 1: Normalize Indicators:
        # Why? Normalizing the indicators will make them comparable and allow us to assign weights to them.
        # How? We will calculate the z-score of each indicator by subtracting the rolling mean and dividing by the
        # rolling standard deviation. This will give us a normalized value that is centered around 0 with a standard
        # deviation of 1.
        dataframe['normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe['stoch'].rolling(window=14).std()
        dataframe['normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe['atr'].rolling(window=14).std()
        dataframe['normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe['obv'].rolling(window=14).std()
        dataframe['normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe['close'].rolling(window=10).std()
        dataframe['normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe['macd'].rolling(window=26).std()
        dataframe['normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe['roc'].rolling(window=2).std()
        dataframe['normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / \
                                           dataframe['momentum'].rolling(window=4).std()
        dataframe['normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe['rsi'].rolling(window=10).std()
        dataframe['normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(
            window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
        dataframe['normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe['cci'].rolling(window=20).std()

        # Dynamic Weights Adjustment
        # Calculate trend strength as the absolute difference between MA and close price
        trend_strength = abs(dataframe['ma'] - dataframe['close'])
        # Calculate rolling mean and stddev once to avoid redundancy
        rolling_mean = trend_strength.rolling(window=14).mean()
        rolling_stddev = trend_strength.rolling(window=14).std()
        # Calculate a more dynamic strong trend threshold
        strong_trend_threshold = rolling_mean + 1.5 * rolling_stddev
        # Determine strong trend condition
        is_strong_trend = trend_strength > strong_trend_threshold
        # Apply dynamic weight adjustment, could also consider a more gradual adjustment
        dataframe['w_momentum'] = self.w3.value * (1 + 0.5 * (trend_strength / strong_trend_threshold))
        # Optional: Clip the w_momentum values to prevent extreme cases
        dataframe['w_momentum'] = dataframe['w_momentum'].clip(lower=self.w3.value, upper=self.w3.value * 2)

        # Step 2: Calculate aggregate score S
        # Calculate aggregate score S
        w = [self.w0.value, self.w1.value, self.w2.value, self.w3.value, self.w4.value, self.w5.value, self.w6.value, self.w7.value, self.w8.value]
      
        dataframe['S'] = w[0] * dataframe['normalized_ma'] + \
                         w[1] * dataframe['normalized_macd'] + \
                         w[2] * dataframe['normalized_roc'] + \
                         w[3] * dataframe['normalized_rsi'] + \
                         w[4] * dataframe['normalized_bb_width'] + \
                         w[5] * dataframe['normalized_cci'] + \
                         dataframe['w_momentum'] * dataframe['normalized_momentum'] + \
                         self.w8.value * dataframe['normalized_stoch'] + \
                         self.w7.value * dataframe['normalized_atr'] + \
                         self.w6.value * dataframe['normalized_obv']

        # Step 3: Market Regime Filter R

        dataframe['R'] = 0
        dataframe.loc[dataframe['close'] > dataframe['bb_upperband'], 'R'] = 1
        dataframe.loc[dataframe['close'] < dataframe['bb_lowerband'], 'R'] = -1
        buffer_pct = 0.01  # 1% buffer
        
        dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'] * (1 + buffer_pct), 1, 
                                    np.where(dataframe['close'] < dataframe['ma_100'] * (1 - buffer_pct), -1, np.nan))

        # Step 4: Volatility Adjustment V
        # EXPLANATION: Calculate the Bollinger Band width and assign it to V. The Bollinger Band width is the
        # difference between the upper and lower Bollinger Bands divided by the middle Bollinger Band. The idea is
        # that when the Bollinger Bands are wide, the market is volatile, and when the Bollinger Bands are narrow,
        # the market is less volatile. So we are using the Bollinger Band width as a measure of volatility. You can
        # use other indicators to measure volatility as well. For example, you can use the ATR (Average True Range)

        bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['V_mean'] = 1 / (bb_width + 1e-8)  # Avoid division by zero

        # ATR-based Volatility Measure
        dataframe['V2_mean'] = 1 / (dataframe['atr'] + 1e-8)  # Avoid division by zero

        # Rolling window size for adaptive normalization
        rolling_window = 50

        # Normalize V_mean using a rolling window
        mean_v = dataframe['V_mean'].rolling(window=rolling_window).mean()
        std_v = dataframe['V_mean'].rolling(window=rolling_window).std()
        dataframe['V_norm'] = (dataframe['V_mean'] - mean_v) / std_v
        dataframe['V_norm'] = dataframe['V_norm'].fillna(0) 

        # Normalize V2_mean using a rolling window
        mean_v2 = dataframe['V2_mean'].rolling(window=rolling_window).mean()
        std_v2 = dataframe['V2_mean'].rolling(window=rolling_window).std()
        dataframe['V2_norm'] = (dataframe['V2_mean'] - mean_v2) / std_v2
        dataframe['V2_norm'] = dataframe['V2_norm'].fillna(0)

        # Signal assignment using hysteresis
        upper_threshold = 1.0
        lower_threshold = -1.0

        dataframe['V'] = np.where(dataframe['V_norm'] > upper_threshold, 1,
                          np.where(dataframe['V_norm'] < lower_threshold, -1, np.nan))
        dataframe['V2'] = np.where(dataframe['V2_norm'] > upper_threshold, 1,
                           np.where(dataframe['V2_norm'] < lower_threshold, -1, np.nan))

        # Forward-fill to maintain the last state of the signal
        dataframe['V'] = dataframe['V'].ffill()  # Correct ffill usage
        dataframe['V2'] = dataframe['V2'].ffill()  # Correct ffill usage



        # Get Final Target Score to incorporate new calculations
        dataframe['T'] = dataframe['S'] * dataframe['R'] * dataframe['R2'] * dataframe['V'] * dataframe['V2']

        # Assign the target score T to the AI target column
        target_horizon = 1  # Define your prediction horizon here
        dataframe['&-target'] = dataframe['T'].shift(-target_horizon)
        return dataframe
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        dataframe = self.freqai.start(dataframe, metadata, self)
        # One can define indicators here if needed and add logic to populate_entry_trend and populate_exit_trend

        #print(f'>>>>>>>>>>>>>>>>>>>>> {dataframe.columns.values}')
        # dataframe["target_roi"] = dataframe["&-target_mean"] + dataframe["&-target_std"] * 1.25
        # dataframe["sell_roi"] = dataframe["&-target_mean"] - dataframe["&-target_std"] * 1.25

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,
            df['&-target'] > self.threshold_buy.value,  #
            df['volume'] > 0
        ]

        enter_short_conditions = [
            df["do_predict"] == 1,
            df['&-target'] < self.threshold_sell.value,
            df["volume"] > 0

        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "long")

        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
            df["do_predict"] == 1,

            df['&-target'] < self.threshold_sell.value
        ]

        exit_short_conditions = [
            df["do_predict"] == 1,
            df['&-target'] > self.threshold_buy.value
        ]

        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]
            ] = (1, "exit_long")

        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), ["exit_short", "exit_tag"]
            ] = (1, "exit_short")

        return df
    
    def trailing_stop_logic(self, dataframe: DataFrame, row: int) -> (float, float):
        """
        Calculate the trailing stop and offset dynamically based on ATR.
        """
        atr = dataframe["atr"].iloc[row]

        # Normalize ATR between min and max values for trailing stop and offset
        atr_norm = (atr - dataframe["atr"].min()) / (dataframe["atr"].max() - dataframe["atr"].min())

        # Dynamically calculate trailing stop and offset based on ATR
        trailing_stop = self.min_trailing_stop.value + atr_norm * (self.max_trailing_stop.value - self.min_trailing_stop.value)
        trailing_offset = self.min_trailing_offset.value + atr_norm * (self.max_trailing_offset.value - self.min_trailing_offset.value)

        return trailing_stop, trailing_offset

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Calculate custom stoploss based on dynamic trailing stop logic.
        """
        dataframe, row = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # If no valid dataframe is available (initial trades)
        if dataframe is None or row is None:
            return self.stoploss

        # Get trailing stop and offset based on ATR
        trailing_stop, trailing_offset = self.trailing_stop_logic(dataframe, row)

        # If the price has increased above the trailing offset, adjust stoploss
        if current_profit > trailing_offset:
            return -trailing_stop

        return self.stoploss

    def leverage(
        self,
        pair: str,
        current_time: "datetime",
        current_rate: float,
        proposed_leverage: float,
        **kwargs,
    ) -> float:
        """
        Dynamically adjust leverage based on volatility (ATR).
        """
        # Get the analyzed dataframe for the pair and timeframe
        dataframe = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    
        if dataframe is None or dataframe.empty:
            return self.leverage_value  # Fallback to default leverage

        # Get the latest ATR value from the last candle
        atr = dataframe['atr'].iat[-1]  # Get the ATR of the most recent candle
    
        if pd.isna(atr):
            return self.leverage_value  # Fallback if ATR is NaN

        # Example: Lower leverage when volatility (ATR) is high
        rolling_mean = dataframe['atr'].rolling(window=14).mean().iat[-1]
        rolling_stddev = dataframe['atr'].rolling(window=14).std().iat[-1]
    
        if atr > rolling_mean + 1.5 * rolling_stddev:
            return self.leverage_value * 0.5  # Reduce leverage in volatile markets

        return self.leverage_value


# # #--------------------------------------
#     overbuy_factor = 1.295 #2

#     position_adjustment_enable = True
#     initial_safety_order_trigger = -0.02
#     max_so_multiplier_orig = 3
#     safety_order_step_scale = 2
#     safety_order_volume_scale = 1.8

#     # just for initialization, now we calculate it...
#     max_so_multiplier = max_so_multiplier_orig
#     # We will store the size of stake of each trade's first order here
#     cust_proposed_initial_stakes = {}
#     # Amount the strategy should compensate previously partially filled orders for successive safety orders (0.0 - 1.0)
#     partial_fill_compensation_scale = 1

#     if (max_so_multiplier_orig > 0):
#         if (safety_order_volume_scale > 1):
#             # print(safety_order_volume_scale * (math.pow(safety_order_volume_scale,(max_so_multiplier - 1)) - 1))

#             firstLine = (safety_order_volume_scale *
#                          (math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)) - 1))
#             divisor = (safety_order_volume_scale - 1)
#             max_so_multiplier = (2 + firstLine / divisor)
#             # max_so_multiplier = (2 +
#             #                     (safety_order_volume_scale *
#             #                      (math.pow(safety_order_volume_scale, (max_so_multiplier - 1)) - 1) /
#             #                      (safety_order_volume_scale - 1)))
#         elif (safety_order_volume_scale < 1):
#             firstLine = safety_order_volume_scale * \
#                         (1 - math.pow(safety_order_volume_scale, (max_so_multiplier_orig - 1)))
#             divisor = 1 - safety_order_volume_scale
#             max_so_multiplier = (2 + firstLine / divisor)
#             # max_so_multiplier = (2 + (safety_order_volume_scale * (
#             #        1 - math.pow(safety_order_volume_scale, (max_so_multiplier - 1))) / (
#             #                                  1 - safety_order_volume_scale)))

#     # Since stoploss can only go up and can't go down, if you set your stoploss here, your lowest stoploss will always be tied to the first buy rate
#     # So disable the hard stoploss here, and use custom_sell or custom_stoploss to handle the stoploss trigger
#     stoploss = -1


#     def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
#                     current_profit: float, **kwargs):

#         tag = super().custom_sell(pair, trade, current_time, current_rate, current_profit, **kwargs)
#         if tag:
#             return tag

#         entry_tag = 'empty'
#         if hasattr(trade, 'entry_tag') and trade.entry_tag is not None:
#             entry_tag = trade.entry_tag

#         if current_profit <= -0.35:
#             return f'stop_loss ({entry_tag})'

#         return None

#     def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
#                            rate: float, time_in_force: str, exit_reason: str,
#                            current_time: datetime, **kwargs) -> bool:
#         # remove pair from custom initial stake dict only if full exit
#         if trade.amount == amount and pair in self.cust_proposed_initial_stakes:
#             del self.cust_proposed_initial_stakes[pair]
#         return True

#     # Let unlimited stakes leave funds open for DCA orders
#     def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
#                             proposed_stake: float, min_stake: float, max_stake: float,
#                             **kwargs) -> float:
#         custom_stake = proposed_stake / self.max_so_multiplier * self.overbuy_factor
#         self.cust_proposed_initial_stakes[pair] = custom_stake  # Setting of first stake size just before each first order of a trade
#         return custom_stake # set to static 10 to simulate partial fills of 10$, etc

#     def adjust_trade_position(self, trade: Trade, current_time: datetime,
#                               current_rate: float, current_profit: float, min_stake: float,
#                               max_stake: float, **kwargs) -> Optional[float]:
#         if current_profit > self.initial_safety_order_trigger:
#             return None

#         filled_buys = trade.select_filled_orders(trade.entry_side)
#         count_of_buys = len(filled_buys)

#         if 1 <= count_of_buys <= self.max_so_multiplier_orig:
#             # if (1 <= count_of_buys) and (open_trade_value < self.stake_amount * self.overbuy_factor):
#             safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
#             if self.safety_order_step_scale > 1:
#                 safety_order_trigger = abs(self.initial_safety_order_trigger) + (
#                         abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
#                         math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / (self.safety_order_step_scale - 1))
#             elif self.safety_order_step_scale < 1:
#                 safety_order_trigger = abs(self.initial_safety_order_trigger) + (
#                         abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
#                         1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) / (1 - self.safety_order_step_scale))

#             if current_profit <= (-1 * abs(safety_order_trigger)):
#                 try:
#                     # This returns first order actual stake size
#                     actual_initial_stake = filled_buys[0].cost

#                     # Fallback for when the initial stake was not set for whatever reason
#                     stake_amount = actual_initial_stake

#                     already_bought = sum(filled_buy.cost for filled_buy in filled_buys)
#                     if trade.pair in self.cust_proposed_initial_stakes:
#                         if self.cust_proposed_initial_stakes[trade.pair] > 0:
#                             # This calculates the amount of stake that will get used for the current safety order,
#                             # including compensation for any partial buys
#                             proposed_initial_stake = self.cust_proposed_initial_stakes[trade.pair]
#                             current_actual_stake = already_bought * math.pow(self.safety_order_volume_scale,
#                                                                              (count_of_buys - 1))
#                             current_stake_preposition = proposed_initial_stake * math.pow(self.safety_order_volume_scale,
#                                                                                           (count_of_buys - 1))
#                             current_stake_preposition_compensation = current_stake_preposition + abs(
#                                 current_stake_preposition - current_actual_stake)
#                             total_so_stake = lerp(current_actual_stake, current_stake_preposition_compensation,
#                                                   self.partial_fill_compensation_scale)
#                             # Set the calculated stake amount
#                             stake_amount = total_so_stake
#                         else:
#                             # Fallback stake amount calculation
#                             stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
#                     else:
#                         # Fallback stake amount calculation
#                         stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))

#                     # amount = stake_amount / current_rate
#                     # logger.info(
#                     #     f"Initiating safety order buy #{count_of_buys} "
#                     #     f"for {trade.pair} with stake amount of {stake_amount}. "
#                     #     f"which equals {amount}. "
#                     #     f"Previously bought: {already_bought}. "
#                     #     f"Now overall:{already_bought + stake_amount}. ")
#                     return stake_amount
#                 except Exception as exception:
#                     # logger.info(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
#                     # print(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
#                     return None

#         return None

# def lerp(a: float, b: float, t: float) -> float:
#     """Linear interpolate on the scale given by a to b, using t as the point on that scale.
#     Examples
#     --------
#         50 == lerp(0, 100, 0.5)
#         4.2 == lerp(1, 5, 0.8)
#     """
#     return (1 - t) * a + t * b

# #--------------------------------------



def chaikin_mf(df, periods=20):
    close = df["close"]
    low = df["low"]
    high = df["high"]
    volume = df["volume"]
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name="cmf")

def top_percent_change(dataframe: DataFrame, length: int) -> float:
    if length == 0:
        return (dataframe["open"] - dataframe["close"]) / dataframe["close"]
    else:
        return (dataframe["open"].rolling(length).max() - dataframe["close"]) / dataframe["close"]
