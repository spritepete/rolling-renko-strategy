import talib.abstract as ta
from pandas import DataFrame, Series, DatetimeIndex, merge
import pandas as pd
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
pd.set_option("display.precision", 10)

class RollingRenko(IStrategy):
    """
    Rolling Renko Strategy
    Uses a percentage-based step size calculated from initial window
    Tracks high/low levels based on price movement reaching step size
    """
    
    minimal_roi = {
        "0": 100
    }

    stoploss = -100
    timeframe = '15m'
    
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.1
    ignore_roi_if_buy_signal = True

    # Strategy parameters
    window_size = 100  # Initial window for calculating average price
    step_percentage = 0.02  # 2% step size

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate initial average price from first window_size values
        initial_avg_price = dataframe['close'].iloc[:self.window_size].mean()
        brick_size = initial_avg_price * self.step_percentage

        # Initialize columns for the rolling Renko
        dataframe['brick_size'] = brick_size
        dataframe['level'] = np.nan
        dataframe['trend'] = np.nan
        dataframe['signal'] = 0

        # Set initial level
        current_level = dataframe['close'].iloc[0]
        current_trend = None
        last_signal_level = current_level

        for i in range(len(dataframe)):
            close = dataframe['close'].iloc[i]
            
            if current_trend is None:
                # Initialize trend
                current_trend = 1 if close >= current_level else -1
                dataframe.loc[dataframe.index[i], 'trend'] = current_trend
                dataframe.loc[dataframe.index[i], 'level'] = current_level
                continue

            if current_trend == 1:
                # In uptrend
                if close >= current_level:
                    # New high, check if step is reached
                    steps_up = int((close - current_level) / brick_size)
                    if steps_up > 0:
                        # Move level up by completed steps
                        current_level += steps_up * brick_size
                        # Generate buy signal if we moved up
                        if current_level > last_signal_level:
                            dataframe.loc[dataframe.index[i], 'signal'] = 1
                            last_signal_level = current_level
                elif close <= (current_level - brick_size):
                    # Trend reversal
                    current_trend = -1
                    current_level -= brick_size
                    dataframe.loc[dataframe.index[i], 'signal'] = -1
                    last_signal_level = current_level

            else:
                # In downtrend
                if close <= current_level:
                    # New low, check if step is reached
                    steps_down = int((current_level - close) / brick_size)
                    if steps_down > 0:
                        # Move level down by completed steps
                        current_level -= steps_down * brick_size
                        # Generate sell signal if we moved down
                        if current_level < last_signal_level:
                            dataframe.loc[dataframe.index[i], 'signal'] = -1
                            last_signal_level = current_level
                elif close >= (current_level + brick_size):
                    # Trend reversal
                    current_trend = 1
                    current_level += brick_size
                    dataframe.loc[dataframe.index[i], 'signal'] = 1
                    last_signal_level = current_level

            dataframe.loc[dataframe.index[i], 'trend'] = current_trend
            dataframe.loc[dataframe.index[i], 'level'] = current_level

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populate the buy signal for the given dataframe
        """
        dataframe['buy'] = 0
        dataframe.loc[dataframe['signal'] == 1, 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populate the sell signal for the given dataframe
        """
        dataframe['sell'] = 0
        dataframe.loc[dataframe['signal'] == -1, 'sell'] = 1
        return dataframe