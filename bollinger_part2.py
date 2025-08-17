import pandas as pd
import numpy as np

def zero_lag_trend_level(df, length=34, sensitivity=2.0):
    """
    Calculate Zero-Lag Trend Level (ZLTL) similar to the Pine Script indicator.

    Parameters:
    - df: DataFrame with 'close' column (prices)
    - length: Trend length (int)
    - sensitivity: Sensitivity factor (float)

    Returns:
    - df with columns: 'zlema' (ZL trend level),
                       'trendUp' (bool),
                       'trendDn' (bool)
    """

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    src = df['close']

    ema1 = ema(src, length)
    zlema_raw = ema(src + (src - ema1) * sensitivity, length)

    smooth_length = max(2, length // 3)
    zlema = ema(zlema_raw, smooth_length)

    trendUp = zlema > zlema.shift(1)
    trendDn = zlema < zlema.shift(1)

    df = df.copy()
    df['zlema'] = zlema
    df['trendUp'] = trendUp
    df['trendDn'] = trendDn

    return df

# Example usage:
# Suppose you have a DataFrame `data` with a 'close' column

# import yfinance as yf
# data = yf.download("AAPL", period="1mo", interval="1h")
# data = zero_lag_trend_level(data)

# Now data has the zero-lag trend level and trend direction flags.
