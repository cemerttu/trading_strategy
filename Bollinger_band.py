import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === STRATEGY CONFIGURATION ===
INITIAL_BALANCE = 1000
STOP_LOSS_PIPS = 20
TAKE_PROFIT_PIPS = 40
PIP_VALUE = 0.0001
SPREAD = 0.0002
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# === SIMULATE MARKET DATA ===
np.random.seed(1)
dates = pd.date_range(start="2024-01-01", periods=300, freq="H")
price = 1.1000 + np.cumsum(np.random.randn(len(dates)) * 0.0015)

df = pd.DataFrame(index=dates)
df["Open"] = price
df["High"] = df["Open"] + np.random.rand(len(df)) * 0.0015
df["Low"] = df["Open"] - np.random.rand(len(df)) * 0.0015
df["Close"] = df["Open"] + (np.random.rand(len(df)) - 0.5) * 0.0015

# === BOLLINGER BANDS ===
df["MA"] = df["Close"].rolling(window=BOLLINGER_WINDOW).mean()
df["STD"] = df["Close"].rolling(window=BOLLINGER_WINDOW).std()
df["Upper"] = df["MA"] + BOLLINGER_STD * df["STD"]
df["Lower"] = df["MA"] - BOLLINGER_STD * df["STD"]

# === SIGNAL GENERATION ===
df["Signal"] = None
position = False

for i in range(BOLLINGER_WINDOW, len(df)):
    close = df["Close"].iloc[i]
    upper = df["Upper"].iloc[i]
    lower = df["Lower"].iloc[i]

    if not position and close < lower:
        df.iloc[i, df.columns.get_loc("Signal")] = "Buy"
        position = True
    elif position and close > upper:
        df.iloc[i, df.columns.get_loc("Signal")] = "Sell"
        position = False

# === BACKTESTING ===
balance = INITIAL_BALANCE
equity = [balance]
open_trade = None
trades = []

for i in range(len(df)):
    row = df.iloc[i]
    signal = row["Signal"]
    price = row["Close"]

    if signal == "Buy" and not open_trade:
        entry = price + SPREAD
        sl = entry - STOP_LOSS_PIPS * PIP_VALUE
        tp = entry + TAKE_PROFIT_PIPS * PIP_VALUE
        open_trade = {"entry": entry, "sl": sl, "tp": tp, "entry_time": row.name}

    elif open_trade:
        low = row["Low"]
        high = row["High"]
        exit_price = None
        result = None

        if low <= open_trade["sl"]:
            exit_price = open_trade["sl"]
            result = "Loss"
        elif high >= open_trade["tp"]:
            exit_price = open_trade["tp"]
            result = "Win"

        if exit_price:
            pips = (exit_price - open_trade["entry"]) / PIP_VALUE
            balance += pips
            trades.append({
                "entry_time": open_trade["entry_time"],
                "exit_time": row.name,
                "entry": open_trade["entry"],
                "exit": exit_price,
                "pips": round(pips, 1),
                "result": result,
                "balance": round(balance, 2)
            })
            open_trade = None

    equity.append(balance)

df["Equity"] = equity[:len(df)]

# === PERFORMANCE METRICS ===
trades_df = pd.DataFrame(trades)

if not trades_df.empty:
    win_rate = 100 * len(trades_df[trades_df["result"] == "Win"]) / len(trades_df)
    net_profit = balance - INITIAL_BALANCE
    max_dd = round(np.max([INITIAL_BALANCE - e for e in df["Equity"]]), 2)
    metrics = [
        ["Win Rate (%)", f"{win_rate:.2f}%"],
        ["Net Profit ($)", f"{net_profit:.2f}"],
        ["Final Balance ($)", f"{balance:.2f}"],
        ["Max Drawdown ($)", f"{max_dd:.2f}"]
    ]
else:
    metrics = [["Note", "No trades executed"]]

# === CHARTING ===
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.25, 0.15],
                    vertical_spacing=0.03,
                    specs=[[{"type": "candlestick"}],
                           [{"type": "scatter"}],
                           [{"type": "table"}]])

# Candles and Bands
fig.add_trace(go.Candlestick(x=df.index,
                             open=df["Open"], high=df["High"],
                             low=df["Low"], close=df["Close"],
                             name="Candles"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Upper Band", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Lower Band", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["MA"], name="Moving Average", line=dict(color="blue")), row=1, col=1)

# Buy/Sell signals
buy_signals = df[df["Signal"] == "Buy"]
sell_signals = df[df["Signal"] == "Sell"]

fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals["Low"],
                         mode="markers", name="Buy", marker=dict(color="green", symbol="arrow-up", size=10)),
              row=1, col=1)

fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals["High"],
                         mode="markers", name="Sell", marker=dict(color="red", symbol="arrow-down", size=10)),
              row=1, col=1)

# Equity Curve
fig.add_trace(go.Scatter(x=df.index, y=df["Equity"], name="Equity", line=dict(color="magenta", width=2)), row=2, col=1)

# Metrics Table
fig.add_trace(go.Table(
    header=dict(values=["Metric", "Value"], fill_color="gray", font=dict(color="white", size=14), align="left"),
    cells=dict(values=[[m[0] for m in metrics], [m[1] for m in metrics]],
               fill_color="lightgray", align="left")), row=3, col=1)

fig.update_layout(
    title="📊 Bollinger Band Strategy Backtest",
    height=900,
    showlegend=True,
    template="plotly_dark"
)

fig.show()
